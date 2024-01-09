//! Local field translations for uniform and adaptive Kernel Indepenent FMMs
use std::collections::HashSet;

use itertools::Itertools;
use num::Float;
use rayon::prelude::*;

use bempp_traits::{
    field::FieldTranslationData,
    fmm::{Fmm, InteractionLists, TargetTranslation},
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
    types::{EvalType, Scalar},
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use crate::{
    constants::L2L_MAX_CHUNK_SIZE,
    helpers::find_chunk_size,
    types::{
        FmmDataAdaptive, FmmDataUniform, FmmDataUniformMatrix, KiFmmLinear, KiFmmLinearMatrix,
    },
};

use rlst_blis::interface::gemm::Gemm;
use rlst_dense::{
    array::empty_array,
    rlst_dynamic_array2,
    traits::{MultIntoResize, RawAccess, RawAccessMut, UnsafeRandomAccessMut},
};

impl<T, U, V> TargetTranslation for FmmDataUniform<KiFmmLinear<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V> + std::marker::Send + std::marker::Sync,
    U: FieldTranslationData<T> + std::marker::Sync + std::marker::Send,
    V: Scalar<Real = V> + Float + Default + std::marker::Sync + std::marker::Send + Gemm,
{
    fn l2l<'a>(&self, level: u64) {
        let Some(child_targets) = self.fmm.tree().get_keys(level) else {
            return;
        };

        let nsiblings = 8;
        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

        let parent_sources: HashSet<MortonKey> =
            child_targets.iter().map(|source| source.parent()).collect();
        let mut parent_sources = parent_sources.into_iter().collect_vec();
        parent_sources.sort();
        let nparents = parent_sources.len();
        let mut parent_locals = Vec::new();
        for parent in parent_sources.iter() {
            let parent_index_pointer = *self.level_index_pointer[(level - 1) as usize]
                .get(parent)
                .unwrap();
            let parent_local = self.level_locals[(level - 1) as usize][parent_index_pointer];
            parent_locals.push(parent_local);
        }

        let mut max_chunk_size = nparents;
        if max_chunk_size > L2L_MAX_CHUNK_SIZE {
            max_chunk_size = L2L_MAX_CHUNK_SIZE
        }
        let chunk_size = find_chunk_size(nparents, max_chunk_size);

        let child_locals = &self.level_locals[level as usize];

        parent_locals
            .par_chunks_exact(chunk_size)
            .zip(child_locals.par_chunks_exact(nsiblings*chunk_size))
            .for_each(|(parent_local_pointer_chunk, child_local_pointers_chunk)| {

                let mut parent_locals = rlst_dynamic_array2!(V, [ncoeffs, chunk_size]);
                for (chunk_idx, parent_local_pointer) in parent_local_pointer_chunk.iter().enumerate().take(chunk_size) {
                    parent_locals.data_mut()[chunk_idx*ncoeffs..(chunk_idx+1)*ncoeffs].copy_from_slice(unsafe{std::slice::from_raw_parts_mut(parent_local_pointer.raw, ncoeffs)});
                }

                for i in 0..nsiblings {
                    let tmp = self.fmm.l2l[i].dot(&parent_locals).eval();

                    for j in 0..chunk_size {
                        let chunk_displacement = j*nsiblings;
                        let child_displacement = chunk_displacement + i;
                        let child_local = unsafe { std::slice::from_raw_parts_mut(child_local_pointers_chunk[child_displacement].raw, ncoeffs)};
                        child_local.iter_mut().zip(&tmp.data()[j*ncoeffs..(j+1)*ncoeffs]).for_each(|(l, t)| *l += *t);
                    }
                }
            });
    }

    fn m2p<'a>(&self) {}

    fn l2p<'a>(&self) {
        let Some(_leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

        let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
        let dim = self.fmm.kernel.space_dimension();
        let surface_size = ncoeffs * dim;

        self.leaf_upward_surfaces
            .par_chunks_exact(surface_size)
            .zip(self.leaf_locals.into_par_iter())
            .zip(&self.charge_index_pointer)
            .zip(&self.potentials_send_pointers)
            .for_each(
                |(
                    ((leaf_downward_equivalent_surface, local_ptr), charge_index_pointer),
                    potential_send_ptr,
                )| {
                    let target_coordinates = &coordinates
                        [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                    let ntargets = target_coordinates.len() / dim;

                    // Compute direct
                    if ntargets > 0 {

                        let target_coordinates = unsafe {
                            rlst_pointer_mat!['a, V, target_coordinates.as_ptr(), (ntargets, dim), (dim, 1)]
                        }.eval();

                        let local_expansion =
                            unsafe { rlst_pointer_mat!['a, V, local_ptr.raw, (ncoeffs, 1), (1, ncoeffs) ]};

                        let result = unsafe { std::slice::from_raw_parts_mut(potential_send_ptr.raw, ntargets)};

                        self.fmm.kernel.evaluate_st(
                            EvalType::Value,
                            leaf_downward_equivalent_surface,
                            target_coordinates.data(),
                            local_expansion.data(),
                            result,
                        );

                    }
                },
            );
    }

    fn p2p<'a>(&self) {
        let Some(leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let dim = self.fmm.kernel.space_dimension();

        let coordinates = self.fmm.tree().get_all_coordinates().unwrap();

        leaves
            .par_iter()
            .zip(&self.charge_index_pointer)
            .zip(&self.potentials_send_pointers)
            .for_each(|((leaf, charge_index_pointer), potential_send_pointer)| {
                let targets =
                    &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                let ntargets = targets.len() / dim;

                if ntargets > 0 {

                    let targets = unsafe {
                        rlst_pointer_mat!['a, V, targets.as_ptr(), (ntargets, dim), (dim, 1)]
                    }.eval();

                    if let Some(u_list) = self.fmm.get_u_list(leaf) {

                        let u_list_indices = u_list
                            .iter()
                            .filter_map(|k| self.fmm.tree().get_leaf_index(k));

                        let charges = u_list_indices
                            .clone()
                            .map(|&idx| {
                                let index_pointer = &self.charge_index_pointer[idx];
                                &self.charges[index_pointer.0..index_pointer.1]
                            })
                            .collect_vec();

                        let sources_coordinates = u_list_indices
                            .into_iter()
                            .map(|&idx| {
                                let index_pointer = &self.charge_index_pointer[idx];
                                &coordinates[index_pointer.0 * dim..index_pointer.1 * dim]
                            })
                            .collect_vec();

                        for (&charges, sources) in charges.iter().zip(sources_coordinates) {
                            let nsources = sources.len() / dim;

                            if nsources > 0 {

                                let sources = unsafe {
                                    rlst_pointer_mat!['a, V, sources.as_ptr(), (nsources, dim), (dim, 1)]
                                }.eval();

                                let result = unsafe { std::slice::from_raw_parts_mut(potential_send_pointer.raw, ntargets)};
                                self.fmm.kernel.evaluate_st(
                                    EvalType::Value,
                                    sources.data(),
                                    targets.data(),
                                    charges,
                                    result,
                                );
                            }
                        }
                    }
                }
            })
    }
}

impl<T, U, V> TargetTranslation for FmmDataAdaptive<KiFmmLinear<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V> + std::marker::Send + std::marker::Sync,
    U: FieldTranslationData<T> + std::marker::Sync + std::marker::Send,
    V: Scalar<Real = V> + Float + Default + std::marker::Sync + std::marker::Send + Gemm,
{
    fn l2l<'a>(&self, level: u64) {
        let Some(child_targets) = self.fmm.tree().get_keys(level) else {
            return;
        };

        let nsiblings = 8;
        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

        let mut child_targets = child_targets.iter().cloned().collect_vec();
        child_targets.sort();

        let parent_sources: HashSet<MortonKey> =
            child_targets.iter().map(|source| source.parent()).collect();
        let mut parent_sources = parent_sources.into_iter().collect_vec();
        parent_sources.sort();

        let mut parent_locals = Vec::new();
        for parent in parent_sources.iter() {
            let parent_index_pointer = *self.level_index_pointer[(level - 1) as usize]
                .get(parent)
                .unwrap();
            let parent_local = self.level_locals[(level - 1) as usize][parent_index_pointer];
            parent_locals.push(parent_local);
        }

        let mut child_locals = Vec::new();
        for child in child_targets.iter() {
            let child_index_pointer = self.level_index_pointer[level as usize].get(child).unwrap();
            let child_local = self.level_locals[level as usize][*child_index_pointer];
            child_locals.push(child_local);
        }

        parent_locals
            .into_par_iter()
            .zip(child_locals.par_chunks_exact(nsiblings))
            .for_each(|(parent_local_pointer, child_local_pointers)| {
                let parent_local = unsafe {
                    rlst_pointer_mat!['a, V, parent_local_pointer.raw, (ncoeffs, 1), (1, ncoeffs)]
                };

                for (i, child_local_pointer) in child_local_pointers.iter().enumerate().take(8) {
                    let tmp = self.fmm.l2l[i].dot(&parent_local).eval();
                    let child_local =
                        unsafe { std::slice::from_raw_parts_mut(child_local_pointer.raw, ncoeffs) };
                    child_local
                        .iter_mut()
                        .zip(tmp.data())
                        .for_each(|(c, p)| *c += *p);
                }
            });
    }

    fn m2p<'a>(&self) {
        let Some(leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let dim = self.fmm.kernel.space_dimension();
        let coordinates = self.fmm.tree().get_all_coordinates().unwrap();

        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

        leaves
            .par_iter()
            .zip(&self.charge_index_pointer)
            .zip(&self.potentials_send_pointers)
            .for_each(|((leaf, charge_index_pointer), potential_send_pointer)| {
                let targets =
                    &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                let ntargets = targets.len() / dim;

                if ntargets > 0 {
                    let targets = unsafe {
                        rlst_pointer_mat!['a, V, targets.as_ptr(), (ntargets, dim), (dim, 1)]
                    }
                    .eval();

                    if let Some(w_list) = self.fmm.get_w_list(leaf) {
                        let result = unsafe {
                            std::slice::from_raw_parts_mut(potential_send_pointer.raw, ntargets)
                        };

                        let w_list_indices = w_list
                            .iter()
                            .filter_map(|k| self.fmm.tree().get_leaf_index(k))
                            .collect_vec();

                        for &source_index in w_list_indices {
                            let multipole_send_ptr = self.leaf_multipoles[source_index];
                            let multipole = unsafe {
                                std::slice::from_raw_parts(multipole_send_ptr.raw, ncoeffs)
                            };
                            let surface = &self.leaf_upward_surfaces
                                [source_index * ncoeffs * dim..(source_index + 1) * ncoeffs * dim];

                            self.fmm.kernel.evaluate_st(
                                EvalType::Value,
                                surface,
                                targets.data(),
                                multipole,
                                result,
                            )
                        }
                    }
                }
            });
    }

    fn l2p<'a>(&self) {
        let Some(_leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);

        let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
        let dim = self.fmm.kernel.space_dimension();
        let surface_size = ncoeffs * dim;

        self.leaf_upward_surfaces
            .par_chunks_exact(surface_size)
            .zip(self.leaf_locals.into_par_iter())
            .zip(&self.charge_index_pointer)
            .zip(&self.potentials_send_pointers)
            .for_each(
                |(
                    ((leaf_downward_equivalent_surface, local_ptr), charge_index_pointer),
                    potential_send_ptr,
                )| {
                    let target_coordinates = &coordinates
                        [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                    let ntargets = target_coordinates.len() / dim;

                    let local_expansion =
                        unsafe { rlst_pointer_mat!['a, V, local_ptr.raw, (ncoeffs, 1), (1, ncoeffs) ]};

                    // Compute direct
                    if ntargets > 0 {

                        let targets = unsafe {
                            rlst_pointer_mat!['a, V, target_coordinates.as_ptr(), (ntargets, dim), (dim, 1)]
                        }.eval();

                        let result = unsafe { std::slice::from_raw_parts_mut(potential_send_ptr.raw, ntargets)};

                        self.fmm.kernel.evaluate_st(
                            EvalType::Value,
                            leaf_downward_equivalent_surface,
                            targets.data(),
                            local_expansion.data(),
                            result,
                        );

                    }
                },
            );
    }

    fn p2p<'a>(&self) {
        let Some(leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let dim = self.fmm.kernel.space_dimension();
        let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
        leaves
            .par_iter()
            .zip(&self.charge_index_pointer)
            .zip(&self.potentials_send_pointers)
            .for_each(|((leaf, charge_index_pointer), potential_send_pointer)| {
                let targets =
                    &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                let ntargets = targets.len() / dim;

                if ntargets > 0 {

                    let targets = unsafe {
                        rlst_pointer_mat!['a, V, targets.as_ptr(), (ntargets, dim), (dim, 1)]
                    }.eval();

                    if let Some(u_list) = self.fmm.get_u_list(leaf) {


                        let u_list_indices = u_list
                            .iter()
                            .filter_map(|k| self.fmm.tree().get_leaf_index(k));

                        let charges = u_list_indices
                            .clone()
                            .map(|&idx| {
                                let index_pointer = &self.charge_index_pointer[idx];
                                &self.charges[index_pointer.0..index_pointer.1]
                            })
                            .collect_vec();

                        let sources_coordinates = u_list_indices
                            .into_iter()
                            .map(|&idx| {
                                let index_pointer = &self.charge_index_pointer[idx];
                                &coordinates[index_pointer.0 * dim..index_pointer.1 * dim]
                            })
                            .collect_vec();

                        for (&charges, sources) in charges.iter().zip(sources_coordinates) {
                            let nsources = sources.len() / dim;

                            if nsources > 0 {

                                let sources = unsafe {
                                    rlst_pointer_mat!['a, V, sources.as_ptr(), (nsources, dim), (dim, 1)]
                                }.eval();

                                let result = unsafe { std::slice::from_raw_parts_mut(potential_send_pointer.raw, ntargets)};
                                self.fmm.kernel.evaluate_st(
                                    EvalType::Value,
                                    sources.data(),
                                    targets.data(),
                                    charges,
                                    result,
                                );
                            }
                        }
                    }
                }
            })
    }
}

impl<T, U, V> TargetTranslation
    for FmmDataUniformMatrix<KiFmmLinearMatrix<SingleNodeTree<V>, T, U, V>, V>
where
    T: Kernel<T = V> + ScaleInvariantKernel<T = V> + std::marker::Send + std::marker::Sync,
    U: FieldTranslationData<T> + std::marker::Sync + std::marker::Send,
    V: Scalar<Real = V> + Float + Default + std::marker::Sync + std::marker::Send,
{
    fn l2l<'a>(&self, level: u64) {
        let Some(child_targets) = self.fmm.tree().get_keys(level) else {
            return;
        };

        let nsiblings = 8;

        let parent_sources: HashSet<MortonKey> =
            child_targets.iter().map(|source| source.parent()).collect();
        let mut parent_sources = parent_sources.into_iter().collect_vec();
        parent_sources.sort();
        let nparents = parent_sources.len();
        let mut parent_locals = vec![Vec::new(); nparents];
        for (parent_idx, parent) in parent_sources.iter().enumerate() {
            for charge_vec_idx in 0..self.ncharge_vectors {
                let parent_index_pointer = *self.level_index_pointer[(level - 1) as usize]
                    .get(parent)
                    .unwrap();
                let parent_local =
                    self.level_locals[(level - 1) as usize][parent_index_pointer][charge_vec_idx];
                parent_locals[parent_idx].push(parent_local);
            }
        }

        let child_locals = &self.level_locals[level as usize];

        parent_locals
            .into_par_iter()
            .zip(child_locals.par_chunks_exact(nsiblings))
            .for_each(|(parent_local_pointers, child_locals_pointers)| {
                let mut parent_locals = rlst_dynamic_array2!(V, [self.ncoeffs, self.ncharge_vectors]);

                for (charge_vec_idx, parent_local_pointer) in parent_local_pointers
                    .iter()
                    .enumerate()
                    .take(self.ncharge_vectors)
                {
                    let tmp = unsafe {
                        std::slice::from_raw_parts(parent_local_pointer.raw, self.ncoeffs)
                    };
                    parent_locals.data_mut()
                        [charge_vec_idx * self.ncoeffs..(charge_vec_idx + 1) * self.ncoeffs]
                        .copy_from_slice(tmp);
                }

                for (i, child_locals_i) in child_locals_pointers.iter().enumerate().take(nsiblings)
                {
                    let result_i = self.fmm.l2l[i].dot(&parent_locals).eval();

                    for (j, child_locals_ij) in
                        child_locals_i.iter().enumerate().take(self.ncharge_vectors)
                    {
                        let child_locals_ij = unsafe {
                            std::slice::from_raw_parts_mut(child_locals_ij.raw, self.ncoeffs)
                        };
                        let result_ij = &result_i.data()[j * self.ncoeffs..(j + 1) * self.ncoeffs];
                        child_locals_ij
                            .iter_mut()
                            .zip(result_ij.iter())
                            .for_each(|(c, r)| *c += *r);
                    }
                }
            });
    }

    fn m2p<'a>(&self) {}

    fn l2p<'a>(&self) {
        let Some(_leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let surface_size = self.ncoeffs * self.fmm.kernel.space_dimension();
        let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
        let dim = self.fmm.kernel.space_dimension();

        for i in 0..self.ncharge_vectors {
            self.leaf_upward_surfaces
            .par_chunks_exact(surface_size)
            .zip(&self.leaf_locals)
            .zip(&self.charge_index_pointer)
            .zip(&self.potentials_send_pointers[i*self.nleaves..(i+1)*self.nleaves])
            .for_each(|(((leaf_downward_equivalent_surface, leaf_locals), charge_index_pointer), potential_send_pointer)| {

                let target_coordinates = &coordinates
                    [charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                let ntargets = target_coordinates.len() / dim;

                if ntargets > 0 {
                    let target_coordinates = unsafe {
                        rlst_pointer_mat!['a, V, target_coordinates.as_ptr(), (ntargets, dim), (dim, 1)]
                    }.eval();

                        let local_expansion_ptr = leaf_locals[i].raw;
                        let local_expansion = unsafe { std::slice::from_raw_parts(local_expansion_ptr, self.ncoeffs) };
                        let result = unsafe { std::slice::from_raw_parts_mut(potential_send_pointer.raw, ntargets)};

                        self.fmm.kernel().evaluate_st(
                                EvalType::Value,
                                leaf_downward_equivalent_surface,
                                target_coordinates.data(),
                        local_expansion,
                                result
                            );

                }
            })
        }
    }

    fn p2p<'a>(&self) {
        let Some(leaves) = self.fmm.tree().get_all_leaves() else {
            return;
        };

        let dim = self.fmm.kernel.space_dimension();
        let coordinates = self.fmm.tree().get_all_coordinates().unwrap();
        let ncoordinates = coordinates.len() / dim;

        for i in 0..self.ncharge_vectors {
            leaves
                .par_iter()
                .zip(&self.charge_index_pointer)
                .zip(&self.potentials_send_pointers[i*self.nleaves..(i+1)*self.nleaves])
                .for_each(|((leaf, charge_index_pointer), potential_send_pointer)| {
                    let targets =
                        &coordinates[charge_index_pointer.0 * dim..charge_index_pointer.1 * dim];
                    let ntargets = targets.len() / dim;

                    if ntargets > 0 {
                        if let Some(u_list) = self.fmm.get_u_list(leaf) {
                            let u_list_indices = u_list
                                .iter()
                                .filter_map(|k| self.fmm.tree().get_leaf_index(k));

                            let charge_vec_displacement = i * ncoordinates;
                            let charges = u_list_indices
                                .clone()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    &self.charges[charge_vec_displacement + index_pointer.0..charge_vec_displacement + index_pointer.1]
                                })
                                .collect_vec();

                            let sources_coordinates = u_list_indices
                                .into_iter()
                                .map(|&idx| {
                                    let index_pointer = &self.charge_index_pointer[idx];
                                    &coordinates[index_pointer.0 * dim..index_pointer.1 * dim]
                                })
                                .collect_vec();

                            for (&charges, sources) in charges.iter().zip(sources_coordinates) {
                                let nsources = sources.len() / dim;

                                if nsources > 0 {
                                    let result = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            potential_send_pointer.raw,
                                            ntargets,
                                        )
                                    };
                                    self.fmm.kernel.evaluate_st(
                                        EvalType::Value,
                                        sources,
                                        targets,
                                        charges,
                                        result,
                                    );
                                }
                            }
                        }
                    }
                })
        }
    }
}
