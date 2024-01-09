//! Multipole to Local field translations for uniform and adaptive Kernel Indepenent FMMs
use itertools::Itertools;
use num::{Complex, Float};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use bempp_field::{
    fft::Fft,
    types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm},
};

use bempp_traits::{
    field::{FieldTranslation, FieldTranslationData},
    fmm::{Fmm, InteractionLists},
    kernel::{Kernel, ScaleInvariantKernel},
    tree::Tree,
    types::EvalType,
};
use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use crate::helpers::find_chunk_size;
use crate::types::{FmmDataAdaptive, FmmDataUniform, KiFmmLinear, SendPtrMut};

use rlst::{
    algorithms::{linalg::DenseMatrixLinAlgBuilder, traits::svd::Svd},
    common::traits::*,
    dense::{rlst_pointer_mat, traits::*, Dot, MultiplyAdd, VectorContainer},
};

use super::hadamard::matmul8x8;

/// Field translations defined on uniformly refined trees.
pub mod uniform {
    use super::*;

    impl<T, U> FmmDataUniform<KiFmmLinear<SingleNodeTree<U>, T, FftFieldTranslationKiFmm<U, T>, U>, U>
    where
        T: Kernel<T = U>
            + ScaleInvariantKernel<T = U>
            + std::marker::Send
            + std::marker::Sync
            + Default,
        U: Scalar<Real = U> + Float + Default + std::marker::Send + std::marker::Sync + Fft,
        Complex<U>: Scalar,
        U: MultiplyAdd<
            U,
            VectorContainer<U>,
            VectorContainer<U>,
            VectorContainer<U>,
            Dynamic,
            Dynamic,
            Dynamic,
        >,
    {
        fn displacements(&self, level: u64) -> Vec<Vec<usize>> {
            let nneighbors = 26;
            let nsiblings = 8;

            let sources = self.fmm.tree().get_keys(level).unwrap();

            let parents: HashSet<MortonKey> =
                sources.iter().map(|source| source.parent()).collect();
            let mut parents = parents.into_iter().collect_vec();
            parents.sort();
            let nparents = parents.len();

            let mut result = vec![Vec::new(); nneighbors];

            let parent_neighbors = parents
                .iter()
                .map(|parent| parent.all_neighbors())
                .collect_vec();

            for i in 0..nneighbors {
                for all_neighbors in parent_neighbors.iter().take(nparents) {
                    // First check if neighbor exists is in the bounds of the tree
                    if let Some(neighbor) = all_neighbors[i] {
                        let first_child = neighbor.first_child();
                        // Then need to check if first child exists in the tree
                        if let Some(first_child_index) =
                            self.level_index_pointer[level as usize].get(&first_child)
                        {
                            result[i].push(*first_child_index)
                        } else {
                            result[i].push(nparents * nsiblings)
                        }
                    } else {
                        result[i].push(nparents * nsiblings);
                    }
                }
            }

            result
        }
    }

    impl<T, U> FieldTranslation<U>
        for FmmDataUniform<KiFmmLinear<SingleNodeTree<U>, T, FftFieldTranslationKiFmm<U, T>, U>, U>
    where
        T: Kernel<T = U>
            + ScaleInvariantKernel<T = U>
            + std::marker::Send
            + std::marker::Sync
            + Default,
        U: Scalar<Real = U> + Float + Default + std::marker::Send + std::marker::Sync + Fft,
        Complex<U>: Scalar,
        U: MultiplyAdd<
            U,
            VectorContainer<U>,
            VectorContainer<U>,
            VectorContainer<U>,
            Dynamic,
            Dynamic,
            Dynamic,
        >,
    {
        fn p2l(&self, _level: u64) {}

        fn m2l<'a>(&self, level: u64) {
            let Some(targets) = self.fmm.tree().get_keys(level) else {
                return;
            };

            let n = 2 * self.fmm.order - 1;
            let npad = n + 1;

            // let nparents = self.fmm.tree().get_keys(level - 1).unwrap().len();
            let parents: HashSet<MortonKey> = targets
                .iter()
                .map(|target: &MortonKey| target.parent())
                .collect();
            let mut parents = parents.into_iter().collect_vec();
            parents.sort();
            let nparents = parents.len();

            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
            let nsiblings = 8;
            let nzeros = 8;
            let size = npad * npad * npad;
            let size_real = npad * npad * (npad / 2 + 1);
            let all_displacements = self.displacements(level);

            let ntargets = targets.len();
            let min = &targets[0];
            let max = &targets[ntargets - 1];
            let min_idx = self.fmm.tree().key_to_index.get(min).unwrap();
            let max_idx = self.fmm.tree().key_to_index.get(max).unwrap();

            let multipoles = &self.multipoles[min_idx * ncoeffs..(max_idx + 1) * ncoeffs];

            ////////////////////////////////////////////////////////////////////////////////////
            // Pre-process to setup data structures for M2L kernel
            ////////////////////////////////////////////////////////////////////////////////////

            // Allocation of FFT of multipoles on convolution grid, in frequency order
            let mut signals_hat_f_buffer = vec![U::zero(); size_real * (ntargets + nzeros) * 2];
            let signals_hat_f: &mut [Complex<U>];
            unsafe {
                let ptr = signals_hat_f_buffer.as_mut_ptr() as *mut Complex<U>;
                signals_hat_f =
                    std::slice::from_raw_parts_mut(ptr, size_real * (ntargets + nzeros));
            }

            // A thread safe mutable pointer for saving to this vector
            let raw = signals_hat_f.as_mut_ptr();
            let signals_hat_f_ptr = SendPtrMut { raw };

            // Pre processing chunk size, in terms of number of parents
            // let chunk_size = 1;
            let max_chunk_size;
            if level == 2 {
                max_chunk_size = 8
            } else if level == 3 {
                max_chunk_size = 64
            } else {
                max_chunk_size = 128
            }
            let chunk_size = find_chunk_size(nparents, max_chunk_size);

            // Pre-processing to find FFT
            multipoles
                .par_chunks_exact(ncoeffs * nsiblings * chunk_size)
                .enumerate()
                .for_each(|(i, multipole_chunk)| {
                    // Place Signal on convolution grid
                    let mut signal_chunk = vec![U::zero(); size * nsiblings * chunk_size];

                    for i in 0..nsiblings * chunk_size {
                        let multipole = &multipole_chunk[i * ncoeffs..(i + 1) * ncoeffs];
                        let signal = &mut signal_chunk[i * size..(i + 1) * size];
                        for (surf_idx, &conv_idx) in
                            self.fmm.m2l.surf_to_conv_map.iter().enumerate()
                        {
                            signal[conv_idx] = multipole[surf_idx]
                        }
                    }

                    // Temporary buffer to hold results of FFT
                    let signal_hat_chunk_buffer =
                        vec![U::zero(); size_real * nsiblings * chunk_size * 2];
                    let signal_hat_chunk_c;
                    unsafe {
                        let ptr = signal_hat_chunk_buffer.as_ptr() as *mut Complex<U>;
                        signal_hat_chunk_c =
                            std::slice::from_raw_parts_mut(ptr, size_real * nsiblings * chunk_size);
                    }

                    U::rfft3_fftw_slice(&mut signal_chunk, signal_hat_chunk_c, &[npad, npad, npad]);

                    // Re-order the temporary buffer into frequency order before flushing to main memory
                    let signal_hat_chunk_f_buffer =
                        vec![U::zero(); size_real * nsiblings * chunk_size * 2];
                    let signal_hat_chunk_f_c;
                    unsafe {
                        let ptr = signal_hat_chunk_f_buffer.as_ptr() as *mut Complex<U>;
                        signal_hat_chunk_f_c =
                            std::slice::from_raw_parts_mut(ptr, size_real * nsiblings * chunk_size);
                    }

                    for i in 0..size_real {
                        for j in 0..nsiblings * chunk_size {
                            signal_hat_chunk_f_c[nsiblings * chunk_size * i + j] =
                                signal_hat_chunk_c[size_real * j + i]
                        }
                    }

                    // Storing the results of the FFT in frequency order
                    unsafe {
                        let sibling_offset = i * nsiblings * chunk_size;

                        // Pointer to storage buffer for frequency ordered FFT of signals
                        let ptr = signals_hat_f_ptr;

                        for i in 0..size_real {
                            let frequency_offset = i * (ntargets + nzeros);

                            // Head of buffer for each frequency
                            let head = ptr.raw.add(frequency_offset).add(sibling_offset);

                            let signal_hat_f_chunk =
                                std::slice::from_raw_parts_mut(head, nsiblings * chunk_size);

                            // Store results for this frequency for this sibling set chunk
                            let results_i = &signal_hat_chunk_f_c
                                [i * nsiblings * chunk_size..(i + 1) * nsiblings * chunk_size];

                            signal_hat_f_chunk
                                .iter_mut()
                                .zip(results_i)
                                .for_each(|(c, r)| *c += *r);
                        }
                    }
                });

            // Allocate check potentials (implicitly in frequency order)
            let mut check_potentials_hat_f_buffer = vec![U::zero(); 2 * size_real * ntargets];
            let check_potentials_hat_f: &mut [Complex<U>];
            unsafe {
                let ptr = check_potentials_hat_f_buffer.as_mut_ptr() as *mut Complex<U>;
                check_potentials_hat_f = std::slice::from_raw_parts_mut(ptr, size_real * ntargets);
            }

            ////////////////////////////////////////////////////////////////////////////////////
            // M2L Kernel
            ////////////////////////////////////////////////////////////////////////////////////
            let scale = Complex::from(self.m2l_scale(level) * self.fmm.kernel.scale(level));
            let kernel_data_f = &self.fmm.m2l.operator_data.kernel_data_f;

            (0..size_real)
                .into_par_iter()
                .zip(signals_hat_f.par_chunks_exact(ntargets + nzeros))
                .zip(check_potentials_hat_f.par_chunks_exact_mut(ntargets))
                .for_each(|((freq, signal_hat_f), check_potential_hat_f)| {
                    (0..nparents).step_by(chunk_size).for_each(|chunk_start| {
                        let chunk_end = std::cmp::min(chunk_start + chunk_size, nparents);

                        let save_locations = &mut check_potential_hat_f
                            [chunk_start * nsiblings..chunk_end * nsiblings];

                        for (i, kernel_f) in kernel_data_f.iter().enumerate().take(26) {
                            let frequency_offset = 64 * freq;
                            let k_f = &kernel_f[frequency_offset..(frequency_offset + 64)].to_vec();

                            // Lookup signals
                            let displacements = &all_displacements[i][chunk_start..chunk_end];

                            for j in 0..(chunk_end - chunk_start) {
                                let displacement = displacements[j];
                                let s_f = &signal_hat_f[displacement..displacement + nsiblings];

                                matmul8x8(
                                    k_f,
                                    s_f,
                                    &mut save_locations[j * nsiblings..(j + 1) * nsiblings],
                                    scale,
                                )
                            }
                        }
                    });
                });

            ////////////////////////////////////////////////////////////////////////////////////
            // Post processing to find local expansions from check potentials
            ////////////////////////////////////////////////////////////////////////////////////

            // Get check potentials back into target order from frequency order
            let mut check_potential_hat = vec![U::zero(); size_real * ntargets * 2];
            let mut check_potential = vec![U::zero(); size * ntargets];
            let check_potential_hat_c;
            unsafe {
                let ptr = check_potential_hat.as_mut_ptr() as *mut Complex<U>;
                check_potential_hat_c = std::slice::from_raw_parts_mut(ptr, size_real * ntargets)
            }

            check_potential_hat_c
                .par_chunks_exact_mut(size_real)
                .enumerate()
                .for_each(|(i, check_potential_hat_chunk)| {
                    // Lookup all frequencies for this target box
                    for j in 0..size_real {
                        check_potential_hat_chunk[j] = check_potentials_hat_f[j * ntargets + i]
                    }
                });

            // Compute inverse FFT
            U::irfft3_fftw_par_slice(
                check_potential_hat_c,
                &mut check_potential,
                &[npad, npad, npad],
            );

            // TODO: Experiment with chunk size for post processing
            check_potential
                .par_chunks_exact(nsiblings * size)
                .zip(self.level_locals[level as usize].par_chunks_exact(nsiblings))
                .for_each(|(check_potential_chunk, local_ptrs)| {
                    // Map to surface grid
                    let mut potential_buffer = vec![U::zero(); ncoeffs * nsiblings];
                    for i in 0..nsiblings {
                        let tmp = &mut potential_buffer[i * ncoeffs..(i + 1) * ncoeffs];
                        let check_potential = &check_potential_chunk[i * size..(i + 1) * size];
                        for (surf_idx, &conv_idx) in self.fmm.m2l.conv_to_surf_map.iter().enumerate() {
                            tmp[surf_idx] = check_potential[conv_idx];
                        }
                    }

                    // Can now find local expansion coefficients
                    let potential_chunk = unsafe {
                        rlst_pointer_mat!['a, U, potential_buffer.as_ptr(), (ncoeffs, nsiblings), (1, ncoeffs)]
                    };

                    let local_chunk = self
                        .fmm
                        .dc2e_inv_1
                        .dot(&self.fmm.dc2e_inv_2.dot(&potential_chunk))
                        .eval();


                    local_chunk
                        .data()
                        .chunks_exact(ncoeffs)
                        .zip(local_ptrs)
                        .for_each(|(result, local)| {
                            let local = unsafe { std::slice::from_raw_parts_mut(local.raw, ncoeffs) };
                            local.iter_mut().zip(result).for_each(|(l, r)| *l += *r);
                        });
                });
        }

        fn m2l_scale(&self, level: u64) -> U {
            if level < 2 {
                panic!("M2L only perfomed on level 2 and below")
            }

            if level == 2 {
                U::from(1. / 2.).unwrap()
            } else {
                let two = U::from(2.0).unwrap();
                Scalar::powf(two, U::from(level - 3).unwrap())
            }
        }
    }

    /// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
    impl<T, U> FieldTranslation<U>
        for FmmDataUniform<KiFmmLinear<SingleNodeTree<U>, T, SvdFieldTranslationKiFmm<U, T>, U>, U>
    where
        T: Kernel<T = U>
            + ScaleInvariantKernel<T = U>
            + std::marker::Send
            + std::marker::Sync
            + Default,
        DenseMatrixLinAlgBuilder<U>: Svd,
        U: Scalar<Real = U>,
        U: Float
            + Default
            + MultiplyAdd<
                U,
                VectorContainer<U>,
                VectorContainer<U>,
                VectorContainer<U>,
                Dynamic,
                Dynamic,
                Dynamic,
            >,
        U: std::marker::Send + std::marker::Sync + Default,
    {
        fn p2l(&self, _level: u64) {}

        fn m2l<'a>(&self, level: u64) {
            let Some(sources) = self.fmm.tree().get_keys(level) else {
                return;
            };

            let nsources = sources.len();

            let mut all_displacements = vec![vec![-1i64; nsources]; 316];

            // Need to identify all save locations in a pre-processing step.
            for (j, source) in sources.iter().enumerate() {
                let v_list = source
                    .parent()
                    .neighbors()
                    .iter()
                    .flat_map(|pn| pn.children())
                    .filter(|pnc| {
                        !source.is_adjacent(pnc) && self.fmm.tree().get_all_keys_set().contains(pnc)
                    })
                    .collect_vec();

                let transfer_vectors = v_list
                    .iter()
                    .map(|target| target.find_transfer_vector(source))
                    .collect_vec();

                let mut transfer_vectors_map = HashMap::new();
                for (i, v) in transfer_vectors.iter().enumerate() {
                    transfer_vectors_map.insert(v, i);
                }

                let transfer_vectors_set: HashSet<_> = transfer_vectors.iter().collect();

                for (i, tv) in self.fmm.m2l.transfer_vectors.iter().enumerate() {
                    if transfer_vectors_set.contains(&tv.hash) {
                        let target = &v_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                        let target_index = self.level_index_pointer[level as usize]
                            .get(target)
                            .unwrap();
                        all_displacements[i][j] = *target_index as i64;
                    }
                }
            }

            // Interpret multipoles as a matrix
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
            let multipoles = unsafe {
                rlst_pointer_mat!['a, U, self.level_multipoles[level as usize][0].raw, (ncoeffs, nsources), (1, ncoeffs)]
            };

            let (nrows, _) = self.fmm.m2l.operator_data.c.shape();
            let c_dim = (nrows, self.fmm.m2l.k);

            let mut compressed_multipoles = self.fmm.m2l.operator_data.st_block.dot(&multipoles);

            compressed_multipoles
                .data_mut()
                .iter_mut()
                .for_each(|d| *d *= self.fmm.kernel.scale(level) * self.m2l_scale(level));

            (0..316).into_par_iter().for_each(|c_idx| {
                let top_left = (0, c_idx * self.fmm.m2l.k);
                let c_sub = self.fmm.m2l.operator_data.c.block(top_left, c_dim);

                let locals = self.fmm.dc2e_inv_1.dot(
                    &self.fmm.dc2e_inv_2.dot(
                        &self
                            .fmm
                            .m2l
                            .operator_data
                            .u
                            .dot(&c_sub.dot(&compressed_multipoles)),
                    ),
                );

                let displacements = &all_displacements[c_idx];

                for (result_idx, &save_idx) in displacements.iter().enumerate() {
                    if save_idx > -1 {
                        let save_idx = save_idx as usize;
                        let local_ptr = self.level_locals[level as usize][save_idx].raw;
                        let local = unsafe { std::slice::from_raw_parts_mut(local_ptr, ncoeffs) };

                        let res = &locals.data()[result_idx * ncoeffs..(result_idx + 1) * ncoeffs];
                        local.iter_mut().zip(res).for_each(|(l, r)| *l += *r);
                    }
                }
            })
        }

        fn m2l_scale(&self, level: u64) -> U {
            if level < 2 {
                panic!("M2L only perfomed on level 2 and below")
            }

            if level == 2 {
                U::from(1. / 2.).unwrap()
            } else {
                let two = U::from(2.0).unwrap();
                Scalar::powf(two, U::from(level - 3).unwrap())
            }
        }
    }

    /// Field translations for uniformly refined trees that take matrix input for charges.
    pub mod matrix {
        use crate::types::{FmmDataUniformMatrix, KiFmmLinearMatrix};

        use super::*;

        /// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
        impl<T, U> FieldTranslation<U>
            for FmmDataUniformMatrix<
                KiFmmLinearMatrix<SingleNodeTree<U>, T, SvdFieldTranslationKiFmm<U, T>, U>,
                U,
            >
        where
            T: Kernel<T = U>
                + ScaleInvariantKernel<T = U>
                + std::marker::Send
                + std::marker::Sync
                + Default,
            DenseMatrixLinAlgBuilder<U>: Svd,
            U: Scalar<Real = U>,
            U: Float
                + Default
                + MultiplyAdd<
                    U,
                    VectorContainer<U>,
                    VectorContainer<U>,
                    VectorContainer<U>,
                    Dynamic,
                    Dynamic,
                    Dynamic,
                >,
            U: std::marker::Send + std::marker::Sync + Default,
        {
            fn p2l(&self, _level: u64) {}

            fn m2l<'a>(&self, level: u64) {
                let Some(sources) = self.fmm.tree().get_keys(level) else { return; };

                let nsources = sources.len();

                let mut all_displacements = vec![vec![-1i64; nsources]; 316];

                // Need to identify all save locations in a pre-processing step.
                for (j, source) in sources.iter().enumerate() {
                    let v_list = source
                        .parent()
                        .neighbors()
                        .iter()
                        .flat_map(|pn| pn.children())
                        .filter(|pnc| {
                            !source.is_adjacent(pnc)
                                && self.fmm.tree().get_all_keys_set().contains(pnc)
                        })
                        .collect_vec();

                    let transfer_vectors = v_list
                        .iter()
                        .map(|target| target.find_transfer_vector(source))
                        .collect_vec();

                    let mut transfer_vectors_map = HashMap::new();
                    for (i, v) in transfer_vectors.iter().enumerate() {
                        transfer_vectors_map.insert(v, i);
                    }

                    let transfer_vectors_set: HashSet<_> = transfer_vectors.iter().collect();

                    for (i, tv) in self.fmm.m2l.transfer_vectors.iter().enumerate() {
                        if transfer_vectors_set.contains(&tv.hash) {
                            let target = &v_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                            let target_index = self.level_index_pointer[level as usize]
                                .get(target)
                                .unwrap();
                            all_displacements[i][j] = *target_index as i64;
                        }
                    }
                }

                // Interpret multipoles as a matrix
                let multipoles = unsafe {
                    rlst_pointer_mat!['a, U, self.level_multipoles[level as usize][0][0].raw, (self.ncoeffs, nsources * self.ncharge_vectors), (1, self.ncoeffs)]
                };

                let (nrows, _) = self.fmm.m2l.operator_data.c.shape();
                let c_dim = (nrows, self.fmm.m2l.k);

                let mut compressed_multipoles =
                    self.fmm.m2l.operator_data.st_block.dot(&multipoles);

                compressed_multipoles
                    .data_mut()
                    .iter_mut()
                    .for_each(|d| *d *= self.fmm.kernel.scale(level) * self.m2l_scale(level));

                (0..316).into_par_iter().for_each(|c_idx| {
                    let top_left = (0, c_idx * self.fmm.m2l.k);
                    let c_sub = self.fmm.m2l.operator_data.c.block(top_left, c_dim);

                    let locals = self.fmm.dc2e_inv_1.dot(
                        &self.fmm.dc2e_inv_2.dot(
                            &self
                                .fmm
                                .m2l
                                .operator_data
                                .u
                                .dot(&c_sub.dot(&compressed_multipoles)),
                        ),
                    );

                    let displacements = &all_displacements[c_idx];

                    for (result_idx, &save_idx) in displacements.iter().enumerate() {
                        if save_idx > -1 {
                            let save_idx = save_idx as usize;
                            for charge_vec_idx in 0..self.ncharge_vectors {
                                let local_ptr =
                                    self.level_locals[level as usize][save_idx][charge_vec_idx].raw;
                                let local = unsafe {
                                    std::slice::from_raw_parts_mut(local_ptr, self.ncoeffs)
                                };

                                let key_displacement =
                                    result_idx * self.ncoeffs * self.ncharge_vectors;
                                let charge_vec_displacement = charge_vec_idx * self.ncoeffs;

                                let result = &locals.data()[key_displacement
                                    + charge_vec_displacement
                                    ..key_displacement + charge_vec_displacement + self.ncoeffs];
                                local.iter_mut().zip(result).for_each(|(l, r)| *l += *r);
                            }
                        }
                    }
                })
            }

            fn m2l_scale(&self, level: u64) -> U {
                if level < 2 {
                    panic!("M2L only perfomed on level 2 and below")
                }

                if level == 2 {
                    U::from(1. / 2.).unwrap()
                } else {
                    let two = U::from(2.0).unwrap();
                    Scalar::powf(two, U::from(level - 3).unwrap())
                }
            }
        }
    }
}

/// Field translations defined on adaptively refined
pub mod adaptive {
    use super::*;

    impl<T, U> FmmDataAdaptive<KiFmmLinear<SingleNodeTree<U>, T, FftFieldTranslationKiFmm<U, T>, U>, U>
    where
        T: Kernel<T = U>
            + ScaleInvariantKernel<T = U>
            + std::marker::Send
            + std::marker::Sync
            + Default,
        U: Scalar<Real = U> + Float + Default + std::marker::Send + std::marker::Sync + Fft,
        Complex<U>: Scalar,
        U: MultiplyAdd<
            U,
            VectorContainer<U>,
            VectorContainer<U>,
            VectorContainer<U>,
            Dynamic,
            Dynamic,
            Dynamic,
        >,
    {
        fn displacements(&self, level: u64) -> Vec<Vec<usize>> {
            let nneighbors = 26;
            let nsiblings = 8;

            let sources = self.fmm.tree().get_keys(level).unwrap();

            let parents: HashSet<MortonKey> =
                sources.iter().map(|source| source.parent()).collect();
            let mut parents = parents.into_iter().collect_vec();
            parents.sort();
            let nparents = parents.len();

            let mut result = vec![Vec::new(); nneighbors];

            let parent_neighbors = parents
                .iter()
                .map(|parent| parent.all_neighbors())
                .collect_vec();

            for i in 0..nneighbors {
                for all_neighbors in parent_neighbors.iter().take(nparents) {
                    // First check if neighbor exists is in the bounds of the tree
                    if let Some(neighbor) = all_neighbors[i] {
                        let first_child = neighbor.first_child();
                        // Then need to check if first child exists in the tree
                        if let Some(first_child_index) =
                            self.level_index_pointer[level as usize].get(&first_child)
                        {
                            result[i].push(*first_child_index)
                        } else {
                            result[i].push(nparents * nsiblings)
                        }
                    } else {
                        result[i].push(nparents * nsiblings);
                    }
                }
            }

            result
        }
    }

    impl<T, U> FieldTranslation<U>
        for FmmDataAdaptive<KiFmmLinear<SingleNodeTree<U>, T, FftFieldTranslationKiFmm<U, T>, U>, U>
    where
        T: Kernel<T = U>
            + ScaleInvariantKernel<T = U>
            + std::marker::Send
            + std::marker::Sync
            + Default,
        U: Scalar<Real = U> + Float + Default + std::marker::Send + std::marker::Sync + Fft,
        Complex<U>: Scalar,
        U: MultiplyAdd<
            U,
            VectorContainer<U>,
            VectorContainer<U>,
            VectorContainer<U>,
            Dynamic,
            Dynamic,
            Dynamic,
        >,
    {
        fn p2l<'a>(&self, level: u64) {
            let Some(targets) = self.fmm.tree().get_keys(level) else {
                return;
            };

            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
            let dim = self.fmm.kernel().space_dimension();
            let surface_size = ncoeffs * dim;
            let min_idx = self.fmm.tree().key_to_index.get(&targets[0]).unwrap();
            let max_idx = self
                .fmm
                .tree()
                .key_to_index
                .get(targets.last().unwrap())
                .unwrap();
            let downward_surfaces =
                &self.downward_surfaces[min_idx * surface_size..(max_idx + 1) * surface_size];
            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();

            let ntargets = targets.len();
            let mut check_potentials = vec![U::zero(); ncoeffs * ntargets];

            // 1. Compute check potentials from x list of each target
            targets
                .par_iter()
                .zip(downward_surfaces.par_chunks_exact(surface_size))
                .zip(check_potentials.par_chunks_exact_mut(ncoeffs))
                .for_each(|((target, downward_surface), check_potential)| {
                    // Find check potential
                    if let Some(x_list) = self.fmm.get_x_list(target) {
                        let x_list_indices = x_list
                            .iter()
                            .filter_map(|k| self.fmm.tree().get_leaf_index(k));
                        let charges = x_list_indices
                            .clone()
                            .map(|&idx| {
                                let index_pointer = &self.charge_index_pointer[idx];
                                &self.charges[index_pointer.0..index_pointer.1]
                            })
                            .collect_vec();

                        let sources_coordinates = x_list_indices
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
                                    rlst_pointer_mat!['a, U, sources.as_ptr(), (nsources, dim), (dim, 1)]
                                }
                                .eval();

                                self.fmm.kernel.evaluate_st(
                                    EvalType::Value,
                                    sources.data(),
                                    downward_surface,
                                    charges,
                                    check_potential,
                                );
                            }
                        }
                    }
                });

            // 2. Compute local expansion from check potential
            self.level_locals[level as usize]
                .par_iter()
                .zip(check_potentials.par_chunks_exact(ncoeffs))
                .for_each(|(local_ptr, check_potential)| {
                    let target_local =
                        unsafe { std::slice::from_raw_parts_mut(local_ptr.raw, ncoeffs) };
                    let check_potential = unsafe {
                        rlst_pointer_mat!['a, U, check_potential.as_ptr(), (ncoeffs, 1), (1, ncoeffs)]
                    };

                    let scale = self.fmm.kernel().scale(level);
                    let mut tmp = self
                        .fmm
                        .dc2e_inv_1
                        .dot(&self.fmm.dc2e_inv_2.dot(&check_potential));
                    tmp.data_mut().iter_mut().for_each(|val| *val *= scale);

                    target_local
                        .iter_mut()
                        .zip(tmp.data())
                        .for_each(|(r, &t)| *r += t);
                });
        }

        fn m2l<'a>(&self, level: u64) {
            let Some(targets) = self.fmm.tree().get_keys(level) else {
                return;
            };

            let n = 2 * self.fmm.order - 1;
            let npad = n + 1;

            let parents: HashSet<MortonKey> = targets
                .iter()
                .map(|target: &MortonKey| target.parent())
                .collect();
            let mut parents = parents.into_iter().collect_vec();
            parents.sort();
            let nparents = parents.len();

            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
            let nsiblings = 8;
            let nzeros = 8;
            let size = npad * npad * npad;
            let size_real = npad * npad * (npad / 2 + 1);
            let all_displacements = self.displacements(level);

            let ntargets = targets.len();
            let min = &targets[0];
            let max = &targets[ntargets - 1];

            let min_idx = self.fmm.tree().key_to_index.get(min).unwrap();
            let max_idx = self.fmm.tree().key_to_index.get(max).unwrap();

            let multipoles = &self.multipoles[min_idx * ncoeffs..(max_idx + 1) * ncoeffs];

            ////////////////////////////////////////////////////////////////////////////////////
            // Pre-process to setup data structures for M2L kernel
            ////////////////////////////////////////////////////////////////////////////////////

            // Allocation of FFT of multipoles on convolution grid, in frequency order
            let mut signals_hat_f_buffer = vec![U::zero(); size_real * (ntargets + nzeros) * 2];
            let signals_hat_f: &mut [Complex<U>];
            unsafe {
                let ptr = signals_hat_f_buffer.as_mut_ptr() as *mut Complex<U>;
                signals_hat_f =
                    std::slice::from_raw_parts_mut(ptr, size_real * (ntargets + nzeros));
            }

            // A thread safe mutable pointer for saving to this vector
            let raw = signals_hat_f.as_mut_ptr();
            let signals_hat_f_ptr = SendPtrMut { raw };

            // Pre processing chunk size, in terms of number of parents
            // let chunk_size = 1;
            let max_chunk_size;
            if level == 2 {
                max_chunk_size = 8
            } else if level == 3 {
                max_chunk_size = 64
            } else {
                max_chunk_size = 128
            }
            let chunk_size = find_chunk_size(nparents, max_chunk_size);

            // Pre-processing to find FFT
            multipoles
                .par_chunks_exact(ncoeffs * nsiblings * chunk_size)
                // .par_chunks_exact(nsiblings * chunk_size)
                .enumerate()
                .for_each(|(i, multipole_chunk)| {
                    // Place Signal on convolution grid
                    let mut signal_chunk = vec![U::zero(); size * nsiblings * chunk_size];

                    for i in 0..nsiblings * chunk_size {
                        let multipole = &multipole_chunk[i * ncoeffs..(i + 1) * ncoeffs];
                        // let multipole = unsafe { std::slice::from_raw_parts(multipole_chunk[i].raw, ncoeffs) };
                        let signal = &mut signal_chunk[i * size..(i + 1) * size];
                        for (surf_idx, &conv_idx) in
                            self.fmm.m2l.surf_to_conv_map.iter().enumerate()
                        {
                            signal[conv_idx] = multipole[surf_idx]
                        }
                    }

                    // Temporary buffer to hold results of FFT
                    let signal_hat_chunk_buffer =
                        vec![U::zero(); size_real * nsiblings * chunk_size * 2];
                    let signal_hat_chunk_c;
                    unsafe {
                        let ptr = signal_hat_chunk_buffer.as_ptr() as *mut Complex<U>;
                        signal_hat_chunk_c =
                            std::slice::from_raw_parts_mut(ptr, size_real * nsiblings * chunk_size);
                    }

                    U::rfft3_fftw_slice(&mut signal_chunk, signal_hat_chunk_c, &[npad, npad, npad]);

                    // Re-order the temporary buffer into frequency order before flushing to main memory
                    let signal_hat_chunk_f_buffer =
                        vec![U::zero(); size_real * nsiblings * chunk_size * 2];
                    let signal_hat_chunk_f_c;
                    unsafe {
                        let ptr = signal_hat_chunk_f_buffer.as_ptr() as *mut Complex<U>;
                        signal_hat_chunk_f_c =
                            std::slice::from_raw_parts_mut(ptr, size_real * nsiblings * chunk_size);
                    }

                    for i in 0..size_real {
                        for j in 0..nsiblings * chunk_size {
                            signal_hat_chunk_f_c[nsiblings * chunk_size * i + j] =
                                signal_hat_chunk_c[size_real * j + i]
                        }
                    }

                    // Storing the results of the FFT in frequency order
                    unsafe {
                        let sibling_offset = i * nsiblings * chunk_size;

                        // Pointer to storage buffer for frequency ordered FFT of signals
                        let ptr = signals_hat_f_ptr;

                        for i in 0..size_real {
                            let frequency_offset = i * (ntargets + nzeros);

                            // Head of buffer for each frequency
                            let head = ptr.raw.add(frequency_offset).add(sibling_offset);

                            let signal_hat_f_chunk =
                                std::slice::from_raw_parts_mut(head, nsiblings * chunk_size);

                            // Store results for this frequency for this sibling set chunk
                            let results_i = &signal_hat_chunk_f_c
                                [i * nsiblings * chunk_size..(i + 1) * nsiblings * chunk_size];

                            signal_hat_f_chunk
                                .iter_mut()
                                .zip(results_i)
                                .for_each(|(c, r)| *c += *r);
                        }
                    }
                });

            // Allocate check potentials (implicitly in frequency order)
            let mut check_potentials_hat_f_buffer = vec![U::zero(); 2 * size_real * ntargets];
            let check_potentials_hat_f: &mut [Complex<U>];
            unsafe {
                let ptr = check_potentials_hat_f_buffer.as_mut_ptr() as *mut Complex<U>;
                check_potentials_hat_f = std::slice::from_raw_parts_mut(ptr, size_real * ntargets);
            }

            ////////////////////////////////////////////////////////////////////////////////////
            // M2L Kernel
            ////////////////////////////////////////////////////////////////////////////////////
            let scale = Complex::from(self.m2l_scale(level) * self.fmm.kernel.scale(level));
            let kernel_data_f = &self.fmm.m2l.operator_data.kernel_data_f;

            (0..size_real)
                .into_par_iter()
                .zip(signals_hat_f.par_chunks_exact(ntargets + nzeros))
                .zip(check_potentials_hat_f.par_chunks_exact_mut(ntargets))
                .for_each(|((freq, signal_hat_f), check_potential_hat_f)| {
                    (0..nparents).step_by(chunk_size).for_each(|chunk_start| {
                        let chunk_end = std::cmp::min(chunk_start + chunk_size, nparents);

                        let save_locations = &mut check_potential_hat_f
                            [chunk_start * nsiblings..chunk_end * nsiblings];

                        for (i, kernel_f) in kernel_data_f.iter().enumerate().take(26) {
                            let frequency_offset = 64 * freq;
                            let k_f = &kernel_f[frequency_offset..(frequency_offset + 64)].to_vec();

                            // Lookup signals
                            let displacements = &all_displacements[i][chunk_start..chunk_end];

                            for j in 0..(chunk_end - chunk_start) {
                                let displacement = displacements[j];
                                let s_f = &signal_hat_f[displacement..displacement + nsiblings];

                                matmul8x8(
                                    k_f,
                                    s_f,
                                    &mut save_locations[j * nsiblings..(j + 1) * nsiblings],
                                    scale,
                                )
                            }
                        }
                    });
                });

            ////////////////////////////////////////////////////////////////////////////////////
            // Post processing to find local expansions from check potentials
            ////////////////////////////////////////////////////////////////////////////////////

            // Get check potentials back into target order from frequency order
            let mut check_potential_hat = vec![U::zero(); size_real * ntargets * 2];
            let mut check_potential = vec![U::zero(); size * ntargets];
            let check_potential_hat_c;
            unsafe {
                let ptr = check_potential_hat.as_mut_ptr() as *mut Complex<U>;
                check_potential_hat_c = std::slice::from_raw_parts_mut(ptr, size_real * ntargets)
            }

            check_potential_hat_c
                .par_chunks_exact_mut(size_real)
                .enumerate()
                .for_each(|(i, check_potential_hat_chunk)| {
                    // Lookup all frequencies for this target box
                    for j in 0..size_real {
                        check_potential_hat_chunk[j] = check_potentials_hat_f[j * ntargets + i]
                    }
                });

            // Compute inverse FFT
            U::irfft3_fftw_par_slice(
                check_potential_hat_c,
                &mut check_potential,
                &[npad, npad, npad],
            );

            // TODO: Experiment with chunk size for post processing
            check_potential
                .par_chunks_exact(nsiblings * size)
                .zip(self.level_locals[level as usize].par_chunks_exact(nsiblings))
                .for_each(|(check_potential_chunk, local_ptrs)| {
                    // Map to surface grid
                    let mut potential_buffer = vec![U::zero(); ncoeffs * nsiblings];
                    for i in 0..nsiblings {
                        let tmp = &mut potential_buffer[i * ncoeffs..(i + 1) * ncoeffs];
                        let check_potential = &check_potential_chunk[i * size..(i + 1) * size];
                        for (surf_idx, &conv_idx) in self.fmm.m2l.conv_to_surf_map.iter().enumerate() {
                            tmp[surf_idx] = check_potential[conv_idx];
                        }
                    }

                    // Can now find local expansion coefficients
                    let potential_chunk = unsafe {
                        rlst_pointer_mat!['a, U, potential_buffer.as_ptr(), (ncoeffs, nsiblings), (1, ncoeffs)]
                    };

                    let local_chunk = self
                        .fmm
                        .dc2e_inv_1
                        .dot(&self.fmm.dc2e_inv_2.dot(&potential_chunk))
                        .eval();

                    local_chunk
                        .data()
                        .chunks_exact(ncoeffs)
                        .zip(local_ptrs)
                        .for_each(|(result, local)| {
                            let local = unsafe { std::slice::from_raw_parts_mut(local.raw, ncoeffs) };
                            local.iter_mut().zip(result).for_each(|(l, r)| *l += *r);
                        });
                });
        }

        fn m2l_scale(&self, level: u64) -> U {
            if level < 2 {
                panic!("M2L only perfomed on level 2 and below")
            }

            if level == 2 {
                U::from(1. / 2.).unwrap()
            } else {
                let two = U::from(2.0).unwrap();
                Scalar::powf(two, U::from(level - 3).unwrap())
            }
        }
    }

    /// Implement the multipole to local translation operator for an SVD accelerated KiFMM on a single node.
    impl<T, U> FieldTranslation<U>
        for FmmDataAdaptive<KiFmmLinear<SingleNodeTree<U>, T, SvdFieldTranslationKiFmm<U, T>, U>, U>
    where
        T: Kernel<T = U>
            + ScaleInvariantKernel<T = U>
            + std::marker::Send
            + std::marker::Sync
            + Default,
        DenseMatrixLinAlgBuilder<U>: Svd,
        U: Scalar<Real = U>,
        U: Float
            + Default
            + MultiplyAdd<
                U,
                VectorContainer<U>,
                VectorContainer<U>,
                VectorContainer<U>,
                Dynamic,
                Dynamic,
                Dynamic,
            >,
        U: std::marker::Send + std::marker::Sync + Default,
    {
        fn p2l<'a>(&self, level: u64) {
            let Some(targets) = self.fmm.tree().get_keys(level) else {
                return;
            };

            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
            let dim = self.fmm.kernel().space_dimension();
            let surface_size = ncoeffs * dim;
            let min_idx = self.fmm.tree().key_to_index.get(&targets[0]).unwrap();
            let max_idx = self
                .fmm
                .tree()
                .key_to_index
                .get(targets.last().unwrap())
                .unwrap();
            let downward_surfaces =
                &self.downward_surfaces[min_idx * surface_size..(max_idx + 1) * surface_size];
            let coordinates = self.fmm.tree().get_all_coordinates().unwrap();

            let ntargets = targets.len();
            let mut check_potentials = vec![U::zero(); ncoeffs * ntargets];

            // 1. Compute check potentials from x list of each target
            targets
                .par_iter()
                .zip(downward_surfaces.par_chunks_exact(surface_size))
                .zip(check_potentials.par_chunks_exact_mut(ncoeffs))
                .for_each(|((target, downward_surface), check_potential)| {
                    // Find check potential
                    if let Some(x_list) = self.fmm.get_x_list(target) {
                        let x_list_indices = x_list
                            .iter()
                            .filter_map(|k| self.fmm.tree().get_leaf_index(k));
                        let charges = x_list_indices
                            .clone()
                            .map(|&idx| {
                                let index_pointer = &self.charge_index_pointer[idx];
                                &self.charges[index_pointer.0..index_pointer.1]
                            })
                            .collect_vec();

                        let sources_coordinates = x_list_indices
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
                                    rlst_pointer_mat!['a, U, sources.as_ptr(), (nsources, dim), (dim, 1)]
                                }
                                .eval();

                                self.fmm.kernel.evaluate_st(
                                    EvalType::Value,
                                    sources.data(),
                                    downward_surface,
                                    charges,
                                    check_potential,
                                );
                            }
                        }
                    }
                });

            // 2. Compute local expansion from check potential
            self.level_locals[level as usize]
                .par_iter()
                .zip(check_potentials.par_chunks_exact(ncoeffs))
                .for_each(|(local_ptr, check_potential)| {
                    let target_local =
                        unsafe { std::slice::from_raw_parts_mut(local_ptr.raw, ncoeffs) };
                    let check_potential = unsafe {
                        rlst_pointer_mat!['a, U, check_potential.as_ptr(), (ncoeffs, 1), (1, ncoeffs)]
                    };

                    let scale = self.fmm.kernel().scale(level);
                    let mut tmp = self
                        .fmm
                        .dc2e_inv_1
                        .dot(&self.fmm.dc2e_inv_2.dot(&check_potential));
                    tmp.data_mut().iter_mut().for_each(|val| *val *= scale);

                    target_local
                        .iter_mut()
                        .zip(tmp.data())
                        .for_each(|(r, &t)| *r += t);
                });
        }

        fn m2l<'a>(&self, level: u64) {
            let Some(sources) = self.fmm.tree().get_keys(level) else {
                return;
            };

            let nsources = sources.len();

            let mut source_map = HashMap::new();

            for (i, t) in sources.iter().enumerate() {
                source_map.insert(t, i);
            }

            let mut target_indices = vec![vec![-1i64; nsources]; 316];

            // Need to identify all save locations in a pre-processing step.
            for (j, source) in sources.iter().enumerate() {
                let v_list = source
                    .parent()
                    .neighbors()
                    .iter()
                    .flat_map(|pn| pn.children())
                    .filter(|pnc| {
                        !source.is_adjacent(pnc) && self.fmm.tree().get_all_keys_set().contains(pnc)
                    })
                    .collect_vec();

                let transfer_vectors = v_list
                    .iter()
                    .map(|target| target.find_transfer_vector(source))
                    .collect_vec();

                let mut transfer_vectors_map = HashMap::new();
                for (i, v) in transfer_vectors.iter().enumerate() {
                    transfer_vectors_map.insert(v, i);
                }

                let transfer_vectors_set: HashSet<_> = transfer_vectors.iter().collect();

                for (i, tv) in self.fmm.m2l.transfer_vectors.iter().enumerate() {
                    if transfer_vectors_set.contains(&tv.hash) {
                        let target = &v_list[*transfer_vectors_map.get(&tv.hash).unwrap()];
                        let target_index = source_map.get(target).unwrap();
                        target_indices[i][j] = *target_index as i64;
                    }
                }
            }

            // Interpret multipoles as a matrix
            let ncoeffs = self.fmm.m2l.ncoeffs(self.fmm.order);
            let multipoles = unsafe {
                rlst_pointer_mat!['a, U, self.level_multipoles[level as usize][0].raw, (ncoeffs, nsources), (1, ncoeffs)]
            };

            let (nrows, _) = self.fmm.m2l.operator_data.c.shape();
            let dim = (nrows, self.fmm.m2l.k);

            let mut compressed_multipoles = self.fmm.m2l.operator_data.st_block.dot(&multipoles);

            compressed_multipoles
                .data_mut()
                .iter_mut()
                .for_each(|d| *d *= self.fmm.kernel.scale(level) * self.m2l_scale(level));

            (0..316).into_par_iter().for_each(|c_idx| {
                let top_left = (0, c_idx * self.fmm.m2l.k);
                let c_sub = self.fmm.m2l.operator_data.c.block(top_left, dim);

                let locals = self.fmm.dc2e_inv_1.dot(
                    &self.fmm.dc2e_inv_2.dot(
                        &self
                            .fmm
                            .m2l
                            .operator_data
                            .u
                            .dot(&c_sub.dot(&compressed_multipoles)),
                    ),
                );

                let displacements = &target_indices[c_idx];

                for (result_idx, &save_idx) in displacements.iter().enumerate() {
                    if save_idx > -1 {
                        let save_idx = save_idx as usize;
                        let local_ptr = self.level_locals[(level) as usize][save_idx].raw;
                        let local = unsafe { std::slice::from_raw_parts_mut(local_ptr, ncoeffs) };

                        let res = &locals.data()[result_idx * ncoeffs..(result_idx + 1) * ncoeffs];
                        local.iter_mut().zip(res).for_each(|(l, r)| *l += *r);
                    }
                }
            })
        }

        fn m2l_scale(&self, level: u64) -> U {
            if level < 2 {
                panic!("M2L only perfomed on level 2 and below")
            }

            if level == 2 {
                U::from(1. / 2.).unwrap()
            } else {
                let two = U::from(2.0).unwrap();
                Scalar::powf(two, U::from(level - 3).unwrap())
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use bempp_field::types::{FftFieldTranslationKiFmm, SvdFieldTranslationKiFmm};
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::{
        implementations::helpers::{points_fixture, points_fixture_sphere},
        types::single_node::SingleNodeTree,
    };
    use float_cmp::assert_approx_eq;

    use crate::{
        charge::build_charge_dict,
        types::{
            FmmDataAdaptive, FmmDataUniform, FmmDataUniformMatrix, KiFmmLinear, KiFmmLinearMatrix,
        },
    };

    use bempp_traits::{field::FieldTranslationData, kernel::ScaleInvariantKernel};
    use bempp_traits::{
        fmm::{Fmm, FmmLoop, InteractionLists},
        kernel::Kernel,
        tree::Tree,
    };
    use itertools::Itertools;
    use rlst::dense::{rlst_col_vec, rlst_pointer_mat, RawAccess};

    use rlst::{
        common::traits::*,
        dense::{traits::*, Dot},
    };

    #[test]
    pub fn test_field_translation_uniform() {
        let npoints = 1000000;
        let points = points_fixture::<f64>(npoints, None, None);

        let global_idxs = (0..npoints).collect_vec();
        let charges = vec![1.0; npoints];

        let order = 5;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;
        let adaptive = false;

        let kernel = Laplace3dKernel::default();

        let tree = SingleNodeTree::new(
            points.data(),
            adaptive,
            None,
            Some(4),
            &global_idxs[..],
            false,
        );

        let m2l_data = SvdFieldTranslationKiFmm::new(
            kernel.clone(),
            Some(1000),
            order,
            *tree.get_domain(),
            alpha_inner,
        );

        let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

        let ncoeffs = fmm.m2l.ncoeffs(fmm.order);
        // Form charge dict, matching charges with their associated global indices
        let charge_dict = build_charge_dict(&global_idxs[..], &charges[..]);

        let datatree = FmmDataUniform::new(fmm, &charge_dict).unwrap();

        let s = Instant::now();
        let times = datatree.run(true);
        println!("single vec svd {:?}", times);
        assert!(false);

        // let depth = datatree.fmm.tree().get_depth();

        // for leaf_node in datatree.fmm.tree.get_keys(depth).unwrap().iter() {
        //     if let Some(v_list) = datatree.fmm.get_v_list(leaf_node) {
        //         let downward_equivalent_surface = leaf_node.compute_surface(
        //             datatree.fmm.tree.get_domain(),
        //             datatree.fmm.order,
        //             datatree.fmm.alpha_outer,
        //         );

        //         let downward_check_surface = leaf_node.compute_surface(
        //             datatree.fmm.tree.get_domain(),
        //             datatree.fmm.order,
        //             datatree.fmm.alpha_inner,
        //         );

        //         let leaf_level = leaf_node.level();
        //         let &level_index = datatree.level_index_pointer[leaf_level as usize]
        //             .get(leaf_node)
        //             .unwrap();
        //         let leaf_local = datatree.level_locals[leaf_level as usize][level_index];
        //         let leaf_local = unsafe { std::slice::from_raw_parts_mut(leaf_local.raw, ncoeffs) };

        //         let mut equivalent = vec![0f64; leaf_local.len()];
        //         datatree.fmm.kernel().evaluate_st(
        //             bempp_traits::types::EvalType::Value,
        //             &downward_equivalent_surface,
        //             &downward_check_surface,
        //             leaf_local,
        //             &mut equivalent,
        //         );

        //         let mut direct = vec![0f64; leaf_local.len()];

        //         for source in v_list.iter() {
        //             assert!(datatree.fmm.tree().get_all_keys_set().contains(source));
        //             let upward_equivalent_surface = source.compute_surface(
        //                 datatree.fmm.tree.get_domain(),
        //                 datatree.fmm.order,
        //                 datatree.fmm.alpha_inner,
        //             );

        //             let source_level = source.level();
        //             let &level_index = datatree.level_index_pointer[source_level as usize]
        //                 .get(source)
        //                 .unwrap();
        //             let source_multipole =
        //                 datatree.level_multipoles[source_level as usize][level_index];
        //             let source_multipole =
        //                 unsafe { std::slice::from_raw_parts(source_multipole.raw, ncoeffs) };

        //             datatree.fmm.kernel().evaluate_st(
        //                 bempp_traits::types::EvalType::Value,
        //                 &upward_equivalent_surface,
        //                 &downward_check_surface,
        //                 source_multipole,
        //                 &mut direct,
        //             )
        //         }

        //         // Add parent contribution
        //         let mut l2l_idx = 0;
        //         for (i, sib) in leaf_node.siblings().iter().enumerate() {
        //             if *sib == *leaf_node {
        //                 l2l_idx = i;
        //                 break;
        //             }
        //         }

        //         let parent = leaf_node.parent();
        //         let &level_index = datatree.level_index_pointer[parent.level() as usize]
        //             .get(&parent)
        //             .unwrap();
        //         let parent_local = datatree.level_locals[parent.level() as usize][level_index];
        //         let parent_local = unsafe {
        //             rlst_pointer_mat!['_, f64, parent_local.raw, (ncoeffs, 1), (1, ncoeffs)]
        //         };

        //         // Parent contribution to check potential
        //         let tmp = datatree.fmm.l2l[l2l_idx].dot(&parent_local).eval();

        //         // Evaluate tmp at child equivalent surface to find potential
        //         datatree.fmm.kernel().evaluate_st(
        //             bempp_traits::types::EvalType::Value,
        //             &downward_equivalent_surface,
        //             &downward_check_surface,
        //             tmp.data(),
        //             &mut direct,
        //         );

        //         // Compare check potential found directly, and via approximation
        //         for (a, b) in equivalent.iter().zip(direct.iter()) {
        //             assert_approx_eq!(f64, *a, *b, epsilon = 1e-5);
        //         }
        //     }
        // }
    }

    #[test]
    pub fn foo_bar() {
        let npoints = 1000000;
        let points = points_fixture::<f64>(npoints, None, None);

        let global_idxs = (0..npoints).collect_vec();
        let ncharge_vecs = 10;

        let mut charge_mat = vec![vec![0.0; npoints]; ncharge_vecs];
        charge_mat
            .iter_mut()
            .enumerate()
            .for_each(|(i, charge_mat_i)| *charge_mat_i = vec![i as f64 + 1.0; npoints]);

        let order = 5;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;
        let adaptive = false;
        let depth = 4;

        let kernel = Laplace3dKernel::default();

        let tree = SingleNodeTree::new(
            points.data(),
            adaptive,
            None,
            Some(depth),
            &global_idxs[..],
            true,
        );

        // let m2l_data =
        // FftFieldTranslationKiFmm::new(kernel.clone(), order, *tree.get_domain(), alpha_inner);
        let m2l_data = SvdFieldTranslationKiFmm::new(
            kernel.clone(),
            Some(1000),
            order,
            *tree.get_domain(),
            alpha_inner,
        );
        let fmm = KiFmmLinearMatrix::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

        let ncoeffs = fmm.m2l.ncoeffs(fmm.order);
        // Form charge dict, matching charges with their associated global indices
        let charge_dicts: Vec<_> = (0..ncharge_vecs)
            .map(|i| build_charge_dict(&global_idxs, &charge_mat[i]))
            .collect();

        let datatree = FmmDataUniformMatrix::new(fmm, &charge_dicts).unwrap();

        let s = Instant::now();
        let times = datatree.run(true);
        println!("multi vector m2l time {:?}", times);
        assert!(false);
        // let depth = datatree.fmm.tree().get_depth();

        // for leaf_node in datatree.fmm.tree.get_keys(depth).unwrap().iter() {
        //     if let Some(v_list) = datatree.fmm.get_v_list(leaf_node) {
        //         let downward_equivalent_surface = leaf_node.compute_surface(
        //             datatree.fmm.tree.get_domain(),
        //             datatree.fmm.order,
        //             datatree.fmm.alpha_outer,
        //         );

        //         let downward_check_surface = leaf_node.compute_surface(
        //             datatree.fmm.tree.get_domain(),
        //             datatree.fmm.order,
        //             datatree.fmm.alpha_inner,
        //         );

        //         let leaf_level = leaf_node.level();
        //         let &level_index = datatree.level_index_pointer[leaf_level as usize]
        //             .get(leaf_node)
        //             .unwrap();
        //         let leaf_local = datatree.level_locals[leaf_level as usize][level_index];
        //         let leaf_local = unsafe { std::slice::from_raw_parts_mut(leaf_local.raw, ncoeffs) };

        //         let mut equivalent = vec![0f64; leaf_local.len()];
        //         datatree.fmm.kernel().evaluate_st(
        //             bempp_traits::types::EvalType::Value,
        //             &downward_equivalent_surface,
        //             &downward_check_surface,
        //             leaf_local,
        //             &mut equivalent,
        //         );

        //         let mut direct = vec![0f64; leaf_local.len()];

        //         for source in v_list.iter() {
        //             assert!(datatree.fmm.tree().get_all_keys_set().contains(source));
        //             let upward_equivalent_surface = source.compute_surface(
        //                 datatree.fmm.tree.get_domain(),
        //                 datatree.fmm.order,
        //                 datatree.fmm.alpha_inner,
        //             );

        //             let source_level = source.level();
        //             let &level_index = datatree.level_index_pointer[source_level as usize]
        //                 .get(source)
        //                 .unwrap();
        //             let source_multipole =
        //                 datatree.level_multipoles[source_level as usize][level_index];
        //             let source_multipole =
        //                 unsafe { std::slice::from_raw_parts(source_multipole.raw, ncoeffs) };

        //             datatree.fmm.kernel().evaluate_st(
        //                 bempp_traits::types::EvalType::Value,
        //                 &upward_equivalent_surface,
        //                 &downward_check_surface,
        //                 source_multipole,
        //                 &mut direct,
        //             )
        //         }

        //         // Add parent contribution
        //         let mut l2l_idx = 0;
        //         for (i, sib) in leaf_node.siblings().iter().enumerate() {
        //             if *sib == *leaf_node {
        //                 l2l_idx = i;
        //                 break;
        //             }
        //         }

        //         let parent = leaf_node.parent();
        //         let &level_index = datatree.level_index_pointer[parent.level() as usize]
        //             .get(&parent)
        //             .unwrap();
        //         let parent_local = datatree.level_locals[parent.level() as usize][level_index];
        //         let parent_local = unsafe {
        //             rlst_pointer_mat!['_, f64, parent_local.raw, (ncoeffs, 1), (1, ncoeffs)]
        //         };

        //         // Parent contribution to check potential
        //         let tmp = datatree.fmm.l2l[l2l_idx].dot(&parent_local).eval();

        //         // Evaluate tmp at child equivalent surface to find potential
        //         datatree.fmm.kernel().evaluate_st(
        //             bempp_traits::types::EvalType::Value,
        //             &downward_equivalent_surface,
        //             &downward_check_surface,
        //             tmp.data(),
        //             &mut direct,
        //         );

        //         // Compare check potential found directly, and via approximation
        //         for (a, b) in equivalent.iter().zip(direct.iter()) {
        //             assert_approx_eq!(f64, *a, *b, epsilon = 1e-5);
        //         }
        //     }
        // }
    }

    #[test]
    pub fn test_field_translation_adaptive() {
        let npoints = 100000;
        // let points = points_fixture::<f64>(npoints, None, None);
        let points = points_fixture_sphere::<f64>(npoints);

        let global_idxs = (0..npoints).collect_vec();
        let charges = vec![1.0; npoints];

        let order = 5;
        let alpha_inner = 1.05;
        let alpha_outer = 2.95;
        let adaptive = true;
        let ncrit = 150;

        let kernel = Laplace3dKernel::default();

        let tree = SingleNodeTree::new(
            points.data(),
            adaptive,
            Some(ncrit),
            None,
            &global_idxs[..],
            true,
        );

        let m2l_data =
            FftFieldTranslationKiFmm::new(kernel.clone(), order, *tree.get_domain(), alpha_inner);

        let fmm = KiFmmLinear::new(order, alpha_inner, alpha_outer, kernel, tree, m2l_data);

        let ncoeffs = fmm.m2l.ncoeffs(fmm.order);
        // Form charge dict, matching charges with their associated global indices
        let charge_dict = build_charge_dict(&global_idxs[..], &charges[..]);

        let datatree = FmmDataAdaptive::new(fmm, &charge_dict).unwrap();

        datatree.run(false);
        let depth = datatree.fmm.tree().get_depth();
        let dim = 3;

        for leaf_node in datatree.fmm.tree.get_keys(depth).unwrap().iter() {
            if let Some(v_list) = datatree.fmm.get_v_list(leaf_node) {
                let downward_equivalent_surface = leaf_node.compute_surface(
                    datatree.fmm.tree.get_domain(),
                    datatree.fmm.order,
                    datatree.fmm.alpha_outer,
                );

                let downward_check_surface = leaf_node.compute_surface(
                    datatree.fmm.tree.get_domain(),
                    datatree.fmm.order,
                    datatree.fmm.alpha_inner,
                );

                let leaf_level = leaf_node.level();
                let &level_index = datatree.level_index_pointer[leaf_level as usize]
                    .get(leaf_node)
                    .unwrap();
                let leaf_local = datatree.level_locals[leaf_level as usize][level_index];
                let leaf_local = unsafe { std::slice::from_raw_parts_mut(leaf_local.raw, ncoeffs) };

                let mut equivalent = vec![0f64; leaf_local.len()];
                datatree.fmm.kernel().evaluate_st(
                    bempp_traits::types::EvalType::Value,
                    &downward_equivalent_surface,
                    &downward_check_surface,
                    leaf_local,
                    &mut equivalent,
                );

                let mut direct = vec![0f64; leaf_local.len()];

                for source in v_list.iter() {
                    assert!(datatree.fmm.tree().get_all_keys_set().contains(source));
                    let upward_equivalent_surface = source.compute_surface(
                        datatree.fmm.tree.get_domain(),
                        datatree.fmm.order,
                        datatree.fmm.alpha_inner,
                    );

                    let source_level = source.level();
                    let &level_index = datatree.level_index_pointer[source_level as usize]
                        .get(source)
                        .unwrap();
                    let source_multipole =
                        datatree.level_multipoles[source_level as usize][level_index];
                    let source_multipole =
                        unsafe { std::slice::from_raw_parts(source_multipole.raw, ncoeffs) };

                    datatree.fmm.kernel().evaluate_st(
                        bempp_traits::types::EvalType::Value,
                        &upward_equivalent_surface,
                        &downward_check_surface,
                        source_multipole,
                        &mut direct,
                    )
                }

                // Add parent contribution
                let mut l2l_idx = 0;
                for (i, sib) in leaf_node.siblings().iter().enumerate() {
                    if *sib == *leaf_node {
                        l2l_idx = i;
                        break;
                    }
                }

                let parent = leaf_node.parent();
                let &level_index = datatree.level_index_pointer[parent.level() as usize]
                    .get(&parent)
                    .unwrap();
                let parent_local = datatree.level_locals[parent.level() as usize][level_index];
                let parent_local = unsafe {
                    rlst_pointer_mat!['_, f64, parent_local.raw, (ncoeffs, 1), (1, ncoeffs)]
                };

                // Parent contribution to check potential
                let tmp = datatree.fmm.l2l[l2l_idx].dot(&parent_local).eval();

                // Evaluate tmp at child equivalent surface to find potential
                datatree.fmm.kernel().evaluate_st(
                    bempp_traits::types::EvalType::Value,
                    &downward_equivalent_surface,
                    &downward_check_surface,
                    tmp.data(),
                    &mut direct,
                );

                if let Some(x_list) = datatree.fmm.get_x_list(leaf_node) {
                    let x_list_indices = x_list
                        .iter()
                        .filter_map(|k| datatree.fmm.tree().get_leaf_index(k));
                    let charges = x_list_indices
                        .clone()
                        .map(|&idx| {
                            let index_pointer = &datatree.charge_index_pointer[idx];
                            &datatree.charges[index_pointer.0..index_pointer.1]
                        })
                        .collect_vec();

                    let sources_coordinates = x_list_indices
                        .into_iter()
                        .map(|&idx| {
                            let index_pointer = &datatree.charge_index_pointer[idx];
                            &datatree.fmm.tree().coordinates
                                [index_pointer.0 * dim..index_pointer.1 * dim]
                        })
                        .collect_vec();

                    for (&charges, sources) in charges.iter().zip(sources_coordinates) {
                        let nsources = sources.len() / dim;
                        if nsources > 0 {
                            let sources = unsafe {
                                rlst_pointer_mat!['_, f64, sources.as_ptr(), (nsources, dim), (dim, 1)]
                            }
                            .eval();

                            let mut check_potential = rlst_col_vec![f64, ncoeffs];
                            datatree.fmm.kernel.evaluate_st(
                                bempp_traits::types::EvalType::Value,
                                sources.data(),
                                &downward_check_surface,
                                charges,
                                check_potential.data_mut(),
                            );
                            let mut tmp = datatree
                                .fmm
                                .dc2e_inv_1
                                .dot(&datatree.fmm.dc2e_inv_2.dot(&check_potential));
                            let scale = datatree.fmm.kernel.scale(leaf_node.level());
                            tmp.data_mut().iter_mut().for_each(|val| *val *= scale);

                            // Evaluate tmp at child equivalent surface to find potential
                            datatree.fmm.kernel().evaluate_st(
                                bempp_traits::types::EvalType::Value,
                                &downward_equivalent_surface,
                                &downward_check_surface,
                                tmp.data(),
                                &mut direct,
                            );
                        }
                    }
                }

                // Compare check potential found directly, and via approximation
                for (a, b) in equivalent.iter().zip(direct.iter()) {
                    assert_approx_eq!(f64, *a, *b, epsilon = 1e-5);
                }
            }
        }
    }
}
