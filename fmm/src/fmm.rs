//! Implementation of FmmData and Fmm traits.
use num::Float;
use rlst_dense::{
    rlst_dynamic_array2,
    traits::{RawAccess, RawAccessMut, Shape},
    types::RlstScalar,
};

use bempp_traits::{
    field::{SourceToTarget, SourceToTargetData},
    fmm::{Fmm, SourceTranslation, TargetTranslation},
    kernel::Kernel,
    tree::{FmmTree, Tree},
    types::EvalType,
};

use bempp_tree::types::{morton::MortonKey, single_node::SingleNodeTree};

use crate::types::{Charges, FmmEvalType, KiFmm, KiFmmDummy, SendPtrMut};

impl<T, U, V, W> Fmm for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTree<W>, Node = MortonKey> + Send + Sync,
    U: SourceToTargetData<V> + Send + Sync,
    V: Kernel<T = W> + Send + Sync,
    W: RlstScalar<Real = W> + Float + Default,
    Self: SourceToTarget,
{
    type NodeIndex = T::Node;
    type Precision = W;
    type Kernel = V;
    type Tree = T;

    fn dim(&self) -> usize {
        self.dim
    }

    fn multipole(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        if let Some(index) = self.tree.source_tree().index(key) {
            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    Some(&self.multipoles[index * self.ncoeffs..(index + 1) * self.ncoeffs])
                }
                FmmEvalType::Matrix(nmatvecs) => Some(
                    &self.multipoles
                        [index * self.ncoeffs * nmatvecs..(index + 1) * self.ncoeffs * nmatvecs],
                ),
            }
        } else {
            None
        }
    }

    fn local(&self, key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        if let Some(index) = self.tree.target_tree().index(key) {
            match self.fmm_eval_type {
                FmmEvalType::Vector => {
                    Some(&self.locals[index * self.ncoeffs..(index + 1) * self.ncoeffs])
                }
                FmmEvalType::Matrix(nmatvecs) => Some(
                    &self.locals
                        [index * self.ncoeffs * nmatvecs..(index + 1) * self.ncoeffs * nmatvecs],
                ),
            }
        } else {
            None
        }
    }

    fn potential(&self, leaf: &Self::NodeIndex) -> Option<Vec<&[Self::Precision]>> {
        if let Some(&leaf_idx) = self.tree.target_tree().leaf_index(leaf) {
            let (l, r) = self.charge_index_pointer_targets[leaf_idx];
            let ntargets = r - l;

            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(vec![
                    &self.potentials[l * self.kernel_eval_size..r * self.kernel_eval_size],
                ]),
                FmmEvalType::Matrix(nmatvecs) => {
                    let nleaves = self.tree.target_tree().nleaves().unwrap();
                    let mut slices = Vec::new();
                    for eval_idx in 0..nmatvecs {
                        let potentials_pointer =
                            self.potentials_send_pointers[eval_idx * nleaves + leaf_idx].raw;
                        slices.push(unsafe {
                            std::slice::from_raw_parts(
                                potentials_pointer,
                                ntargets * self.kernel_eval_size,
                            )
                        });
                    }
                    Some(slices)
                }
            }
        } else {
            None
        }
    }

    fn expansion_order(&self) -> usize {
        self.expansion_order
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn tree(&self) -> &Self::Tree {
        &self.tree
    }

    fn evaluate(&self) {
        // println!("HERE {:?}", self.charges);

        // Upward pass
        {
            self.p2m();
            for level in (1..=self.tree.source_tree().get_depth()).rev() {
                self.m2m(level);
            }
        }

        // Downward pass
        {
            for level in 2..=self.tree.target_tree().get_depth() {
                if level > 2 {
                    self.l2l(level);
                }
                self.m2l(level);
                self.p2l(level)
            }

            // Leaf level computations
            self.m2p();
            self.p2p();
            self.l2p();
        }
    }

    fn clear(&mut self) {
        let nmatvecs = match self.fmm_eval_type {
            FmmEvalType::Vector => 1,
            FmmEvalType::Matrix(nmatvecs) => nmatvecs,
        };
        // Clear buffers
        let multipoles_shape = self.multipoles.len();
        self.multipoles = vec![W::default(); multipoles_shape];

        let locals_shape = self.locals.len();
        self.locals = vec![W::default(); locals_shape];

        let potentials_shape = self.potentials.len();
        self.potentials = vec![W::default(); potentials_shape];

        let charges_shape = self.charges.len();
        self.charges = vec![W::default(); charges_shape];

        // Recreate mutable pointers
        let ntarget_points = self.tree().target_tree().all_coordinates().unwrap().len() / self.dim;
        let nsource_leaves = self.tree().source_tree().nleaves().unwrap();
        let ntarget_leaves = self.tree().target_tree().nleaves().unwrap();

        let mut potentials_send_pointers = vec![SendPtrMut::default(); ntarget_leaves * nmatvecs];
        let mut leaf_multipoles = vec![Vec::new(); nsource_leaves];
        let mut leaf_locals = vec![Vec::new(); ntarget_leaves];

        let mut level_multipoles = vec![
            Vec::new();
            (self.tree().source_tree().get_depth() + 1)
                .try_into()
                .unwrap()
        ];
        let mut level_locals = vec![
            Vec::new();
            (self.tree().target_tree().get_depth() + 1)
                .try_into()
                .unwrap()
        ];

        {
            let mut potential_raw_pointers = Vec::new();
            for eval_idx in 0..nmatvecs {
                let ptr = unsafe {
                    self.potentials
                        .as_mut_ptr()
                        .add(eval_idx * ntarget_points * self.kernel_eval_size)
                };
                potential_raw_pointers.push(ptr)
            }

            for (i, leaf) in self
                .tree
                .target_tree()
                .all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let npoints;
                let nevals;

                if let Some(coordinates) = self.tree.target_tree().coordinates(leaf) {
                    npoints = coordinates.len() / self.dim;
                    nevals = npoints * self.kernel_eval_size;
                } else {
                    nevals = 0;
                }

                for j in 0..nmatvecs {
                    potentials_send_pointers[ntarget_leaves * j + i] = SendPtrMut {
                        raw: potential_raw_pointers[j],
                    }
                }

                // Update raw pointer with number of points at this leaf
                for ptr in potential_raw_pointers.iter_mut() {
                    *ptr = unsafe { ptr.add(nevals) }
                }
            }
        }

        for level in 0..=self.tree.source_tree().get_depth() {
            let mut tmp_multipoles = Vec::new();

            let keys = self.tree.source_tree().keys(level).unwrap();
            for key in keys.into_iter() {
                let &key_idx = self.tree.source_tree().index(key).unwrap();
                let key_displacement = self.ncoeffs * nmatvecs * key_idx;
                let mut key_multipoles = Vec::new();
                for eval_idx in 0..nmatvecs {
                    let eval_displacement = self.ncoeffs * eval_idx;
                    let raw = unsafe {
                        self.multipoles
                            .as_ptr()
                            .add(key_displacement + eval_displacement)
                            as *mut W
                    };
                    key_multipoles.push(SendPtrMut { raw });
                }
                tmp_multipoles.push(key_multipoles)
            }
            level_multipoles[level as usize] = tmp_multipoles
        }

        for level in 0..=self.tree.target_tree().get_depth() {
            let mut tmp_locals = Vec::new();

            let keys = self.tree.target_tree().keys(level).unwrap();
            for key in keys.into_iter() {
                let &key_idx = self.tree.target_tree().index(key).unwrap();
                let key_displacement = self.ncoeffs * nmatvecs * key_idx;
                let mut key_locals = Vec::new();
                for eval_idx in 0..nmatvecs {
                    let eval_displacement = self.ncoeffs * eval_idx;
                    let raw = unsafe {
                        self.locals
                            .as_ptr()
                            .add(key_displacement + eval_displacement)
                            as *mut W
                    };
                    key_locals.push(SendPtrMut { raw });
                }
                tmp_locals.push(key_locals)
            }
            level_locals[level as usize] = tmp_locals
        }

        {
            for (leaf_idx, leaf) in self
                .tree
                .source_tree()
                .all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let key_idx = self.tree.source_tree().index(leaf).unwrap();
                let key_displacement = self.ncoeffs * nmatvecs * key_idx;
                for eval_idx in 0..nmatvecs {
                    let eval_displacement = self.ncoeffs * eval_idx;
                    let raw = unsafe {
                        self.multipoles
                            .as_ptr()
                            .add(eval_displacement + key_displacement)
                            as *mut W
                    };

                    leaf_multipoles[leaf_idx].push(SendPtrMut { raw });
                }
            }

            for (leaf_idx, leaf) in self
                .tree
                .target_tree()
                .all_leaves()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let key_idx = self.tree.target_tree().index(leaf).unwrap();
                let key_displacement = self.ncoeffs * nmatvecs * key_idx;
                for eval_idx in 0..nmatvecs {
                    let eval_displacement = self.ncoeffs * eval_idx;
                    let raw = unsafe {
                        self.locals
                            .as_ptr()
                            .add(eval_displacement + key_displacement)
                            as *mut W
                    };
                    leaf_locals[leaf_idx].push(SendPtrMut { raw });
                }
            }
        }

        self.level_locals = level_locals;
        self.level_multipoles = level_multipoles;
        self.leaf_locals = leaf_locals;
        self.leaf_multipoles = leaf_multipoles;
        self.potentials_send_pointers = potentials_send_pointers;
    }

    fn set_charges(&mut self, charges: &Charges<W>) {
        let [ncharges, nmatvec] = charges.shape();

        let mut reordered_charges = rlst_dynamic_array2!(W, [ncharges, nmatvec]);
        let global_idxs = self.tree.source_tree().all_global_indices().unwrap();

        for eval_idx in 0..nmatvec {
            let eval_displacement = eval_idx * ncharges;
            for (new_idx, old_idx) in global_idxs.iter().enumerate() {
                reordered_charges.data_mut()[new_idx + eval_displacement] =
                    charges.data()[old_idx + eval_displacement];
            }
        }

        self.charges = reordered_charges.data().to_vec();
    }
}

impl<T, U, V, W> Default for KiFmm<T, U, V, W>
where
    T: FmmTree<Tree = SingleNodeTree<W>> + Default,
    U: SourceToTargetData<V> + Default,
    V: Kernel + Default,
    W: RlstScalar<Real = W> + Float + Default,
{
    fn default() -> Self {
        let uc2e_inv_1 = rlst_dynamic_array2!(W, [1, 1]);
        let uc2e_inv_2 = rlst_dynamic_array2!(W, [1, 1]);
        let dc2e_inv_1 = rlst_dynamic_array2!(W, [1, 1]);
        let dc2e_inv_2 = rlst_dynamic_array2!(W, [1, 1]);
        let source = rlst_dynamic_array2!(W, [1, 1]);
        KiFmm {
            tree: T::default(),
            source_to_target_translation_data: U::default(),
            kernel: V::default(),
            expansion_order: 0,
            fmm_eval_type: FmmEvalType::Vector,
            kernel_eval_type: EvalType::Value,
            kernel_eval_size: 0,
            dim: 0,
            ncoeffs: 0,
            uc2e_inv_1,
            uc2e_inv_2,
            dc2e_inv_1,
            dc2e_inv_2,
            source_data: source,
            source_translation_data_vec: Vec::default(),
            target_data: Vec::default(),
            multipoles: Vec::default(),
            locals: Vec::default(),
            leaf_multipoles: Vec::default(),
            level_multipoles: Vec::default(),
            leaf_locals: Vec::default(),
            level_locals: Vec::default(),
            level_index_pointer_locals: Vec::default(),
            level_index_pointer_multipoles: Vec::default(),
            potentials: Vec::default(),
            potentials_send_pointers: Vec::default(),
            leaf_upward_surfaces_sources: Vec::default(),
            leaf_upward_surfaces_targets: Vec::default(),
            leaf_downward_surfaces: Vec::default(),
            charges: Vec::default(),
            charge_index_pointer_sources: Vec::default(),
            charge_index_pointer_targets: Vec::default(),
            leaf_scales_sources: Vec::default(),
            global_indices: Vec::default(),
        }
    }
}

impl<T, U, V> Fmm for KiFmmDummy<T, U, V>
where
    T: FmmTree<Tree = SingleNodeTree<U>, Node = MortonKey>,
    U: RlstScalar<Real = U> + Float + Default,
    V: Kernel<T = U> + Send + Sync,
{
    type NodeIndex = T::Node;
    type Precision = U;
    type Tree = T;
    type Kernel = V;

    fn dim(&self) -> usize {
        3
    }

    fn multipole(&self, _key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        None
    }

    fn local(&self, _key: &Self::NodeIndex) -> Option<&[Self::Precision]> {
        None
    }

    fn potential(&self, leaf: &Self::NodeIndex) -> Option<Vec<&[Self::Precision]>> {
        let ntarget_coordinates =
            self.tree.target_tree().all_coordinates().unwrap().len() / self.dim();

        if let Some(&leaf_idx) = self.tree.target_tree().leaf_index(leaf) {
            let (l, r) = self.charge_index_pointer_targets[leaf_idx];

            match self.fmm_eval_type {
                FmmEvalType::Vector => Some(vec![
                    &self.potentials[l * self.kernel_eval_size..r * self.kernel_eval_size],
                ]),
                FmmEvalType::Matrix(nmatvecs) => {
                    let mut slices = Vec::new();
                    for eval_idx in 0..nmatvecs {
                        let vec_displacement = eval_idx * ntarget_coordinates;
                        let slice = &self.potentials[vec_displacement + l..vec_displacement + r];
                        slices.push(slice);
                    }
                    Some(slices)
                }
            }
        } else {
            None
        }
    }

    fn evaluate(&self) {
        let all_target_coordinates = self.tree.target_tree().all_coordinates().unwrap();
        let ntarget_coordinates = all_target_coordinates.len() / self.dim();
        let all_source_coordinates = self.tree.source_tree().all_coordinates().unwrap();
        let nsource_coordinates = all_target_coordinates.len() / self.dim();

        match self.fmm_eval_type {
            FmmEvalType::Vector => {
                let charges = &self.charges;
                let res = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.potentials.as_ptr() as *mut U,
                        ntarget_coordinates,
                    )
                };
                self.kernel.evaluate_st(
                    self.kernel_eval_type,
                    all_source_coordinates,
                    all_target_coordinates,
                    charges,
                    res,
                )
            }

            FmmEvalType::Matrix(nmatvec) => {
                for i in 0..nmatvec {
                    let charges_i =
                        &self.charges[i * nsource_coordinates..(i + 1) * nsource_coordinates];
                    let res_i = unsafe {
                        std::slice::from_raw_parts_mut(
                            self.potentials.as_ptr().add(ntarget_coordinates) as *mut U,
                            ntarget_coordinates,
                        )
                    };
                    self.kernel.evaluate_st(
                        self.kernel_eval_type,
                        all_source_coordinates,
                        all_target_coordinates,
                        charges_i,
                        res_i,
                    )
                }
            }
        }
    }

    fn expansion_order(&self) -> usize {
        self.expansion_order
    }

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn tree(&self) -> &Self::Tree {
        &self.tree
    }

    fn clear(&mut self) {}

    fn set_charges(&mut self, charges: &Charges<U>) {}
}

#[cfg(test)]
mod test {

    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_tree::constants::{ALPHA_INNER, ROOT};
    use bempp_tree::implementations::helpers::points_fixture;
    use num::Float;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rlst_dense::array::Array;
    use rlst_dense::base_array::BaseArray;
    use rlst_dense::data_container::VectorContainer;
    use rlst_dense::rlst_array_from_slice2;
    use rlst_dense::traits::{RawAccess, RawAccessMut, Shape};

    use crate::{tree::SingleNodeFmmTree, types::KiFmmBuilderSingleNode};
    use bempp_field::types::{BlasFieldTranslationKiFmm, FftFieldTranslationKiFmm};

    use super::*;

    fn test_single_node_fmm_vector_helper<T: RlstScalar<Real = T> + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Precision = T,
                NodeIndex = MortonKey,
                Kernel = Laplace3dKernel<T>,
                Tree = SingleNodeFmmTree<T>,
            >,
        >,
        eval_type: EvalType,
        sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T,
    ) {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf: MortonKey = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];
        let potential = fmm.potential(&leaf).unwrap()[0];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();
        let mut direct = vec![T::zero(); ntargets * eval_size];

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(T, leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);
        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T, [ntargets, fmm.dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        fmm.kernel().evaluate_st(
            eval_type,
            sources.data(),
            leaf_coordinates_col_major.data(),
            charges.data(),
            &mut direct,
        );

        direct.iter().zip(potential).for_each(|(&d, &p)| {
            let abs_error = num::Float::abs(d - p);
            let rel_error = abs_error / p;
            println!("err {:?} \nd {:?} \np {:?}", rel_error, direct, potential);
            assert!(rel_error <= threshold)
        });
    }

    fn test_single_node_fmm_matrix_helper<T: RlstScalar<Real = T> + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Precision = T,
                NodeIndex = MortonKey,
                Kernel = Laplace3dKernel<T>,
                Tree = SingleNodeFmmTree<T>,
            >,
        >,
        eval_type: EvalType,
        sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T,
    ) {
        let eval_size = match eval_type {
            EvalType::Value => 1,
            EvalType::ValueDeriv => 4,
        };

        let leaf_idx = 0;
        let leaf: MortonKey = fmm.tree().target_tree().all_leaves().unwrap()[leaf_idx];

        let leaf_targets = fmm.tree().target_tree().coordinates(&leaf).unwrap();

        let ntargets = leaf_targets.len() / fmm.dim();

        let leaf_coordinates_row_major =
            rlst_array_from_slice2!(T, leaf_targets, [ntargets, fmm.dim()], [fmm.dim(), 1]);

        let mut leaf_coordinates_col_major = rlst_dynamic_array2!(T, [ntargets, fmm.dim()]);
        leaf_coordinates_col_major.fill_from(leaf_coordinates_row_major.view());

        let [nsources, nmatvec] = charges.shape();

        for i in 0..nmatvec {
            let potential_i = fmm.potential(&leaf).unwrap()[i];
            let charges_i = &charges.data()[nsources * i..nsources * (i + 1)];
            let mut direct_i = vec![T::zero(); ntargets * eval_size];
            fmm.kernel().evaluate_st(
                eval_type,
                sources.data(),
                leaf_coordinates_col_major.data(),
                charges_i,
                &mut direct_i,
            );

            println!(
                "i {:?} \n direct_i {:?}\n potential_i {:?}",
                i, direct_i, potential_i
            );
            direct_i.iter().zip(potential_i).for_each(|(&d, &p)| {
                let abs_error = num::Float::abs(d - p);
                let rel_error = abs_error / p;
                assert!(rel_error <= threshold)
            })
        }
    }

    #[test]
    fn test_fmm_api() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;
        let threshold_pot = 1e-5;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let mut fmm = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                FftFieldTranslationKiFmm::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm.evaluate();

        // Reset Charge data and re-evaluate potential
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(1);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        fmm.clear();
        fmm.set_charges(&charges);
        fmm.evaluate();
        let fmm = Box::new(fmm);

        test_single_node_fmm_vector_helper(
            fmm,
            bempp_traits::types::EvalType::Value,
            &sources,
            &charges,
            threshold_pot,
        );
    }

    #[test]
    fn test_laplace_fmm_vector() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;
        let threshold_pot = 1e-5;
        let threshold_deriv = 1e-4;
        let threshold_deriv_blas = 1e-3;
        let singular_value_threshold = Some(1e-2);

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        // fmm with fft based field translation
        {
            // Evaluate potentials
            let fmm_fft = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    bempp_traits::types::EvalType::Value,
                    FftFieldTranslationKiFmm::new(),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_fft.evaluate();
            let eval_type = fmm_fft.kernel_eval_type;
            let fmm_fft = Box::new(fmm_fft);
            test_single_node_fmm_vector_helper(
                fmm_fft,
                eval_type,
                &sources,
                &charges,
                threshold_pot,
            );

            // Evaluate potentials + derivatives
            let fmm_fft = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    bempp_traits::types::EvalType::ValueDeriv,
                    FftFieldTranslationKiFmm::new(),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_fft.evaluate();
            let eval_type = fmm_fft.kernel_eval_type;
            let fmm_fft = Box::new(fmm_fft);
            test_single_node_fmm_vector_helper(
                fmm_fft,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
            );
        }

        // fmm with BLAS based field translation
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let fmm_blas = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationKiFmm::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_fmm_vector_helper(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_pot,
            );

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let fmm_blas = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationKiFmm::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_fmm_vector_helper(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_deriv_blas,
            );
        }
    }

    #[test]
    fn test_laplace_fmm_matrix() {
        // Setup random sources and targets
        let nsources = 9000;
        let ntargets = 10000;

        let min = None;
        let max = None;
        let sources = points_fixture::<f64>(nsources, min, max, Some(0));
        let targets = points_fixture::<f64>(ntargets, min, max, Some(1));
        // FMM parameters
        let n_crit = Some(10);
        let expansion_order = 6;
        let sparse = true;
        let threshold = 1e-5;
        let threshold_deriv = 1e-3;
        let singular_value_threshold = Some(1e-2);

        // Charge data
        let nvecs = 5;
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        let mut rng = StdRng::seed_from_u64(0);
        charges
            .data_mut()
            .chunks_exact_mut(nsources)
            .for_each(|chunk| chunk.iter_mut().for_each(|elem| *elem += rng.gen::<f64>()));

        // fmm with blas based field translation
        {
            // Evaluate potentials
            let eval_type = EvalType::Value;
            let fmm_blas = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationKiFmm::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();

            let fmm_blas = Box::new(fmm_blas);
            test_single_node_fmm_matrix_helper(fmm_blas, eval_type, &sources, &charges, threshold);

            // Evaluate potentials + derivatives
            let eval_type = EvalType::ValueDeriv;
            let fmm_blas = KiFmmBuilderSingleNode::new()
                .tree(&sources, &targets, n_crit, sparse)
                .unwrap()
                .parameters(
                    &charges,
                    expansion_order,
                    Laplace3dKernel::new(),
                    eval_type,
                    BlasFieldTranslationKiFmm::new(singular_value_threshold),
                )
                .unwrap()
                .build()
                .unwrap();
            fmm_blas.evaluate();
            let fmm_blas = Box::new(fmm_blas);
            test_single_node_fmm_matrix_helper(
                fmm_blas,
                eval_type,
                &sources,
                &charges,
                threshold_deriv,
            );
        }
    }

    fn test_root_multipole_laplace_single_node<T: RlstScalar<Real = T> + Float + Default>(
        fmm: Box<
            dyn Fmm<
                Precision = T,
                NodeIndex = MortonKey,
                Kernel = Laplace3dKernel<T>,
                Tree = SingleNodeFmmTree<T>,
            >,
        >,
        sources: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        charges: &Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        threshold: T,
    ) {
        let multipole = fmm.multipole(&ROOT).unwrap();
        let upward_equivalent_surface = ROOT.compute_surface(
            fmm.tree().domain(),
            fmm.expansion_order(),
            T::from(ALPHA_INNER).unwrap(),
        );

        let test_point = vec![T::from(100000.).unwrap(), T::zero(), T::zero()];
        let mut expected = vec![T::zero()];
        let mut found = vec![T::zero()];

        fmm.kernel().evaluate_st(
            bempp_traits::types::EvalType::Value,
            sources.data(),
            &test_point,
            charges.data(),
            &mut expected,
        );

        fmm.kernel().evaluate_st(
            bempp_traits::types::EvalType::Value,
            &upward_equivalent_surface,
            &test_point,
            multipole,
            &mut found,
        );

        let abs_error = num::Float::abs(expected[0] - found[0]);
        let rel_error = abs_error / expected[0];

        assert!(rel_error <= threshold);
    }

    #[test]
    fn test_upward_pass_vector() {
        // Setup random sources and targets
        let nsources = 10000;
        let ntargets = 10000;
        let sources = points_fixture::<f64>(nsources, None, None, Some(1));
        let targets = points_fixture::<f64>(ntargets, None, None, Some(1));

        // FMM parameters
        let n_crit = Some(100);
        let expansion_order = 6;
        let sparse = true;

        // Charge data
        let nvecs = 1;
        let mut rng = StdRng::seed_from_u64(0);
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().iter_mut().for_each(|c| *c = rng.gen());

        let fmm_fft = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                FftFieldTranslationKiFmm::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_fft.evaluate();

        let svd_threshold = Some(1e-5);
        let fmm_svd = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, n_crit, sparse)
            .unwrap()
            .parameters(
                &charges,
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                BlasFieldTranslationKiFmm::new(svd_threshold),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_svd.evaluate();

        let fmm_fft = Box::new(fmm_fft);
        let fmm_svd = Box::new(fmm_svd);
        test_root_multipole_laplace_single_node(fmm_fft, &sources, &charges, 1e-5);
        test_root_multipole_laplace_single_node(fmm_svd, &sources, &charges, 1e-5);
    }
}
