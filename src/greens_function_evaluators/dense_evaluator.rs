//! A dense evaluator for Green's functions.

use std::rc::Rc;

use green_kernels::{traits::DistributedKernelEvaluator, types::GreenKernelEvalType};
use mpi::traits::{Communicator, Equivalence};
use rlst::{
    operator::{interface::DistributedArrayVectorSpace, Operator},
    rlst_dynamic_array1, AsApply, IndexLayout, OperatorBase, RawAccess, RawAccessMut, RlstScalar,
};

use crate::{evaluator_tools::grid_points_from_space, function::FunctionSpaceTrait};

/// Wrapper for a dense Green's function evaluator.
pub struct DenseEvaluator<
    'a,
    C: Communicator,
    T: RlstScalar + Equivalence,
    K: DistributedKernelEvaluator<T = T>,
> where
    T::Real: Equivalence,
{
    sources: Vec<T::Real>,
    targets: Vec<T::Real>,
    eval_type: GreenKernelEvalType,
    use_multithreaded: bool,
    kernel: K,
    domain_space: Rc<DistributedArrayVectorSpace<'a, C, T>>,
    range_space: Rc<DistributedArrayVectorSpace<'a, C, T>>,
}

impl<C: Communicator, T: RlstScalar + Equivalence, K: DistributedKernelEvaluator<T = T>>
    std::fmt::Debug for DenseEvaluator<'_, C, T, K>
where
    T::Real: Equivalence,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DenseEvaluator with {} sources and {} targets",
            self.domain_space.index_layout().number_of_global_indices(),
            self.range_space.index_layout().number_of_global_indices()
        )
    }
}

impl<'a, C: Communicator, T: RlstScalar + Equivalence, K: DistributedKernelEvaluator<T = T>>
    DenseEvaluator<'a, C, T, K>
where
    T::Real: Equivalence,
{
    /// Create a new dense evaluator.
    pub fn new(
        sources: &[T::Real],
        targets: &[T::Real],
        eval_type: GreenKernelEvalType,
        use_multithreaded: bool,
        kernel: K,
        comm: &'a C,
    ) -> Self {
        // We want that both layouts have the same communicator.

        assert_eq!(
            sources.len() % 3,
            0,
            "Source vector length must be a multiple of 3."
        );
        assert_eq!(
            targets.len() % 3,
            0,
            "Target vector length must be a multiple of 3."
        );

        let n_sources = sources.len() / 3;
        let n_targets = targets.len() / 3;

        let domain_space = DistributedArrayVectorSpace::from_index_layout(Rc::new(
            IndexLayout::from_local_counts(n_sources, comm),
        ));

        let range_space = DistributedArrayVectorSpace::from_index_layout(Rc::new(
            IndexLayout::from_local_counts(n_targets, comm),
        ));

        Self {
            sources: sources.to_vec(),
            targets: targets.to_vec(),
            eval_type,
            kernel,
            use_multithreaded,
            domain_space,
            range_space,
        }
    }

    /// Create a new dense assembler from a trial space and test space.
    pub fn from_spaces<Space: FunctionSpaceTrait<T = T, C = C>>(
        trial_space: &'a Space,
        test_space: &'a Space,
        eval_type: GreenKernelEvalType,
        use_multithreaded: bool,
        kernel: K,
        quad_points: &[T::Real],
    ) -> Operator<Self> {
        let source_points = grid_points_from_space(trial_space, quad_points);
        let target_points = grid_points_from_space(test_space, quad_points);

        Operator::new(Self::new(
            &source_points,
            &target_points,
            eval_type,
            use_multithreaded,
            kernel,
            trial_space.comm(),
        ))
    }
}

impl<'a, C: Communicator, T: RlstScalar + Equivalence, K: DistributedKernelEvaluator<T = T>>
    OperatorBase for DenseEvaluator<'a, C, T, K>
where
    T::Real: Equivalence,
{
    type Domain = DistributedArrayVectorSpace<'a, C, T>;

    type Range = DistributedArrayVectorSpace<'a, C, T>;

    fn domain(&self) -> Rc<Self::Domain> {
        self.domain_space.clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.range_space.clone()
    }
}

impl<C: Communicator, T: RlstScalar + Equivalence, K: DistributedKernelEvaluator<T = T>> AsApply
    for DenseEvaluator<'_, C, T, K>
where
    T::Real: Equivalence,
{
    fn apply_extended<
        ContainerIn: rlst::ElementContainer<E = <Self::Domain as rlst::LinearSpace>::E>,
        ContainerOut: rlst::ElementContainerMut<E = <Self::Range as rlst::LinearSpace>::E>,
    >(
        &self,
        alpha: <Self::Range as rlst::LinearSpace>::F,
        x: rlst::Element<ContainerIn>,
        beta: <Self::Range as rlst::LinearSpace>::F,
        mut y: rlst::Element<ContainerOut>,
    ) {
        y.scale_inplace(beta);
        let mut charges = rlst_dynamic_array1!(
            T,
            [self.domain_space.index_layout().number_of_local_indices()]
        );

        charges.fill_from(x.view().local().r().scalar_mul(alpha));

        self.kernel.evaluate_distributed(
            self.eval_type,
            self.sources.as_slice(),
            self.targets.as_slice(),
            charges.data(),
            y.view_mut().local_mut().data_mut(),
            self.use_multithreaded,
            self.domain_space.comm(), // domain space and range space have the same communicator
        );
    }
}
