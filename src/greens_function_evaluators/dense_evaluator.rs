//! A dense evaluator for Green's functions.

use std::rc::Rc;

use green_kernels::{traits::DistributedKernelEvaluator, types::GreenKernelEvalType};
use mpi::traits::{Communicator, Equivalence};
use num::One;
use rlst::{
    operator::{interface::DistributedArrayVectorSpace, zero_element},
    rlst_dynamic_array1, AsApply, Element, IndexLayout, OperatorBase, RawAccess, RawAccessMut,
    RlstScalar,
};

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
    eval_mode: GreenKernelEvalType,
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
        eval_mode: GreenKernelEvalType,
        use_multithreaded: bool,
        kernel: K,
        domain_space: Rc<DistributedArrayVectorSpace<'a, C, T>>,
        range_space: Rc<DistributedArrayVectorSpace<'a, C, T>>,
    ) -> Self {
        // We want that both layouts have the same communicator.
        assert!(std::ptr::addr_eq(domain_space.comm(), range_space.comm()));

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

        // The length of the source vector must be 3 times the length of the local source indices.
        assert_eq!(
            sources.len(),
            3 * domain_space.index_layout().number_of_local_indices(),
            "Number of sources ({}) does not match number of local indices ({}).",
            sources.len() / 3,
            domain_space.index_layout().number_of_local_indices(),
        );

        // The length of the target vector must be 3 times the length of the local target indices.
        assert_eq!(
            targets.len(),
            3 * range_space.index_layout().number_of_local_indices(),
            "Number of targets ({}) does not match number of local indices ({}).",
            targets.len() / 3,
            range_space.index_layout().number_of_local_indices(),
        );

        Self {
            sources: sources.to_vec(),
            targets: targets.to_vec(),
            eval_mode,
            kernel,
            use_multithreaded,
            domain_space,
            range_space,
        }
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
    fn apply_extended(
        &self,
        alpha: <Self::Range as rlst::LinearSpace>::F,
        x: &<Self::Domain as rlst::LinearSpace>::E,
        beta: <Self::Range as rlst::LinearSpace>::F,
        y: &mut <Self::Range as rlst::LinearSpace>::E,
    ) {
        y.scale_inplace(beta);
        let mut charges = rlst_dynamic_array1!(
            T,
            [self.domain_space.index_layout().number_of_local_indices()]
        );

        charges.fill_from(x.view().local().r().scalar_mul(alpha));

        self.kernel.evaluate_distributed(
            self.eval_mode,
            self.sources.as_slice(),
            self.targets.as_slice(),
            charges.data(),
            y.view_mut().local_mut().data_mut(),
            self.use_multithreaded,
            self.domain_space.comm(), // domain space and range space have the same communicator
        );
    }

    fn apply(
        &self,
        x: &<Self::Domain as rlst::LinearSpace>::E,
    ) -> <Self::Range as rlst::LinearSpace>::E {
        let mut y = zero_element(self.range());
        self.apply_extended(<T as One>::one(), x, T::zero(), &mut y);
        y
    }
}
