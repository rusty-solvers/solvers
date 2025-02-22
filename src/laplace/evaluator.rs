//! Laplace evaluators

use green_kernels::laplace_3d::Laplace3dKernel;
use itertools::Itertools;
use ndgrid::traits::Grid;
use rlst::prelude::*;

use crate::{
    boundary_assemblers::BoundaryAssemblerOptions,
    evaluator_tools::{space_to_point_map, NeighbourEvaluator},
    function::{FunctionSpaceTrait, LocalFunctionSpaceTrait},
};

/// Implement a single layer operator
#[measure_duration(id = "laplace_single_layer_evaluate")]
#[allow(clippy::type_complexity)]
pub fn single_layer<
    'a,
    KernelEval: AsApply<
        Domain = DistributedArrayVectorSpace<'a, Space::C, Space::T>,
        Range = DistributedArrayVectorSpace<'a, Space::C, Space::T>,
    >,
    Space: FunctionSpaceTrait,
>(
    trial_space: &'a Space,
    test_space: &'a Space,
    kernel_evaluator: Operator<KernelEval>,
    options: &BoundaryAssemblerOptions,
) -> Operator<
    impl AsApply<
        Domain = DistributedArrayVectorSpace<'a, Space::C, Space::T>,
        Range = DistributedArrayVectorSpace<'a, Space::C, Space::T>,
    >,
>
where
    Space::T: MatrixInverse + RlstScalar<Real = Space::T>,
    Space::LocalFunctionSpace: Sync,
    Space::LocalGrid: Sync,
    for<'b> <Space::LocalGrid as Grid>::GeometryMap<'b>: Sync,
{
    // First we setup the point maps

    let quad_rule = options.get_regular_quadrature_rule(trial_space.local_space().cell_type());

    let quad_points = quad_rule
        .points
        .iter()
        .map(|&elem| num::cast(elem).unwrap())
        .collect_vec();
    let quad_weights = quad_rule
        .weights
        .iter()
        .map(|&elem| num::cast(elem).unwrap())
        .collect_vec();

    let space_to_point = space_to_point_map(trial_space, &quad_points, &quad_weights, false);
    let point_to_space = space_to_point_map(trial_space, &quad_points, &quad_weights, true);

    // Now we need to setup the assembler
    let assembler = super::assembler::single_layer::<Space::T>(options);

    // And the neighbour correction

    let correction = NeighbourEvaluator::from_spaces_and_kernel(
        trial_space,
        test_space,
        &quad_points,
        Laplace3dKernel::default(),
        green_kernels::types::GreenKernelEvalType::Value,
    );

    // We also need the sparse matrix with the actual singular integrals

    let singular_operator = Operator::from(assembler.assemble_singular(trial_space, test_space));

    // We have all the operators. We can now put the evaluator together

    point_to_space * (kernel_evaluator - correction) * space_to_point + singular_operator
}
