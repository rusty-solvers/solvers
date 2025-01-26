//! Test the neighbour evaluator

use bempp::{
    boundary_assemblers::BoundaryAssemblerOptions,
    evaluator_tools::{grid_points_from_space, NeighbourEvaluator},
};
use green_kernels::laplace_3d::Laplace3dKernel;
use ndelement::{ciarlet::LagrangeElementFamily, types::ReferenceCellType};
use rlst::{operator::zero_element, AsApply, OperatorBase, RandomAccessMut};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let grid = bempp::shapes::regular_sphere::<f64, _>(5, 1, &world);

    let quad_degree = 6;
    // Get the number of cells in the grid.

    let space = bempp::function::FunctionSpace::new(
        &grid,
        &LagrangeElementFamily::<f64>::new(1, ndelement::types::Continuity::Discontinuous),
    );

    let mut options = BoundaryAssemblerOptions::default();
    options.set_regular_quadrature_degree(ReferenceCellType::Triangle, quad_degree);

    let quad_degree = options
        .get_regular_quadrature_degree(ReferenceCellType::Triangle)
        .unwrap();

    let qrule = bempp_quadrature::simplex_rules::simplex_rule_triangle(quad_degree).unwrap();

    let neighbour_evaluator = NeighbourEvaluator::new(
        &space,
        &qrule.points,
        Laplace3dKernel::default(),
        green_kernels::types::GreenKernelEvalType::Value,
    );

    // We now manually test the evaluator. For that we first create a dense evaluator so that we have comparison.

    // For the evaluator we need all the points.

    let physical_points = grid_points_from_space(&space, &qrule.points);

    let kernel_evaluator = bempp::greens_function_evaluators::dense_evaluator::DenseEvaluator::new(
        &physical_points,
        &physical_points,
        green_kernels::types::GreenKernelEvalType::Value,
        true,
        Laplace3dKernel::default(),
        &world,
    );

    let mut x = zero_element(kernel_evaluator.domain());

    *x.view_mut().local_mut().get_mut([0]).unwrap() = 1.0;
    *x.view_mut().local_mut().get_mut([1]).unwrap() = 2.0;

    let actual = neighbour_evaluator.apply(x.r());
    let expected = kernel_evaluator.apply(x.r());

    println!("Actual: {}", actual.view().local()[[0]]);
    println!("Expected: {}", expected.view().local()[[0]]);
}
