//! This file implements an example Laplace evaluator test the different involved operators.

use bempp::{
    boundary_assemblers::BoundaryAssemblerOptions,
    evaluator_tools::NeighbourEvaluator,
    function::{FunctionSpaceTrait, LocalFunctionSpaceTrait},
};
use green_kernels::laplace_3d::Laplace3dKernel;
use mpi::traits::{Communicator, CommunicatorCollectives};
use ndelement::{ciarlet::LagrangeElementFamily, types::ReferenceCellType};
use ndgrid::traits::{Grid, ParallelGrid};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rlst::{
    operator::{zero_element, Operator},
    rlst_dynamic_array1, AsApply, MultInto, OperatorBase,
};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let mut rng = ChaCha8Rng::seed_from_u64(world.rank() as u64);

    let grid = bempp::shapes::regular_sphere::<f64, _>(4, 1, &world);
    println!(
        "Number of elements: {}",
        grid.cell_layout().number_of_global_indices()
    );

    let quad_degree = 6;
    // Get the number of cells in the grid.

    let space = bempp::function::FunctionSpace::new(
        &grid,
        &LagrangeElementFamily::<f64>::new(1, ndelement::types::Continuity::Standard),
    );

    let mut options = BoundaryAssemblerOptions::default();
    options.set_regular_quadrature_degree(ReferenceCellType::Triangle, quad_degree);

    let quad_degree = options
        .get_regular_quadrature_degree(ReferenceCellType::Triangle)
        .unwrap();

    let assembler = bempp::laplace::assembler::single_layer::<f64>(&options);

    //let dense_matrix = assembler.assemble(&space, &space);

    // Now let's build an evaluator.

    // Instantiate function spaces.

    let qrule = bempp_quadrature::simplex_rules::simplex_rule_triangle(quad_degree).unwrap();

    let space_to_point =
        bempp::evaluator_tools::basis_to_point_map(&space, &qrule.points, &qrule.weights, false);

    let point_to_space =
        bempp::evaluator_tools::basis_to_point_map(&space, &qrule.points, &qrule.weights, true);

    // We now have to get all the points from the grid. We do this by iterating through all cells.

    let points = bempp::evaluator_tools::grid_points_from_space(&space, &qrule.points);

    let kernel_evaluator = bempp::greens_function_evaluators::dense_evaluator::DenseEvaluator::new(
        &points,
        &points,
        green_kernels::types::GreenKernelEvalType::Value,
        true,
        Laplace3dKernel::default(),
        &world,
    );

    // let kernel_evaluator = bempp::greens_function_evaluators::kifmm_evaluator::KiFmmEvaluator::new(
    //     &points, &points, 1, 3, 5, &world,
    // );

    let correction = NeighbourEvaluator::new(
        &space,
        &qrule.points,
        Laplace3dKernel::default(),
        green_kernels::types::GreenKernelEvalType::Value,
    );

    let corrected_evaluator = kernel_evaluator.r().sum(correction.r().scale(-1.0));

    let prod1 = corrected_evaluator.r().product(space_to_point.r());

    let singular_operator = Operator::from(assembler.assemble_singular(&space, &space));

    let laplace_evaluator = (point_to_space.product(prod1)).sum(singular_operator.r());

    let mut x = zero_element(laplace_evaluator.domain());
    x.view_mut()
        .local_mut()
        .fill_from_equally_distributed(&mut rng);

    let res = laplace_evaluator.apply(x.r());

    println!("Finished.");

    // let res_local = res.view().local();

    // let mut expected = rlst_dynamic_array1!(f64, [space.local_space().global_size()]);

    // expected
    //     .r_mut()
    //     .simple_mult_into(dense_matrix.r(), x.view().local().r());

    // let rel_diff = (expected.r() - res_local.r()).norm_2() / expected.r().norm_2();

    // println!("Relative difference: {}", rel_diff);
}
