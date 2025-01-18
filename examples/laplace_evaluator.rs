//! This file implements an example Laplace evaluator test the different involved operators.

use std::rc::Rc;

use bempp::{
    boundary_assemblers::BoundaryAssemblerOptions, evaluator_tools::NeighbourEvaluator,
    function::LocalFunctionSpaceTrait,
};
use green_kernels::laplace_3d::Laplace3dKernel;
use mpi::traits::Communicator;
use ndelement::{ciarlet::LagrangeElementFamily, types::ReferenceCellType};
use ndgrid::{
    traits::{Entity, GeometryMap, Grid},
    types::Ownership,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rlst::{
    operator::{interface::DistributedArrayVectorSpace, zero_element, Operator},
    rlst_dynamic_array1, AsApply, MultInto, OperatorBase,
};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let mut rng = ChaCha8Rng::seed_from_u64(world.rank() as u64);

    let grid = bempp::shapes::regular_sphere::<f64, _>(5, 1, &world);

    let quad_degree = 6;
    // Get the number of cells in the grid.

    let n_cells = grid.entity_iter(2).count();

    println!("Number of cells: {}", n_cells);

    let space = bempp::function::FunctionSpace::new(
        &grid,
        &LagrangeElementFamily::<f64>::new(0, ndelement::types::Continuity::Discontinuous),
    );

    let mut options = BoundaryAssemblerOptions::default();
    options.set_regular_quadrature_degree(ReferenceCellType::Triangle, quad_degree);

    let quad_degree = options
        .get_regular_quadrature_degree(ReferenceCellType::Triangle)
        .unwrap();

    let assembler = bempp::laplace::assembler::single_layer::<f64>(&options);

    let dense_matrix = assembler.assemble(&space, &space);

    // Now let's build an evaluator.

    //First initialise the index layouts.

    let space_layout = Rc::new(bempp_distributed_tools::IndexLayout::from_local_counts(
        space.global_size(),
        &world,
    ));

    let point_layout = Rc::new(bempp_distributed_tools::IndexLayout::from_local_counts(
        quad_degree * n_cells,
        &world,
    ));

    // Instantiate function spaces.

    let array_function_space =
        DistributedArrayVectorSpace::<_, f64>::from_index_layout(space_layout.clone());
    let point_function_space =
        DistributedArrayVectorSpace::<_, f64>::from_index_layout(point_layout.clone());

    let qrule = bempp_quadrature::simplex_rules::simplex_rule_triangle(quad_degree).unwrap();

    let space_to_point =
        bempp::evaluator_tools::basis_to_point_map(&space, &qrule.points, &qrule.weights, false);

    let point_to_space =
        bempp::evaluator_tools::basis_to_point_map(&space, &qrule.points, &qrule.weights, true);

    // We now have to get all the points from the grid. We do this by iterating through all cells.

    let mut points = vec![0 as f64; 3 * quad_degree * n_cells];

    let geometry_map = grid.geometry_map(ReferenceCellType::Triangle, &qrule.points);

    for cell in grid
        .entity_iter(2)
        .filter(|e| matches!(e.ownership(), Ownership::Owned))
    {
        let start_index = 3 * point_function_space
            .index_layout()
            .global2local(world.rank() as usize, qrule.npoints * cell.global_index())
            .unwrap();
        geometry_map.points(
            cell.local_index(),
            &mut points[start_index..start_index + 3 * qrule.npoints],
        );
    }

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
        &qrule.points,
        Laplace3dKernel::default(),
        green_kernels::types::GreenKernelEvalType::Value,
        space.support_cells(),
        &grid,
    );

    let corrected_evaluator = kernel_evaluator.r().sum(correction.r().scale(-1.0));

    let prod1 = corrected_evaluator.r().product(space_to_point.r());

    let singular_operator = Operator::from(assembler.assemble_singular(&space, &space));

    let laplace_evaluator = (point_to_space.product(prod1)).sum(singular_operator.r());

    let mut x = zero_element(array_function_space.clone());
    x.view_mut()
        .local_mut()
        .fill_from_equally_distributed(&mut rng);

    let res = laplace_evaluator.apply(x.r());

    let res_local = res.view().local();

    let mut expected = rlst_dynamic_array1!(f64, [space.global_size()]);

    expected
        .r_mut()
        .simple_mult_into(dense_matrix.r(), x.view().local().r());

    let rel_diff = (expected.r() - res_local.r()).norm_2() / expected.r().norm_2();

    println!("Relative difference: {}", rel_diff);
}
