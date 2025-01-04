//! This file implements an example Laplace evaluator test the different involved operators.

use bempp::{boundary_assemblers::BoundaryAssemblerOptions, function::LocalFunctionSpaceTrait};
use bempp_quadrature::types::NumericalQuadratureGenerator;
use ndelement::{ciarlet::LagrangeElementFamily, types::ReferenceCellType};
use ndgrid::traits::Grid;
use rlst::operator::interface::DistributedArrayVectorSpace;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let grid = bempp::shapes::regular_sphere::<f64, _>(3, 1, &world);

    // Get the number of cells in the grid.

    let n_cells = grid.entity_iter(2).count();

    let space = bempp::function::FunctionSpace::new(
        &grid,
        &LagrangeElementFamily::<f64>::new(1, ndelement::types::Continuity::Standard),
    );

    let mut options = BoundaryAssemblerOptions::default();
    options.set_regular_quadrature_degree(ReferenceCellType::Triangle, 6);

    let quad_degree = options
        .get_regular_quadrature_degree(ReferenceCellType::Triangle)
        .unwrap();

    let assembler = bempp::laplace::assembler::single_layer::<f64>(&options);

    let dense_matrix = assembler.assemble(&space, &space);

    // Now let's build an evaluator.

    //First initialise the index layouts.

    let space_layout =
        bempp_distributed_tools::SingleProcessIndexLayout::new(0, space.global_size(), &world);

    let point_layout = bempp_distributed_tools::SingleProcessIndexLayout::new(
        0,
        quad_degree * space.global_size(),
        &world,
    );

    // Instantiate function spaces.

    let array_function_space = DistributedArrayVectorSpace::<_, f64>::new(&space_layout);
    let point_function_space = DistributedArrayVectorSpace::<_, f64>::new(&point_layout);

    let qrule = bempp_quadrature::simplex_rules::simplex_rule_triangle(6).unwrap();

    println!("Here");
    let space_to_point = bempp::evaluator_tools::basis_to_point_map(
        &space,
        &array_function_space,
        &point_function_space,
        &qrule.points,
        &qrule.weights,
        false,
    );
}
