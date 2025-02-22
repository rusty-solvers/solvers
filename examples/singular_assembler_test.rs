//! This file implements an example Laplace evaluator test the different involved operators.

use bempp::{
    boundary_assemblers::BoundaryAssemblerOptions,
    function::{FunctionSpaceTrait, LocalFunctionSpaceTrait, SpaceEvaluator},
    laplace,
};
use green_kernels::laplace_3d::Laplace3dKernel;
use itertools::izip;
use mpi::traits::{Communicator, Equivalence};
use ndelement::{ciarlet::LagrangeElementFamily, types::ReferenceCellType};
use ndgrid::traits::{GeometryMap, Grid, ParallelGrid};
use num::{One, Zero};
use rlst::{prelude::*, tracing::trace_call};

fn main() {
    // We first evaluate the operator on a parallel grid

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    if rank == 0 {
        env_logger::init();
    }

    let refinement_level = 7;

    let grid = bempp::shapes::regular_sphere::<f64, _>(refinement_level, 1, &world);
    if rank == 0 {
        println!(
            "Number of elements: {}",
            grid.cell_layout().number_of_global_indices()
        );
    }

    let quad_degree = 6;
    // Get the number of cells in the grid.

    let space = bempp::function::FunctionSpace::new(
        &grid,
        &LagrangeElementFamily::<f64>::new(1, ndelement::types::Continuity::Standard),
    );

    let mut options = BoundaryAssemblerOptions::default();
    options.set_regular_quadrature_degree(ReferenceCellType::Triangle, quad_degree);

    // We now have to get all the points from the grid. We do this by iterating through all cells.

    let assembler = laplace::single_layer(&options);

    let _sing_mat = trace_call("sing_assembly_time", || {
        assembler.assemble_singular(&space, &space)
    });
}
