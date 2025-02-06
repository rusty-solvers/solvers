//! Test function space generation

use std::time::Instant;

use mpi::traits::Communicator;
use ndelement::ciarlet::LagrangeElementFamily;
use ndgrid::traits::ParallelGrid;

fn main() {
    // We first evaluate the operator on a parallel grid

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let refinement_level = 9;

    let grid = bempp::shapes::regular_sphere::<f64, _>(refinement_level, 1, &world);
    if rank == 0 {
        println!(
            "Number of elements: {}",
            grid.cell_layout().number_of_global_indices()
        );
    }

    // Get the number of cells in the grid.

    let now = Instant::now();
    let _space = bempp::function::FunctionSpace::new(
        &grid,
        &LagrangeElementFamily::<f64>::new(1, ndelement::types::Continuity::Standard),
    );
    let elapsed = now.elapsed();

    if rank == 0 {
        println!("Function space generated in {} ms", elapsed.as_millis());
    }
}
