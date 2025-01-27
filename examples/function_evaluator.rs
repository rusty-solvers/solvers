//! Demo the evaluation of functions in Bempp-rs

use bempp::function::FunctionSpaceTrait;
use bempp::function::LocalFunctionSpaceTrait;
use bempp::function::SpaceEvaluator;
use itertools::izip;
use itertools::Itertools;
use mpi::traits::Communicator;
use ndelement::ciarlet::LagrangeElementFamily;
use ndelement::types::ReferenceCellType;
use ndgrid::traits::Entity;
use ndgrid::traits::GeometryMap;
use ndgrid::traits::Grid;
use ndgrid::traits::ParallelGrid;
use rlst::{operator::zero_element, prelude::*};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as usize;

    // We create a sphere grid

    let grid = bempp::shapes::regular_sphere::<f64, _>(5, 1, &world);

    // We have the grid. Now create the function space.

    println!(
        "Number of vertices: {}",
        grid.local_grid().entity_count(ReferenceCellType::Point)
    );

    let space = bempp::function::FunctionSpace::<f64, _>::new(
        &grid,
        &LagrangeElementFamily::new(1, ndelement::types::Continuity::Standard),
    );

    // Let us setup the evaluator. We will evaluate on the midpoint of each cell.

    let evaluator = SpaceEvaluator::new(&space, &[1. / 3.0, 1. / 3.0], false);

    // Let us sample a function on the grid. We iterate through each cell, evaluate the function on the vertices and save
    // in the corresponding dof position.

    let mut x = zero_element(evaluator.domain());

    let geometry_map = space.grid().local_grid().geometry_map(
        ReferenceCellType::Triangle,
        &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );

    let mut points = rlst_dynamic_array2!(f64, [3, 3]);

    // We evaluate the function e^{ikx} for k = 1 on the grid points and save them in the
    // coefficient vector x

    for &cell in space.local_space().owned_support_cells() {
        let cell_dofs = space.local_space().cell_dofs(cell).unwrap();

        geometry_map.points(cell, points.data_mut());

        for (&cell_dof, p) in izip!(cell_dofs.iter(), points.col_iter()) {
            let global_dof = space.local_space().global_dof_index(cell_dof);
            if let Some(local_owned_index) = space.index_layout().global2local(rank, global_dof) {
                x.view_mut().local_mut()[[local_owned_index]] = p[[0]].cos();
            }
        }
    }

    // Let us now evaluate the function at the centre of the triangles.

    let result = evaluator.apply(x.r());

    // We now go through the grid owned grid cells and compare the computed result against the actual result.

    let mut expected = zero_element(evaluator.range());

    let geometry_map = space
        .grid()
        .local_grid()
        .geometry_map(ReferenceCellType::Triangle, &[1.0 / 3.0, 1.0 / 3.0]);

    let mut point = vec![0_f64; 3];

    for (cell, e) in izip!(
        space
            .grid()
            .local_grid()
            .cell_iter()
            .take_while(|cell| cell.is_owned()),
        expected.view_mut().local_mut().data_mut().iter_mut()
    ) {
        geometry_map.points(cell.local_index(), &mut point);

        *e = point[0].cos();
    }

    // Now compute the relative error

    let rel_err = (result.r() - expected.r()) / expected.r().norm();

    println!(
        "Max distance on rank {}: {}",
        rank,
        rel_err
            .view()
            .local()
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    );

    println!(
        "Max position on rank {}: {}",
        rank,
        rel_err
            .view()
            .local()
            .iter()
            .position_max_by(|a, b| a.total_cmp(b))
            .unwrap()
    );

    if world.rank() == 1 {
        println!("First value {}", result.r().view().local().data()[8]);
        println!("Expected value {}", expected.r().view().local().data()[8]);
    }
}
