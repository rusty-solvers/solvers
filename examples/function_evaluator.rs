//! Demo the evaluation of functions in Bempp-rs

use std::rc::Rc;

use bempp::function::FunctionSpaceTrait;
use bempp::function::LocalFunctionSpaceTrait;
use bempp::function::SpaceEvaluator;
use itertools::izip;
use mpi::traits::Communicator;
use ndelement::ciarlet::LagrangeElementFamily;
use ndelement::types::ReferenceCellType;
use ndgrid::traits::Entity;
use ndgrid::traits::GeometryMap;
use ndgrid::traits::Grid;
use ndgrid::traits::ParallelGrid;
use rlst::operator::interface::DistributedArrayVectorSpace;
use rlst::{operator::zero_element, prelude::*};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as usize;

    let refinement_level = 5;

    // We create a sphere grid

    let grid = bempp::shapes::regular_sphere::<f64, _>(refinement_level, 1, &world);

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

    let evaluator = SpaceEvaluator::new(&space, &[1. / 3.0, 1. / 3.0], true);

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

    // We now evaluate the same on the first process and check that the results are identical.

    // We need to send the result vector to process zero.

    if rank == 0 {
        let self_comm = mpi::topology::SimpleCommunicator::self_comm();
        let grid = bempp::shapes::regular_sphere::<f64, _>(refinement_level, 1, &self_comm);

        let space = bempp::function::FunctionSpace::new(
            &grid,
            &LagrangeElementFamily::<f64>::new(1, ndelement::types::Continuity::Standard),
        );

        // Now evaluate the comparison vector here.

        let array_space = DistributedArrayVectorSpace::<_, f64>::from_index_layout(Rc::new(
            IndexLayout::from_local_counts(grid.local_grid().cell_count(), &self_comm),
        ));

        // Let's go through the grid now, and save in the correct order of cell ids.

        let mut expected = zero_element(array_space.clone());

        let geometry_map = space
            .grid()
            .local_grid()
            .geometry_map(ReferenceCellType::Triangle, &[1.0 / 3.0, 1.0 / 3.0]);

        let mut point = vec![0_f64; 3];

        for cell in izip!(space
            .grid()
            .local_grid()
            .cell_iter()
            .take_while(|cell| cell.is_owned()),)
        {
            geometry_map.points(cell.local_index(), &mut point);

            let id = cell.id().unwrap();

            expected.view_mut().local_mut()[[id]] = point[0].cos();
        }

        // We have the expected vector. Now gather the distributed vector to the root and compare with the expected.

        let mut root_result = zero_element(array_space.clone());

        result
            .view()
            .gather_to_rank_root(root_result.view_mut().local_mut().r_mut());

        // Let us now compare the results.
        let rel_err = (root_result.r() - expected.r()).norm() / expected.r().norm();

        println!("Relative error: {}", rel_err);
    } else {
        result.view().gather_to_rank(0);
    }
}
