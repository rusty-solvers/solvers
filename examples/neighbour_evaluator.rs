//! Test the neighbour evaluator

use std::rc::Rc;

use bempp::evaluator_tools::NeighbourEvaluator;
use green_kernels::laplace_3d::Laplace3dKernel;
use itertools::Itertools;
use mpi::traits::Communicator;
use ndelement::types::ReferenceCellType;
use ndgrid::{
    traits::{Entity, GeometryMap, Grid},
    types::Ownership,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rlst::{
    operator::{interface::DistributedArrayVectorSpace, zero_element},
    rlst_dynamic_array2, AsApply, IndexLayout, RandomAccessMut, RawAccess,
};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let n_points = 5;

    let mut rng = ChaCha8Rng::seed_from_u64(world.rank() as u64);

    let mut points = rlst_dynamic_array2!(f64, [2, n_points]);
    points.fill_from_equally_distributed(&mut rng);

    let grid = bempp::shapes::regular_sphere::<f64, _>(3, 1, &world);

    // Now get the active cells on the current process.

    let n_cells = grid
        .entity_iter(2)
        .filter(|e| matches!(e.ownership(), Ownership::Owned))
        .count();

    let index_layout = Rc::new(IndexLayout::from_local_counts(n_cells * n_points, &world));

    let space = DistributedArrayVectorSpace::<_, f64>::from_index_layout(index_layout.clone());

    let neighbour_evaluator = NeighbourEvaluator::new(
        points.data(),
        Laplace3dKernel::default(),
        green_kernels::types::GreenKernelEvalType::Value,
        &(0..grid.entity_count(ReferenceCellType::Triangle)).collect_vec(),
        &grid,
    );

    // We now manually test the evaluator. For that we first create a dense evaluator so that we have comparison.

    // For the evaluator we need all the points.

    let mut physical_points = vec![0 as f64; 3 * n_points * n_cells];

    let geometry_map = grid.geometry_map(ReferenceCellType::Triangle, points.data());

    for cell in grid
        .entity_iter(2)
        .filter(|e| matches!(e.ownership(), Ownership::Owned))
    {
        let start_index = 3 * index_layout
            .global2local(world.rank() as usize, n_points * cell.global_index())
            .unwrap();
        geometry_map.points(
            cell.local_index(),
            &mut physical_points[start_index..start_index + 3 * n_points],
        );
    }

    let kernel_evaluator = bempp::greens_function_evaluators::dense_evaluator::DenseEvaluator::new(
        &physical_points,
        &physical_points,
        green_kernels::types::GreenKernelEvalType::Value,
        true,
        Laplace3dKernel::default(),
        space.clone(),
        space.clone(),
    );

    let mut x = zero_element(space.clone());

    *x.view_mut().local_mut().get_mut([0]).unwrap() = 1.0;
    *x.view_mut().local_mut().get_mut([1]).unwrap() = 2.0;

    let actual = neighbour_evaluator.apply(x.r());
    let expected = kernel_evaluator.apply(x.r());

    println!("Actual: {}", actual.view().local()[[0]]);
    println!("Expected: {}", expected.view().local()[[0]]);
}
