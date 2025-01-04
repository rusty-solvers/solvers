//! Test the neighbour evaluator

use bempp::evaluator_tools::NeighbourEvaluator;
use bempp_distributed_tools::IndexLayoutFromLocalCounts;
use green_kernels::laplace_3d::Laplace3dKernel;
use mpi::traits::Communicator;
use ndgrid::{
    traits::{Entity, Grid},
    types::Ownership,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rlst::{
    operator::interface::DistributedArrayVectorSpace, rlst_dynamic_array2, AsApply, Element,
    LinearSpace, RawAccess,
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

    let index_layout = IndexLayoutFromLocalCounts::new(n_cells * n_points, &world);

    let space = DistributedArrayVectorSpace::<_, f64>::new(&index_layout);

    let evaluator = NeighbourEvaluator::new(
        points.data(),
        Laplace3dKernel::default(),
        green_kernels::types::GreenKernelEvalType::Value,
        &space,
        &space,
        &grid,
    );

    // Create an element in the space.

    let mut x = space.zero();

    x.view_mut()
        .local_mut()
        .fill_from_equally_distributed(&mut rng);

    let _res = evaluator.apply(&x);
}
