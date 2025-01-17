// Compare dense and FMM Green's function evaluators.

use std::rc::Rc;

use bempp::greens_function_evaluators::kifmm_evaluator::KiFmmEvaluator;
use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use mpi::traits::{Communicator, Root};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rlst::{
    operator::{interface::DistributedArrayVectorSpace, zero_element},
    rlst_dynamic_array1, AsApply, IndexLayout, RawAccessMut,
};

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as usize;
    let rank = world.rank() as usize;

    // Number of points per process.
    let npoints = 10000;

    // Seed the random number generator with a different seed for each process.
    let mut rng = ChaCha8Rng::seed_from_u64(world.rank() as u64);

    // Create random sources and targets.

    let sources = (0..3 * npoints)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect::<Vec<f64>>();

    let targets = sources.clone();

    // Initialise MPI

    // Initalise the index layout.

    let index_layout = Rc::new(IndexLayout::from_local_counts(npoints, &world));

    // Create the vector space.

    let space = DistributedArrayVectorSpace::<_, f64>::from_index_layout(index_layout.clone());

    // Create a random vector of charges.

    let mut charges = zero_element(space.clone());

    // charges
    //     .view_mut()
    //     .local_mut()
    //     .fill_from_equally_distributed(&mut rng);

    charges.view_mut().local_mut().set_one();

    // Create the dense evaluator.

    let dense_evaluator = bempp::greens_function_evaluators::dense_evaluator::DenseEvaluator::new(
        &sources,
        &targets,
        GreenKernelEvalType::Value,
        false,
        Laplace3dKernel::default(),
        &world,
    );

    // Create the FMM evaluator.

    let fmm_evaluator = KiFmmEvaluator::new(&sources, &targets, 3, 1, 5, &world);

    // Apply the dense evaluator.

    let output_dense = dense_evaluator.apply(charges.r());

    // Apply the FMM evaluator

    let output_fmm = fmm_evaluator.apply(charges.r());

    // Compare the results.

    let dense_norm = output_dense.norm();

    let rel_diff = (output_dense.r() - output_fmm.r()).norm() / dense_norm;

    if world.rank() == 0 {
        println!("The relative error is: {}", rel_diff);
    }

    // We now gather back the data to the root process and repeat the calculation on just a single node.

    if rank != 0 {
        world.process_at_rank(0).gather_into(&sources);
        world.process_at_rank(0).gather_into(&targets);
        charges.view().gather_to_rank(0);
        output_fmm.view().gather_to_rank(0);
        output_dense.view().gather_to_rank(0);
    } else {
        let mut gathered_sources = vec![0.0; 3 * npoints * size];
        let mut gathered_targets = vec![0.0; 3 * npoints * size];
        let mut gathered_charges = rlst_dynamic_array1!(f64, [npoints * size]);
        let mut gathered_output_fmm = rlst_dynamic_array1!(f64, [npoints * size]);
        let mut gathered_output_dense = rlst_dynamic_array1!(f64, [npoints * size]);

        world
            .this_process()
            .gather_into_root(&sources, &mut gathered_sources);

        world
            .this_process()
            .gather_into_root(&targets, &mut gathered_targets);

        charges.view().gather_to_rank_root(gathered_charges.r_mut());
        output_fmm
            .view()
            .gather_to_rank_root(gathered_output_fmm.r_mut());
        output_dense
            .view()
            .gather_to_rank_root(gathered_output_dense.r_mut());

        // Now we have everything on root. Let's create a self communicator just on root.

        let root_comm = mpi::topology::SimpleCommunicator::self_comm();

        let index_layout_root = Rc::new(IndexLayout::from_local_counts(npoints * size, &root_comm));
        let space_root =
            DistributedArrayVectorSpace::<_, f64>::from_index_layout(index_layout_root);
        let evaluator_dense_on_root =
            bempp::greens_function_evaluators::dense_evaluator::DenseEvaluator::new(
                &gathered_sources,
                &gathered_targets,
                GreenKernelEvalType::Value,
                false,
                Laplace3dKernel::default(),
                &root_comm,
            );
        let fmm_evaluator_on_root =
            KiFmmEvaluator::new(&gathered_sources, &gathered_targets, 3, 1, 5, &root_comm);

        // Create the charge vector on root.

        let mut charges_on_root = zero_element(space_root);

        charges_on_root
            .view_mut()
            .local_mut()
            .data_mut()
            .copy_from_slice(gathered_charges.data_mut());

        // Now apply the fmm evaluator
        let fmm_result_on_root = fmm_evaluator_on_root.apply(charges_on_root.r());
        // Now apply the dense evaluator
        let dense_result_on_root = evaluator_dense_on_root.apply(charges_on_root.r());

        // Now compare the dense result on root with the global dense result.

        let dense_rel_diff = (gathered_output_dense.r() - dense_result_on_root.view().local().r())
            .norm_2()
            / gathered_output_dense.r().norm_2();

        println!(
            "Dense difference between MPI and single node: {}",
            dense_rel_diff
        );

        let fmm_rel_diff = (gathered_output_fmm.r() - fmm_result_on_root.view().local().r())
            .norm_2()
            / gathered_output_fmm.r().norm_2();

        println!(
            "FMM difference between MPI and single node: {}",
            fmm_rel_diff
        );
    }
}
