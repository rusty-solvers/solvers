//! This file implements an example Laplace evaluator test the different involved operators.

use bempp::{
    boundary_assemblers::BoundaryAssemblerOptions,
    evaluator_tools::NeighbourEvaluator,
    function::{FunctionSpaceTrait, LocalFunctionSpaceTrait, SpaceEvaluator},
    laplace,
};
use green_kernels::laplace_3d::Laplace3dKernel;
use itertools::izip;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};
use ndelement::{ciarlet::LagrangeElementFamily, types::ReferenceCellType};
use ndgrid::traits::{Geometry, GeometryMap, Grid, ParallelGrid};
use num::{One, Zero};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rlst::prelude::*;
use rlst::{
    operator::{zero_element, Operator},
    rlst_dynamic_array1, rlst_dynamic_array2, AsApply, DistributedVector, MultInto, OperatorBase,
    RawAccessMut, RlstScalar,
};

// Sample a function on a continuous P1 grid
fn sample_function<'a, T: RlstScalar + Equivalence, Space: FunctionSpaceTrait<T = T>>(
    space: &'a Space,
    fun: &impl Fn(&[T::Real]) -> T,
) -> DistributedVector<'a, Space::C, Space::T>
where
    T::Real: RlstScalar,
{
    let rank = space.comm().rank() as usize;

    let geometry_map = space.grid().local_grid().geometry_map(
        ReferenceCellType::Triangle,
        &[
            <T::Real as Zero>::zero(),
            <T::Real as Zero>::zero(),
            <T::Real as One>::one(),
            <T::Real as Zero>::zero(),
            <T::Real as Zero>::zero(),
            <T::Real as One>::one(),
        ],
    );

    let x = DistributedVector::<_, T>::new(space.index_layout());

    let mut points = rlst_dynamic_array2!(T::Real, [3, 3]);

    for &cell in space.local_space().owned_support_cells() {
        let cell_dofs = space.local_space().cell_dofs(cell).unwrap();

        geometry_map.points(cell, points.data_mut());

        for (&cell_dof, p) in izip!(cell_dofs.iter(), points.col_iter()) {
            let global_dof = space.local_space().global_dof_index(cell_dof);
            if let Some(local_owned_index) = space.index_layout().global2local(rank, global_dof) {
                x.local_mut()[[local_owned_index]] = fun(p.data());
            }
        }
    }

    x
}

fn main() {
    // We first evaluate the operator on a parallel grid

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let refinement_level = 5;

    let mut rng = ChaCha8Rng::seed_from_u64(world.rank() as u64);

    let grid = bempp::shapes::regular_sphere::<f64, _>(refinement_level, 1, &world);
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

    // let kernel_evaluator = bempp::greens_function_evaluators::dense_evaluator::DenseEvaluator::new(
    //     &points,
    //     &points,
    //     green_kernels::types::GreenKernelEvalType::Value,
    //     true,
    //     Laplace3dKernel::default(),
    //     &world,
    // );

    let kernel_evaluator = bempp::greens_function_evaluators::kifmm_evaluator::KiFmmEvaluator::new(
        &points, &points, 1, 3, 5, &world,
    );

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
        .fill_from(sample_function(&space, &|p| p[0].cos()).local().r());

    let res = laplace_evaluator.apply(x.r());

    // We now evaluate the result vector on each triangle.

    let evaluator = SpaceEvaluator::new(&space, &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], true);

    let evaluated_result = evaluator.apply(res.r());

    println!("Finished parallel evaluation.");

    // We now evaluate the dense operator on just the root process and compare the results.

    if rank == 0 {
        let self_comm = mpi::topology::SimpleCommunicator::self_comm();

        let grid = bempp::shapes::regular_sphere::<f64, _>(refinement_level, 1, &self_comm);
        println!(
            "Number of elements: {}",
            grid.cell_layout().number_of_global_indices()
        );

        let space = bempp::function::FunctionSpace::new(
            &grid,
            &LagrangeElementFamily::<f64>::new(1, ndelement::types::Continuity::Standard),
        );

        let laplace = laplace::assembler::single_layer::<f64>(&options);
        let laplace_mat = laplace.assemble(&space, &space);

        // We need an evaluator for the root result

        let evaluator = SpaceEvaluator::new(&space, &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], true);

        let mut x = rlst_dynamic_array1!(f64, [space.global_dof_count()]);

        x.fill_from(sample_function(&space, &|p| p[0].cos()).local().r());

        let mut root_res = zero_element(evaluator.domain());

        root_res
            .view_mut()
            .local_mut()
            .r_mut()
            .simple_mult_into(laplace_mat.r(), x.r());

        let root_evaluated_result = evaluator.apply(root_res);

        // We have now created the evaluated result on root. We now copy over the result from the distributed vector to root

        let mut actual = zero_element(evaluator.range());

        evaluated_result
            .view()
            .gather_to_rank_root(actual.view_mut().local_mut().r_mut());

        // Now compute the relative difference between the distributed result and the rank zero result

        let rel_diff =
            (actual - root_evaluated_result.r()).norm() / root_evaluated_result.r().norm();

        println!("Relative error: {}", rel_diff);
    } else {
        evaluated_result.view().gather_to_rank(0);
    }
}
