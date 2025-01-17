use bempp::boundary_assemblers::BoundaryAssemblerOptions;
use bempp::function::FunctionSpace;
use bempp::function::LocalFunctionSpaceTrait;
use bempp::laplace::assembler::single_layer;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndelement::ciarlet::LagrangeElementFamily;
use ndelement::types::{Continuity, ReferenceCellType};

pub fn assembly_parts_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly");
    group.sample_size(20);

    let _ = mpi::initialize().unwrap();
    let comm = mpi::topology::SimpleCommunicator::self_comm();

    for i in 3..5 {
        let grid = bempp::shapes::regular_sphere(i, 1, &comm);
        let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);

        let space = FunctionSpace::new(&grid, &element);
        let mut options = BoundaryAssemblerOptions::default();
        options.set_regular_quadrature_degree(ReferenceCellType::Triangle, 16);
        options.set_singular_quadrature_degree(
            (ReferenceCellType::Triangle, ReferenceCellType::Triangle),
            4,
        );
        options.set_batch_size(128);

        let assembler = single_layer(&options);

        group.bench_function(
            format!(
                "Assembly of singular terms of {}x{} matrix",
                space.global_size(),
                space.global_size()
            ),
            |b| b.iter(|| black_box(assembler.assemble_singular(&space, &space))),
        );
    }
    group.finish();
}

criterion_group!(benches, assembly_parts_benchmark);
criterion_main!(benches);
