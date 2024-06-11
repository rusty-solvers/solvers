// use bempp::assembly::{batched, batched::BatchedAssembler};
// use bempp::element::ciarlet::LagrangeElementFamily;
// use bempp::function::SerialFunctionSpace;
// use bempp::grid::shapes::regular_sphere;
// use bempp::traits::{element::Continuity, function::FunctionSpace, types::ReferenceCellType};
// use criterion::{criterion_group, criterion_main, Criterion};
// use rlst::rlst_dynamic_array2;

// extern crate blas_src;
// extern crate lapack_src;

// pub fn assembly_parts_benchmark(c: &mut Criterion) {
//     let mut group = c.benchmark_group("assembly");
//     group.sample_size(20);

//     for i in 3..5 {
//         let grid = regular_sphere(i);
//         let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);

//         let space = SerialFunctionSpace::new(&grid, &element);
//         let mut matrix = rlst_dynamic_array2!(f64, [space.global_size(), space.global_size()]);

//         let colouring = space.cell_colouring();
//         let mut a = batched::LaplaceSingleLayerAssembler::<f64>::default();
//         a.quadrature_degree(ReferenceCellType::Triangle, 16);
//         a.singular_quadrature_degree(
//             (ReferenceCellType::Triangle, ReferenceCellType::Triangle),
//             4,
//         );
//         a.batch_size(128);

//         group.bench_function(
//             &format!(
//                 "Assembly of singular terms of {}x{} matrix",
//                 space.global_size(),
//                 space.global_size()
//             ),
//             |b| b.iter(|| a.assemble_singular_into_dense(&mut matrix, &space, &space)),
//         );
//         group.bench_function(
//             &format!(
//                 "Assembly of non-singular terms of {}x{} matrix",
//                 space.global_size(),
//                 space.global_size()
//             ),
//             |b| {
//                 b.iter(|| {
//                     a.assemble_nonsingular_into_dense(
//                         &mut matrix,
//                         &space,
//                         &space,
//                         &colouring,
//                         &colouring,
//                     )
//                 })
//             },
//         );
//     }
//     group.finish();
// }

// criterion_group!(benches, assembly_parts_benchmark);
// criterion_main!(benches);

pub fn main() {}
