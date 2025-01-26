//! Demo the evaluation of functions in Bempp-rs

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    // We create a sphere grid

    let grid = bempp::shapes::regular_sphere::<f64, _>(5, 1, &world);

    // We have the grid.
}
