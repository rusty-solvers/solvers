use bempp::function::{assign_dofs, FunctionSpace, FunctionSpaceTrait, LocalFunctionSpaceTrait};
use bempp::shapes::{regular_sphere, screen_triangles};
use mpi::traits::Communicator;
use ndelement::ciarlet::{LagrangeElementFamily, RaviartThomasElementFamily};
use ndelement::types::{Continuity, ReferenceCellType};
use ndgrid::traits::{Entity, Grid, ParallelGrid, Topology};
use std::sync::LazyLock;

use mpi::environment::Universe;

static MPI_UNIVERSE: LazyLock<Universe> = std::sync::LazyLock::new(|| {
    mpi::initialize_with_threading(mpi::Threading::Multiple)
        .unwrap()
        .0
});

fn run_test<C: Communicator, GridImpl: ParallelGrid<C = C, T = f64>>(
    grid: &GridImpl,
    degree: usize,
    continuity: Continuity,
) where
    GridImpl::LocalGrid: Grid<EntityDescriptor = ReferenceCellType> + Sync,
{
    let family = LagrangeElementFamily::<f64>::new(degree, continuity);
    let (cell_dofs, entity_dofs, size, owner_data) = assign_dofs(0, grid.local_grid(), &family);

    for o in &owner_data {
        assert_eq!(o.0, 0);
    }
    for d in &cell_dofs {
        for (i, n) in d.iter().enumerate() {
            assert!(*n < size);
            for m in d.iter().skip(i + 1) {
                assert!(*n != *m);
            }
        }
    }
    for i in &entity_dofs {
        for j in i {
            for k in j {
                assert!(*k < size);
            }
        }
    }
}

fn run_test_rt<C: Communicator, GridImpl: ParallelGrid<C = C, T = f64>>(
    grid: &GridImpl,
    degree: usize,
    continuity: Continuity,
) where
    GridImpl::LocalGrid: Grid<EntityDescriptor = ReferenceCellType> + Sync,
{
    let family = RaviartThomasElementFamily::<f64>::new(degree, continuity);
    let (cell_dofs, entity_dofs, size, owner_data) = assign_dofs(0, grid.local_grid(), &family);

    for o in &owner_data {
        assert_eq!(o.0, 0);
    }
    for d in &cell_dofs {
        for (i, n) in d.iter().enumerate() {
            assert!(*n < size);
            for m in d.iter().skip(i + 1) {
                assert!(*n != *m);
            }
        }
    }
    for i in &entity_dofs {
        for j in i {
            for k in j {
                assert!(*k < size);
            }
        }
    }
}

#[test]
fn test_dp0_triangles() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = screen_triangles::<f64, _>(8, &comm);
    run_test(&grid, 0, Continuity::Discontinuous);
}
#[test]
fn test_dp2_triangles() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = screen_triangles::<f64, _>(8, &comm);
    run_test(&grid, 2, Continuity::Discontinuous);
}
#[test]
fn test_p2_triangles() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = screen_triangles::<f64, _>(8, &comm);
    run_test(&grid, 2, Continuity::Standard);
}
#[test]
fn test_p3_triangles() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = screen_triangles::<f64, _>(8, &comm);
    run_test(&grid, 3, Continuity::Standard);
}
#[test]
fn test_rt1_triangles() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = screen_triangles::<f64, _>(8, &comm);
    run_test_rt(&grid, 1, Continuity::Discontinuous);
}

#[test]
fn test_dp0_quadrilaterals() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = screen_triangles::<f64, _>(8, &comm);
    run_test(&grid, 0, Continuity::Discontinuous);
}
#[test]
fn test_dp2_quadrilaterals() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = screen_triangles::<f64, _>(8, &comm);
    run_test(&grid, 2, Continuity::Discontinuous);
}
#[test]
fn test_p2_quadrilaterals() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = screen_triangles::<f64, _>(8, &comm);
    run_test(&grid, 2, Continuity::Standard);
}
#[test]
fn test_p3_quadrilaterals() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = screen_triangles::<f64, _>(8, &comm);
    run_test(&grid, 3, Continuity::Standard);
}

#[test]
fn test_dofmap_lagrange0() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = regular_sphere::<f64, _>(2, 1, &comm);
    //let grid = regular_sphere::<f64, _>(2, 1, &comm);
    let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);
    let space = FunctionSpace::new(&grid, &element);
    assert_eq!(
        space.local_space().local_size(),
        space.local_space().global_size()
    );
    assert_eq!(
        space.local_space().local_size(),
        grid.local_grid().entity_count(ReferenceCellType::Triangle)
    );
}

#[test]
fn test_dofmap_lagrange1() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = regular_sphere::<f64, _>(2, 1, &comm);
    let element = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
    let space = FunctionSpace::new(&grid, &element);
    assert_eq!(
        space.local_space().local_size(),
        space.local_space().global_size()
    );
    assert_eq!(
        space.local_space().local_size(),
        grid.local_grid().entity_count(ReferenceCellType::Point)
    );
}

#[test]
fn test_dofmap_lagrange2() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = regular_sphere::<f64, _>(2, 1, &comm);
    let element = LagrangeElementFamily::<f64>::new(2, Continuity::Standard);
    let space = FunctionSpace::new(&grid, &element);
    assert_eq!(
        space.local_space().local_size(),
        space.local_space().global_size()
    );
    assert_eq!(
        space.local_space().local_size(),
        grid.local_grid().entity_count(ReferenceCellType::Point)
            + grid.local_grid().entity_count(ReferenceCellType::Interval)
    );
}

#[test]
fn test_colouring_p1() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = regular_sphere::<f64, _>(2, 1, &comm);
    let element = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
    let space = FunctionSpace::new(&grid, &element);
    let colouring = &space.local_space().cell_colouring()[&ReferenceCellType::Triangle];
    let cells = grid.local_grid().entity_iter(2).collect::<Vec<_>>();
    let mut n = 0;
    for i in colouring {
        n += i.len()
    }
    assert_eq!(
        n,
        grid.local_grid().entity_count(ReferenceCellType::Triangle)
    );
    for (i, ci) in colouring.iter().enumerate() {
        for (j, cj) in colouring.iter().enumerate() {
            if i != j {
                for cell0 in ci {
                    for cell1 in cj {
                        assert!(cell0 != cell1);
                    }
                }
            }
        }
    }
    for ci in colouring {
        for cell0 in ci {
            for cell1 in ci {
                if cell0 != cell1 {
                    for v0 in cells[*cell0].topology().sub_entity_iter(0) {
                        for v1 in cells[*cell1].topology().sub_entity_iter(0) {
                            assert!(v0 != v1);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_colouring_dp0() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = regular_sphere::<f64, _>(2, 1, &comm);
    let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);
    let space = FunctionSpace::new(&grid, &element);
    let colouring = &space.local_space().cell_colouring()[&ReferenceCellType::Triangle];
    let mut n = 0;
    for i in colouring {
        n += i.len()
    }
    assert_eq!(
        n,
        grid.local_grid().entity_count(ReferenceCellType::Triangle)
    );
    for (i, ci) in colouring.iter().enumerate() {
        for (j, cj) in colouring.iter().enumerate() {
            if i != j {
                for cell0 in ci {
                    for cell1 in cj {
                        assert!(cell0 != cell1);
                    }
                }
            }
        }
    }
    assert_eq!(colouring.len(), 1);
}

#[test]
fn test_colouring_rt1() {
    let _ = *MPI_UNIVERSE;
    let comm = mpi::topology::SimpleCommunicator::self_comm();
    let grid = regular_sphere::<f64, _>(2, 1, &comm);
    let element = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
    let space = FunctionSpace::new(&grid, &element);
    let colouring = &space.local_space().cell_colouring()[&ReferenceCellType::Triangle];
    let mut n = 0;
    for i in colouring {
        n += i.len()
    }
    assert_eq!(
        n,
        grid.local_grid().entity_count(ReferenceCellType::Triangle)
    );
    for (i, ci) in colouring.iter().enumerate() {
        for (j, cj) in colouring.iter().enumerate() {
            if i != j {
                for cell0 in ci {
                    for cell1 in cj {
                        assert!(cell0 != cell1);
                    }
                }
            }
        }
    }
    for ci in colouring {
        for cell0 in ci {
            for cell1 in ci {
                if cell0 != cell1 {
                    for e0 in grid
                        .local_grid()
                        .entity(2, *cell0)
                        .unwrap()
                        .topology()
                        .sub_entity_iter(1)
                    {
                        for e1 in grid
                            .local_grid()
                            .entity(2, *cell1)
                            .unwrap()
                            .topology()
                            .sub_entity_iter(1)
                        {
                            assert!(e0 != e1);
                        }
                    }
                }
            }
        }
    }
}

/*
#[test]
fn test_dp0_mixed() {
    let grid = screen_mixed::<f64>(8);
    run_test(&grid, 0, Continuity::Discontinuous);
}
#[test]
fn test_dp2_mixed() {
    let grid = screen_mixed::<f64>(8);
    run_test(&grid, 2, Continuity::Discontinuous);
}
#[test]
fn test_p2_mixed() {
    let grid = screen_mixed::<f64>(8);
    run_test(&grid, 2, Continuity::Standard);
}
#[test]
fn test_p3_mixed() {
    let grid = screen_mixed::<f64>(8);
    run_test(&grid, 3, Continuity::Standard);
}
*/
