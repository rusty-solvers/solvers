//! Functions and function spaces

//mod function_space;

use bempp_distributed_tools::IndexLayoutFromLocalCounts;
use mpi::request::WaitGuard;
use mpi::traits::{Communicator, Destination, Source};
use ndelement::ciarlet::CiarletElement;
use ndelement::traits::ElementFamily;
use ndelement::{traits::FiniteElement, types::ReferenceCellType};
use ndgrid::traits::ParallelGrid;
use ndgrid::traits::{Entity, Topology};
use ndgrid::{traits::Grid, types::Ownership};
use rlst::{MatrixInverse, RlstScalar};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Index;

type DofList = Vec<Vec<usize>>;
type OwnerData = Vec<(usize, usize, usize, usize)>;

/// A local function space
pub trait LocalFunctionSpaceTrait {
    /// Scalar type
    type T: RlstScalar;
    /// The grid type
    type LocalGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType>;
    /// The finite element type
    type FiniteElement: FiniteElement<T = Self::T> + Sync;

    /// Get the grid that the element is defined on
    fn grid(&self) -> &Self::LocalGrid;

    /// Get the finite element used to define this function space
    fn element(&self, cell_type: ReferenceCellType) -> &Self::FiniteElement;

    /// Get the DOF numbers on the local process associated with the given entity
    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize];

    /// Get the number of DOFs associated with the local process
    fn local_size(&self) -> usize;

    /// Get the number of DOFs on all processes
    fn global_size(&self) -> usize;

    /// Get the local DOF numbers associated with a cell
    fn cell_dofs(&self, cell: usize) -> Option<&[usize]>;

    /// Get the local DOF numbers associated with a cell
    ///
    /// # Safety
    /// The function uses unchecked array access
    unsafe fn cell_dofs_unchecked(&self, cell: usize) -> &[usize];

    /// Compute a colouring of the cells so that no two cells that share an entity with DOFs associated with it are assigned the same colour
    fn cell_colouring(&self) -> HashMap<ReferenceCellType, Vec<Vec<usize>>>;

    /// Get the global DOF index associated with a local DOF index
    fn global_dof_index(&self, local_dof_index: usize) -> usize;

    /// Get ownership of a local DOF
    fn ownership(&self, local_dof_index: usize) -> Ownership;

    /// Get the local indices of the support cells associated with this space.
    ///
    /// The vector of support cells is sorted in ascending order and may contain
    /// ghost cells who are not owned by the current process.
    fn support_cells(&self) -> &[usize];
}

/// A function space
pub trait FunctionSpaceTrait: LocalFunctionSpaceTrait {
    /// Communicator
    type C: Communicator;

    /// The grid type
    type Grid: ParallelGrid<Self::C, LocalGrid = Self::LocalGrid>;
    /// Local Function Space
    type LocalFunctionSpace: LocalFunctionSpaceTrait<
        T = Self::T,
        LocalGrid = Self::LocalGrid,
        FiniteElement = Self::FiniteElement,
    >;

    /// Get the communicator
    fn comm(&self) -> &Self::C;

    /// Get the local function space
    fn local_space(&self) -> &Self::LocalFunctionSpace;

    /// Return the index layout associated with the function space.
    fn index_layout(&self) -> &IndexLayoutFromLocalCounts<'_, Self::C>;
}

/// Definition of a local function space.
pub struct LocalFunctionSpace<
    'a,
    T: RlstScalar + MatrixInverse,
    GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
> {
    grid: &'a GridImpl,
    elements: HashMap<ReferenceCellType, CiarletElement<T>>,
    entity_dofs: [Vec<Vec<usize>>; 4],
    cell_dofs: Vec<Vec<usize>>,
    local_size: usize,
    global_size: usize,
    global_dof_numbers: Vec<usize>,
    ownership: Vec<Ownership>,
    support_cells: Vec<usize>,
}

impl<
        'a,
        T: RlstScalar + MatrixInverse,
        GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    > LocalFunctionSpace<'a, T, GridImpl>
{
    /// Create new local function space
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grid: &'a GridImpl,
        elements: HashMap<ReferenceCellType, CiarletElement<T>>,
        entity_dofs: [Vec<Vec<usize>>; 4],
        cell_dofs: Vec<Vec<usize>>,
        local_size: usize,
        global_size: usize,
        global_dof_numbers: Vec<usize>,
        ownership: Vec<Ownership>,
    ) -> Self {
        Self {
            grid,
            elements,
            entity_dofs,
            cell_dofs,
            local_size,
            global_size,
            global_dof_numbers,
            ownership,
            support_cells: (0..local_size).collect(),
        }
    }
}

impl<
        T: RlstScalar + MatrixInverse,
        GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    > LocalFunctionSpaceTrait for LocalFunctionSpace<'_, T, GridImpl>
{
    type T = T;

    type LocalGrid = GridImpl;

    type FiniteElement = CiarletElement<T>;

    fn grid(&self) -> &Self::LocalGrid {
        self.grid
    }

    fn element(&self, cell_type: ReferenceCellType) -> &Self::FiniteElement {
        &self.elements[&cell_type]
    }

    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        &self.entity_dofs[entity_dim][entity_number]
    }

    fn local_size(&self) -> usize {
        self.local_size
    }

    fn global_size(&self) -> usize {
        self.global_size
    }
    unsafe fn cell_dofs_unchecked(&self, cell: usize) -> &[usize] {
        self.cell_dofs.get_unchecked(cell)
    }
    fn cell_dofs(&self, cell: usize) -> Option<&[usize]> {
        if cell < self.cell_dofs.len() {
            Some(unsafe { self.cell_dofs_unchecked(cell) })
        } else {
            None
        }
    }
    fn cell_colouring(&self) -> HashMap<ReferenceCellType, Vec<Vec<usize>>> {
        let mut colouring = HashMap::new();
        //: HashMap<ReferenceCellType, Vec<Vec<usize>>>
        for cell in self.grid.entity_types(2) {
            colouring.insert(*cell, vec![]);
        }
        let mut edim = 0;
        while self.elements[&self.grid.entity_types(2)[0]]
            .entity_dofs(edim, 0)
            .unwrap()
            .is_empty()
        {
            edim += 1;
        }

        let mut entity_colours = vec![
            vec![];
            if edim == 0 {
                self.grid.entity_count(ReferenceCellType::Point)
            } else if edim == 1 {
                self.grid.entity_count(ReferenceCellType::Interval)
            } else if edim == 2 && self.grid.topology_dim() == 2 {
                self.grid
                    .entity_types(2)
                    .iter()
                    .map(|&i| self.grid.entity_count(i))
                    .sum::<usize>()
            } else {
                unimplemented!();
            }
        ];

        for cell in self.grid.entity_iter(2) {
            let cell_type = cell.entity_type();
            let indices = cell.topology().sub_entity_iter(edim).collect::<Vec<_>>();

            let c = {
                let mut c = 0;
                while c < colouring[&cell_type].len() {
                    let mut found = false;
                    for v in &indices {
                        if entity_colours[*v].contains(&c) {
                            found = true;
                            break;
                        }
                    }

                    if !found {
                        break;
                    }
                    c += 1;
                }
                c
            };
            if c == colouring[&cell_type].len() {
                for ct in self.grid.entity_types(2) {
                    colouring.get_mut(ct).unwrap().push(if *ct == cell_type {
                        vec![cell.local_index()]
                    } else {
                        vec![]
                    });
                }
            } else {
                colouring.get_mut(&cell_type).unwrap()[c].push(cell.local_index());
            }
            for v in &indices {
                entity_colours[*v].push(c);
            }
        }
        colouring
    }
    fn global_dof_index(&self, local_dof_index: usize) -> usize {
        self.global_dof_numbers[local_dof_index]
    }
    fn ownership(&self, local_dof_index: usize) -> Ownership {
        self.ownership[local_dof_index]
    }

    fn support_cells(&self) -> &[usize] {
        self.support_cells.as_slice()
    }
}

/// Implementation of a general function space.
pub struct FunctionSpace<
    'a,
    C: Communicator,
    T: RlstScalar + MatrixInverse,
    GridImpl: ParallelGrid<C, T = T::Real, EntityDescriptor = ReferenceCellType>,
> {
    grid: &'a GridImpl,
    local_space: LocalFunctionSpace<'a, T, GridImpl::LocalGrid>,
    index_layout: IndexLayoutFromLocalCounts<'a, C>,
    _marker: PhantomData<(C, T)>,
}

impl<
        'a,
        C: Communicator,
        T: RlstScalar + MatrixInverse,
        GridImpl: ParallelGrid<C, T = T::Real, EntityDescriptor = ReferenceCellType>,
    > FunctionSpace<'a, C, T, GridImpl>
where
    GridImpl::LocalGrid: Sync,
{
    /// Create new function space
    pub fn new(
        grid: &'a GridImpl,
        e_family: &impl ElementFamily<
            T = T,
            FiniteElement = CiarletElement<T>,
            CellType = ReferenceCellType,
        >,
    ) -> Self {
        let comm = grid.comm();
        let rank = comm.rank();
        let size = comm.size();

        // Create local space on current process
        let (cell_dofs, entity_dofs, dofmap_size, owner_data) =
            assign_dofs(rank as usize, grid.local_grid(), e_family);

        let mut elements = HashMap::new();
        for cell in grid.entity_types(grid.topology_dim()) {
            elements.insert(*cell, e_family.element(*cell));
        }

        // Assign global DOF numbers
        let mut global_dof_numbers = vec![0; dofmap_size];
        let mut ghost_indices = vec![vec![]; size as usize];
        let mut ghost_dims = vec![vec![]; size as usize];
        let mut ghost_entities = vec![vec![]; size as usize];
        let mut ghost_entity_dofs = vec![vec![]; size as usize];

        let local_offset = if rank == 0 {
            0
        } else {
            let (value, _status) = comm.process_at_rank(rank - 1).receive::<usize>();
            value
        };
        let mut dof_n = local_offset;
        for (i, ownership) in owner_data.iter().enumerate() {
            if ownership.0 == rank as usize {
                global_dof_numbers[i] = dof_n;
                dof_n += 1;
            } else {
                ghost_indices[ownership.0].push(i);
                ghost_dims[ownership.0].push(ownership.1);
                ghost_entities[ownership.0].push(ownership.2);
                ghost_entity_dofs[ownership.0].push(ownership.3);
            }
        }
        if rank < size - 1 {
            mpi::request::scope(|scope| {
                let _ =
                    WaitGuard::from(comm.process_at_rank(rank + 1).immediate_send(scope, &dof_n));
            });
        }

        let global_size = if rank == size - 1 {
            for p in 0..rank {
                mpi::request::scope(|scope| {
                    let _ = WaitGuard::from(comm.process_at_rank(p).immediate_send(scope, &dof_n));
                });
            }
            dof_n
        } else {
            let (gs, _status) = comm.process_at_rank(size - 1).receive::<usize>();
            gs
        };

        // Communicate information about ghosts
        // send requests for ghost info
        for p in 0..size {
            if p != rank {
                mpi::request::scope(|scope| {
                    let process = comm.process_at_rank(p);
                    let _ = WaitGuard::from(process.immediate_send(scope, &ghost_dims[p as usize]));
                    let _ =
                        WaitGuard::from(process.immediate_send(scope, &ghost_entities[p as usize]));
                    let _ = WaitGuard::from(
                        process.immediate_send(scope, &ghost_entity_dofs[p as usize]),
                    );
                });
            }
        }
        // accept requests and send ghost info
        for p in 0..size {
            if p != rank {
                let process = comm.process_at_rank(p);
                let (gdims, _status) = process.receive_vec::<usize>();
                let (gentities, _status) = process.receive_vec::<usize>();
                let (gentity_dofs, _status) = process.receive_vec::<usize>();
                let local_ghost_dofs = gdims
                    .iter()
                    .zip(gentities.iter().zip(&gentity_dofs))
                    .map(|(c, (e, d))| entity_dofs[*c][*e][*d])
                    .collect::<Vec<_>>();
                let global_ghost_dofs = local_ghost_dofs
                    .iter()
                    .map(|i| global_dof_numbers[*i])
                    .collect::<Vec<_>>();
                mpi::request::scope(|scope| {
                    let _ = WaitGuard::from(process.immediate_send(scope, &local_ghost_dofs));
                    let _ = WaitGuard::from(process.immediate_send(scope, &global_ghost_dofs));
                });
            }
        }

        // receive ghost info
        let mut ownership = vec![Ownership::Owned; dofmap_size];
        for p in 0..size {
            if p != rank {
                let process = comm.process_at_rank(p);
                let (local_ghost_dofs, _status) = process.receive_vec::<usize>();
                let (global_ghost_dofs, _status) = process.receive_vec::<usize>();
                for (i, (l, g)) in ghost_indices[p as usize]
                    .iter()
                    .zip(local_ghost_dofs.iter().zip(&global_ghost_dofs))
                {
                    global_dof_numbers[*i] = *g;
                    ownership[*i] = Ownership::Ghost(p as usize, *l);
                }
            }
        }

        Self {
            grid,
            local_space: LocalFunctionSpace::new(
                grid.local_grid(),
                elements,
                entity_dofs,
                cell_dofs,
                dofmap_size,
                global_size,
                global_dof_numbers,
                ownership,
            ),
            index_layout: IndexLayoutFromLocalCounts::new(dofmap_size, comm),
            _marker: PhantomData,
        }

        // Self {
        //     grid,
        //     elements,
        //     entity_dofs,
        //     cell_dofs,
        //     local_size: dofmap_size,
        //     global_size,
        //     global_dof_numbers,
        //     ownership,
        //     _marker: PhantomData,
        // }
    }
}

impl<
        C: Communicator,
        T: RlstScalar + MatrixInverse,
        GridImpl: ParallelGrid<C, T = T::Real, EntityDescriptor = ReferenceCellType>,
    > LocalFunctionSpaceTrait for FunctionSpace<'_, C, T, GridImpl>
{
    type T = T;

    type LocalGrid = GridImpl::LocalGrid;

    type FiniteElement = CiarletElement<T>;

    fn grid(&self) -> &Self::LocalGrid {
        self.local_space.grid()
    }

    fn element(&self, cell_type: ReferenceCellType) -> &Self::FiniteElement {
        self.local_space.element(cell_type)
    }

    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        self.local_space
            .get_local_dof_numbers(entity_dim, entity_number)
    }

    fn local_size(&self) -> usize {
        self.local_space.local_size()
    }

    fn global_size(&self) -> usize {
        self.local_space.global_size()
    }

    fn cell_dofs(&self, cell: usize) -> Option<&[usize]> {
        self.local_space.cell_dofs(cell)
    }

    unsafe fn cell_dofs_unchecked(&self, cell: usize) -> &[usize] {
        self.local_space.cell_dofs_unchecked(cell)
    }

    fn cell_colouring(&self) -> HashMap<ReferenceCellType, Vec<Vec<usize>>> {
        self.local_space.cell_colouring()
    }

    fn global_dof_index(&self, local_dof_index: usize) -> usize {
        self.local_space.global_dof_index(local_dof_index)
    }

    fn ownership(&self, local_dof_index: usize) -> Ownership {
        // Syntactical workaround as rust-analyzer mixed up this ownership with entity ownership.
        LocalFunctionSpaceTrait::ownership(&self.local_space, local_dof_index)
    }

    fn support_cells(&self) -> &[usize] {
        self.local_space.support_cells()
    }
}

impl<
        'a,
        C: Communicator,
        T: RlstScalar + MatrixInverse,
        GridImpl: ParallelGrid<C, T = T::Real, EntityDescriptor = ReferenceCellType>,
    > FunctionSpaceTrait for FunctionSpace<'a, C, T, GridImpl>
{
    fn comm(&self) -> &C {
        self.grid.comm()
    }

    type C = C;

    type Grid = GridImpl;

    type LocalFunctionSpace = LocalFunctionSpace<'a, T, GridImpl::LocalGrid>;

    fn local_space(&self) -> &Self::LocalFunctionSpace {
        &self.local_space
    }

    fn index_layout(&self) -> &IndexLayoutFromLocalCounts<'_, Self::C> {
        &self.index_layout
    }
}

/// Assign DOFs to entities.
pub fn assign_dofs<
    T: RlstScalar + MatrixInverse,
    GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
>(
    rank: usize,
    grid: &GridImpl,
    e_family: &impl ElementFamily<
        T = T,
        FiniteElement = CiarletElement<T>,
        CellType = ReferenceCellType,
    >,
) -> (DofList, [DofList; 4], usize, OwnerData) {
    let mut size = 0;
    let mut entity_dofs: [Vec<Vec<usize>>; 4] = [vec![], vec![], vec![], vec![]];
    let mut owner_data = vec![];
    let tdim = grid.topology_dim();

    let mut elements = HashMap::new();
    let mut element_dims = HashMap::new();
    for cell in grid.entity_types(2) {
        elements.insert(*cell, e_family.element(*cell));
        element_dims.insert(*cell, elements[cell].dim());
    }

    let entity_counts = (0..tdim + 1)
        .map(|d| {
            grid.entity_types(d)
                .iter()
                .map(|&i| grid.entity_count(i))
                .sum::<usize>()
        })
        .collect::<Vec<_>>();
    if tdim > 2 {
        unimplemented!("DOF maps not implemented for cells with tdim > 2.");
    }

    for d in 0..tdim + 1 {
        entity_dofs[d] = vec![vec![]; entity_counts[d]];
    }
    let mut cell_dofs = vec![vec![]; entity_counts[tdim]];

    let mut max_rank = rank;
    for cell in grid.entity_iter(tdim) {
        if let Ownership::Ghost(process, _index) = cell.ownership() {
            if process > max_rank {
                max_rank = process;
            }
        }
    }
    for cell in grid.entity_iter(tdim) {
        cell_dofs[cell.local_index()] = vec![0; element_dims[&cell.entity_type()]];
        let element = &elements[&cell.entity_type()];
        let topology = cell.topology();

        // Assign DOFs to entities
        for (d, edofs_d) in entity_dofs.iter_mut().take(tdim + 1).enumerate() {
            for (i, e) in topology.sub_entity_iter(d).enumerate() {
                let e_dofs = element.entity_dofs(d, i).unwrap();
                if !e_dofs.is_empty() {
                    if edofs_d[e].is_empty() {
                        for (dof_i, _d) in e_dofs.iter().enumerate() {
                            edofs_d[e].push(size);
                            if let Ownership::Ghost(process, index) =
                                grid.entity(d, e).unwrap().ownership()
                            {
                                owner_data.push((process, d, index, dof_i));
                            } else {
                                owner_data.push((rank, d, e, dof_i));
                            }
                            size += 1;
                        }
                    }
                    for (local_dof, dof) in e_dofs.iter().zip(&edofs_d[e]) {
                        cell_dofs[cell.local_index()][*local_dof] = *dof;
                    }
                }
            }
        }
    }
    (cell_dofs, entity_dofs, size, owner_data)
}
