//! Various helper functions to support evaluators.

use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    rc::Rc,
};

use bempp_distributed_tools::GhostCommunicator;
use green_kernels::{traits::Kernel, types::GreenKernelEvalType};
use itertools::{izip, Itertools};
use mpi::traits::{Communicator, Equivalence};
use ndelement::traits::FiniteElement;
use ndgrid::{
    traits::{Entity, GeometryMap, Grid, ParallelGrid, Topology},
    types::Ownership,
};

use rlst::{
    operator::{
        interface::{
            distributed_sparse_operator::DistributedCsrMatrixOperatorImpl,
            DistributedArrayVectorSpace,
        },
        Operator,
    },
    rlst_array_from_slice2, rlst_dynamic_array1, rlst_dynamic_array2, rlst_dynamic_array3,
    rlst_dynamic_array4, Array, AsApply, DefaultIterator, DistributedCsrMatrix, DistributedVector,
    Element, IndexLayout, OperatorBase, RawAccess, RawAccessMut, RlstScalar, Shape,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

use rayon::prelude::*;

use crate::function::FunctionSpaceTrait;

/// Create a linear operator from the map of a basis to points. The points are sorted by global
/// index of the corresponding cells.
pub fn basis_to_point_map<
    'a,
    T: RlstScalar + Equivalence,
    C: Communicator,
    Space: FunctionSpaceTrait<T = T, C = C>,
>(
    function_space: &'a Space,
    quadrature_points: &[T::Real],
    quadrature_weights: &[T::Real],
    return_transpose: bool,
) -> Operator<DistributedCsrMatrixOperatorImpl<'a, T, C>>
where
    T::Real: Equivalence,
{
    // Get the grid.
    let grid = function_space.grid();

    // Topological dimension of the grid.
    let tdim = grid.topology_dim();

    // The method is currently restricted to single element type grids.
    // So let's make sure that this is the case.

    assert_eq!(grid.entity_types(tdim).len(), 1);

    let reference_cell = grid.entity_types(tdim)[0];

    let owned_active_cells: Vec<usize> = function_space
        .support_cells()
        .iter()
        .filter(|index| {
            matches!(
                grid.entity(tdim, **index).unwrap().ownership(),
                Ownership::Owned
            )
        })
        .copied()
        .sorted_by_key(|&index| grid.entity(tdim, index).unwrap().global_index())
        .collect_vec();

    // Number of cells. We are only interested in owned cells.
    let n_cells = owned_active_cells.len();

    // Get the number of quadrature points and check that weights
    // and points have compatible dimensions.

    let n_quadrature_points = quadrature_weights.len();
    assert_eq!(quadrature_points.len() % tdim, 0);
    assert_eq!(quadrature_points.len() / tdim, n_quadrature_points);

    // Assign number of domain and range dofs.

    let n_domain_dofs = function_space.local_size();
    let n_range_dofs = n_cells * n_quadrature_points;

    // Create the index layouts.

    let domain_layout = Rc::new(IndexLayout::from_local_counts(
        n_domain_dofs,
        function_space.comm(),
    ));
    let range_layout = Rc::new(IndexLayout::from_local_counts(
        n_range_dofs,
        function_space.comm(),
    ));

    // All the dimensions are OK. Let's get to work. We need to iterate through the elements,
    // get the attached global dofs and the corresponding jacobian map.

    // Let's first tabulate the basis function values at the quadrature points on the reference element.

    // Quadrature point is needed here as a RLST matrix.

    let quadrature_points = rlst_array_from_slice2!(quadrature_points, [tdim, n_quadrature_points]);

    let mut table = rlst_dynamic_array4!(
        T,
        function_space
            .element(reference_cell)
            .tabulate_array_shape(0, n_quadrature_points)
    );
    function_space
        .element(reference_cell)
        .tabulate(&quadrature_points, 0, &mut table);

    // We have tabulated the basis functions on the reference element. Now need
    // the map to physical elements.

    let geometry_evaluator = grid.geometry_map(reference_cell, quadrature_points.data());

    // The following arrays hold the jacobians, their determinants and the normals.

    let mut jacobians =
        rlst_dynamic_array3![T::Real, [grid.geometry_dim(), tdim, n_quadrature_points]];
    let mut jdets = rlst_dynamic_array1![T::Real, [n_quadrature_points]];
    let mut normals = rlst_dynamic_array2![T::Real, [grid.geometry_dim(), n_quadrature_points]];

    // Now iterate through the cells of the grid, get the attached dofs and evaluate the geometry map.

    // These arrays store the data of the transformation matrix.
    let mut rows = Vec::<usize>::default();
    let mut cols = Vec::<usize>::default();
    let mut data = Vec::<T>::default();

    for cell_index in owned_active_cells.iter() {
        let cell = grid.entity(tdim, *cell_index).unwrap();

        // Get the Jacobians
        geometry_evaluator.jacobians_dets_normals(
            cell.local_index(),
            jacobians.data_mut(),
            jdets.data_mut(),
            normals.data_mut(),
        );
        // Get the global dofs of the cell.
        let global_dofs = function_space
            .cell_dofs(cell.local_index())
            .unwrap()
            .iter()
            .map(|local_dof_index| function_space.global_dof_index(*local_dof_index))
            .collect::<Vec<_>>();

        for (qindex, (jdet, qweight)) in izip!(jdets.iter(), quadrature_weights.iter()).enumerate()
        {
            for (i, global_dof) in global_dofs.iter().enumerate() {
                rows.push(n_quadrature_points * cell.global_index() + qindex);
                cols.push(*global_dof);
                data.push(T::from_real(jdet * *qweight) * table[[0, qindex, i, 0]]);
            }
        }
    }

    if return_transpose {
        Operator::from(DistributedCsrMatrix::from_aij(
            range_layout,
            domain_layout,
            &cols,
            &rows,
            &data,
        ))
    } else {
        Operator::from(DistributedCsrMatrix::from_aij(
            domain_layout,
            range_layout,
            &rows,
            &cols,
            &data,
        ))
    }
}

/// Create a linear operator from the map of a basis to points.
pub struct NeighbourEvaluator<
    'a,
    T: RlstScalar + Equivalence,
    K: Kernel<T = T>,
    GridImpl: ParallelGrid<C, T = T::Real>,
    C: Communicator,
> {
    eval_points: Vec<T::Real>,
    n_points: usize,
    kernel: K,
    eval_type: GreenKernelEvalType,
    array_space: Rc<DistributedArrayVectorSpace<'a, C, T>>,
    grid: &'a GridImpl,
    active_cells: Vec<usize>,
    owned_active_cells: Vec<usize>,
    ghost_communicator: GhostCommunicator<usize>,
    receive_local_indices: Vec<usize>,
    global_cell_index_to_active_cell_index: HashMap<usize, usize>,
    global_cell_index_to_owned_active_cell_index: HashMap<usize, usize>,
    _marker: PhantomData<C>,
}

impl<
        'a,
        T: RlstScalar + Equivalence,
        K: Kernel<T = T>,
        GridImpl: ParallelGrid<C, T = T::Real>,
        C: Communicator,
    > NeighbourEvaluator<'a, T, K, GridImpl, C>
{
    /// Create a new neighbour evaluator.
    pub fn new(
        eval_points: &[T::Real],
        kernel: K,
        eval_type: GreenKernelEvalType,
        active_cells: &[usize],
        grid: &'a GridImpl,
    ) -> Self {
        let tdim = grid.topology_dim();

        let comm = grid.comm();

        // The method is currently restricted to single element type grids.
        // So let's make sure that this is the case.

        assert_eq!(grid.entity_types(tdim).len(), 1);

        // Get the number of points
        assert_eq!(eval_points.len() % tdim, 0);
        let n_points = eval_points.len() / tdim;

        // The active cells are those that we need to iterate over.
        // We need all active cells (including non-owned) and the
        // owned active cells.

        let active_cells = active_cells
            .iter()
            .copied()
            .sorted_by_key(|&index| grid.entity(tdim, index).unwrap().global_index())
            .collect_vec();

        let owned_active_cells: Vec<usize> = active_cells
            .iter()
            .filter(|index| {
                matches!(
                    grid.entity(tdim, **index).unwrap().ownership(),
                    Ownership::Owned
                )
            })
            .copied()
            .collect_vec();

        let n_cells = owned_active_cells.len();

        // We now need to setup the ghost communicator structure. When cells are a target that interface process
        // boundaries, their sources are on another process, hence need ghost communicators to get the data.

        // This map stores the local index associated with a global index.
        let mut global_cell_index_to_active_cell_index = HashMap::<usize, usize>::default();

        // We iterate through the cells to figure out ghost cells and their originating ranks.
        let mut ghost_global_indices: Vec<usize> = Vec::new();
        let mut ghost_ranks: Vec<usize> = Vec::new();

        for (active_index, cell_index) in active_cells.iter().enumerate() {
            let cell = grid.entity(tdim, *cell_index).unwrap();
            // This maps each active cell to a local index in order.
            global_cell_index_to_active_cell_index.insert(cell.global_index(), active_index);
            if let Ownership::Ghost(owning_rank, _) = cell.ownership() {
                ghost_global_indices.push(cell.global_index());
                ghost_ranks.push(owning_rank);
            }
        }

        // We also want to create a map from global cell indices to owned active cell indices.
        let mut global_cell_index_to_owned_active_cell_index = HashMap::<usize, usize>::default();

        for (owned_active_index, &cell_index) in owned_active_cells.iter().enumerate() {
            let cell = grid.entity(tdim, cell_index).unwrap();
            global_cell_index_to_owned_active_cell_index
                .insert(cell.global_index(), owned_active_index);
        }

        // We now setup the array space. This is a distributed array space with a layout corresponding
        // to the number of points on each process.

        let array_space = DistributedArrayVectorSpace::from_index_layout(Rc::new(
            IndexLayout::from_local_counts(n_cells * n_points, comm),
        ));

        // We now setup the ghost communicator. The chunk sizes is `n_points` per triangle.

        let ghost_communicator = GhostCommunicator::new(
            // This is the actual global ghost indices.
            &ghost_global_indices,
            &ghost_ranks,
            comm,
        );

        // We now need to get the local indices of the receive values too. The ghost communicator
        // only stores global indices.

        let receive_local_indices = ghost_communicator
            .receive_indices()
            .iter()
            .map(|global_index| global_cell_index_to_active_cell_index[global_index])
            .collect_vec();

        Self {
            eval_points: eval_points.to_vec(),
            n_points,
            kernel,
            eval_type,
            array_space,
            grid,
            active_cells,
            owned_active_cells,
            ghost_communicator,
            receive_local_indices,
            global_cell_index_to_active_cell_index,
            global_cell_index_to_owned_active_cell_index,
            _marker: PhantomData,
        }
    }

    fn communicate_dofs<
        ArrayImpl: UnsafeRandomAccessByValue<1, Item = T>
            + Shape<1>
            + UnsafeRandomAccessMut<1, Item = T>
            + RawAccessMut<Item = T>,
    >(
        &self,
        x: &DistributedVector<'_, C, T>,
        mut out: Array<T, ArrayImpl, 1>,
    ) {
        let tdim = self.grid.topology_dim();
        let reference_cell = self.grid.entity_types(tdim)[0];
        // Check that `out` has the correct size. It must be n_points * the number of cells in the grid.
        assert_eq!(
            out.shape()[0],
            self.grid.entity_count(reference_cell) * self.n_points
        );

        // We first setup the send data.

        let mut send_data =
            vec![T::zero(); self.ghost_communicator.total_send_count() * self.n_points];

        // We now need to fill the send data with the values of the local dofs.

        for (send_chunk, &send_global_index) in izip!(
            send_data.chunks_mut(self.n_points),
            self.ghost_communicator.send_indices().iter()
        ) {
            let local_start_index = self.global_cell_index_to_owned_active_cell_index
                [&send_global_index]
                * self.n_points;
            let local_end_index = local_start_index + self.n_points;
            send_chunk.copy_from_slice(&x.local().r().data()[local_start_index..local_end_index]);
        }

        // We need an array for the receive data.

        let mut receive_data =
            vec![T::zero(); self.ghost_communicator.total_receive_count() * self.n_points];

        // Now we need to communicate the data.

        self.ghost_communicator.forward_send_values_by_chunks(
            send_data.as_slice(),
            &mut receive_data,
            self.n_points,
        );

        // We have done the ghost exchange. Now we build the local vector of dofs. Each chunk of n_points corresponds
        // to one local index.

        for (receive_chunk, &receive_local_index) in izip!(
            receive_data.chunks(self.n_points),
            self.receive_local_indices.iter(),
        ) {
            let local_start_index = receive_local_index * self.n_points;
            let local_end_index = local_start_index + self.n_points;
            out.data_mut()[local_start_index..local_end_index].copy_from_slice(receive_chunk);
        }

        // After filling the ghost data we need to fill the local data in the right cell order.
        // We go through the cells, get the global index of the cell, multiply it with `n_points` and
        // use that as global start index for the data in x.

        for cell in self.active_cells.iter() {
            let x_start = self.global_cell_index_to_active_cell_index[cell] * self.n_points;
            let x_end = x_start + self.n_points;
            out.data_mut()[x_start..x_end].copy_from_slice(&x.local().data()[x_start..x_end]);
        }
    }
}

impl<
        T: RlstScalar + Equivalence,
        K: Kernel<T = T>,
        GridImpl: ParallelGrid<C, T = T::Real>,
        C: Communicator,
    > std::fmt::Debug for NeighbourEvaluator<'_, T, K, GridImpl, C>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Neighbourhood Evaluator with dimenion [{}, {}].",
            self.array_space.index_layout().number_of_global_indices(),
            self.array_space.index_layout().number_of_global_indices()
        )
    }
}

impl<
        'a,
        T: RlstScalar + Equivalence,
        K: Kernel<T = T>,
        GridImpl: ParallelGrid<C, T = T::Real>,
        C: Communicator,
    > OperatorBase for NeighbourEvaluator<'a, T, K, GridImpl, C>
{
    type Domain = DistributedArrayVectorSpace<'a, C, T>;

    type Range = DistributedArrayVectorSpace<'a, C, T>;

    fn domain(&self) -> Rc<Self::Domain> {
        self.array_space.clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.array_space.clone()
    }
}

impl<
        T: RlstScalar + Equivalence,
        K: Kernel<T = T>,
        GridImpl: ParallelGrid<C, T = T::Real>,
        C: Communicator,
    > AsApply for NeighbourEvaluator<'_, T, K, GridImpl, C>
where
    GridImpl::LocalGrid: Sync,
    for<'b> <GridImpl::LocalGrid as Grid>::GeometryMap<'b>: Sync,
{
    fn apply_extended<
        ContainerIn: rlst::ElementContainer<E = <Self::Domain as rlst::LinearSpace>::E>,
        ContainerOut: rlst::ElementContainerMut<E = <Self::Range as rlst::LinearSpace>::E>,
    >(
        &self,
        alpha: <Self::Range as rlst::LinearSpace>::F,
        x: Element<ContainerIn>,
        beta: <Self::Range as rlst::LinearSpace>::F,
        mut y: Element<ContainerOut>,
    ) {
        // We need to iterate through the elements.

        let tdim = self.grid.topology_dim();
        let gdim = self.grid.geometry_dim();

        // We need the chunk size of the targets. This is the chunk size of the domain multiplied
        // with the range component count of the kernel.
        let target_chunk_size = self.n_points * self.kernel.range_component_count(self.eval_type);

        // In the new function we already made sure that eval_points is a multiple of tdim.
        let n_points = self.n_points;

        // We need the reference cell
        let reference_cell = self.grid.entity_types(tdim)[0];

        // Let's get the number of cells on this rank. This includes local and ghost cells.
        // This is different from `n_cells` in `new` which only counts owned cells.
        let n_cells = self.active_cells.len();

        // Let us allocate space for the local charges. Each chunk of charges is associated with a cell.
        let mut charges = rlst_dynamic_array1![T, [n_cells * n_points]];

        // We now need to communicate the data.

        self.communicate_dofs(x.view(), charges.r_mut());

        // The `charges` vector now has all the charges for each cell on our process, including ghost cells.
        // Next we iterate through the targets from the active cells and evaluate the kernel with sources coming
        // from the neighbouring cells and the self interactions.

        // We go through groups of target dofs in chunks of lenth n_points.
        // This corresponds to iteration in active cells since we ordered those
        // already by global index.

        let local_grid = self.grid.local_grid();
        let active_cells: HashSet<usize> =
            HashSet::from_iter(self.active_cells.as_slice().iter().copied());
        let owned_active_cells = self.owned_active_cells.as_slice();
        let eval_points = self.eval_points.as_slice();
        let geometry_map = local_grid.geometry_map(reference_cell, eval_points);
        let kernel = &self.kernel;
        let eval_type = self.eval_type;
        let global_to_active_index = &self.global_cell_index_to_active_cell_index;

        y.view_mut()
            .local_mut()
            .data_mut()
            .par_chunks_mut(target_chunk_size)
            .zip(owned_active_cells)
            .for_each(|(result_chunk, &owned_target_cell_index)| {
                let cell_entity = local_grid.entity(tdim, owned_target_cell_index).unwrap();

                // Get the target points for the target cell.
                let mut target_points = rlst_dynamic_array2![T::Real, [gdim, n_points]];
                geometry_map.points(owned_target_cell_index, target_points.data_mut());

                // Get all the neighbouring celll indices.
                // This is a bit cumbersome. We go through the points of the target cell and add all cells that are connected
                // to each point, using a `unique` iterator to remove duplicates. However, we also need to make sure
                // that we only add cells that are in the active set.
                let source_cells = cell_entity
                    .topology()
                    .sub_entity_iter(0)
                    .flat_map(|v| {
                        local_grid
                            .entity(0, v)
                            .unwrap()
                            .topology()
                            .connected_entity_iter(tdim)
                            .collect_vec()
                    })
                    .filter(|&cell| active_cells.contains(&cell))
                    .unique()
                    .collect_vec();

                // The next array will store the set of source points for each source cell.
                let mut source_points = rlst_dynamic_array2![T::Real, [gdim, n_points]];
                let mut source_charges = rlst_dynamic_array1![T, [n_points]];

                // We need to multiply the targets with beta
                for value in result_chunk.iter_mut() {
                    *value *= beta;
                }

                // We can now go through the source cells and evaluate the kernel.
                for &source_cell in source_cells.iter() {
                    // Get the points of the other cell.
                    geometry_map.points(source_cell, source_points.data_mut());
                    // Now get the right charges.
                    let charge_start = global_to_active_index[&source_cell] * n_points;
                    let charge_end = charge_start + n_points;
                    source_charges
                        .data_mut()
                        .copy_from_slice(&charges.data()[charge_start..charge_end]);
                    // We need to multiply the source charges with alpha
                    source_charges.scale_inplace(alpha);

                    // Now we can evaluate into the result chunks. The evaluate routine always adds to the result array.
                    // We use the single threaded routine here as threading is done on the chunk level.
                    kernel.evaluate_st(
                        eval_type,
                        source_points.data(),
                        target_points.data(),
                        source_charges.data(),
                        result_chunk,
                    );
                }
            });
    }
}
