//! Various helper functions to support evaluators.

use std::{collections::HashSet, marker::PhantomData, rc::Rc};

use bempp_distributed_tools::{index_embedding::IndexEmbedding, Global2LocalDataMapper};
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
    rlst_dynamic_array4, AsApply, DefaultIterator, DistributedCsrMatrix, DistributedVector,
    Element, IndexLayout, OperatorBase, RawAccess, RawAccessMut, RlstScalar,
};

use rayon::prelude::*;

use crate::function::{FunctionSpaceTrait, LocalFunctionSpaceTrait};

/// Create a linear operator from the map of a basis to points. The points are sorted by global
/// index of the corresponding cells in the support. The output space is the space obtained through
/// the owned support cells on each process.
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

    // Get the rank

    // Topological dimension of the grid.
    let tdim = grid.local_grid().topology_dim();

    // The method is currently restricted to single element type grids.
    // So let's make sure that this is the case.

    assert_eq!(grid.local_grid().entity_types(tdim).len(), 1);

    let reference_cell = grid.local_grid().entity_types(tdim)[0];

    let owned_support_cells: Vec<usize> = function_space
        .local_space()
        .support_cells()
        .iter()
        .filter(|index| {
            matches!(
                grid.local_grid().entity(tdim, **index).unwrap().ownership(),
                Ownership::Owned
            )
        })
        .copied()
        .collect_vec();

    // Number of cells. We are only interested in owned cells.
    let n_cells = owned_support_cells.len();

    // Get the number of quadrature points and check that weights
    // and points have compatible dimensions.

    let n_quadrature_points = quadrature_weights.len();
    assert_eq!(quadrature_points.len() % tdim, 0);
    assert_eq!(quadrature_points.len() / tdim, n_quadrature_points);

    // Assign number of domain and range dofs.

    let n_range_dofs = n_cells * n_quadrature_points;

    // Create an index embedding from support cells to all local cells.

    let index_embedding =
        IndexEmbedding::new(grid.cell_layout(), &owned_support_cells, grid.comm());

    // Create the index layouts.

    let domain_layout = function_space.index_layout();
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
            .local_space()
            .element(reference_cell)
            .tabulate_array_shape(0, n_quadrature_points)
    );
    function_space
        .local_space()
        .element(reference_cell)
        .tabulate(&quadrature_points, 0, &mut table);

    // We have tabulated the basis functions on the reference element. Now need
    // the map to physical elements.

    let geometry_evaluator = grid
        .local_grid()
        .geometry_map(reference_cell, quadrature_points.data());

    // The following arrays hold the jacobians, their determinants and the normals.

    let mut jacobians = rlst_dynamic_array3![
        T::Real,
        [grid.local_grid().geometry_dim(), tdim, n_quadrature_points]
    ];
    let mut jdets = rlst_dynamic_array1![T::Real, [n_quadrature_points]];
    let mut normals = rlst_dynamic_array2![
        T::Real,
        [grid.local_grid().geometry_dim(), n_quadrature_points]
    ];

    // Now iterate through the cells of the grid, get the attached dofs and evaluate the geometry map.

    // These arrays store the data of the transformation matrix.
    let mut rows = Vec::<usize>::default();
    let mut cols = Vec::<usize>::default();
    let mut data = Vec::<T>::default();

    for (embedded_index, &cell_index) in owned_support_cells.iter().enumerate() {
        let cell = grid.local_grid().entity(tdim, cell_index).unwrap();
        // We need to get the global embeddeded index of the owned support cell.
        // First we get the global index of the owned support cell.

        let global_embedded_index = index_embedding
            .embedded_layout()
            .local2global(embedded_index)
            .unwrap();

        // Get the Jacobians
        geometry_evaluator.jacobians_dets_normals(
            cell.local_index(),
            jacobians.data_mut(),
            jdets.data_mut(),
            normals.data_mut(),
        );
        // Get the global dofs of the cell.
        let global_dofs = function_space
            .local_space()
            .cell_dofs(cell_index)
            .unwrap()
            .iter()
            .map(|local_dof_index| {
                function_space
                    .local_space()
                    .global_dof_index(*local_dof_index)
            })
            .collect::<Vec<_>>();

        for (qindex, (jdet, qweight)) in izip!(jdets.iter(), quadrature_weights.iter()).enumerate()
        {
            for (i, global_dof) in global_dofs.iter().enumerate() {
                rows.push(n_quadrature_points * global_embedded_index + qindex);
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
    GridImpl: ParallelGrid<C = C, T = T::Real>,
    C: Communicator,
> {
    eval_points: Vec<T::Real>,
    n_points: usize,
    kernel: K,
    eval_type: GreenKernelEvalType,
    array_space: Rc<DistributedArrayVectorSpace<'a, C, T>>,
    grid: &'a GridImpl,
    support_cells: Vec<usize>,
    owned_support_cells: Vec<usize>,
    global_to_local_mapper: Global2LocalDataMapper<'a, C>,
    index_embedding: IndexEmbedding<'a, C>,
    _marker: PhantomData<C>,
}

impl<
        'a,
        T: RlstScalar + Equivalence,
        K: Kernel<T = T>,
        GridImpl: ParallelGrid<C = C, T = T::Real>,
        C: Communicator,
    > NeighbourEvaluator<'a, T, K, GridImpl, C>
{
    /// Create a new neighbour evaluator.
    pub fn new(
        eval_points: &[T::Real],
        kernel: K,
        eval_type: GreenKernelEvalType,
        support_cells: &[usize],
        grid: &'a GridImpl,
    ) -> Self {
        let tdim = grid.local_grid().topology_dim();

        let comm = grid.comm();

        // The method is currently restricted to single element type grids.
        // So let's make sure that this is the case.

        assert_eq!(grid.local_grid().entity_types(tdim).len(), 1);

        // Get the number of points
        assert_eq!(eval_points.len() % tdim, 0);
        let n_points = eval_points.len() / tdim;

        // The active cells are all relevant cells on a process,
        // typically the support of a function space.

        // The owned active cells provide the relevant dofs.

        // The operator gets the relevant dofs for the owned active cells
        // and then needs to communicate for the rest of the active cells
        // the data from the other processes.

        let support_cells = support_cells
            .iter()
            .copied()
            .sorted_by_key(|&index| {
                grid.local_grid()
                    .entity(tdim, index)
                    .unwrap()
                    .global_index()
            })
            .collect_vec();

        let owned_support_cells: Vec<usize> = support_cells
            .iter()
            .filter(|cell_index| {
                matches!(
                    grid.local_grid()
                        .entity(tdim, **cell_index)
                        .unwrap()
                        .ownership(),
                    Ownership::Owned
                )
            })
            .copied()
            .collect_vec();

        // The data will be coming in with respect to owned support cells. However, to make data distribution
        // easier we embed the owned support cells into the set of all owned cells.

        let index_embedding =
            IndexEmbedding::new(grid.cell_layout(), &owned_support_cells, grid.comm());

        let n_owned_support_cells = owned_support_cells.len();

        // We now setup the array space. This is a distributed array space with a layout corresponding
        // to the number of points on each process.

        let array_space = DistributedArrayVectorSpace::from_index_layout(Rc::new(
            IndexLayout::from_local_counts(n_owned_support_cells * n_points, comm),
        ));

        // We now setup the data mapper with respect to all owned cells on each process.

        let global_to_local_mapper = Global2LocalDataMapper::new(
            grid.cell_layout(),
            &support_cells
                .iter()
                .map(|index| {
                    grid.local_grid()
                        .entity(tdim, *index)
                        .unwrap()
                        .global_index()
                })
                .collect_vec(),
        );

        Self {
            eval_points: eval_points.to_vec(),
            n_points,
            kernel,
            eval_type,
            array_space,
            grid,
            support_cells,
            owned_support_cells,
            global_to_local_mapper,
            index_embedding,
            _marker: PhantomData,
        }
    }

    fn communicate_dofs(&self, x: &DistributedVector<'_, C, T>) -> Vec<T> {
        // The data vector x has dofs associated with the owned support cells.
        // We embed this data into a larger vector associated with all owned cells
        // and then map this vector globally across all processes to the active cells
        // on each process.

        let mut all_cells_vec =
            vec![T::zero(); self.grid.cell_layout().number_of_local_indices() * self.n_points];

        self.index_embedding
            .embed_data(x.local().data(), &mut all_cells_vec, self.n_points);

        // We can now communicate the data. The returned data is the data for all support cells on each process.

        self.global_to_local_mapper
            .map_data(x.local().data(), self.n_points)
    }
}

impl<
        T: RlstScalar + Equivalence,
        K: Kernel<T = T>,
        GridImpl: ParallelGrid<C = C, T = T::Real>,
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
        GridImpl: ParallelGrid<C = C, T = T::Real>,
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
        GridImpl: ParallelGrid<C = C, T = T::Real>,
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

        let tdim = self.grid.local_grid().topology_dim();
        let gdim = self.grid.local_grid().geometry_dim();

        // We need the chunk size of the targets. This is the chunk size of the domain multiplied
        // with the range component count of the kernel.
        let target_chunk_size = self.n_points * self.kernel.range_component_count(self.eval_type);

        // In the new function we already made sure that eval_points is a multiple of tdim.
        let n_points = self.n_points;

        // We need the reference cell
        let reference_cell = self.grid.local_grid().entity_types(tdim)[0];

        // We now need to communicate the data. The `charges` vector contains all charges for the active cells on this process.

        let charges = self.communicate_dofs(x.view());

        // The `charges` vector now has all the charges for each cell on our process, including ghost cells.
        // Next we iterate through the targets from the active cells and evaluate the kernel with sources coming
        // from the neighbouring cells and the self interactions.

        // We go through groups of target dofs in chunks of lenth n_points.
        // This corresponds to iteration in active cells since we ordered those
        // already by global index.

        let local_grid = self.grid.local_grid();
        let support_cells: HashSet<usize> =
            HashSet::from_iter(self.support_cells.as_slice().iter().copied());
        let owned_support_cells = self.owned_support_cells.as_slice();
        let eval_points = self.eval_points.as_slice();
        let geometry_map = local_grid.geometry_map(reference_cell, eval_points);
        let kernel = &self.kernel;
        let eval_type = self.eval_type;
        let dof_to_position_map = &self.global_to_local_mapper.dof_to_position_map();

        y.view_mut()
            .local_mut()
            .data_mut()
            .par_chunks_mut(target_chunk_size)
            .zip(owned_support_cells)
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
                    .filter(|&cell| support_cells.contains(&cell))
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
                    let charge_start = dof_to_position_map[&source_cell] * n_points;
                    let charge_end = charge_start + n_points;
                    source_charges
                        .data_mut()
                        .copy_from_slice(&charges[charge_start..charge_end]);
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
