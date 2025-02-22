//! Various helper functions to support evaluators.

use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use bempp_distributed_tools::{index_embedding::IndexEmbedding, Global2LocalDataMapper};
use green_kernels::{traits::Kernel, types::GreenKernelEvalType};
use itertools::{izip, Itertools};
use mpi::traits::{Communicator, Equivalence};
use ndelement::traits::FiniteElement;
use ndgrid::traits::{Entity, GeometryMap, Grid, ParallelGrid, Topology};

use num::Zero;
use rlst::{
    measure_duration,
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
#[measure_duration(id = "space_to_point_map")]
pub fn space_to_point_map<
    'a,
    T: RlstScalar + Equivalence,
    C: Communicator,
    Space: FunctionSpaceTrait<T = T, C = C>,
>(
    function_space: &'a Space,
    quadrature_points: &[T::Real],
    quadrature_weights: &[T::Real],
    return_transpose: bool,
) -> Operator<DistributedCsrMatrixOperatorImpl<'a, T, C>> {
    // Get the grid.
    let grid = function_space.grid();

    // Get the rank

    // Topological dimension of the grid.
    let tdim = grid.local_grid().topology_dim();

    // The method is currently restricted to single element type grids.
    // So let's make sure that this is the case.

    assert_eq!(grid.local_grid().entity_types(tdim).len(), 1);

    let reference_cell = grid.local_grid().entity_types(tdim)[0];

    // The quadrature points will be taken on the owned support cells.
    let owned_support_cells = function_space.local_space().owned_support_cells();

    // Number of owned support cells.
    let n_owned_support_cells = owned_support_cells.len();

    // Get the number of quadrature points and check that weights
    // and points have compatible dimensions.

    let n_quadrature_points = quadrature_weights.len();
    assert_eq!(quadrature_points.len() % tdim, 0);
    assert_eq!(quadrature_points.len() / tdim, n_quadrature_points);

    // Assign number of domain and range dofs.

    // Create an index embedding from support cells to all owned cells. The index embedding is necessary
    // since the created sparse matrix will map onto the subset of the owned cells that are given by
    // the owned support cells.

    let index_embedding = IndexEmbedding::new(grid.cell_layout(), owned_support_cells, grid.comm());

    // Create the index layouts. The domain index layout is just the function space layout.
    // The range layout is given by the number of owned support cells times the number of quadrature
    // points on each cell.

    let domain_layout = function_space.index_layout();
    let range_layout = Rc::new(IndexLayout::from_local_counts(
        n_owned_support_cells * n_quadrature_points,
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

    // In the tabulated data the dimensions are as follows.
    // - First dimension is derivative
    // - Second dimension is point index
    // - Third dimension is basis function index
    // - Fourth dimension is basis function component
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
pub struct NeighbourEvaluator<'a, Space: FunctionSpaceTrait, K: Kernel<T = Space::T>> {
    /// The source function space on which the evaluator acts
    source_space: &'a Space,
    /// The target function space on which the evaluator acts
    target_space: &'a Space,
    /// The element evaluation points (usually quadrature points)
    eval_points: Vec<<Space::T as RlstScalar>::Real>,
    /// The number of evaluation points
    n_points: usize,
    /// The Green's function kernel to be evaluated
    kernel: K,
    /// The evaluation type (only values or also derivatives)
    eval_type: GreenKernelEvalType,
    /// The domain vector space
    domain_space: Rc<DistributedArrayVectorSpace<'a, Space::C, Space::T>>,
    /// The range vector space
    range_space: Rc<DistributedArrayVectorSpace<'a, Space::C, Space::T>>,
    /// The mapper from global vectors to local data
    global_to_local_mapper: Global2LocalDataMapper<'a, Space::C>,
    /// The required index embedding
    index_embedding: IndexEmbedding<'a, Space::C>,
    /// Mapping of cell index to position in the support cells
    cell_to_position: HashMap<usize, usize>,
}

impl<'a, K: Kernel<T = Space::T>, Space: FunctionSpaceTrait> NeighbourEvaluator<'a, Space, K> {
    /// Create a new neighbour evaluator.
    #[measure_duration(id = "neighbour_evaluator_from_spaces_and_kernel")]
    pub fn from_spaces_and_kernel(
        source_space: &'a Space,
        target_space: &'a Space,
        eval_points: &[<Space::T as RlstScalar>::Real],
        kernel: K,
        eval_type: GreenKernelEvalType,
    ) -> Operator<Self> {
        assert!(
            std::ptr::addr_eq(source_space.grid(), target_space.grid()),
            "Source and target space must have the same grid."
        );
        let grid = target_space.grid();
        let tdim = grid.local_grid().topology_dim();

        let comm = grid.comm();

        // The method is currently restricted to single element type grids.
        // So let's make sure that this is the case.

        assert_eq!(grid.local_grid().entity_types(tdim).len(), 1);

        // Get the number of points
        assert_eq!(eval_points.len() % tdim, 0);
        let n_points = eval_points.len() / tdim;

        let owned_source_support_cells = source_space.local_space().owned_support_cells();
        let owned_target_support_cells = target_space.local_space().owned_support_cells();

        // The data will be coming in with respect to owned source support cells. However, to make data distribution
        // easier we embed the owned support cells into the set of all owned cells.

        let index_embedding =
            IndexEmbedding::new(grid.cell_layout(), owned_source_support_cells, grid.comm());

        let n_owned_target_support_cells = owned_target_support_cells.len();
        let n_owned_source_support_cells = owned_source_support_cells.len();

        // We now setup the array space. This is a distributed array space with a layout corresponding
        // to the number of points on each process.

        let domain_space = DistributedArrayVectorSpace::from_index_layout(Rc::new(
            IndexLayout::from_local_counts(n_owned_source_support_cells * n_points, comm),
        ));

        let range_space = DistributedArrayVectorSpace::from_index_layout(Rc::new(
            IndexLayout::from_local_counts(
                n_owned_target_support_cells * n_points * kernel.range_component_count(eval_type),
                comm,
            ),
        ));

        // We now setup the data mapper that maps from all owned cells to the right order of all owned source support cells.
        // The data mapper acts on global indices, so have to map the indices of the support cells to their respective
        // global indices. Note that we use `support_cells` and not `owned_support_cells` since we need the action of all
        // owned support cells against all support cells.

        let required_dofs = source_space
            .local_space()
            .support_cells()
            .iter()
            .map(|index| {
                grid.local_grid()
                    .entity(tdim, *index)
                    .unwrap()
                    .global_index()
            })
            .collect_vec();

        let global_to_local_mapper =
            Global2LocalDataMapper::new(grid.cell_layout(), &required_dofs);

        // We now need to setup a mapping from local cell index to position in the require dofs.
        // This is needed later to copy out the correct data slice from the charges vector.

        let cell_to_position: HashMap<usize, usize> = source_space
            .local_space()
            .support_cells()
            .iter()
            .enumerate()
            .map(|(i, d)| (*d, i))
            .collect();

        Operator::new(Self {
            source_space,
            target_space,
            eval_points: eval_points.to_vec(),
            n_points,
            kernel,
            eval_type,
            domain_space,
            range_space,
            global_to_local_mapper,
            index_embedding,
            cell_to_position,
        })
    }

    fn communicate_dofs(&self, x: &DistributedVector<'_, Space::C, Space::T>) -> Vec<Space::T> {
        // The data vector x has dofs associated with the owned support cells.
        // We embed this data into a larger vector associated with all owned cells
        // and then map this vector globally across all processes to the support cells
        // on each process. This ensures that each process also has the data for cells that
        // are in the support but not owned by it.

        let mut all_owned_cells_vec = vec![
            <Space::T as Zero>::zero();
            self.source_space
                .grid()
                .cell_layout()
                .number_of_local_indices()
                * self.n_points
        ];

        self.index_embedding
            .embed_data(x.local().data(), &mut all_owned_cells_vec, self.n_points);

        // We can now communicate the data. The returned data is the data for all support cells on each process.

        self.global_to_local_mapper
            .map_data(&all_owned_cells_vec, self.n_points)
    }
}

impl<K: Kernel<T = Space::T>, Space: FunctionSpaceTrait> std::fmt::Debug
    for NeighbourEvaluator<'_, Space, K>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Neighbourhood Evaluator with dimenion [{}, {}].",
            self.domain_space.index_layout().number_of_global_indices(),
            self.range_space.index_layout().number_of_global_indices()
        )
    }
}

impl<'a, K: Kernel<T = Space::T>, Space: FunctionSpaceTrait> OperatorBase
    for NeighbourEvaluator<'a, Space, K>
{
    type Domain = DistributedArrayVectorSpace<'a, Space::C, Space::T>;

    type Range = DistributedArrayVectorSpace<'a, Space::C, Space::T>;

    fn domain(&self) -> Rc<Self::Domain> {
        self.domain_space.clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.range_space.clone()
    }
}

impl<K: Kernel<T = Space::T>, Space: FunctionSpaceTrait> AsApply
    for NeighbourEvaluator<'_, Space, K>
where
    Space::LocalGrid: Sync,
    for<'a> <Space::LocalGrid as Grid>::GeometryMap<'a>: Sync,
{
    #[measure_duration(id = "neighbour_evaluator_apply")]
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
        let tdim = self.source_space.grid().local_grid().topology_dim();
        let gdim = self.source_space.grid().local_grid().geometry_dim();

        // We need the chunk size of the targets. This is the chunk size of the domain multiplied
        // with the range component count of the kernel.
        let target_chunk_size = self.n_points * self.kernel.range_component_count(self.eval_type);

        let n_points = self.n_points;

        // We need the reference cell
        let reference_cell = self.source_space.grid().local_grid().entity_types(tdim)[0];

        // We now need to communicate the data. The `charges` vector contains all charges for the support cells on this process.

        let charges = self.communicate_dofs(x.view());

        // The `charges` vector now has all the charges for each cell on our process, including ghost cells.
        // Next we iterate through the targets from the owned support cells and evaluate the kernel with sources coming
        // from the neighbouring cells and the self interactions.

        // We go through groups of target dofs in chunks of lenth n_points.
        // But first we need to get references to all requires variables since `self` is not
        // thread safe due to the communicator that is not necessarily thread safe.

        // The local grid
        let local_grid = self.source_space.grid().local_grid();
        // All support cells (including ghosts). We need this as set to check if a given cell is a support cell.
        let source_support_cells: HashSet<usize> = HashSet::from_iter(
            self.source_space
                .local_space()
                .support_cells()
                .iter()
                .copied(),
        );
        // The owned support cells
        let owned_target_support_cells = self.target_space.local_space().owned_support_cells();
        // Evaluation points
        let eval_points = self.eval_points.as_slice();
        // Geometry map that maps from reference points to physical points
        let geometry_map = local_grid.geometry_map(reference_cell, eval_points);
        // The kernel to evaluate
        let kernel = &self.kernel;
        // The evaluation type
        let eval_type = self.eval_type;
        // Map of cell index to position in `support_cells` array
        let cell_to_position = &self.cell_to_position;

        // We have a parallel loop over chunks of the result vector zipped with the owned support cells
        // The owned support cells are the targets. The support cells are the sources. This ensures that the interaction
        // from ghost to owned cell is covered.

        y.view_mut()
            .local_mut()
            .data_mut()
            .par_chunks_mut(target_chunk_size)
            .zip(owned_target_support_cells)
            .for_each(|(result_chunk, &owned_target_cell_index)| {
                let cell_entity = local_grid.entity(tdim, owned_target_cell_index).unwrap();

                // Get the target points for the target cell.
                let mut target_points =
                    rlst_dynamic_array2![<Space::T as RlstScalar>::Real, [gdim, n_points]];
                geometry_map.points(owned_target_cell_index, target_points.data_mut());

                // Get all the neighbouring celll indices.
                // This is a bit cumbersome. We go through the points of the target cell and add all cells that are connected
                // to each point, using a `unique` iterator to remove duplicates. However, we also need to make sure
                // that we only add cells that are in the support set.
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
                    .filter(|&cell| source_support_cells.contains(&cell))
                    .unique()
                    .collect_vec();

                // The next array will store the set of source points for each source cell.
                let mut source_points =
                    rlst_dynamic_array2![<Space::T as RlstScalar>::Real, [gdim, n_points]];
                // We also need to store the source charges.
                let mut source_charges = rlst_dynamic_array1![Space::T, [n_points]];

                // We need to multiply the targets with beta
                for value in result_chunk.iter_mut() {
                    *value *= beta;
                }

                // We can now go through the source cells and evaluate the kernel.
                for &source_cell in source_cells.iter() {
                    // Get the points of the other cell.
                    geometry_map.points(source_cell, source_points.data_mut());
                    // Now get the right charges.
                    let charge_start = cell_to_position[&source_cell] * n_points;
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

/// Return the associated grid points from a function space
pub fn grid_points_from_space<Space: FunctionSpaceTrait>(
    space: &Space,
    eval_points: &[<Space::T as RlstScalar>::Real],
) -> Vec<<Space::T as RlstScalar>::Real> {
    let tdim = space.grid().local_grid().topology_dim();
    let gdim = space.grid().local_grid().geometry_dim();

    assert_eq!(eval_points.len() % tdim, 0);

    assert_eq!(space.grid().local_grid().entity_types(tdim).len(), 1);
    let reference_cell = space.grid().local_grid().entity_types(tdim)[0];

    let n_eval_points = eval_points.len() / tdim;

    let mut points = vec![
        <<Space::T as RlstScalar>::Real as Zero>::zero();
        gdim * n_eval_points * space.local_space().owned_support_cells().len()
    ];

    let geometry_map = space
        .grid()
        .local_grid()
        .geometry_map(reference_cell, eval_points);

    for (cell, point_chunk) in izip!(
        space
            .local_space()
            .owned_support_cells()
            .iter()
            .map(|cell_index| space.grid().local_grid().entity(tdim, *cell_index).unwrap()),
        points.chunks_mut(gdim * n_eval_points)
    ) {
        geometry_map.points(cell.local_index(), point_chunk);
    }
    points
}
