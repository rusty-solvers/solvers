//! Boundary operator assembly
mod cell_pair_assemblers;
pub(crate) mod helpers;
pub(crate) mod integrands;

use crate::boundary_assemblers::cell_pair_assemblers::{
    NonsingularCellPairAssemblerWithTestCaching, SingularCellPairAssembler,
};
use crate::boundary_assemblers::helpers::KernelEvaluator;
use crate::boundary_assemblers::helpers::{equal_grids, RawData2D, RlstArray, SparseMatrixData};
use crate::function::{FunctionSpaceTrait, LocalFunctionSpaceTrait};
use bempp_quadrature::duffy::{
    quadrilateral_duffy, quadrilateral_triangle_duffy, triangle_duffy, triangle_quadrilateral_duffy,
};
use bempp_quadrature::types::{
    CellToCellConnectivity, NumericalQuadratureDefinition, TestTrialNumericalQuadratureDefinition,
};
use green_kernels::traits::Kernel;
use integrands::BoundaryIntegrand;
use itertools::izip;
use mpi::traits::{Communicator, Equivalence};
use ndelement::quadrature::simplex_rule;
use ndelement::reference_cell;
use ndelement::traits::FiniteElement;
use ndelement::types::ReferenceCellType;
use ndgrid::traits::{Entity, Grid, Topology};
use ndgrid::types::Ownership;
use rayon::prelude::*;
use rlst::{
    measure_duration, rlst_dynamic_array2, rlst_dynamic_array4, DefaultIterator,
    DistributedCsrMatrix, DynamicArray, MatrixInverse, RandomAccessMut, RawAccess, RawAccessMut,
    RlstScalar, Shape,
};
use std::collections::HashMap;

/// Options for a boundary assembler
#[derive(Clone)]
pub struct BoundaryAssemblerOptions {
    /// Number of points used in quadrature for non-singular integrals
    pub quadrature_degrees: HashMap<ReferenceCellType, usize>,
    /// Quadrature degrees to be used for singular integrals
    pub singular_quadrature_degrees: HashMap<(ReferenceCellType, ReferenceCellType), usize>,
    /// Maximum size of each batch of cells to send to an assembly function
    pub batch_size: usize,
}

impl Default for BoundaryAssemblerOptions {
    fn default() -> Self {
        use ReferenceCellType::{Quadrilateral, Triangle};
        Self {
            quadrature_degrees: HashMap::from([(Triangle, 37), (Quadrilateral, 37)]),
            singular_quadrature_degrees: HashMap::from([
                ((Triangle, Triangle), 4),
                ((Quadrilateral, Quadrilateral), 4),
                ((Quadrilateral, Triangle), 4),
                ((Triangle, Quadrilateral), 4),
            ]),
            batch_size: 128,
        }
    }
}

impl BoundaryAssemblerOptions {
    /// Set the regular quadrature order.
    pub fn set_regular_quadrature_degree(&mut self, cell_type: ReferenceCellType, npoints: usize) {
        self.quadrature_degrees
            .entry(cell_type)
            .and_modify(|x| *x = npoints);
    }

    /// Get the regular quadrature order.
    pub fn get_regular_quadrature_degree(&self, cell_type: ReferenceCellType) -> Option<usize> {
        self.quadrature_degrees.get(&cell_type).copied()
    }

    /// Set the singular quadrature order.
    pub fn set_singular_quadrature_degree(
        &mut self,
        cell_type: (ReferenceCellType, ReferenceCellType),
        npoints: usize,
    ) {
        self.singular_quadrature_degrees
            .entry(cell_type)
            .and_modify(|x| *x = npoints);
    }

    /// Get the singular quadrature order.
    pub fn get_singular_quadrature_degree(
        &self,
        cell_type: (ReferenceCellType, ReferenceCellType),
    ) -> Option<usize> {
        self.singular_quadrature_degrees.get(&cell_type).copied()
    }

    /// Set the batch size.
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }

    /// Set the batch size.
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the regular quadrature rule.
    pub fn get_regular_quadrature_rule(
        &self,
        cell_type: ReferenceCellType,
    ) -> NumericalQuadratureDefinition {
        match cell_type {
            ReferenceCellType::Triangle => bempp_quadrature::simplex_rules::simplex_rule_triangle(
                self.get_regular_quadrature_degree(cell_type).unwrap(),
            )
            .unwrap(),
            ReferenceCellType::Quadrilateral => {
                bempp_quadrature::simplex_rules::simplex_rule_quadrilateral(
                    self.get_regular_quadrature_degree(cell_type).unwrap(),
                )
                .unwrap()
            }
            _ => panic!("Quadrature rules not implemented for cell type."),
        }
    }
}

/// Boundary assembler
///
/// Assembles operators by processing batches of cells in parallel
pub struct BoundaryAssembler<
    'o,
    T: RlstScalar + MatrixInverse,
    Integrand: BoundaryIntegrand<T = T>,
    K: Kernel<T = T>,
> {
    pub(crate) integrand: Integrand,
    pub(crate) kernel: KernelEvaluator<T, K>,
    pub(crate) options: &'o BoundaryAssemblerOptions,
    pub(crate) deriv_size: usize,
    pub(crate) table_derivs: usize,
}

impl<'o, T: RlstScalar + MatrixInverse, Integrand: BoundaryIntegrand<T = T>, K: Kernel<T = T>>
    BoundaryAssembler<'o, T, Integrand, K>
{
    /// Assemble the singular part into a CSR matrix.
    #[measure_duration(id = "assemble_singular")]
    pub fn assemble_singular<'a, C: Communicator, Space: FunctionSpaceTrait<T = T, C = C>>(
        &self,
        trial_space: &'a Space,
        test_space: &'a Space,
    ) -> DistributedCsrMatrix<'a, T, C>
    where
        Space::LocalFunctionSpace: Sync,
        T: Equivalence,
    {
        let shape = [
            test_space.local_space().global_size(),
            trial_space.local_space().global_size(),
        ];
        let sparse_matrix =
            self.assemble_singular_part(shape, trial_space.local_space(), test_space.local_space());

        // Instantiate the CSR matrix.

        DistributedCsrMatrix::from_aij(
            trial_space.index_layout(),
            test_space.index_layout(),
            &sparse_matrix.rows,
            &sparse_matrix.cols,
            &sparse_matrix.data,
        )
    }

    /// Assemble into a dense matrix.
    pub fn assemble<Space: FunctionSpaceTrait<T = T>>(
        &self,
        trial_space: &Space,
        test_space: &Space,
    ) -> DynamicArray<T, 2>
    where
        Space::LocalFunctionSpace: Sync,
    {
        if trial_space.comm().size() > 1 || test_space.comm().size() > 1 {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }

        let mut output = rlst_dynamic_array2!(
            T,
            [
                test_space.local_space().global_size(),
                trial_space.local_space().global_size()
            ]
        );

        self.assemble_into_memory(trial_space, test_space, output.data_mut());

        output
    }

    /// Assemble into a dense matrix.
    pub fn assemble_into_memory<Space: FunctionSpaceTrait<T = T>>(
        &self,
        trial_space: &Space,
        test_space: &Space,
        output: &mut [T],
    ) where
        Space::LocalFunctionSpace: Sync,
    {
        assert_eq!(
            output.len(),
            test_space.local_space().global_size() * trial_space.local_space().global_size()
        );
        if trial_space.comm().size() > 1 || test_space.comm().size() > 1 {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }

        let test_colouring = test_space.local_space().cell_colouring();
        let trial_colouring = trial_space.local_space().cell_colouring();
        let shape = [
            test_space.local_space().global_size(),
            trial_space.local_space().global_size(),
        ];
        let output_raw = RawData2D {
            data: output.as_mut_ptr(),
            shape,
        };

        self.assemble_nonsingular_part(
            &output_raw,
            trial_space.local_space(),
            test_space.local_space(),
            &trial_colouring,
            &test_colouring,
        );

        let sparse_matrix =
            self.assemble_singular_part(shape, trial_space.local_space(), test_space.local_space());

        let data = sparse_matrix.data;
        let rows = sparse_matrix.rows;
        let cols = sparse_matrix.cols;
        for ((i, j), value) in rows.iter().zip(cols.iter()).zip(data.iter()) {
            *output.get_mut(*i + shape[0] * *j).unwrap() += *value;
        }
    }

    /// Create new Boundary assembler
    pub(crate) fn new(
        integrand: Integrand,
        kernel: KernelEvaluator<T, K>,
        options: &'o BoundaryAssemblerOptions,
        deriv_size: usize,
        table_derivs: usize,
    ) -> Self {
        Self {
            integrand,
            kernel,
            options,
            deriv_size,
            table_derivs,
        }
    }

    /// Assemble the singular contributions
    fn assemble_singular_part<Space: LocalFunctionSpaceTrait<T = T> + Sync>(
        &self,
        shape: [usize; 2],
        trial_space: &Space,
        test_space: &Space,
    ) -> SparseMatrixData<T> {
        if !equal_grids(test_space.grid(), trial_space.grid()) {
            // If the test and trial grids are different, there are no neighbouring triangles
            return SparseMatrixData::new(shape);
        }

        if shape[0] != test_space.global_size() || shape[1] != trial_space.global_size() {
            panic!("Matrix has wrong shape");
        }

        let grid = test_space.grid();

        let mut qweights = vec![];
        let mut trial_points = vec![];
        let mut test_points = vec![];
        let mut trial_tables = vec![];
        let mut test_tables = vec![];
        let mut test_cell_types = vec![];
        let mut trial_cell_types = vec![];

        let mut pair_indices = HashMap::new();

        let mut max_test_basis_fun = 0;
        let mut max_trial_basis_fun = 0;

        for test_cell_type in grid.entity_types(2) {
            for trial_cell_type in grid.entity_types(2) {
                let qdegree =
                    self.options.singular_quadrature_degrees[&(*test_cell_type, *trial_cell_type)];
                let offset = qweights.len();

                let mut possible_pairs = vec![];
                // Vertex-adjacent
                for i in 0..reference_cell::entity_counts(*test_cell_type)[0] {
                    for j in 0..reference_cell::entity_counts(*trial_cell_type)[0] {
                        possible_pairs.push(vec![(i, j)]);
                    }
                }
                // edge-adjacent
                for test_e in reference_cell::edges(*test_cell_type) {
                    for trial_e in reference_cell::edges(*trial_cell_type) {
                        possible_pairs.push(vec![(test_e[0], trial_e[0]), (test_e[1], trial_e[1])]);
                        possible_pairs.push(vec![(test_e[1], trial_e[0]), (test_e[0], trial_e[1])]);
                    }
                }
                // Same cell
                if test_cell_type == trial_cell_type {
                    possible_pairs.push(
                        (0..reference_cell::entity_counts(*test_cell_type)[0])
                            .map(&|i| (i, i))
                            .collect::<Vec<_>>(),
                    );
                }

                for (i, pairs) in possible_pairs.iter().enumerate() {
                    pair_indices.insert(
                        (*test_cell_type, *trial_cell_type, pairs.clone()),
                        offset + i,
                    );
                    test_cell_types.push(*test_cell_type);
                    trial_cell_types.push(*trial_cell_type);
                }

                for pairs in &possible_pairs {
                    let qrule = get_singular_quadrature_rule(
                        *test_cell_type,
                        *trial_cell_type,
                        pairs,
                        qdegree,
                    );
                    let npts = qrule.weights.len();

                    let mut points = rlst_dynamic_array2!(<T as RlstScalar>::Real, [2, npts]);
                    for i in 0..npts {
                        for j in 0..2 {
                            *points.get_mut([j, i]).unwrap() =
                                num::cast::<f64, <T as RlstScalar>::Real>(
                                    qrule.trial_points[2 * i + j],
                                )
                                .unwrap();
                        }
                    }
                    let trial_element = trial_space.element(*trial_cell_type);
                    max_trial_basis_fun = std::cmp::max(max_trial_basis_fun, trial_element.dim());
                    let mut table = rlst_dynamic_array4!(
                        T,
                        trial_element.tabulate_array_shape(self.table_derivs, points.shape()[1])
                    );
                    trial_element.tabulate(&points, self.table_derivs, &mut table);
                    trial_points.push(points);
                    trial_tables.push(table);

                    let mut points = rlst_dynamic_array2!(<T as RlstScalar>::Real, [2, npts]);
                    for i in 0..npts {
                        for j in 0..2 {
                            *points.get_mut([j, i]).unwrap() =
                                num::cast::<f64, <T as RlstScalar>::Real>(
                                    qrule.test_points[2 * i + j],
                                )
                                .unwrap();
                        }
                    }
                    let test_element = test_space.element(*test_cell_type);
                    max_test_basis_fun = std::cmp::max(max_test_basis_fun, test_element.dim());
                    let mut table = rlst_dynamic_array4!(
                        T,
                        test_element.tabulate_array_shape(self.table_derivs, points.shape()[1])
                    );
                    test_element.tabulate(&points, self.table_derivs, &mut table);
                    test_points.push(points);
                    test_tables.push(table);
                    qweights.push(
                        qrule
                            .weights
                            .iter()
                            .map(|w| num::cast::<f64, <T as RlstScalar>::Real>(*w).unwrap())
                            .collect::<Vec<_>>(),
                    );
                }
            }
        }
        let cell_blocks = make_cell_blocks(
            |test_cell_type, trial_cell_type, pairs| {
                pair_indices[&(test_cell_type, trial_cell_type, pairs)]
            },
            pair_indices.len(),
            grid,
            self.options.batch_size,
        );

        let npairs = cell_blocks
            .iter()
            .map(|(_first, second)| second.len())
            .sum::<usize>();

        let map = cell_blocks.into_par_iter().map(|(i, cell_block)| {
            assemble_batch_singular(
                self,
                self.deriv_size,
                shape,
                trial_cell_types[i],
                test_cell_types[i],
                trial_space,
                test_space,
                &cell_block,
                &trial_points[i],
                &test_points[i],
                &qweights[i],
                &trial_tables[i],
                &test_tables[i],
            )
        });
        // For some reason rust analyzer threw an error when simply writing
        // map.reduce(...) even though the code compiled fine. Doing it this
        // way allows rust analyer to see that the `reduce` method is from
        // `ParallelIterator` and not from the std::core Iterator
        ParallelIterator::reduce(
            map,
            || {
                SparseMatrixData::<T>::new_known_size(
                    shape,
                    max_trial_basis_fun * max_test_basis_fun * npairs,
                )
            },
            |mut a, b| {
                a.add(b);
                a
            },
        )
    }

    /// Assemble the non-singular contributions into a dense matrix
    fn assemble_nonsingular_part<Space: LocalFunctionSpaceTrait<T = T> + Sync>(
        &self,
        output: &RawData2D<T>,
        trial_space: &Space,
        test_space: &Space,
        trial_colouring: &HashMap<ReferenceCellType, Vec<Vec<usize>>>,
        test_colouring: &HashMap<ReferenceCellType, Vec<Vec<usize>>>,
    ) {
        if output.shape[0] != test_space.global_size()
            || output.shape[1] != trial_space.global_size()
        {
            panic!("Matrix has wrong shape");
        }

        let batch_size = self.options.batch_size;

        for test_cell_type in test_space.grid().entity_types(2) {
            let npts_test = self.options.quadrature_degrees[test_cell_type];
            for trial_cell_type in trial_space.grid().entity_types(2) {
                let npts_trial = self.options.quadrature_degrees[trial_cell_type];
                let qrule_test = simplex_rule(*test_cell_type, npts_test).unwrap();
                let mut qpoints_test =
                    rlst_dynamic_array2!(<T as RlstScalar>::Real, [2, npts_test]);
                for i in 0..npts_test {
                    for j in 0..2 {
                        *qpoints_test.get_mut([j, i]).unwrap() =
                            num::cast::<f64, <T as RlstScalar>::Real>(qrule_test.points[2 * i + j])
                                .unwrap();
                    }
                }
                let qweights_test = qrule_test
                    .weights
                    .iter()
                    .map(|w| num::cast::<f64, <T as RlstScalar>::Real>(*w).unwrap())
                    .collect::<Vec<_>>();
                let qrule_trial = simplex_rule(*trial_cell_type, npts_trial).unwrap();
                let mut qpoints_trial =
                    rlst_dynamic_array2!(<T as RlstScalar>::Real, [2, npts_trial]);
                for i in 0..npts_trial {
                    for j in 0..2 {
                        *qpoints_trial.get_mut([j, i]).unwrap() =
                            num::cast::<f64, <T as RlstScalar>::Real>(
                                qrule_trial.points[2 * i + j],
                            )
                            .unwrap();
                    }
                }
                let qweights_trial = qrule_trial
                    .weights
                    .iter()
                    .map(|w| num::cast::<f64, <T as RlstScalar>::Real>(*w).unwrap())
                    .collect::<Vec<_>>();

                let test_element = test_space.element(*test_cell_type);
                let mut test_table = rlst_dynamic_array4!(
                    T,
                    test_element.tabulate_array_shape(self.table_derivs, npts_test)
                );
                test_element.tabulate(&qpoints_test, self.table_derivs, &mut test_table);

                let trial_element = trial_space.element(*trial_cell_type);
                let mut trial_table = rlst_dynamic_array4!(
                    T,
                    trial_element.tabulate_array_shape(self.table_derivs, npts_trial)
                );
                trial_element.tabulate(&qpoints_test, self.table_derivs, &mut trial_table);

                for test_c in &test_colouring[test_cell_type] {
                    for trial_c in &trial_colouring[trial_cell_type] {
                        let mut test_cells: Vec<&[usize]> = vec![];
                        let mut trial_cells: Vec<&[usize]> = vec![];

                        let mut test_start = 0;
                        while test_start < test_c.len() {
                            let test_end = if test_start + batch_size < test_c.len() {
                                test_start + batch_size
                            } else {
                                test_c.len()
                            };

                            let mut trial_start = 0;
                            while trial_start < trial_c.len() {
                                let trial_end = if trial_start + batch_size < trial_c.len() {
                                    trial_start + batch_size
                                } else {
                                    trial_c.len()
                                };
                                test_cells.push(&test_c[test_start..test_end]);
                                trial_cells.push(&trial_c[trial_start..trial_end]);
                                trial_start = trial_end;
                            }
                            test_start = test_end
                        }

                        let numtasks = test_cells.len();
                        let r: usize = (0..numtasks)
                            .into_par_iter()
                            .map(&|t| {
                                assemble_batch_nonadjacent(
                                    self,
                                    self.deriv_size,
                                    output,
                                    *test_cell_type,
                                    *trial_cell_type,
                                    trial_space,
                                    trial_cells[t],
                                    test_space,
                                    test_cells[t],
                                    &qpoints_trial,
                                    &qweights_trial,
                                    &qpoints_test,
                                    &qweights_test,
                                    &trial_table,
                                    &test_table,
                                )
                            })
                            .sum();
                        assert_eq!(r, numtasks);
                    }
                }
            }
        }
    }
}

fn get_singular_quadrature_rule(
    test_celltype: ReferenceCellType,
    trial_celltype: ReferenceCellType,
    pairs: &[(usize, usize)],
    npoints: usize,
) -> TestTrialNumericalQuadratureDefinition {
    if pairs.is_empty() {
        panic!("Non-singular rule requested.");
    }
    let con = CellToCellConnectivity {
        connectivity_dimension: match pairs.len() {
            1 => 0,
            2 => 1,
            _ => 2,
        },
        local_indices: pairs.to_vec(),
    };
    match test_celltype {
        ReferenceCellType::Triangle => match trial_celltype {
            ReferenceCellType::Triangle => triangle_duffy(&con, npoints).unwrap(),
            ReferenceCellType::Quadrilateral => {
                triangle_quadrilateral_duffy(&con, npoints).unwrap()
            }
            _ => {
                unimplemented!("Only triangles and quadrilaterals are currently supported");
            }
        },
        ReferenceCellType::Quadrilateral => match trial_celltype {
            ReferenceCellType::Triangle => quadrilateral_triangle_duffy(&con, npoints).unwrap(),
            ReferenceCellType::Quadrilateral => quadrilateral_duffy(&con, npoints).unwrap(),
            _ => {
                unimplemented!("Only triangles and quadrilaterals are currently supported");
            }
        },
        _ => {
            unimplemented!("Only triangles and quadrilaterals are currently supported");
        }
    }
}

fn make_cell_blocks<F>(
    f: F,
    size: usize,
    grid: &impl Grid<EntityDescriptor = ReferenceCellType>,
    batch_size: usize,
) -> Vec<(usize, Vec<(usize, usize)>)>
where
    F: Fn(ReferenceCellType, ReferenceCellType, Vec<(usize, usize)>) -> usize,
{
    let mut cell_pairs = vec![vec![]; size];

    for vertex in grid.entity_iter(0) {
        for test_cell_index in vertex.topology().connected_entity_iter(2) {
            let test_cell = grid.entity(2, test_cell_index).unwrap();
            let test_cell_type = test_cell.entity_type();
            if test_cell.ownership() == Ownership::Owned {
                for trial_cell_index in vertex.topology().connected_entity_iter(2) {
                    let trial_cell = grid.entity(2, trial_cell_index).unwrap();
                    let trial_cell_type = trial_cell.entity_type();

                    if let Some(pairs) =
                        get_pairs_if_smallest(&test_cell, &trial_cell, vertex.local_index())
                    {
                        cell_pairs[f(test_cell_type, trial_cell_type, pairs)]
                            .push((test_cell_index, trial_cell_index));
                    }
                }
            }
        }
    }
    let mut cell_blocks = vec![];

    for (i, cells) in cell_pairs.iter().enumerate() {
        let mut start = 0;
        while start < cells.len() {
            let end = std::cmp::min(start + batch_size, cells.len());
            cell_blocks.push((i, cells[start..end].to_vec()));
            start = end;
        }
    }

    cell_blocks
}

/// Assemble the contribution to the terms of a matrix for a batch of pairs of adjacent cells
#[allow(clippy::too_many_arguments)]
fn assemble_batch_singular<
    T: RlstScalar + MatrixInverse,
    Space: LocalFunctionSpaceTrait<T = T> + Sync,
    Integrand: BoundaryIntegrand<T = T>,
    K: Kernel<T = T>,
>(
    assembler: &BoundaryAssembler<T, Integrand, K>,
    deriv_size: usize,
    shape: [usize; 2],
    trial_cell_type: ReferenceCellType,
    test_cell_type: ReferenceCellType,
    trial_space: &Space,
    test_space: &Space,
    cell_pairs: &[(usize, usize)],
    trial_points: &RlstArray<T::Real, 2>,
    test_points: &RlstArray<T::Real, 2>,
    weights: &[T::Real],
    trial_table: &RlstArray<T, 4>,
    test_table: &RlstArray<T, 4>,
) -> SparseMatrixData<T> {
    let mut output = SparseMatrixData::<T>::new_known_size(
        shape,
        cell_pairs.len()
            * trial_space.element(trial_cell_type).dim()
            * test_space.element(test_cell_type).dim(),
    );
    let npts = weights.len();
    debug_assert!(weights.len() == npts);
    debug_assert!(test_points.shape()[1] == npts);
    debug_assert!(trial_points.shape()[1] == npts);

    let grid = test_space.grid();
    assert_eq!(grid.geometry_dim(), 3);
    assert_eq!(grid.topology_dim(), 2);

    let test_evaluator = grid.geometry_map(test_cell_type, test_points.data());
    let trial_evaluator = grid.geometry_map(trial_cell_type, trial_points.data());

    let mut a = SingularCellPairAssembler::new(
        npts,
        deriv_size,
        &assembler.integrand,
        &assembler.kernel,
        test_evaluator,
        trial_evaluator,
        test_table,
        trial_table,
        weights,
    );

    let mut local_mat = rlst_dynamic_array2!(
        T,
        [
            test_space.element(test_cell_type).dim(),
            trial_space.element(trial_cell_type).dim()
        ]
    );
    for (test_cell, trial_cell) in cell_pairs {
        a.set_test_cell(*test_cell);
        a.set_trial_cell(*trial_cell);
        a.assemble(&mut local_mat);

        let test_dofs = unsafe { test_space.cell_dofs_unchecked(*test_cell) };
        let trial_dofs = unsafe { trial_space.cell_dofs_unchecked(*trial_cell) };

        for (trial_dof, col) in izip!(trial_dofs, local_mat.col_iter()) {
            for (test_dof, entry) in izip!(test_dofs, col.iter()) {
                output.rows.push(test_space.global_dof_index(*test_dof));
                output.cols.push(trial_space.global_dof_index(*trial_dof));
                output.data.push(entry);
            }
        }
    }

    output
}

/// Assemble the contribution to the terms of a matrix for a batch of non-adjacent cells
#[allow(clippy::too_many_arguments)]
fn assemble_batch_nonadjacent<
    T: RlstScalar + MatrixInverse,
    Space: LocalFunctionSpaceTrait<T = T>,
    Integrand: BoundaryIntegrand<T = T>,
    K: Kernel<T = T>,
>(
    assembler: &BoundaryAssembler<T, Integrand, K>,
    deriv_size: usize,
    output: &RawData2D<T>,
    trial_cell_type: ReferenceCellType,
    test_cell_type: ReferenceCellType,
    trial_space: &Space,
    trial_cells: &[usize],
    test_space: &Space,
    test_cells: &[usize],
    trial_points: &RlstArray<T::Real, 2>,
    trial_weights: &[T::Real],
    test_points: &RlstArray<T::Real, 2>,
    test_weights: &[T::Real],
    trial_table: &RlstArray<T, 4>,
    test_table: &RlstArray<T, 4>,
) -> usize {
    let npts_test = test_weights.len();
    let npts_trial = trial_weights.len();
    debug_assert!(test_points.shape()[1] == npts_test);
    debug_assert!(trial_points.shape()[1] == npts_trial);

    let test_grid = test_space.grid();
    let trial_grid = trial_space.grid();

    assert_eq!(test_grid.geometry_dim(), 3);
    assert_eq!(test_grid.topology_dim(), 2);
    assert_eq!(trial_grid.geometry_dim(), 3);
    assert_eq!(trial_grid.topology_dim(), 2);

    let test_evaluator = test_grid.geometry_map(test_cell_type, test_points.data());
    let trial_evaluator = trial_grid.geometry_map(trial_cell_type, trial_points.data());

    let mut a = NonsingularCellPairAssemblerWithTestCaching::new(
        npts_test,
        npts_trial,
        deriv_size,
        test_cells,
        &assembler.integrand,
        &assembler.kernel,
        test_evaluator,
        trial_evaluator,
        test_table,
        trial_table,
        test_weights,
        trial_weights,
    );

    let mut local_mat = rlst_dynamic_array2!(
        T,
        [
            test_space.element(test_cell_type).dim(),
            trial_space.element(trial_cell_type).dim()
        ]
    );

    for trial_cell in trial_cells {
        a.set_trial_cell(*trial_cell);
        let trial_dofs = unsafe { trial_space.cell_dofs_unchecked(*trial_cell) };
        for test_cell in test_cells.iter() {
            if neighbours(test_grid, trial_grid, *test_cell, *trial_cell) {
                continue;
            }

            a.set_test_cell(*test_cell);
            a.assemble(&mut local_mat);

            let test_dofs = unsafe { test_space.cell_dofs_unchecked(*test_cell) };

            for (trial_dof, col) in izip!(trial_dofs, local_mat.col_iter()) {
                for (test_dof, entry) in izip!(test_dofs, col.iter()) {
                    unsafe {
                        *output.data.add(*test_dof + output.shape[0] * *trial_dof) += entry;
                    }
                }
            }
        }
    }
    1
}

fn get_pairs_if_smallest(
    test_cell: &impl Entity,
    trial_cell: &impl Entity,
    vertex: usize,
) -> Option<Vec<(usize, usize)>> {
    let mut pairs = vec![];
    for (trial_i, trial_v) in trial_cell.topology().sub_entity_iter(0).enumerate() {
        for (test_i, test_v) in test_cell.topology().sub_entity_iter(0).enumerate() {
            if test_v == trial_v {
                if test_v < vertex {
                    return None;
                }
                pairs.push((test_i, trial_i));
            }
        }
    }
    Some(pairs)
}

fn neighbours<TestGrid: Grid, TrialGrid: Grid>(
    test_grid: &TestGrid,
    trial_grid: &TrialGrid,
    test_cell: usize,
    trial_cell: usize,
) -> bool {
    if !equal_grids(test_grid, trial_grid) {
        false
    } else {
        let test_vertices = trial_grid
            .entity(2, test_cell)
            .unwrap()
            .topology()
            .sub_entity_iter(0)
            .collect::<Vec<_>>();
        for v in trial_grid
            .entity(2, trial_cell)
            .unwrap()
            .topology()
            .sub_entity_iter(0)
        {
            if test_vertices.contains(&v) {
                return true;
            }
        }
        false
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::{helmholtz, laplace};
//     use cauchy::{c32, c64};
//     use ndelement::ciarlet::CiarletElement;
//     use ndelement::ciarlet::LagrangeElementFamily;
//     use ndelement::types::{Continuity, ReferenceCellType};
//     use ndgrid::{
//         grid::serial::{SingleElementGrid, SingleElementGridBuilder},
//         shapes::regular_sphere,
//         traits::Builder,
//         types::RealScalar,
//     };
//     use paste::paste;
//     use rlst::{MatrixInverse, RlstScalar};

//     fn quadrilateral_grid<T: RealScalar + MatrixInverse>() -> SingleElementGrid<T, CiarletElement<T>>
//     {
//         let mut b = SingleElementGridBuilder::<T>::new(3, (ReferenceCellType::Quadrilateral, 1));
//         for j in 0..4 {
//             for i in 0..4 {
//                 b.add_point(
//                     4 * j + i,
//                     &[
//                         num::cast::<usize, T>(i).unwrap() / num::cast::<f64, T>(3.0).unwrap(),
//                         num::cast::<usize, T>(j).unwrap() / num::cast::<f64, T>(3.0).unwrap(),
//                         num::cast::<f64, T>(0.0).unwrap(),
//                     ],
//                 );
//             }
//         }
//         for j in 0..3 {
//             for i in 0..3 {
//                 b.add_cell(
//                     3 * j + i,
//                     &[4 * j + i, 4 * j + i + 1, 4 * j + i + 4, 4 * j + i + 5],
//                 );
//             }
//         }
//         b.create_grid()
//     }

//     /*
//     fn mixed_grid<T: Float + RlstScalar<Real = T>>() -> MixedGrid<T>
//     where
//         for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>:
//             MatrixInverse,
//     {
//         let mut b = MixedGridBuilder::<3, T>::new(());
//         for j in 0..4 {
//             for i in 0..4 {
//                 b.add_point(
//                     4 * j + i,
//                     [
//                         num::cast::<usize, T>(i).unwrap() / num::cast::<f64, T>(3.0).unwrap(),
//                         num::cast::<usize, T>(j).unwrap() / num::cast::<f64, T>(3.0).unwrap(),
//                         num::cast::<f64, T>(0.0).unwrap(),
//                     ],
//                 );
//             }
//         }
//         for j in 0..3 {
//             b.add_cell(
//                 j,
//                 (
//                     vec![4 * j, 4 * j + 1, 4 * j + 4, 4 * j + 5],
//                     ReferenceCellType::Quadrilateral,
//                     1,
//                 ),
//             );
//         }
//         for j in 0..3 {
//             b.add_cell(
//                 3 + 2 * j,
//                 (
//                     vec![4 * j + 1, 4 * j + 2, 4 * j + 6],
//                     ReferenceCellType::Triangle,
//                     1,
//                 ),
//             );
//             b.add_cell(
//                 4 + 2 * j,
//                 (
//                     vec![4 * j + 1, 4 * j + 6, 4 * j + 5],
//                     ReferenceCellType::Triangle,
//                     1,
//                 ),
//             );
//         }
//         for j in 0..3 {
//             b.add_cell(
//                 9 + j,
//                 (
//                     vec![4 * j + 2, 4 * j + 3, 4 * j + 6, 4 * j + 7],
//                     ReferenceCellType::Quadrilateral,
//                     1,
//                 ),
//             );
//         }
//         b.create_grid()
//     }
//     */
//     macro_rules! example_grid {
//         (Triangle, $dtype:ident) => {
//             regular_sphere(0)
//         };
//         (Quadrilateral, $dtype:ident) => {
//             quadrilateral_grid::<<$dtype as RlstScalar>::Real>()
//         }; //(Mixed, $dtype:ident) => {
//            //    mixed_grid::<<$dtype as RlstScalar>::Real>()
//            //};
//     }
//     macro_rules! test_assembly {

//         ($dtype:ident, Helmholtz, $operator:ident, $cell:ident) => {
//             paste! {

//                 #[test]
//                 fn [<test_assembly_helmholtz_ $operator:lower _ $cell:lower _ $dtype>]() {

//                     let grid = example_grid!($cell, $dtype);
//                     let element = LagrangeElementFamily::<[<$dtype>]>::new(0, Continuity::Discontinuous);
//                     let space = DefaultFunctionSpace::new(&grid, &element);

//                     let options = BoundaryAssemblerOptions::default();
//                     let a = helmholtz::assembler::[<$operator>]::<[<$dtype>]>(3.0, &options);
//                     let _matrix = a.assemble(&space, &space);
//                 }

//             }
//         };
//         ($dtype:ident, $pde:ident, $operator:ident, $cell:ident) => {
//             paste! {

//                 #[test]
//                 fn [<test_assembly_ $pde:lower _ $operator:lower _ $cell:lower _ $dtype>]() {

//                     let grid = example_grid!($cell, $dtype);
//                     let element = LagrangeElementFamily::<[<$dtype>]>::new(0, Continuity::Discontinuous);
//                     let space = LocalFunctionSpace::new(&grid, &element);

//                     let options = BoundaryAssemblerOptions::default();
//                     let a = laplace::assembler::[<$operator>]::<[<$dtype>]>(&options);
//                     let _matrix = a.assemble(&space, &space);
//                 }

//             }
//         };
//     }

//     test_assembly!(f64, Laplace, single_layer, Triangle);
//     test_assembly!(f32, Laplace, single_layer, Triangle);
//     //test_assembly!(c64, Laplace, single_layer, Triangle);
//     //test_assembly!(c32, Laplace, single_layer, Triangle);
//     test_assembly!(f64, Laplace, double_layer, Triangle);
//     test_assembly!(f32, Laplace, double_layer, Triangle);
//     //test_assembly!(c64, Laplace, double_layer, Triangle);
//     //test_assembly!(c32, Laplace, double_layer, Triangle);
//     test_assembly!(f64, Laplace, adjoint_double_layer, Triangle);
//     test_assembly!(f32, Laplace, adjoint_double_layer, Triangle);
//     //test_assembly!(c64, Laplace, adjoint_double_layer, Triangle);
//     //test_assembly!(c32, Laplace, adjoint_double_layer, Triangle);
//     test_assembly!(f64, Laplace, hypersingular, Triangle);
//     test_assembly!(f32, Laplace, hypersingular, Triangle);
//     //test_assembly!(c64, Laplace, hypersingular, Triangle);
//     //test_assembly!(c32, Laplace, hypersingular, Triangle);
//     test_assembly!(c64, Helmholtz, single_layer, Triangle);
//     test_assembly!(c32, Helmholtz, single_layer, Triangle);
//     test_assembly!(c64, Helmholtz, double_layer, Triangle);
//     test_assembly!(c32, Helmholtz, double_layer, Triangle);
//     test_assembly!(c64, Helmholtz, adjoint_double_layer, Triangle);
//     test_assembly!(c32, Helmholtz, adjoint_double_layer, Triangle);
//     test_assembly!(c64, Helmholtz, hypersingular, Triangle);
//     test_assembly!(c32, Helmholtz, hypersingular, Triangle);
//     test_assembly!(f64, Laplace, single_layer, Quadrilateral);
//     test_assembly!(f32, Laplace, single_layer, Quadrilateral);
//     //test_assembly!(c64, Laplace, single_layer, Quadrilateral);
//     //test_assembly!(c32, Laplace, single_layer, Quadrilateral);
//     test_assembly!(f64, Laplace, double_layer, Quadrilateral);
//     test_assembly!(f32, Laplace, double_layer, Quadrilateral);
//     //test_assembly!(c64, Laplace, double_layer, Quadrilateral);
//     //test_assembly!(c32, Laplace, double_layer, Quadrilateral);
//     test_assembly!(f64, Laplace, adjoint_double_layer, Quadrilateral);
//     test_assembly!(f32, Laplace, adjoint_double_layer, Quadrilateral);
//     //test_assembly!(c64, Laplace, adjoint_double_layer, Quadrilateral);
//     //test_assembly!(c32, Laplace, adjoint_double_layer, Quadrilateral);
//     test_assembly!(f64, Laplace, hypersingular, Quadrilateral);
//     test_assembly!(f32, Laplace, hypersingular, Quadrilateral);
//     //test_assembly!(c64, Laplace, hypersingular, Quadrilateral);
//     //test_assembly!(c32, Laplace, hypersingular, Quadrilateral);
//     test_assembly!(c64, Helmholtz, single_layer, Quadrilateral);
//     test_assembly!(c32, Helmholtz, single_layer, Quadrilateral);
//     test_assembly!(c64, Helmholtz, double_layer, Quadrilateral);
//     test_assembly!(c32, Helmholtz, double_layer, Quadrilateral);
//     test_assembly!(c64, Helmholtz, adjoint_double_layer, Quadrilateral);
//     test_assembly!(c32, Helmholtz, adjoint_double_layer, Quadrilateral);
//     test_assembly!(c64, Helmholtz, hypersingular, Quadrilateral);
//     test_assembly!(c32, Helmholtz, hypersingular, Quadrilateral);
//     //(f64, Laplace, single_layer, Mixed);
//     //(f32, Laplace, single_layer, Mixed);
//     //(c64, Laplace, single_layer, Mixed);
//     //(c32, Laplace, single_layer, Mixed);
//     //(f64, Laplace, double_layer, Mixed);
//     //(f32, Laplace, double_layer, Mixed);
//     //(c64, Laplace, double_layer, Mixed);
//     //(c32, Laplace, double_layer, Mixed);
//     //(f64, Laplace, adjoint_double_layer, Mixed);
//     //(f32, Laplace, adjoint_double_layer, Mixed);
//     //(c64, Laplace, adjoint_double_layer, Mixed);
//     //(c32, Laplace, adjoint_double_layer, Mixed);
//     //(f64, Laplace, hypersingular, Mixed);
//     //(f32, Laplace, hypersingular, Mixed);
//     //(c64, Laplace, hypersingular, Mixed);
//     //(c32, Laplace, hypersingular, Mixed);
//     //(c64, Helmholtz, single_layer, Mixed);
//     //(c32, Helmholtz, single_layer, Mixed);
//     //(c64, Helmholtz, double_layer, Mixed);
//     //(c32, Helmholtz, double_layer, Mixed);
//     //(c64, Helmholtz, adjoint_double_layer, Mixed);
//     //(c32, Helmholtz, adjoint_double_layer, Mixed);
//     //(c64, Helmholtz, hypersingular, Mixed);
//     //(c32, Helmholtz, hypersingular, Mixed);
// }
