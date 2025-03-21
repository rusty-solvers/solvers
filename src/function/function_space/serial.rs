//! Serial function space

use crate::function::function_space::assign_dofs;
use crate::function::FunctionSpace;
use ndelement::ciarlet::CiarletElement;
use ndelement::traits::{ElementFamily, FiniteElement};
use ndelement::types::ReferenceCellType;
use ndgrid::{
    traits::{Entity, Grid, Topology},
    types::Ownership,
};
use rlst::{MatrixInverse, RlstScalar};
use std::collections::HashMap;

/// A serial function space
pub struct LocalFunctionSpace<
    'a,
    T: RlstScalar + MatrixInverse,
    GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
> {
    pub(crate) grid: &'a GridImpl,
    pub(crate) elements: HashMap<ReferenceCellType, CiarletElement<T>>,
    pub(crate) entity_dofs: [Vec<Vec<usize>>; 4],
    pub(crate) cell_dofs: Vec<Vec<usize>>,
    pub(crate) size: usize,
}

impl<
        'a,
        T: RlstScalar + MatrixInverse,
        GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
    > LocalFunctionSpace<'a, T, GridImpl>
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
        let (cell_dofs, entity_dofs, size, _) = assign_dofs(0, grid, e_family);

        let mut elements = HashMap::new();
        for cell in grid.entity_types(2) {
            elements.insert(*cell, e_family.element(*cell));
        }

        Self {
            grid,
            elements,
            entity_dofs,
            cell_dofs,
            size,
        }
    }
}

impl<
        T: RlstScalar + MatrixInverse,
        GridImpl: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
    > FunctionSpace for LocalFunctionSpace<'_, T, GridImpl>
{
    type T = T;
    type Grid = GridImpl;
    type FiniteElement = CiarletElement<T>;
    type LocalSpace<'b>
        = LocalFunctionSpace<'b, T, GridImpl>
    where
        Self: 'b;

    fn local_space(&self) -> &Self::LocalSpace<'_> {
        self
    }
    fn grid(&self) -> &Self::Grid {
        self.grid
    }
    fn element(&self, cell_type: ReferenceCellType) -> &CiarletElement<T> {
        &self.elements[&cell_type]
    }
    fn get_local_dof_numbers(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
        &self.entity_dofs[entity_dim][entity_number]
    }
    fn is_serial(&self) -> bool {
        true
    }
    fn local_size(&self) -> usize {
        self.size
    }
    fn global_size(&self) -> usize {
        self.size
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
        local_dof_index
    }
    fn ownership(&self, _local_dof_index: usize) -> Ownership {
        Ownership::Owned
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndelement::ciarlet::LagrangeElementFamily;
    use ndelement::types::Continuity;
    use ndgrid::shapes::regular_sphere;
}
