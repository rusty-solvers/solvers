//! Dense Assemblers for Laplace integral operators

use green_kernels::{laplace_3d::Laplace3dKernel, types::GreenKernelEvalType};
use rlst::{MatrixInverse, RlstScalar};

use crate::boundary_assemblers::{
    helpers::KernelEvaluator,
    integrands::{
        AdjointDoubleLayerBoundaryIntegrand, DoubleLayerBoundaryIntegrand,
        HypersingularCurlCurlBoundaryIntegrand, SingleLayerBoundaryIntegrand,
    },
    BoundaryAssembler, BoundaryAssemblerOptions,
};

/// Laplace single layer assembler type.
pub type SingleLayer3dAssembler<'o, T> =
    BoundaryAssembler<'o, T, SingleLayerBoundaryIntegrand<T>, Laplace3dKernel<T>>;

/// Laplace double layer assembler type.
pub type DoubleLayer3dAssembler<'o, T> =
    BoundaryAssembler<'o, T, DoubleLayerBoundaryIntegrand<T>, Laplace3dKernel<T>>;

/// Laplace adjoint double layer assembler type.
pub type AdjointDoubleLayer3dAssembler<'o, T> =
    BoundaryAssembler<'o, T, AdjointDoubleLayerBoundaryIntegrand<T>, Laplace3dKernel<T>>;

/// Laplace hypersingular double layer assembler type.
pub type Hypersingular3dAssembler<'o, T> =
    BoundaryAssembler<'o, T, HypersingularCurlCurlBoundaryIntegrand<T>, Laplace3dKernel<T>>;

/// Assembler for the Laplace single layer operator.
pub fn single_layer<T: RlstScalar<Real = T> + MatrixInverse>(
    options: &BoundaryAssemblerOptions,
) -> SingleLayer3dAssembler<T> {
    let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::Value);
    BoundaryAssembler::new(SingleLayerBoundaryIntegrand::new(), kernel, options, 1, 0)
}

/// Assembler for the Laplace double layer operator.
pub fn double_layer<T: RlstScalar<Real = T> + MatrixInverse>(
    options: &BoundaryAssemblerOptions,
) -> DoubleLayer3dAssembler<T> {
    let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::ValueDeriv);

    BoundaryAssembler::new(DoubleLayerBoundaryIntegrand::new(), kernel, options, 4, 0)
}

/// Assembler for the Laplace adjoint double layer operator.
pub fn adjoint_double_layer<T: RlstScalar<Real = T> + MatrixInverse>(
    options: &BoundaryAssemblerOptions,
) -> AdjointDoubleLayer3dAssembler<T> {
    let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::ValueDeriv);

    BoundaryAssembler::new(
        AdjointDoubleLayerBoundaryIntegrand::new(),
        kernel,
        options,
        4,
        0,
    )
}

/// Assembler for the Laplace hypersingular operator.
pub fn hypersingular<T: RlstScalar<Real = T> + MatrixInverse>(
    options: &BoundaryAssemblerOptions,
) -> Hypersingular3dAssembler<T> {
    let kernel = KernelEvaluator::new(Laplace3dKernel::new(), GreenKernelEvalType::ValueDeriv);

    BoundaryAssembler::new(
        HypersingularCurlCurlBoundaryIntegrand::new(),
        kernel,
        options,
        4,
        1,
    )
}
