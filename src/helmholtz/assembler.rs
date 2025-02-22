//! Implementation of Helmholtz assemblers

use green_kernels::{helmholtz_3d::Helmholtz3dKernel, types::GreenKernelEvalType};
use rlst::{MatrixInverse, RlstScalar};

use crate::boundary_assemblers::{
    helpers::KernelEvaluator,
    integrands::{
        AdjointDoubleLayerBoundaryIntegrand, BoundaryIntegrandSum, BoundaryIntegrandTimesScalar,
        DoubleLayerBoundaryIntegrand, HypersingularCurlCurlBoundaryIntegrand,
        HypersingularNormalNormalBoundaryIntegrand, SingleLayerBoundaryIntegrand,
    },
    BoundaryAssembler, BoundaryAssemblerOptions,
};

/// Helmholtz single layer assembler type.
pub type SingleLayer3dAssembler<'o, T> =
    BoundaryAssembler<'o, T, SingleLayerBoundaryIntegrand<T>, Helmholtz3dKernel<T>>;

/// Helmholtz double layer assembler type.
pub type DoubleLayer3dAssembler<'o, T> =
    BoundaryAssembler<'o, T, DoubleLayerBoundaryIntegrand<T>, Helmholtz3dKernel<T>>;

/// Helmholtz adjoint double layer assembler type.
pub type AdjointDoubleLayer3dAssembler<'o, T> =
    BoundaryAssembler<'o, T, AdjointDoubleLayerBoundaryIntegrand<T>, Helmholtz3dKernel<T>>;

/// Helmholtz hypersingular double layer assembler type.
pub type Hypersingular3dAssembler<'o, T> = BoundaryAssembler<
    'o,
    T,
    BoundaryIntegrandSum<
        T,
        HypersingularCurlCurlBoundaryIntegrand<T>,
        BoundaryIntegrandTimesScalar<T, HypersingularNormalNormalBoundaryIntegrand<T>>,
    >,
    Helmholtz3dKernel<T>,
>;

/// Assembler for the Helmholtz single layer operator.
pub fn single_layer<T: RlstScalar<Complex = T> + MatrixInverse>(
    wavenumber: T::Real,
    options: &BoundaryAssemblerOptions,
) -> SingleLayer3dAssembler<T> {
    let kernel = KernelEvaluator::new(
        Helmholtz3dKernel::new(wavenumber),
        GreenKernelEvalType::Value,
    );

    BoundaryAssembler::new(SingleLayerBoundaryIntegrand::new(), kernel, options, 1, 0)
}

/// Assembler for the Helmholtz double layer operator.
pub fn double_layer<T: RlstScalar<Complex = T> + MatrixInverse>(
    wavenumber: T::Real,
    options: &BoundaryAssemblerOptions,
) -> DoubleLayer3dAssembler<T> {
    let kernel = KernelEvaluator::new(
        Helmholtz3dKernel::new(wavenumber),
        GreenKernelEvalType::ValueDeriv,
    );

    BoundaryAssembler::new(DoubleLayerBoundaryIntegrand::new(), kernel, options, 4, 0)
}

/// Assembler for the Helmholtz adjoint double layer operator.
pub fn adjoint_double_layer<T: RlstScalar<Complex = T> + MatrixInverse>(
    wavenumber: T::Real,
    options: &BoundaryAssemblerOptions,
) -> AdjointDoubleLayer3dAssembler<T> {
    let kernel = KernelEvaluator::new(
        Helmholtz3dKernel::new(wavenumber),
        GreenKernelEvalType::ValueDeriv,
    );

    BoundaryAssembler::new(
        AdjointDoubleLayerBoundaryIntegrand::new(),
        kernel,
        options,
        4,
        0,
    )
}

/// Assembler for the Helmholtz hypersingular operator.
pub fn hypersingular<T: RlstScalar<Complex = T> + MatrixInverse>(
    wavenumber: T::Real,
    options: &BoundaryAssemblerOptions,
) -> Hypersingular3dAssembler<T> {
    let kernel = KernelEvaluator::new(
        Helmholtz3dKernel::new(wavenumber),
        GreenKernelEvalType::ValueDeriv,
    );

    let integrand = BoundaryIntegrandSum::new(
        HypersingularCurlCurlBoundaryIntegrand::new(),
        BoundaryIntegrandTimesScalar::new(
            num::cast::<T::Real, T>(-wavenumber.powi(2)).unwrap(),
            HypersingularNormalNormalBoundaryIntegrand::new(),
        ),
    );

    BoundaryAssembler::new(integrand, kernel, options, 4, 1)
}
