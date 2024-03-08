use bempp_field::types::FftFieldTranslationKiFmm;
use bempp_fmm::builder::KiFmmBuilderSingleNode;
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::fmm::Fmm;
use bempp_tree::implementations::helpers::points_fixture;
use rlst_dense::{rlst_dynamic_array2, traits::RawAccessMut};

fn main() {
    // Setup random sources and targets
    let nsources = 50000;
    let ntargets = 10000;
    let sources = points_fixture::<f64>(nsources, None, None, Some(0));
    let targets = points_fixture::<f64>(ntargets, None, None, Some(3));

    // FMM parameters
    let n_crit = Some(10);
    let expansion_order = 7;
    let sparse = true;

    // FFT based M2L for a vector of charges
    {
        // Charge data
        let nvecs = 1;
        let tmp = vec![1.0; nsources * nvecs];
        let mut charges = rlst_dynamic_array2!(f64, [nsources, nvecs]);
        charges.data_mut().copy_from_slice(&tmp);

        let fmm_fft = KiFmmBuilderSingleNode::new()
            .tree(&sources, &targets, &charges, n_crit, sparse)
            .parameters(
                expansion_order,
                Laplace3dKernel::new(),
                bempp_traits::types::EvalType::Value,
                FftFieldTranslationKiFmm::new(),
            )
            .unwrap()
            .build()
            .unwrap();
        fmm_fft.evaluate();
    }

    // BLAS based M2L for a vector of charges
    {
        let svd_threshold = Some(1e-5);
    }
}
