//! 'On a Fast, Compact Approximation of the Exponential Function'
//! * https://www.schraudolph.org/pubs/Schraudolph99.pdf

// https://github.com/ekmett/approximate/blob/7f1aba0b8632008d5bdab1a844754f16026b731f/cbits/fast.c#L112
// https://martin.ankerl.com/2007/02/11/optimized-exponential-functions-for-java/
// https://martin.ankerl.com/2007/10/04/optimized-pow-approximation-for-java-and-c-c/
// https://github.com/gingerBill/gb/blob/52a3a542ef6d398d541d5083aa878598189425ef/gb_math.h#L935

// https://godbolt.org/
// Rust: -C opt-level=3 --target aarch64-apple-darwin
//  C++: -std=c++20 -O3 # -target aarch64-apple-darwin

use std::f32::consts::*;

// TODO(toms): explain -> exp(x / 2) / exp(-x / 2)
#[cfg(target_endian = "little")]
pub fn expf(x: f32) -> f32 {
    const BIAS: u32 = f32::MAX_EXP as u32 - 1;
    const MANTISSA_BITS: u32 = f32::MANTISSA_DIGITS - 1;

    const A: f32 = (1 << MANTISSA_BITS) as f32 / LN_2;
    const B: f32 = (BIAS << MANTISSA_BITS) as f32;

    debug_assert!(A * x < B); // x ~< 176 (prevent issue in denominator)

    #[cfg(not(target_feature = "neon"))]
    {
        f32::from_bits((A / 2. * x + B) as u32) / f32::from_bits((-A / 2. * x + B) as u32)
    }

    #[cfg(target_feature = "neon")]
    unsafe {
        use std::arch::aarch64::*;
        use std::mem::transmute;

        let a = vcreate_f32(transmute([A / 2., -A / 2.]));
        let x = vdup_n_f32(x);
        let b = vdup_n_f32(B);

        let y = vfma_f32(b, a, x);
        let y = vcvt_u32_f32(y);
        let y = vreinterpret_f32_u32(y);

        vget_lane_f32::<0>(y) / vget_lane_f32::<1>(y)
    }
}

#[test]
fn test_expf() {
    let eps = 0.05;
    assert_float_eq!(expf(0.), 1., eps);
    assert_float_eq!(expf(1.), E, eps);
    assert_float_eq!(expf(-1.), 1. / E, eps);
}

pub fn tanhf(x: f32) -> f32 {
    let y = expf(2. * x);
    (y - 1.) / (y + 1.)
}
