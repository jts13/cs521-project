//! 'On a Fast, Compact Approximation of the Exponential Function'
//! * https://www.schraudolph.org/pubs/Schraudolph99.pdf

/*
#define EXP_A (1048576 / M_LN2)
#define EXP_C 60801
inline double exponential(double y) {
    union {
        double d;
    #ifdef LITTLE_ENDIAN
        struct { int j, i; } n;
    #elseif
        struct { int i, j; } n;
    #endif
    } _eco;
    _eco.n.i = (int)(EXP_A * (y)) + (1072693248 - EXP_C);
    _eco.n.j = 0;
    return _eco.d;
}
*/

pub fn exp(y: f64) -> f64 {
    use core::f64::consts::LN_2;

    const BIAS: i32 = f64::MAX_EXP - 1;
    const MANTISSA_BITS: i32 = f64::MANTISSA_DIGITS as i32 - 1;
    const OFFSET_BITS: i32 = i32::BITS as i32;

    const X: i32 = 1 << (MANTISSA_BITS - OFFSET_BITS);
    const A: f64 = X as f64 / LN_2;
    const B: i32 = X * BIAS;
    const C: i32 = 60_801; // tuning parameter
    const D: i32 = B - C;

    let y = (A * y) as i32 + D;

    unsafe {
        core::mem::transmute(
            #[cfg(target_endian = "little")]
            [0, y],
            #[cfg(target_endian = "big")]
            [y, 0],
        )
    }
}

pub fn expf(y: f32) -> f32 {
    use core::f32::consts::LN_2;

    const BIAS: i16 = f32::MAX_EXP as i16 - 1;
    const MANTISSA_BITS: i16 = f32::MANTISSA_DIGITS as i16 - 1;
    const OFFSET_BITS: i16 = i16::BITS as i16;

    const X: i16 = 1 << (MANTISSA_BITS - OFFSET_BITS);
    const A: f32 = X as f32 / LN_2;
    const B: i16 = X * BIAS;
    const C: i16 = 8; // tuning parameter
    const D: i16 = B - C;

    unsafe {
        let y = (A * y).to_int_unchecked::<i16>() + D;

        core::mem::transmute(
            #[cfg(target_endian = "little")]
            [0, y],
            #[cfg(target_endian = "big")]
            [y, 0],
        )
    }
}

#[test]
fn test_exp() {
    use core::f64::consts::E;

    let eps = 0.1;
    assert_float_eq!(exp(0.0), 1., eps);
    assert_float_eq!(exp(1.0), E, eps);
    assert_float_eq!(exp(-1.0), 1. / E, eps);
}

#[test]
fn test_expf() {
    use core::f32::consts::E;

    let eps = 0.05;
    assert_float_eq!(expf(0.0), 1., eps);
    assert_float_eq!(expf(1.0), E, eps);
    assert_float_eq!(expf(-1.0), 1. / E, eps);
}

// https://git.musl-libc.org/cgit/musl/tree/src/math/tanh.c
/* tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 *         = (exp(2*x) - 1) / (exp(2*x) - 1 + 2)
 *         = (1 - exp(-2*x)) / (exp(-2*x) - 1 + 2)
 */
/*
 * tanh(x) = sinh(x) / cosh(x)
 *         = (exp(2 * x) - 1) / (exp(2 * x) + 1)
 */
pub fn tanhf(x: f32) -> f32 {
    let y = expf(2. * x);
    (y - 1.) / (y + 1.)
}
