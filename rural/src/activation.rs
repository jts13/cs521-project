// * 'Approximating Activation Functions' (Timmons, et al.)
// * 'Optimizing Deep Learning RNN Topologies on Intel Architecture' (Banerjee, et al.)

// TODO(toms): implement `sigmoid`
// TODO(toms): implement `softmax`?

// https://github.com/rutgers-apl/rlibm-32/blob/main/source/float/exp.c
// TODO(toms): ...

// TODO(toms): move this? use crate?
#[cfg(test)]
macro_rules! assert_float_eq {
    ($left:expr, $right:expr $(,)?) => {
        assert!(($left - $right).abs() < 1e-6, "{} != {}", $left, $right)
    };

    ($left:expr, $right:expr, $epsilon:expr $(,)?) => {
        assert!(($left - $right).abs() < $epsilon, "{} != {}", $left, $right)
    };
}

pub mod ktanh;
pub mod pade;
pub mod schraudolph;
pub mod schraudolph_ng;
pub mod spline;
pub mod taylor;

#[cfg(test)]
mod control {
    use core::f32::consts::E;
    use libm::{expf, tanhf};

    #[test]
    fn test_exp() {
        assert_float_eq!(expf(0.0), 1.);
        assert_float_eq!(expf(1.0), E);
        assert_float_eq!(expf(-1.0), 1. / E);
    }

    #[test]
    fn test_tanh() {
        assert_float_eq!(tanhf(0.0), 0.);
        assert_float_eq!(tanhf(0.2), 0.19737533);
        assert_float_eq!(tanhf(0.5), 0.46211714);
        assert_float_eq!(tanhf(1.0), 0.7615942);
        assert_float_eq!(tanhf(1.5), 0.90514827);
        assert_float_eq!(tanhf(2.0), 0.9640276);
        assert_float_eq!(tanhf(2.5), 0.9866143);
        assert_float_eq!(tanhf(3.0), 0.9950548);
        assert_float_eq!(tanhf(8.5), 1.);
    }
}
