//! Efficiently inaccurate approximation of hyperbolic tangent used as transfer function in artificial neural networks
//! * Simos, Tsitouras

pub fn tanhf3(xin: f32) -> f32 {
    // const N0: f32 = 0.;
    const N1: f32 = 0.371025186672900;
    const N2: f32 = 2.572153900248530;
    const N3: f32 = 18.;

    // s0 = -3.695076086125492e-1 * x.powi(3) + 1.987219343897867e-2 * x.powi(2) + x;
    // s1 = 5.928356367224758e-2 * (x - N1).powi(3) - 3.914176949486042e-1 * (x - N1).powi(2) + 8.621472609449146e-1 * (x - N1) + 3.548881072496229e-1;
    // s2 = -3.347599023061577e-6 * (x - N2).powi(3) + 5.456777761558641e-5 * (x - N2).powi(2) + 7.066442941005233e-4 * (x - N2) + 9.884026213740197e-1;
    // s3 = 1.

    // TODO(toms): use Horner method
    match xin.abs() {
        x if x <= N1 => -3.695076086125492e-1 * x.powi(3) + 1.987219343897867e-2 * x.powi(2) + x,
        x if x <= N2 => {
            5.928356367224758e-2 * (x - N1).powi(3) - 3.914176949486042e-1 * (x - N1).powi(2)
                + 8.621472609449146e-1 * (x - N1)
                + 3.548881072496229e-1
        }
        x if x <= N3 => {
            -3.347599023061577e-6 * (x - N2).powi(3)
                + 5.456777761558641e-5 * (x - N2).powi(2)
                + 7.066442941005233e-4 * (x - N2)
                + 9.884026213740197e-1
        }
        _ => 1.,
    }
    .copysign(xin)
}
