// TODO(toms): refactor!

// (1 / 1, 1 / 3, 2 / 15, 17 / 315, 62 / 2835, 1382 / 155925)
pub fn tanhf(x: f32) -> f32 {
    if x.abs() > 1.365 {
        return 1f32.copysign(x);
    }

    let x3 = x * x * x;
    let x5 = x * x * x3;
    let x7 = x * x * x5;
    let x9 = x * x * x7;
    let x11 = x * x * x9;

    let t1 = x;
    let t2 = x3 * (1. / 3.);
    let t3 = x5 * (2. / 15.);
    let t4 = x7 * (17. / 315.);
    let t5 = x9 * (62. / 2835.);
    let t6 = x11 * (1382. / 155925.);

    t1 - t2 + t3 - t4 + t5 - t6
}
