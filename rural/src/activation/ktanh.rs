//! K-TanH: Efficient TanH For Deep Learning
//! * https://arxiv.org/abs/1909.07729

// Parameter Tables TE, Tr, Tb
//
// Index t  - Et   rt  bt
// 00000    - 126  2   119
// 00001    - 126  4   122
// 00010    - 126  4   123
// 00011    - 126  4   123
// 00100    - 126  6   126
// 00101    - 126  6   126
// 00110    - 126  6   126
// 00111    - 126  6   126
// 01000    - 125  1   1
// 01001    - 125  0   -4
// 01010    - 125  0   -6
// 01011    - 125  0   -7
// 01100    - 125  0   -10
// 01101    - 125  0   -12
// 01110    - 125  0   -15
// 01111    - 125  0   -18
// 10000    - 125  0   112
// 10001    - 126  1   -4
// 10010    - 126  1   -1
// 10011    - 126  1   2
// 10100    - 126  1   3
// 10101    - 126  1   4
// 10110    - 126  1   4
// 10111    - 126  1   4
// 11000    - 126  0   65
// 11001    - 126  1   72
// 11010    - 126  1   73
// 11011    - 126  1   73
// 11100    - 126  2   88
// 11101    - 126  2   89
// 11110    - 126  2   89
// 11111    - 126  4   110
//
// Note: according to the paper, this should all fit into a single 512-bit register.
const LUT: [(u8, u8, i8); 32] = [
    (126, 2, 119),
    (126, 4, 122),
    (126, 4, 123),
    (126, 4, 123),
    (126, 6, 126),
    (126, 6, 126),
    (126, 6, 126),
    (126, 6, 126),
    (125, 1, 1),
    (125, 0, -4),
    (125, 0, -6),
    (125, 0, -7),
    (125, 0, -10),
    (125, 0, -12),
    (125, 0, -15),
    (125, 0, -18),
    (125, 0, 112),
    (126, 1, -4),
    (126, 1, -1),
    (126, 1, 2),
    (126, 1, 3),
    (126, 1, 4),
    (126, 1, 4),
    (126, 1, 4),
    (126, 0, 65),
    (126, 1, 72),
    (126, 1, 73),
    (126, 1, 73),
    (126, 2, 88),
    (126, 2, 89),
    (126, 2, 89),
    (126, 4, 110),
];

// K-TanH
//
// Pseudocode:
// 1. Input: Input xi = (si,Ei,Mi), Parameter Tables TE, Tr, Tb.
// 2. Output: Output yo = (so, Eo, Mo)
// 3. If |xi| < T1:
//      yo <- xi, i.e. (so,Eo,Mo) = (si, Ei, Mi).
// 4. Else If |xi| > T2,
//      yo <- si · 1, i.e. (so,Eo,Mo) = (si, Ebias, 0).
// 5. Else,
// 6.   Form bit string t using lower bits of Ei and higher bits of Mi.
// 7.   Fetch parameters θt = (Et,rt,bt) from TE, Tr, Tb using index t.
// 8.   so <- si, Eo <- Et, Mo <- (Mi >> rt) + bt
// 9. Return yo
//
//  float32: S EEEEEEEE MMMMMMM MMMMMMMM MMMMMMMM
// bfloat16: S EEEEEEEE MMMMMMM
// interval:         tt ttt
//
pub fn tanhf(x: f32) -> f32 {
    let xa = x.abs();

    if xa < 0.25 {
        x
    } else if xa > 3.75 {
        1f32.copysign(x)
    } else {
        let x: u32 = x.to_bits();

        let t = (x >> 20) & 0b11_111;
        let mi = (x >> 16) & 0b0111_1111;
        let so = x & 0x8000_0000;

        let (et, rt, bt) = LUT[t as usize];

        let eo = (et as u32) << 23;
        let mo = (((mi >> (rt as u32)) as i32 + bt as i32) as u32) << 16;

        f32::from_bits(so | eo | mo)
    }
}

#[test]
fn test_tanh() {
    assert_eq!(96, core::mem::size_of_val(&LUT));

    assert_float_eq!(tanhf(0.0), 0.);
    assert_float_eq!(tanhf(0.2), 0.19737533, 1e-2);
    assert_float_eq!(tanhf(0.5), 0.46211714, 1e-2);
    assert_float_eq!(tanhf(1.0), 0.7615942, 1e-2);
    assert_float_eq!(tanhf(1.5), 0.90514827, 1e-2);
    assert_float_eq!(tanhf(2.0), 0.9640276, 1e-2);
    assert_float_eq!(tanhf(2.5), 0.9866143, 1e-2);
    assert_float_eq!(tanhf(3.0), 0.9950548, 1e-2);
    assert_float_eq!(tanhf(8.5), 1.);
}
