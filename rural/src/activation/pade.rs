// TODO(toms): Pade approximant continued fraction
// * https://github.com/juce-framework/JUCE/blob/a8ae6edda6d3be78a139ec5e429dc57ef047e82a/modules/juce_dsp/maths/juce_FastMathApproximations.h#L99
// /** Provides a fast approximation of the function tanh(x) using a Pade approximant
//      continued fraction, calculated sample by sample.
//
//      Note: This is an approximation which works on a limited range. You are
//      advised to use input values only between -5 and +5 for limiting the error.
//  */
//  template <typename FloatType>
//  static FloatType tanh (FloatType x) noexcept
//  {
//      auto x2 = x * x;
//      auto numerator = x * (135135 + x2 * (17325 + x2 * (378 + x2)));
//      auto denominator = 135135 + x2 * (62370 + x2 * (3150 + 28 * x2));
//      return numerator / denominator;
//  }

// TODO(toms): add conditional to snap to [-1, 1] at the edges?

pub fn tanhf(x: f32) -> f32 {
    if x.abs() > 5. {
        return 1f32.copysign(x);
    }

    let x2 = x * x;
    let numerator = x * (135135. + x2 * (17325. + x2 * (378. + x2)));
    let denominator = 135135. + x2 * (62370. + x2 * (3150. + 28. * x2));
    numerator / denominator
}
