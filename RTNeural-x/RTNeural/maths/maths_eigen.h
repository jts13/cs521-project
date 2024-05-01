#pragma once

#include "maths_approx.h"
#include <cmath>

namespace RTNEURAL_NAMESPACE
{
struct DefaultMathsProvider
{
    template <typename Matrix>
    static auto exp(const Matrix& x)
    {
#ifdef RTNEURAL_USE_APPROX
        using T = typename Matrix::Scalar;
        return x.unaryExpr(&approx::xexp<T>);
#else
        return x.array().exp();
#endif
    }

    template <typename Matrix>
    static auto tanh(const Matrix& m)
    {
#ifdef RTNEURAL_USE_APPROX
        using T = typename Matrix::Scalar;
        const auto x = m.array();
        const auto y = exp((T)2 * x);
        return (y - (T)1) / (y + (T)1);
#else
        return m.array().tanh();
#endif
    }

    template <typename Matrix>
    static auto sigmoid(const Matrix& x)
    {
        using T = typename Matrix::Scalar;
        constexpr T one = (T)1;
        return one / exp(((-one * x.array())) + one);
    }
};
}
