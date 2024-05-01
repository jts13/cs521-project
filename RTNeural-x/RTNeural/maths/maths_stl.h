#pragma once

#include "maths_approx.h"
#include <bit>
#include <cmath>

namespace RTNEURAL_NAMESPACE
{
struct DefaultMathsProvider
{
private:
public:
#ifdef RTNEURAL_USE_APPROX
    template <typename T>
    static T exp(T x)
    {
        return approx::xexp(x);
    }

    template <typename T>
    static T tanh(T x)
    {
        const auto y = exp((T)2 * x);
        return (y - (T)1) / (y + (T)1);
    }

    template <typename T>
    static T sigmoid(T x)
    {
        return (T)1 / ((T)1 + exp(-x));
    }
#else
    template <typename T>
    static T exp(T x)
    {
        return std::exp(x);
    }

    template <typename T>
    static T tanh(T x)
    {
        return std::tanh(x);
    }

    template <typename T>
    static T sigmoid(T x)
    {
        return (T)1 / ((T)1 + exp(-x));
    }
#endif
};
}
