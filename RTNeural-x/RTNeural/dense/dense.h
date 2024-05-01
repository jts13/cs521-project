#ifndef DENSE_H_INCLUDED
#define DENSE_H_INCLUDED

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#if RTNEURAL_USE_EIGEN
#include "dense_eigen.h"
#elif RTNEURAL_USE_XSIMD
#include "dense_xsimd.h"
#else
#include "../Layer.h"
#include "../config.h"

namespace RTNEURAL_NAMESPACE
{

/**
 * Dynamic implementation of a fully-connected (dense) layer,
 * with no activation.
 */
template <typename T>
class Dense final : public Layer<T>
{
public:
    /** Constructs a dense layer for a given input and output size. */
    Dense(int in_size, int out_size)
        : Layer<T>(in_size, out_size)
        , weights(std::make_unique<T[]>(in_size * out_size))
        , biases(std::make_unique<T[]>(out_size))
#ifdef RTNEURAL_USE_RAND_MATMUL
        , w_norm(std::make_unique<T[]>(in_size))
        , i_norm(std::make_unique<T[]>(in_size))
        , prob(std::make_unique<T[]>(in_size))
        , idxs(in_size)
#endif
    {
    }

    Dense(const Dense& other) = delete;
    Dense& operator=(const Dense& other) = delete;

    virtual ~Dense()
    {
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "dense"; }

    /** Performs forward propagation for this layer. */
    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
    {
#ifndef RTNEURAL_USE_RAND_MATMUL
        for(int j = 0; j < Layer<T>::out_size; ++j)
        {
            out[j] = biases[j];

            for(int k = 0; k < Layer<T>::in_size; ++k)
            {
                out[j] += input[k] * weights[k * Layer<T>::out_size + j];
            }
        }
#else
        // const int m = 1;
        const int n = Layer<T>::in_size;
        const int p = Layer<T>::out_size;

        // special case for in_size=1 (i.e. do normal mat-mul)
        if(n == 1)
        {
            for(int j = 0; j < p; ++j)
            {
                out[j] = biases[j];

                for(int k = 0; k < n; ++k)
                {
                    out[j] += input[k] * weights[k * p + j];
                }
            }

            return;
        }

        // TODO(toms): move most of this into members and setters

        for(int k = 0; k < n; ++k)
        {
            const auto x = input[k];
            i_norm[k] = std::sqrt(x * x);
        }

        const auto sum_norm = std::inner_product(i_norm.get(), i_norm.get() + n, w_norm.get(), (T)0);

        for(int k = 0; k < n; ++k)
        {
            prob[k] = i_norm[k] * w_norm[k] / sum_norm;
        }

        const float factor = 0.90f;
        const int C = std::ceil(n * factor);

        std::iota(idxs.begin(), idxs.end(), 0);
        std::stable_sort(idxs.begin(), idxs.end(),
            [this](size_t i, size_t j)
            { return prob[i] > prob[j]; });

        for(int j = 0; j < p; ++j)
        {
            out[j] = biases[j];

            for(int c = 0; c < C; ++c)
            {
                const auto t = idxs[c];
                const float pt = 1.0 / (C * prob[t]);
                out[j] += pt * input[t] * weights[t * p + j];
            }
        }
#endif
    }

    /**
     * Sets the layer weights from a given vector.
     * The dimension of the weights vector must be weights[out_size][in_size]
     */
    RTNEURAL_REALTIME void
    setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        const int n = Layer<T>::in_size;
        const int p = Layer<T>::out_size;

        for(int j = 0; j < p; ++j)
        {
            for(int k = 0; k < n; ++k)
            {
                weights[k * p + j] = newWeights[j][k];
            }
        }

#ifdef RTNEURAL_USE_RAND_MATMUL
        for(int k = 0; k < n; ++k)
        {
            T y = 0;
            for(int j = 0; j < p; ++j)
            {
                const auto x = weights[k * p + j];
                y += x * x;
            }
            w_norm[k] = std::sqrt(y);
        }
#endif
    }

    /**
     * Sets the layer weights from a given array.
     * The dimension of the weights array must be weights[out_size][in_size]
     */
    RTNEURAL_REALTIME void setWeights(T** newWeights)
    {
        const int n = Layer<T>::in_size;
        const int p = Layer<T>::out_size;

        for(int j = 0; j < p; ++j)
        {
            for(int k = 0; k < n; ++k)
            {
                weights[k * p + j] = newWeights[j][k];
            }
        }

#ifdef RTNEURAL_USE_RAND_MATMUL
        for(int k = 0; k < n; ++k)
        {
            T y = 0;
            for(int j = 0; j < p; ++j)
            {
                const auto x = weights[k * p + j];
                y += x * x;
            }
            w_norm[k] = std::sqrt(y);
        }
#endif
    }

    /**
     * Sets the layer bias from a given array of size
     * bias[out_size]
     */
    RTNEURAL_REALTIME void setBias(const T* b)
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            biases[i] = b[i];
        }
    }

    /** Returns the weights value at the given indices. */
    RTNEURAL_REALTIME T getWeight(int i, int k) const noexcept
    {
        return 0.0; // TODO(toms): weights[i * Layer<T>::in_size + k];
    }

    /** Returns the bias value at the given index. */
    RTNEURAL_REALTIME T getBias(int i) const noexcept { return biases[i]; }

private:
    std::unique_ptr<T[]> weights;
    std::unique_ptr<T[]> biases;

#ifdef RTNEURAL_USE_RAND_MATMUL
    std::unique_ptr<T[]> w_norm;
    std::unique_ptr<T[]> i_norm;
    std::unique_ptr<T[]> prob;
    std::vector<size_t> idxs;
#endif
};

//====================================================
/**
 * Static implementation of a fully-connected (dense) layer,
 * with no activation.
 */
template <typename T, int in_sizet, int out_sizet>
class DenseT
{
    static constexpr auto weights_size = in_sizet * out_sizet;

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    DenseT()
    {
        for(int i = 0; i < weights_size; ++i)
            weights[i] = (T)0.0;

        for(int i = 0; i < out_size; ++i)
            bias[i] = (T)0.0;

        for(int i = 0; i < out_size; ++i)
            outs[i] = (T)0.0;
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "dense"; }

    /** Returns false since dense is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Reset is a no-op, since Dense does not have state. */
    RTNEURAL_REALTIME void reset() { }

    /** Performs forward propagation for this layer. */
    RTNEURAL_REALTIME inline void forward(const T (&ins)[in_size]) noexcept
    {
        for(int i = 0; i < out_size; ++i)
            outs[i] = std::inner_product(ins, ins + in_size, &weights[i * in_size], (T)0) + bias[i];
    }

    /**
     * Sets the layer weights from a given vector.
     *
     * The dimension of the weights vector must be
     * weights[out_size][in_size]
     */
    RTNEURAL_REALTIME void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(int i = 0; i < out_size; ++i)
        {
            for(int k = 0; k < in_size; ++k)
            {
                auto idx = i * in_size + k;
                weights[idx] = newWeights[i][k];
            }
        }
    }

    /**
     * Sets the layer weights from a given vector.
     *
     * The dimension of the weights array must be
     * weights[out_size][in_size]
     */
    RTNEURAL_REALTIME void setWeights(T** newWeights)
    {
        for(int i = 0; i < out_size; ++i)
        {
            for(int k = 0; k < in_size; ++k)
            {
                auto idx = i * in_size + k;
                weights[idx] = newWeights[i][k];
            }
        }
    }

    /**
     * Sets the layer bias from a given array of size
     * bias[out_size]
     */
    RTNEURAL_REALTIME void setBias(const T* b)
    {
        for(int i = 0; i < out_size; ++i)
            bias[i] = b[i];
    }

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

private:
    T bias[out_size];
    T weights[weights_size];
};

} // namespace RTNEURAL_NAMESPACE

#endif // RTNEURAL_USE_STL

#endif // DENSE_H_INCLUDED
