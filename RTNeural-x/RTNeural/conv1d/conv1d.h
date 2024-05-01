#ifndef CONV1D_H_INCLUDED
#define CONV1D_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#ifdef RTNEURAL_STL_PERF_CNN
namespace RTNEURAL_NAMESPACE
{
template <typename T>
std::optional<size_t> find_index_of_nearest_neighbor(const std::vector<T>& arr, size_t index, T value)
{
    size_t left = index;
    size_t right = index;

    while(left > 0 || right < arr.size() - 1)
    {
        if(left > 0)
        {
            if(arr[left - 1] == value)
            {
                return left - 1;
            }
            left -= 1;
        }

        if(right < arr.size() - 1)
        {
            if(arr[right + 1] == value)
            {
                return right + 1;
            }
            right += 1;
        }
    }

    return std::nullopt;
}

std::pair<std::vector<bool>, std::vector<std::pair<size_t, size_t>>> pseudo_mask(size_t length)
{
    const float alpha = 1.5;
    const std::pair<float, float> uniform_range = std::make_pair(0.0f, 1.0f);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> uniform(uniform_range.first, uniform_range.second);

    std::vector<bool> sequence(length, false);
    for(size_t i = 0; true; ++i)
    {
        float u = uniform(rng);
        size_t a = static_cast<size_t>(std::ceil(alpha * (i + u)));
        if(a >= length)
            break;
        sequence[a] = true;
    }

    std::vector<std::pair<size_t, size_t>> neighbors;
    for(size_t i = 0; i < length; ++i)
    {
        if(!sequence[i])
        {
            size_t neighbor = find_index_of_nearest_neighbor(sequence, i, true).value();
            neighbors.emplace_back(std::make_pair(i, neighbor));
        }
    }

    return std::make_pair(sequence, neighbors);
}
}
#endif

#if RTNEURAL_USE_EIGEN
#include "conv1d_eigen.h"
#include "conv1d_eigen.tpp"
#elif RTNEURAL_USE_XSIMD
#include "conv1d_xsimd.h"
#include "conv1d_xsimd.tpp"
#else
#include "../Layer.h"
#include "../common.h"
#include "../config.h"

namespace RTNEURAL_NAMESPACE
{

/**
 * Dynamic implementation of a 1-dimensional convolution layer
 * with no activation.
 *
 * This implementation was designed to be used for "temporal
 * convolution", so the layer has a "state" made up of past inputs
 * to the layer. To ensure that the state is initialized to zero,
 * please make sure to call `reset()` before your first call to
 * the `forward()` method.
 */
template <typename T>
class Conv1D final : public Layer<T>
{
public:
    /**
     * Constructs a convolution layer for the given dimensions.
     *
     * @param in_size: the input size for the layer
     * @param out_size: the output size for the layer
     * @param kernel_size: the size of the convolution kernel
     * @param dilation: the dilation rate to use for dilated convolution
     */
    Conv1D(int in_size, int out_size, int kernel_size, int dilation, int groups = 1);
    Conv1D(std::initializer_list<int> sizes);
    Conv1D(const Conv1D& other);
    Conv1D& operator=(const Conv1D& other);
    virtual ~Conv1D();

    /** Resets the layer state. */
    RTNEURAL_REALTIME void reset() override;

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "conv1d"; }

    /** Performs forward propagation for this layer. */
    RTNEURAL_REALTIME inline void forward(const T* input, T* h) noexcept override
    {
        // insert input into a circular buffer
        std::copy(input, input + Layer<T>::in_size, state[state_ptr]);

        // set state pointers to particular columns of the buffer
        setStatePointers();

#ifdef RTNEURAL_STL_PERF_CNN
        assert(groups == 1);
        assert(this->out_size == (this->in_size + this->kernel_size - 1));

        // TODO(toms): preallocate this!
        static const auto perf_mask(pseudo_mask(this->out_size));
        const auto [mask, neighbors] = perf_mask;

        // copy selected columns to a helper variable
        for(int k = 0; k < kernel_size; ++k)
        {
            const auto& col = state[state_ptrs[k]];
            std::copy(col, col + Layer<T>::in_size, state_cols[k]);
        }

        // perform multi-channel convolution
        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            h[i] = bias[i];

            if(mask[i])
            {
                for(int k = 0; k < kernel_size; ++k)
                    h[i] = std::inner_product(
                        weights[i][k],
                        weights[i][k] + filters_per_group,
                        state_cols[k],
                        h[i]);
            }
        }

        for(auto [i, j] : neighbors)
        {
            h[i] = h[j];
        }
#else
        if(groups == 1)
        {
            // copy selected columns to a helper variable
            for(int k = 0; k < kernel_size; ++k)
            {
                const auto& col = state[state_ptrs[k]];
                std::copy(col, col + Layer<T>::in_size, state_cols[k]);
            }

            // perform multi-channel convolution
            for(int i = 0; i < Layer<T>::out_size; ++i)
            {
                h[i] = bias[i];
                for(int k = 0; k < kernel_size; ++k)
                    h[i] = std::inner_product(
                        weights[i][k],
                        weights[i][k] + filters_per_group,
                        state_cols[k],
                        h[i]);
            }
        }
        else
        {
            // perform multi-channel convolution
            for(int i = 0; i < Layer<T>::out_size; ++i)
            {
                h[i] = bias[i];
                const auto ii = ((i / channels_per_group) * filters_per_group);
                for(int k = 0; k < kernel_size; ++k)
                {
                    // copy selected columns to a helper variable
                    const auto& column = state[state_ptrs[k]];

                    const auto column_begin = column + ii;
                    const auto column_end = column_begin + filters_per_group;
                    std::copy(column_begin, column_end, state_cols[k]);

                    h[i] = std::inner_product(
                        weights[i][k],
                        weights[i][k] + filters_per_group,
                        state_cols[k],
                        h[i]);
                }
            }
        }
#endif

        state_ptr = (state_ptr == state_size - 1 ? 0 : state_ptr + 1); // iterate state pointer forwards
    }

    /**
     * Sets the layer weights.
     *
     * The weights vector must have size weights[out_size][in_size][kernel_size * dilation]
     */
    RTNEURAL_REALTIME void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);

    /**
     * Sets the layer biases.
     *
     * The bias vector must have size bias[out_size]
     */
    RTNEURAL_REALTIME void setBias(const std::vector<T>& biasVals);

    /** Returns the size of the convolution kernel. */
    RTNEURAL_REALTIME int getKernelSize() const noexcept { return kernel_size; }

    /** Returns the convolution dilation rate. */
    RTNEURAL_REALTIME int getDilationRate() const noexcept { return dilation_rate; }

    /** Returns the number of "groups" in the convolution. */
    int getGroups() const noexcept { return groups; }

private:
    const int dilation_rate;
    const int kernel_size;
    const int state_size;
    const int groups;
    const int filters_per_group;
    const int channels_per_group;

    T*** weights;
    T* bias;

    T** state;
    T** state_cols;

    int* state_ptrs;
    int state_ptr = 0;

    /** Sets pointers to state array columns. */
    inline void setStatePointers()
    {
        for(int k = 0; k < kernel_size; ++k)
            state_ptrs[k] = (state_ptr + state_size - k * dilation_rate) % state_size;
    }
};

//====================================================
/**
 * Static implementation of a 1-dimensional convolution layer
 * with no activation.
 *
 * This implementation was designed to be used for "temporal
 * convolution", so the layer has a "state" made up of past inputs
 * to the layer. To ensure that the state is initialized to zero,
 * please make sure to call `reset()` before your first call to
 * the `forward()` method.
 *
 * @param in_sizet: the input size for the layer
 * @param out_sizet: the output size for the layer
 * @param kernel_size: the size of the convolution kernel
 * @param dilation_rate: the dilation rate to use for dilated convolution
 * @param dynamic_state: use dynamically allocated layer state
 * @param groups: controls connections between inputs and outputs
 */
template <typename T, int in_sizet, int out_sizet, int kernel_size, int dilation_rate, int groups = 1, bool dynamic_state = false>
class Conv1DT
{
    static_assert((in_sizet % groups == 0) && (out_sizet % groups == 0), "in_size and out_size must be divisible by groups!");

    static constexpr auto state_size = (kernel_size - 1) * dilation_rate + 1;

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;
    static constexpr auto filters_per_group = in_size / groups;
    static constexpr auto channels_per_group = out_size / groups;

    Conv1DT();

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "conv1d"; }

    /** Returns false since convolution is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Resets the layer state. */
    RTNEURAL_REALTIME void reset();

    template <int _groups = groups, std::enable_if_t<_groups == 1, bool> = true>
    /** Performs forward propagation for this layer. */
    RTNEURAL_REALTIME inline void forward(const T (&ins)[in_size]) noexcept
    {
        // insert input into a circular buffer
        std::copy(std::begin(ins), std::end(ins), state[state_ptr].begin());

        // set state pointers to particular columns of the buffer
        setStatePointers();

        // copy selected columns to a helper variable
        for(int k = 0; k < kernel_size; ++k)
        {
            const auto& col = state[state_ptrs[k]];
            std::copy(col.begin(), col.end(), state_cols[k].begin());
        }

        // perform multi-channel convolution
        for(int i = 0; i < out_size; ++i)
        {
            outs[i] = bias[i];
            for(int k = 0; k < kernel_size; ++k)
                outs[i] = std::inner_product(
                    weights[i][k].begin(),
                    weights[i][k].end(),
                    state_cols[k].begin(),
                    outs[i]);
        }

        state_ptr = (state_ptr == state_size - 1 ? 0 : state_ptr + 1); // iterate state pointer forwards
    }

    template <int _groups = groups, std::enable_if_t<_groups != 1, bool> = true>
    /** Performs forward propagation for this layer. */
    inline void forward(const T (&ins)[in_size]) noexcept
    {
        // insert input into a circular buffer
        std::copy(std::begin(ins), std::end(ins), state[state_ptr].begin());

        // set state pointers to particular columns of the buffer
        setStatePointers();

        // perform multi-channel convolution
        for(int i = 0; i < out_size; ++i)
        {
            outs[i] = bias[i];

            const auto ii = ((i / channels_per_group) * filters_per_group);
            for(int k = 0; k < kernel_size; ++k)
            {
                // copy selected columns to a helper variable
                const auto& column = state[state_ptrs[k]];
                const auto column_begin = column.begin() + ii;
                const auto column_end = column_begin + filters_per_group;
                std::copy(column_begin, column_end, state_cols[k].begin());

                outs[i] = std::inner_product(
                    weights[i][k].begin(),
                    weights[i][k].end(),
                    state_cols[k].begin(),
                    outs[i]);
            }
        }

        state_ptr = (state_ptr == state_size - 1 ? 0 : state_ptr + 1); // iterate state pointer forwards
    }

    /**
     * Sets the layer weights.
     *
     * The weights vector must have size weights[out_size][group_count][kernel_size * dilation]
     */
    RTNEURAL_REALTIME void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);

    /**
     * Sets the layer biases.
     *
     * The bias vector must have size bias[out_size]
     */
    RTNEURAL_REALTIME void setBias(const std::vector<T>& biasVals);

    /** Returns the size of the convolution kernel. */
    RTNEURAL_REALTIME int getKernelSize() const noexcept { return kernel_size; }

    /** Returns the convolution dilation rate. */
    RTNEURAL_REALTIME int getDilationRate() const noexcept { return dilation_rate; }

    /** Returns the number of "groups" in the convolution. */
    int getGroups() const noexcept { return groups; }

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

private:
    template <int DS = dynamic_state>
    typename std::enable_if<DS, void>::type resize_state()
    {
        state.resize(state_size, {});
    }

    template <int DS = dynamic_state>
    typename std::enable_if<!DS, void>::type resize_state() { }

    using state_type = typename std::conditional<dynamic_state, std::vector<std::array<T, in_size>>, std::array<std::array<T, in_size>, state_size>>::type;
    using weights_type = std::array<std::array<T, filters_per_group>, kernel_size>;

    alignas(RTNEURAL_DEFAULT_ALIGNMENT) state_type state;
    alignas(RTNEURAL_DEFAULT_ALIGNMENT) weights_type state_cols;

    int state_ptr = 0;
    std::array<int, kernel_size> state_ptrs;

    alignas(RTNEURAL_DEFAULT_ALIGNMENT) weights_type weights[out_size];
    alignas(RTNEURAL_DEFAULT_ALIGNMENT) std::array<T, out_size> bias;

    /** Sets pointers to state array columns. */
    inline void setStatePointers()
    {
        for(int k = 0; k < kernel_size; ++k)
            state_ptrs[k] = (state_ptr + state_size - k * dilation_rate) % state_size;
    }
};
} // namespace RTNEURAL_NAMESPACE
#endif
#endif // CONV1D_H_INCLUDED
