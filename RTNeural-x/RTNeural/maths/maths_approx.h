#pragma once

#ifdef RTNEURAL_USE_APPROX

#include <bit>
#include <cmath>
#include <cstdint>

namespace RTNEURAL_NAMESPACE
{
namespace approx
{
    template <typename T>
    static T ktanh(T x_in)
    {
        struct X
        {
            uint8_t e;
            uint8_t r;
            int8_t b;
        };

        static constexpr X LUT[32] = {
            { 126, 2, 119 },
            { 126, 4, 122 },
            { 126, 4, 123 },
            { 126, 4, 123 },
            { 126, 6, 126 },
            { 126, 6, 126 },
            { 126, 6, 126 },
            { 126, 6, 126 },
            { 125, 1, 1 },
            { 125, 0, -4 },
            { 125, 0, -6 },
            { 125, 0, -7 },
            { 125, 0, -10 },
            { 125, 0, -12 },
            { 125, 0, -15 },
            { 125, 0, -18 },
            { 125, 0, 112 },
            { 126, 1, -4 },
            { 126, 1, -1 },
            { 126, 1, 2 },
            { 126, 1, 3 },
            { 126, 1, 4 },
            { 126, 1, 4 },
            { 126, 1, 4 },
            { 126, 0, 65 },
            { 126, 1, 72 },
            { 126, 1, 73 },
            { 126, 1, 73 },
            { 126, 2, 88 },
            { 126, 2, 89 },
            { 126, 2, 89 },
            { 126, 4, 110 },
        };

        const auto xa = std::fabsf(x_in);
        if(xa < 0.25)
        {
            return x_in;
        }
        else if(xa > 3.75)
        {
            return std::copysignf(1.0f, x_in);
        }
        else
        {
            const auto x = std::bit_cast<uint32_t>(x_in);

            const auto t = (x >> 20) & 0b11111;
            const auto mi = (x >> 16) & 0b01111111;
            const auto so = x & 0x80000000;

            const auto lut = LUT[t];
            const uint8_t et = lut.e;
            const uint8_t rt = lut.r;
            const int8_t bt = lut.b;

            const auto eo = uint32_t(et) << 23;
            const auto mo = (int32_t(mi >> rt) + bt) << 16;

            return std::bit_cast<float>(so | eo | mo);
        }
    }

    template <typename T>
    static T xexp(T x)
    {
        // TODO(toms): assert T == f32 (SFINAE?)

        const auto u = (uint32_t)(6051101.5f * x + 1065353216);
        const auto v = (uint32_t)(-6051101.5f * x + 1065353216);
        return std::bit_cast<float>(u) / std::bit_cast<float>(v);
    }
}
}

#endif
