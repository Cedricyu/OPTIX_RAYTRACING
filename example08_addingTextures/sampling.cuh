#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>   // CUDART_PI_F
#include "gdt/math/vec.h"

using namespace gdt;

// ------------------------------------------------------------------
// Simple RNG (LCG)
// ------------------------------------------------------------------
__device__ inline float rnd(uint32_t& seed)
{
    seed = 1664525u * seed + 1013904223u;
    return float(seed & 0x00FFFFFF) / float(0x01000000);
}

// ------------------------------------------------------------------
// Cosine-weighted hemisphere sampling (local space, z = normal)
// pdf = cos(theta) / pi
// ------------------------------------------------------------------
__device__ inline vec3f cosineSampleHemisphere(uint32_t& seed)
{
    float u1 = rnd(seed);
    float u2 = rnd(seed);

    float r = sqrtf(u1);
    float phi = 2.f * CUDART_PI_F * u2;

    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.f, 1.f - u1));

    return vec3f(x, y, z); // local space
}

// ------------------------------------------------------------------
// Build TBN and transform local direction to world
// ------------------------------------------------------------------
__device__ inline vec3f toWorld(const vec3f& v, const vec3f& N)
{
    vec3f T, B;

    // Robust tangent construction
    if (fabs(N.x) > fabs(N.z))
        T = normalize(cross(vec3f(0.f, 0.f, 1.f), N));
    else
        T = normalize(cross(vec3f(1.f, 0.f, 0.f), N));

    B = cross(N, T);

    return normalize(
        v.x * T +
        v.y * B +
        v.z * N
    );
}

__device__ __forceinline__ uint32_t tea(uint32_t v0, uint32_t v1)
{
    uint32_t s0 = 0;
    for (int i = 0; i < 4; ++i)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

__device__ __forceinline__ vec3f clamp(const vec3f& c, float lo, float hi)
{
    return vec3f(
        fminf(hi, fmaxf(lo, c.x)),
        fminf(hi, fmaxf(lo, c.y)),
        fminf(hi, fmaxf(lo, c.z))
    );
}

