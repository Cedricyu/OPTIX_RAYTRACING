// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include <cuda_runtime.h>
#include "radiance_prd.h"
#include "sampling.cuh"
#include "LaunchParams.h"

#define MAX_DEPTH 10

using namespace osc;

enum RayType {
    SURFACE_RAY_TYPE = 0,
    SHADOW_RAY_TYPE = 1,
    RAY_TYPE_COUNT
};

__device__ inline float clamp01(float x) {
    return fminf(fmaxf(x, 0.f), 1.f);
}

__device__ inline vec3f tonemap_exposure_gamma(vec3f c, float exposure)
{
    // exposure tone map: 1 - exp(-c * exposure)
    c.x = 1.f - expf(-c.x * exposure);
    c.y = 1.f - expf(-c.y * exposure);
    c.z = 1.f - expf(-c.z * exposure);

    // gamma (sRGB-ish)
    const float invGamma = 1.f / 2.2f;
    c.x = powf(clamp01(c.x), invGamma);
    c.y = powf(clamp01(c.y), invGamma);
    c.z = powf(clamp01(c.z), invGamma);

    return c;
}

namespace osc {

    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;

    // for this simple example, we have a single ray type
    enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

    static __forceinline__ __device__
        void* unpackPointer(uint32_t i0, uint32_t i1)
    {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }

    static __forceinline__ __device__
        void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T* getPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }

    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------

    extern "C" __global__ void __closesthit__radiance()
    {
        const TriangleMeshSBTData& sbtData =
            *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

        const int primID = optixGetPrimitiveIndex();
        const vec3i index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        // ---------------- normal ----------------
        vec3f N;
        if (sbtData.normal) {
            N = (1.f - u - v) * sbtData.normal[index.x]
                + u * sbtData.normal[index.y]
                + v * sbtData.normal[index.z];
        }
        else {
            const vec3f& A = sbtData.vertex[index.x];
            const vec3f& B = sbtData.vertex[index.y];
            const vec3f& C = sbtData.vertex[index.z];
            N = normalize(cross(B - A, C - A));
        }
        N = normalize(N);

        // ---------------- albedo ----------------
        vec3f diffuseColor = sbtData.color;
        if (sbtData.hasTexture && sbtData.texcoord) {
            const vec2f tc =
                (1.f - u - v) * sbtData.texcoord[index.x] +
                u * sbtData.texcoord[index.y] +
                v * sbtData.texcoord[index.z];
            vec4f tex = tex2D<float4>(sbtData.texture, tc.x, tc.y);
            diffuseColor *= (vec3f)tex;
        }

        RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();

        const vec3f rayOrg = optixGetWorldRayOrigin();
        const vec3f rayDir = optixGetWorldRayDirection();
        const float tHit = optixGetRayTmax();
        const vec3f hitPoint = rayOrg + tHit * rayDir;

        const float invPi = 1.0f / 3.14159265f;

        // ---------------- hit light ----------------
        if (sbtData.isLight) {
            prd.radiance += prd.throughput * sbtData.emission;
            return;
        }

        // ---------------- terminate ----------------
        if (prd.depth >= MAX_DEPTH) {
            prd.radiance += prd.throughput * vec3f(0.05f);
            return;
        }

        vec3f wr = cosineSampleHemisphere(prd.seed);
        wr = toWorld(wr, N);

        float cosTheta = dot(wr, N);
        if (cosTheta <= 0.f) return;

        prd.throughput *= diffuseColor;
        prd.depth++;

        uint32_t p0, p1;
        packPointer(&prd, p0, p1);

        optixTrace(
            optixLaunchParams.traversable,
            hitPoint + 1e-4f * N,
            wr,
            1e-4f,
            1e20f,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            SURFACE_RAY_TYPE,
            RAY_TYPE_COUNT,
            SURFACE_RAY_TYPE,
            p0, p1
        );
    }

    extern "C" __global__ void __anyhit__radiance()
    { /*! for this simple example, this will remain empty */
        ShadowPRD& prd = *(ShadowPRD*)getPRD<ShadowPRD>();
        prd.visible = false;

        optixTerminateRay();
    }


    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // as with the anyhit/closest hit programs, in this example we only
    // need to have _some_ dummy function to set up a valid SBT
    // ------------------------------------------------------------------------------

    extern "C" __global__ void __miss__radiance()
    {
        RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();

        vec3f envColor = vec3f(0.6f, 0.7f, 0.9f);
        prd.radiance += prd.throughput * envColor;
        //payload_visible = true;
    }

    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;
        const int si = optixGetLaunchIndex().z;

        const auto& camera = optixLaunchParams.camera;

        // our per-ray data for this example. what we initialize it to
        // won't matter, since this value will be overwritten by either
        // the miss or hit program, anyway
        RadiancePRD prd;

        // the values we store the PRD pointer in:
        vec3f pixelColor = vec3f(0.f);

        prd.radiance = vec3f(0.f);
        prd.throughput = vec3f(1.f);
        prd.depth = 0;
        uint32_t pixelId = ix + iy * optixLaunchParams.frame.size.x;
        prd.seed = tea(pixelId, si);

        uint32_t u0, u1;
        packPointer(&prd, u0, u1);

        // normalized screen plane position, in [0,1]^2
        const vec2f screen(vec2f(ix + .5f, iy + .5f)
            / vec2f(optixLaunchParams.frame.size));

        // generate ray direction
        vec3f rayDir = normalize(camera.direction
            + (screen.x - 0.5f) * camera.horizontal
            + (screen.y - 0.5f) * camera.vertical);

        optixTrace(optixLaunchParams.traversable,
            camera.position,
            rayDir,
            0.f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
            SURFACE_RAY_TYPE,             // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            SURFACE_RAY_TYPE,             // missSBTIndex 
            u0, u1);

        pixelColor = prd.radiance;
        vec3f pixelColorPRD = pixelColor;

        vec3f* AccumPixel = optixLaunchParams.accumBuffer;
        AccumPixel[pixelId] += pixelColor;

        if (si == 99) {
            vec3f pixel = optixLaunchParams.accumBuffer[pixelId];
            vec3f color = pixel / float(optixLaunchParams.frame.spp);

            // exposure tone mapping
            float exposure = 1.0f;
            color = tonemap_exposure_gamma(color, /*exposure*/ 1.0f);


            // clamp 只是保險，不是主角
            color.x = fminf(color.x, 1.f);
            color.y = fminf(color.y, 1.f);
            color.z = fminf(color.z, 1.f);

            const int r = int(255.99f * color.x);
            const int g = int(255.99f * color.y);
            const int b = int(255.99f * color.z);

            optixLaunchParams.frame.colorBuffer[pixelId] =
                0xff000000 | (r << 0) | (g << 8) | (b << 16);

        }
    }
} // ::osc