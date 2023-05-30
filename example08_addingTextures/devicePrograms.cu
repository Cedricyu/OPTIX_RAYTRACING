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

#include <curand_kernel.h>

#include "LaunchParams.h"
#include "stdio.h"




using namespace osc;

namespace osc {
  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  // for this simple example, we have a single ray type

  __device__ __host__ float3 operator+=(float3& a, const float3& b) {
      a.x += b.x; a.y += b.y; a.z += b.z;
      return a;
  }

  __device__ __host__ float3 operator-=(float3& a, const float3& b) {
      a.x -= b.x; a.y -= b.y; a.z -= b.z;
      return a;
  }

  __device__ __host__ float3 operator*=(float3& a, const float3& b) {
      a.x *= b.x; a.y *= b.y; a.z *= b.z;
      return a;
  }

  __device__ __host__ float3 operator-(const float3& a) {
      return make_float3(-a.x, -a.y , -a.z );
  }

  __device__ __host__ float3 operator*(const float3& a, const float& b) {
      return make_float3(a.x * b, a.y * b, a.z * b);
  }

  inline __both__ float len(const float3& v)
  {
      return sqrt(powf(v.x, 2) + powf(v.y, 2) + powf(v.z, 2));
  }

  inline __both__ float3 normalize(float3& v)
  {
      return v * (1.f / len(v));
  }

  inline __both__ float3 cross_float(const float3& a, const float3& b)
  {
      return make_float3(a.y * b.z - b.y * a.z,
          a.z * b.x - b.z * a.x,
          a.x * b.y - b.x * a.y);
  }

  __device__ __host__ vec3f float_to_vec(float3& a) {
      return vec3f(a.x, a.y, a.z);
  }

  __device__ __host__ float3 operator+(const float3& a, const float3& b) {
      return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
  }

  __device__ __host__ float3 operator-(const float3& a, const float3& b) {
      return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  __device__ __host__ float3 operator*(const float3& a, const float3& b) {
      return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
  }
  __forceinline__ __device__ float dot(const float3& i, const float3& n)
  {
      return i.x*n.x + i.y*n.y + i.z *n.z ;
  }
  __forceinline__ __device__ float3 reflect(const float3& i, const float3& n)
  {
      return i -  n * dot(n, i) * 2.0f;
  }

  __forceinline__ __device__ float3 transformNormal(const float4* m, float3 const& v)
  {
      float3 r;

      r.x = m[0].x * v.x + m[1].x * v.y + m[2].x * v.z;
      r.y = m[0].y * v.x + m[1].y * v.y + m[2].y * v.z;
      r.z = m[0].z * v.x + m[1].z * v.y + m[2].z * v.z;

      return r;
  }


  inline __both__ float3 unit_vector(const float3& v)
  {
      float3 f = v * (1.f / len(v));
      return f;
  }

  struct RadiancePRD
  {
      // these are produced by the caller, passed into trace, consumed/modified by CH and MS and consumed again by the caller after trace returned.
      float3       attenuation;
      unsigned int seed;
      int          depth;

      // these are produced by CH and MS, and consumed by the caller after trace returned.
      float3       emitted;
      float3       radiance;
      float3       origin;
      float3      direction;
      int          done;
  };

  class onb
  {
  public:
      __device__ onb() {}
      __device__ inline float3 operator[](int i) const { return axis[i]; }
      __device__ float3 u() const { return axis[0]; }
      __device__ float3 v() const { return axis[1]; }
      __device__ float3 w() const { return axis[2]; }
      __device__ float3 local(float a, float b, float c) const {

          float3 f;
          f.x = a * u().x + a * u().y + a * u().z;
          f.y = b * v().x + b * v().y + b * v().z;
          f.z = c * w().x + c * w().y + c * w().z;
          return f;
      }
      __device__ float3 local(const float3& a) const {

          float3 f;
          f.x = a.x * u().x + a.x * u().y + a.x * u().z;
          f.y = a.y * v().x + a.y * v().y + a.y * v().z;
          f.z = a.z * w().x + a.z * w().y + a.z * w().z;
          return f;

      }
      __device__ void build_from_w(const float3&);
      float3 axis[3];
  };

   __forceinline__ __device__
      void onb::build_from_w(const float3& n) {
      axis[2] = unit_vector(n);
      float3 a;
      if (fabs(w().x) > 0.9)
          a = make_float3(0, 1, 0);
      else
          a = make_float3(1, 0, 0);
      axis[1] = unit_vector(cross_float(w(), a));
      axis[0] = cross_float(w(), v());
  }


   __forceinline__ __device__ float random_float()
   {
       int i = threadIdx.x + blockIdx.x * blockDim.x;
       // printf("%d i = %d\n", i);

       curandState state;
       curand_init(clock64(), i, 0, &state);

       return curand_uniform(&state);

   }

   __forceinline__ __device__ float3 ramdom_float3()
   {
       return make_float3(random_float(), random_float(), random_float());

   }


   __forceinline__ __device__ float3 randomCosineDirection()
   {
       float r1 = random_float();
       float r2 = random_float();
       float z = sqrt(1 - r2);
       float phi = 2 * M_PI * r1;
       float x = cos(phi) * sqrt(r2);
       float y = sin(phi) * sqrt(r2);
       return make_float3(x, y, z);

   }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  
  static __forceinline__ __device__ RadiancePRD loadClosesthitRadiancePRD()
  {
      RadiancePRD prd = {};

      prd.attenuation.x = __uint_as_float(optixGetPayload_0());
      prd.attenuation.y = __uint_as_float(optixGetPayload_1());
      prd.attenuation.z = __uint_as_float(optixGetPayload_2());
      prd.seed = optixGetPayload_3();
      prd.depth = optixGetPayload_4();
      return prd;
  }


  static __forceinline__ __device__ void storeClosesthitRadiancePRD(RadiancePRD prd)
  {
      optixSetPayload_0(__float_as_uint(prd.attenuation.x));
      optixSetPayload_1(__float_as_uint(prd.attenuation.y));
      optixSetPayload_2(__float_as_uint(prd.attenuation.z));

      optixSetPayload_3(prd.seed);
      optixSetPayload_4(prd.depth);

      optixSetPayload_5(__float_as_uint(prd.emitted.x));
      optixSetPayload_6(__float_as_uint(prd.emitted.y));
      optixSetPayload_7(__float_as_uint(prd.emitted.z));

      optixSetPayload_8(__float_as_uint(prd.radiance.x));
      optixSetPayload_9(__float_as_uint(prd.radiance.y));
      optixSetPayload_10(__float_as_uint(prd.radiance.z));

      optixSetPayload_11(__float_as_uint(prd.origin.x));
      optixSetPayload_12(__float_as_uint(prd.origin.y));
      optixSetPayload_13(__float_as_uint(prd.origin.z));

      optixSetPayload_14(__float_as_uint(prd.direction.x));
      optixSetPayload_15(__float_as_uint(prd.direction.y));
      optixSetPayload_16(__float_as_uint(prd.direction.z));

      optixSetPayload_17(prd.done);
  }

  static __forceinline__ __device__ void storeMissRadiancePRD(RadiancePRD prd)
  {
      optixSetPayload_5(__float_as_uint(prd.emitted.x));
      optixSetPayload_6(__float_as_uint(prd.emitted.y));
      optixSetPayload_7(__float_as_uint(prd.emitted.z));

      optixSetPayload_8(__float_as_uint(prd.radiance.x));
      optixSetPayload_9(__float_as_uint(prd.radiance.y));
      optixSetPayload_10(__float_as_uint(prd.radiance.z));

      optixSetPayload_17(prd.done);
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

  extern "C" __global__ void __closesthit__shadow()
  {
      RadiancePRD prd = {};
      prd = loadClosesthitRadiancePRD();

      prd.attenuation = make_float3(0.f, 0.f, 0.f);

      storeClosesthitRadiancePRD(prd);
  }
  
  extern "C" __global__ void __closesthit__radiance()
  {
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
    
    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int   primID = optixGetPrimitiveIndex();
    const vec3i index  = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    const float3& A = static_cast<float3>(sbtData.vertex[index.x]);
    const float3& B = static_cast<float3>(sbtData.vertex[index.y]);
    const float3& C = static_cast<float3>(sbtData.vertex[index.z]);

    float3 Ng = cross_float(B-A,C -A);
    float3 Ns = (sbtData.normal)
        ? static_cast<float3>((1.f - u - v) * sbtData.normal[index.x]
            + u * sbtData.normal[index.y]
            + v * sbtData.normal[index.z])
        : Ng;
    
    const float3 rayDir = optixGetWorldRayDirection();
    /*
    if (dot(rayDir, Ng) > 0.f) Ng = Ng*-1.f;
 
    if (dot(Ng, Ns) < 0.f)
        Ns -= Ng * 2.f * dot(Ng, Ns);
     */

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available
    // ------------------------------------------------------------------
    const float3 surfPos
        = static_cast<float3>((1.f - u - v) * sbtData.vertex[index.x]
        + u * sbtData.vertex[index.y]
        + v * sbtData.vertex[index.z]);
    
    const float3 lightPos = make_float3(-907.108f, 2205.875f, -400.0267f);
    const float3 lightDir = lightPos - surfPos;


    float3  lightVisibility = make_float3(0.0f, 0.0f, 0.0f);
    // the values we store the PRD pointer in:
    uint32_t u0, u1, u2;

    u0 = __float_as_uint(lightVisibility.x);
    u1 = __float_as_uint(lightVisibility.y);
    u2 = __float_as_uint(lightVisibility.z);

   
    optixTrace(optixLaunchParams.traversable,
        surfPos ,
        lightDir,
        1e-3f,      // tmin
        1.f - 1e-3f,  // tmax
        0.0f,       // rayTime
        OptixVisibilityMask(255),
        // For shadow rays: skip any/closest hit shaders and terminate on first
        // intersection with anything. The miss shader is used to mark if the
        // light was visible.
        OPTIX_RAY_FLAG_DISABLE_ANYHIT
        | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
        | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        SHADOW_RAY_TYPE,            // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        SHADOW_RAY_TYPE,            // missSBTIndex 
        u0, u1, u2);

    lightVisibility = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));

    float3 diffuseColor = static_cast<float3>(sbtData.color);
    if (sbtData.dissolve == 0.500000)
        diffuseColor = make_float3(0, 0, 0);
    if (sbtData.hasTexture && sbtData.texcoord) {
      const vec2f tc
        = (1.f-u-v) * sbtData.texcoord[index.x]
        +         u * sbtData.texcoord[index.y]
        +         v * sbtData.texcoord[index.z];
      
      vec4f fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
      diffuseColor = static_cast<float3>((vec3f)fromTexture);
    }
    if (sbtData.dissolve == 0.500000)
        diffuseColor = make_float3(1.0f, 1.0f, 1.0f);
    // ------------------------------------------------------------------
    // perform some simple "NdotD" shading
    // ------------------------------------------------------------------

 
    const float cosDN = .8f * fabsf(dot(rayDir, Ns));

    const float3 P = optixGetWorldRayOrigin() + rayDir * optixGetRayTmax();

   
    RadiancePRD prd = loadClosesthitRadiancePRD();

    prd.attenuation = (((lightVisibility * .8f) * cosDN) * diffuseColor + diffuseColor * 0.1f  + diffuseColor * cosDN * .2f );


    prd.emitted = make_float3(0.9f,0.9f,0.9f);

    //printf("N = %f %f %f\n", N.x, N.y, N.z);
    onb uvw;
    uvw.build_from_w(Ns);
    const float3 reflectionWorld = (reflect(rayDir, Ns));
    const float3 random_dir = randomCosineDirection();
    //printf("N = %f %f %f\n", random_dir.x, random_dir.y, random_dir.z);

    prd.direction = reflectionWorld;
    prd.origin = P;
    //printf("direction %f %f %f\n", reflectionWorld.x, reflectionWorld.y , reflectionWorld.z);

    prd.done = false;
    storeClosesthitRadiancePRD(prd);
  }
  
  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */ }

  extern "C" __global__ void __anyhit__shadow()
  { /*! not going to be used */
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
    RadiancePRD prd = {};
    prd.radiance = make_float3(0.f, 0.f, 0.f);
    prd.emitted = make_float3(0.f, 0.f, 0.f);
    prd.done = true;

    storeMissRadiancePRD(prd);
  }


  extern "C" __global__ void __miss__shadow()
  {
      // we didn't hit anything, so the light is visible
      RadiancePRD prd = {};
      prd = loadClosesthitRadiancePRD();

      prd.attenuation = make_float3(1.f, 1.f, 1.f);

      storeClosesthitRadiancePRD(prd);
  }

  static __forceinline__ __device__ void traceRadiance(
      OptixTraversableHandle handle,
      float3                 ray_origin,
      float3                 ray_direction,
      float                  tmin,
      float                  tmax,
      RadiancePRD& prd
  )
  {
      uint32_t u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17;

      //packPointer(&prd.attenuation, u0, u1);
      u0 = __float_as_uint(prd.attenuation.x);
      u1 = __float_as_uint(prd.attenuation.y);
      u2 = __float_as_uint(prd.attenuation.z);
      u3 = prd.seed;
      u4 = prd.depth;

      optixTrace(
          optixLaunchParams.traversable,
          ray_origin,
          ray_direction,
          tmin,    // tmin
          tmax,  // tmax
          0.0f,   // rayTime
          OptixVisibilityMask(255),
          OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
          RADIANCE_RAY_TYPE,             // SBT offset
          RAY_TYPE_COUNT,               // SBT stride
          RADIANCE_RAY_TYPE,             // missSBTIndex 
          u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17);

      prd.attenuation = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));

      prd.seed = u3;
      prd.depth = u4;

      prd.emitted = make_float3(__uint_as_float(u5), __uint_as_float(u6), __uint_as_float(u7));
      prd.origin = make_float3(__uint_as_float(u11), __uint_as_float(u12), __uint_as_float(u13));
      prd.direction = make_float3(__uint_as_float(u14), __uint_as_float(u15), __uint_as_float(u16));
      prd.done = u17;
  
  }
 
  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto &camera = optixLaunchParams.camera;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    
    //vec3f origin = vec3f(0.f);
    //vec3f direction = vec3f(0.f);

    // the values we store the PRD pointer in:
    //uint32_t u0, u1;
    //packPointer( &pixelColorPRD, u0, u1 );

    // normalized screen plane position, in [0,1]^2
    const vec2f screen(vec2f(ix+.5f,iy+.5f)
                       / vec2f(optixLaunchParams.frame.size));
    
    // generate ray direction
    vec3f rayDir = normalize(camera.direction
                             + (screen.x - 0.5f) * camera.horizontal
                             + (screen.y - 0.5f) * camera.vertical);

    vec3f ray_origin = camera.position;

    int sample_per_pixel = 1;
   
    float3 pixelColorPRD = make_float3(1.f, 1.f, 1.f);
    for (int i = 0; i < sample_per_pixel ; i++) {

        // the miss or hit program, anyway
        RadiancePRD prd;
        prd.attenuation = make_float3(1.f, 1.f, 1.f);
        prd.seed = 1;
        prd.depth = 0;

        for (; prd.depth < 2 ; prd.depth ++ ) {


            traceRadiance(optixLaunchParams.traversable,
                static_cast<float3>(ray_origin),
                static_cast<float3>(rayDir),
                0.f,    // tmin
                1e20f,  // tmax
                prd
            );

            pixelColorPRD += prd.emitted;
            pixelColorPRD *= prd.attenuation;

            if (prd.done) // TODO RR, variable for depth
                break;
            ray_origin = float_to_vec(prd.origin);
            rayDir = float_to_vec(prd.direction);

            //++prd.depth;
            //pixelColorPRD += prd.emitted;
        }
    }
    auto scale = 1.0 / sample_per_pixel;
    pixelColorPRD.x = sqrt(scale * pixelColorPRD.x);
    pixelColorPRD.y = sqrt(scale * pixelColorPRD.y);
    pixelColorPRD.z = sqrt(scale * pixelColorPRD.z);

    const int r = int(255.99f*pixelColorPRD.x);
    const int g = int(255.99f*pixelColorPRD.y);
    const int b = int(255.99f*pixelColorPRD.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
  }
  
} // ::osc
