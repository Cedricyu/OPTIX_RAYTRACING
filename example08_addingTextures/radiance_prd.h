#pragma once
#pragma once
#include <cuda_runtime.h>
#include "LaunchParams.h"
#include "gdt/math/vec.h"

struct RadiancePRD {
    gdt::vec3f radiance;
    gdt::vec3f throughput;
    int    depth;        
    uint32_t seed;      
};

struct ShadowPRD {
    bool inShadow;
};
