#include <cuda_runtime.h>
#include "LaunchParams.h"

__global__ void resolveKernel(
    const osc::AccumPixel* accum,
    uint32_t* colorBuffer,
    int width, int height,
    int spp)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int id = x + y * width;

    float r = accum[id].r / spp;
    float g = accum[id].g / spp;
    float b = accum[id].b / spp;

    r = fminf(r, 1.f);
    g = fminf(g, 1.f);
    b = fminf(b, 1.f);

    colorBuffer[id] =
        0xff000000 |
        (int(255.99f * r) << 0) |
        (int(255.99f * g) << 8) |
        (int(255.99f * b) << 16);
}

extern "C" void launchResolveKernel(
    const osc::AccumPixel* accum,
    uint32_t* colorBuffer,
    int width, int height,
    int spp)
{
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    resolveKernel << <grid, block >> > (
        accum, colorBuffer, width, height, spp
        );
}
