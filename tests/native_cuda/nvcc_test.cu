#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError error, const char *file, int line, bool abort=true) {
    if (error != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(error), file, line);
        if (abort) exit(error);
    }
}

__global__ void __cluster_dims__(8) hello(int a, int b) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Hello, CUDA! %i + %i = %i\n", a, b, a + b);
    }
}

int main() {
    size_t maxDynamicSharedMemory = 114 * (1 << 10);
    cudaError_t error = cudaFuncSetAttribute(hello, cudaFuncAttributeMaxDynamicSharedMemorySize, maxDynamicSharedMemory);

    int i = 0;
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    void * args[] = {reinterpret_cast<void*>(&i), reinterpret_cast<void*>(&i)};
    gpuErrchk(cudaLaunchCooperativeKernel(hello, 128, 256, args, maxDynamicSharedMemory));
    gpuErrchk(cudaDeviceSynchronize());
    std::cout << "Done." << std::endl;
    return 0;
}

/*
pwsh tests/native_cuda/compile.ps1
*/