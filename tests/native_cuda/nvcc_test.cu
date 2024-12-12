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

__global__ void __cluster_dims__(16) hello(int a, int b) {
    if (threadIdx.x == 0) {
        printf("Hello, CUDA from %i! %i + %i = %i\n", blockIdx.x, a, b, a + b);
    }
}

int main() {
    size_t maxDynamicSharedMemory = 214 * (1 << 10);
    gpuErrchk(cudaFuncSetAttribute(hello, cudaFuncAttributeMaxDynamicSharedMemorySize, maxDynamicSharedMemory));
    gpuErrchk(cudaFuncSetAttribute(hello, cudaFuncAttributeNonPortableClusterSizeAllowed, 16));

    int i = 0;
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    void * args[] = {reinterpret_cast<void*>(&i), reinterpret_cast<void*>(&i)};
    gpuErrchk(cudaLaunchCooperativeKernel(hello, 112, 256, args, maxDynamicSharedMemory));
    gpuErrchk(cudaDeviceSynchronize());
    std::cout << "Done." << std::endl;
    return 0;
}

/*
pwsh tests/native_cuda/compile.ps1
*/