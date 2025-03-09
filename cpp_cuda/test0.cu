#include "test0.auto.cu"
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
namespace Device {
    extern "C" __global__ void entry0() {        int v0;        v0 = 5;        int v1;        v1 = 3;        int v2;        v2 = v0 + v1;        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v3 = console_lock;        auto v4 = cooperative_groups::coalesced_threads();        v3.acquire();        printf("%s\n","hello");        v3.release();        v4.sync() ;        return ;    }}
void run_cpp_0();
void run_cpp_0(){
    auto kernel = Device::entry0;
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaLaunchConfig_t v0 = {0};
    v0.gridDim = 1;
    v0.blockDim = 1;
    v0.dynamicSmemBytes = 98304;
    cudaLaunchAttribute v1;
    v1.id = cudaLaunchAttributeCooperative;
    v1.val.cooperative = 1;
    v0.numAttrs = 1;
    cudaLaunchAttribute v2[] = { v1 };
    v0.attrs = v2;
    gpuErrchk(cudaLaunchKernelEx(&v0, kernel));
    return ;
}
int main() {
    int v0;
    v0 = 2;
    int v1;
    v1 = 8;
    int v2;
    v2 = v0 * v1;
    run_cpp_0();
    gpuErrchk(cudaDeviceSynchronize());
    return 0;
}
