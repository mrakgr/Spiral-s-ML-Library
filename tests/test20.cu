#include "test20.auto.cu"
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
namespace Device {
    extern "C" __global__ void __cluster_dims__(12,1,1) entry0() {
        int v0;
        v0 = threadIdx.x;
        int v1;
        v1 = blockIdx.x;
        int v2;
        v2 = v1 * 256;
        int v3;
        v3 = v0 + v2;
        bool v4;
        v4 = v3 == 0;
        if (v4){
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v5 = console_lock;
            auto v6 = cooperative_groups::coalesced_threads();
            v5.acquire();
            printf("%s\n","Hello World!");
            v5.release();
            v6.sync() ;
        } else {
        }
        return ;
    }
}
void run_cuda_host_0();
void run_cuda_host_0(){
    auto kernel = Device::entry0;
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 12));
    cudaLaunchConfig_t v0 = {0};
    v0.gridDim = 84;
    v0.blockDim = 256;
    v0.dynamicSmemBytes = 98304;
    cudaLaunchAttribute v1;
    v1.id = cudaLaunchAttributeCooperative;
    v1.val.cooperative = 1;
    v0.numAttrs = 1;
    cudaLaunchAttribute v2[] = { v1 };
    v0.attrs = v2;
    int v3;
    v3 = 0;
    cudaOccupancyMaxPotentialClusterSize(&v3, (void *)kernel, &v0);
    bool v4;
    v4 = v3 >= 12;
    bool v5;
    v5 = v4 == false;
    if (v5){
        assert("Max potential cluster size must be greater than or equal to the given cluster size." && v4);
    } else {
    }
    gpuErrchk(cudaLaunchKernelEx(&v0, kernel));
    return ;
}
int main() {
    run_cuda_host_0();
    gpuErrchk(cudaDeviceSynchronize());
    return 0;
}
