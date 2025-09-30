#include "test13.auto.cu"
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
namespace Device {
    struct HeapRefs0;
    struct HeapRefs0 {
        int refc{0};
        int v0;
        int v1;
        int v2;
        __device__ HeapRefs0() = default;
        __device__ HeapRefs0(int t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    };
    extern "C" __global__ void __cluster_dims__(12,1,1) entry0() {
        sptr<HeapRefs0> v0;
        v0 = sptr<HeapRefs0>{new HeapRefs0{1, 2, 3}};
        int v1;
        v1 = threadIdx.x;
        int v2;
        v2 = blockIdx.x;
        int v3;
        v3 = v2 * 256;
        int v4;
        v4 = v1 + v3;
        bool v5;
        v5 = v4 == 0;
        if (v5){
            int & v6 = v0.base->v0; int & v7 = v0.base->v1; int & v8 = v0.base->v2;
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v9 = console_lock;
            auto v10 = cooperative_groups::coalesced_threads();
            v9.acquire();
            printf("%d, %d, %d\n",v6, v7, v8);
            v9.release();
            v10.sync() ;
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
