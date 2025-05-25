#include "test1.auto.cu"
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
namespace Device {
    extern "C" __global__ void __cluster_dims__(12,1,1) entry0(int * v0) {
        int v1;
        v1 = v0[0];
        int v2;
        v2 = v0[1];
        int v3;
        v3 = v1 + v2;
        int v4;
        v4 = threadIdx.x;
        int v5;
        v5 = blockIdx.x;
        int v6;
        v6 = v5 * 256;
        int v7;
        v7 = v4 + v6;
        bool v8;
        v8 = v7 == 0;
        if (v8){
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v9 = console_lock;
            auto v10 = cooperative_groups::coalesced_threads();
            v9.acquire();
            printf("{%s = %s; %s = %d}\n","message", "hello from cuda", "result", v3);
            v9.release();
            v10.sync() ;
        } else {
        }
        return ;
    }
}
void run_cuda_host_0(thrust::device_vector<int> v0);
void run_cuda_host_0(thrust::device_vector<int> v0){
    auto kernel = Device::entry0;
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 12));
    cudaLaunchConfig_t v1 = {0};
    v1.gridDim = 84;
    v1.blockDim = 256;
    v1.dynamicSmemBytes = 58304;
    cudaLaunchAttribute v2;
    v2.id = cudaLaunchAttributeCooperative;
    v2.val.cooperative = 1;
    v1.numAttrs = 1;
    cudaLaunchAttribute v3[] = { v2 };
    v1.attrs = v3;
    int v4;
    v4 = 0;
    cudaOccupancyMaxPotentialClusterSize(&v4, (void *)kernel, &v1);
    bool v5;
    v5 = v4 >= 12;
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("Max potential cluster size must be greater than or equal to the given cluster size." && v5);
    } else {
    }
    int * v8 = v0.data().get();
    gpuErrchk(cudaLaunchKernelEx(&v1, kernel, v8));
    return ;
}
int main_body() {
    thrust::device_vector<int> v0 = {1,2,3,4};
    int v1;
    v1 = v0.size();
    int v2;
    v2 = v0[2];
    printf("{%s = %d; %s = %d; %s = %s}\n","index_2", v2, "length_of_array", v1, "message", "hello from host");
    run_cuda_host_0(v0);
    return 0;
}
int main(){
    auto r = main_body();
    gpuErrchk(cudaDeviceSynchronize()); // This line is here so the `__trap()` calls on the kernel aren't missed.
    return r;
}
