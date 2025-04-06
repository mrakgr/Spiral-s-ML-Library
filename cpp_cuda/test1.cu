#include "test1.auto.cu"
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
namespace Device {
    extern "C" __global__ void entry0(int * v0) {
        int v1;
        v1 = v0[0];
        int v2;
        v2 = v0[1];
        int v3;
        v3 = v1 + v2;
        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v4 = console_lock;
        auto v5 = cooperative_groups::coalesced_threads();
        v4.acquire();
        printf("%s\n","hello");
        v4.release();
        v5.sync() ;
        return ;
    }
}
void run_cuda_host_0(thrust::device_vector<int> v0);
void run_cuda_host_0(thrust::device_vector<int> v0){
    auto kernel = Device::entry0;
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaLaunchConfig_t v1 = {0};
    v1.gridDim = 1;
    v1.blockDim = 1;
    v1.dynamicSmemBytes = 98304;
    cudaLaunchAttribute v2;
    v2.id = cudaLaunchAttributeCooperative;
    v2.val.cooperative = 1;
    v1.numAttrs = 1;
    cudaLaunchAttribute v3[] = { v2 };
    v1.attrs = v3;
    int * v4 = v0.data().get();
    gpuErrchk(cudaLaunchKernelEx(&v1, kernel, v4));
    return ;
}
int main_body() {
    thrust::device_vector<int> v0 = {1,2,3,4};
    int v1;
    v1 = v0.size();
    printf("%d\n",v1);
    int v5;
    v5 = v0[2];
    printf("%d\n",v5);
    run_cuda_host_0(v0);
    return 0;
}
int main(){
    auto r = main_body();
    gpuErrchk(cudaDeviceSynchronize()); // This line is here so the `__trap()` calls on the kernel aren't missed.
    return r;
}
