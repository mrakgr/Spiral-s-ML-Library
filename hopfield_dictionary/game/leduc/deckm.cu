#include "deckm.auto.cu"
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
namespace Device {
    extern "C" __global__ void __cluster_dims__(12,1,1) entry0(unsigned int v0) {
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
            unsigned int v6;
            v6 = __fns(v0,0u,2);
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v7 = console_lock;
            auto v8 = cooperative_groups::coalesced_threads();
            v7.acquire();
            printf("%u\n",v6);
            v7.release();
            v8.sync() ;
        } else {
        }
        return ;
    }
}
struct StackMut0;
struct StackMut1;
unsigned int find_nth_set_bit_0(unsigned int v0, unsigned int v1, int v2);
void run_cuda_host_1(unsigned int v0);
struct StackMut0 {
    int v0;
    StackMut0() = default;
    StackMut0(int t0) : v0(t0) {}
};
struct StackMut1 {
    unsigned int v0;
    StackMut1() = default;
    StackMut1(unsigned int t0) : v0(t0) {}
};
inline bool while_method_0(unsigned int v0, unsigned int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
unsigned int find_nth_set_bit_0(unsigned int v0, unsigned int v1, int v2){
    int v3;
    v3 = (int)v1;
    unsigned int v4;
    v4 = v0 >> v3;
    StackMut0 v5{0};
    StackMut1 v6{4294967295u};
    unsigned int v7;
    v7 = 32u - v1;
    unsigned int v8;
    v8 = 0u;
    while (while_method_0(v7, v8)){
        int v10;
        v10 = (int)v8;
        unsigned int v11;
        v11 = v4 >> v10;
        unsigned int v12;
        v12 = v11 & 1u;
        bool v13;
        v13 = v12 == 0u;
        bool v14;
        v14 = v13 != true;
        if (v14){
            int v15 = v5.v0;
            int v16;
            v16 = v15 + 1;
            v5.v0 = v16;
            bool v17;
            v17 = v16 == v2;
            if (v17){
                unsigned int v18;
                v18 = v1 + v8;
                v6.v0 = v18;
                break;
            } else {
            }
        } else {
        }
        v8 += 1u ;
    }
    unsigned int v19 = v6.v0;
    return v19;
}
void run_cuda_host_1(unsigned int v0){
    auto kernel = Device::entry0;
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 12));
    cudaLaunchConfig_t v1 = {0};
    v1.gridDim = 84;
    v1.blockDim = 256;
    v1.dynamicSmemBytes = 98304;
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
    gpuErrchk(cudaLaunchKernelEx(&v1, kernel, v0));
    return ;
}
int main() {
    unsigned int v0;
    v0 = 63u;
    unsigned int v1;
    v1 = v0 ^ 1u;
    unsigned int v2;
    v2 = v1 ^ 4u;
    int v3;
    v3 = v2;
    printf("%d\n",v3);
    unsigned int v7;
    v7 = 0u;
    int v8;
    v8 = 2;
    unsigned int v9;
    v9 = find_nth_set_bit_0(v2, v7, v8);
    int v10;
    v10 = v9;
    printf("%d\n",v10);
    run_cuda_host_1(v2);
    gpuErrchk(cudaDeviceSynchronize());
    return 0;
}
