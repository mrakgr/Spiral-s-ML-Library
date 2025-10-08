#include "test21.auto.cu"
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
namespace Device {
    struct Union0;
    __device__ void method_0(Union0 v0);
    struct Union0_0 { // None
    };
    struct Union0_1 { // Some
        int v0;
        __device__ Union0_1(int t0) : v0(t0) {}
        __device__ Union0_1() = delete;
    };
    struct Union0 {
        union {
            Union0_0 case0; // None
            Union0_1 case1; // Some
        };
        unsigned char tag{255};
        __device__ Union0() {}
        __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // None
        __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Some
        __device__ Union0(Union0 & x) : tag(x.tag) {
            switch(x.tag){
                case 0: new (&this->case0) Union0_0(x.case0); break; // None
                case 1: new (&this->case1) Union0_1(x.case1); break; // Some
            }
        }
        __device__ Union0(Union0 && x) : tag(x.tag) {
            switch(x.tag){
                case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // None
                case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Some
            }
        }
        __device__ Union0 & operator=(Union0 & x) {
            if (this->tag == x.tag) {
                switch(x.tag){
                    case 0: this->case0 = x.case0; break; // None
                    case 1: this->case1 = x.case1; break; // Some
                }
            } else {
                this->~Union0();
                new (this) Union0{x};
            }
            return *this;
        }
        __device__ Union0 & operator=(Union0 && x) {
            if (this->tag == x.tag) {
                switch(x.tag){
                    case 0: this->case0 = std::move(x.case0); break; // None
                    case 1: this->case1 = std::move(x.case1); break; // Some
                }
            } else {
                this->~Union0();
                new (this) Union0{std::move(x)};
            }
            return *this;
        }
        __device__ ~Union0() {
            switch(this->tag){
                case 0: this->case0.~Union0_0(); break; // None
                case 1: this->case1.~Union0_1(); break; // Some
            }
            this->tag = 255;
        }
    };
    __device__ void method_0(Union0 v0){
        switch (v0.tag) {
            case 0: { // None
                printf("%s","None");
                return ;
                break;
            }
            case 1: { // Some
                int v1 = v0.case1.v0;
                printf("%s(%d)","Some", v1);
                return ;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                __trap();
            }
        }
    }
    extern "C" __global__ void __cluster_dims__(12,1,1) entry0(Union0 v0) {
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
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v6 = console_lock;
            auto v7 = cooperative_groups::coalesced_threads();
            v6.acquire();
            printf("");
            method_0(v0);
            printf("\n");
            v6.release();
            v7.sync() ;
        } else {
        }
        return ;
    }
}
struct Union0;
void run_cuda_host_0(Union0 v0);
struct Union0_0 { // None
};
struct Union0_1 { // Some
    int v0;
    Union0_1(int t0) : v0(t0) {}
    Union0_1() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // None
        Union0_1 case1; // Some
    };
    unsigned char tag{255};
    Union0() {}
    Union0(Union0_0 t) : tag(0), case0(t) {} // None
    Union0(Union0_1 t) : tag(1), case1(t) {} // Some
    Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // None
            case 1: new (&this->case1) Union0_1(x.case1); break; // Some
        }
    }
    Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Some
        }
    }
    Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{x};
        }
        return *this;
    }
    Union0 & operator=(Union0 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // None
            case 1: this->case1.~Union0_1(); break; // Some
        }
        this->tag = 255;
    }
};
void run_cuda_host_0(Union0 v0){
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
    int v0;
    v0 = 1;
    Union0 v1;
    v1 = Union0{Union0_1{v0}};
    run_cuda_host_0(v1);
    gpuErrchk(cudaDeviceSynchronize());
    return 0;
}
