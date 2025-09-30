#include "tests.auto.cu"
#include <thrust/device_vector.h>
#include <xoshiro.h>
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
unsigned int find_nth_set_bit_0(int v0, unsigned int v1, unsigned int v2);
void run_cuda_host_1(unsigned int v0);
struct Union0;
struct Tuple0;
struct Tuple1;
struct Tuple2;
unsigned int loop_4(unsigned int v0, xso::rng & v1);
Tuple2 draw_card_3(xso::rng & v0, unsigned int v1);
Tuple0 draw_cards_2(xso::rng & v0, unsigned int v1);
void method_5(Union0 v0);
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
struct Union0_0 { // Jack
};
struct Union0_1 { // King
};
struct Union0_2 { // Queen
};
struct Union0 {
    union {
        Union0_0 case0; // Jack
        Union0_1 case1; // King
        Union0_2 case2; // Queen
    };
    unsigned char tag{255};
    Union0() {}
    Union0(Union0_0 t) : tag(0), case0(t) {} // Jack
    Union0(Union0_1 t) : tag(1), case1(t) {} // King
    Union0(Union0_2 t) : tag(2), case2(t) {} // Queen
    Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union0_1(x.case1); break; // King
            case 2: new (&this->case2) Union0_2(x.case2); break; // Queen
        }
    }
    Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // Queen
        }
    }
    Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Jack
                case 1: this->case1 = x.case1; break; // King
                case 2: this->case2 = x.case2; break; // Queen
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
                case 0: this->case0 = std::move(x.case0); break; // Jack
                case 1: this->case1 = std::move(x.case1); break; // King
                case 2: this->case2 = std::move(x.case2); break; // Queen
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // Jack
            case 1: this->case1.~Union0_1(); break; // King
            case 2: this->case2.~Union0_2(); break; // Queen
        }
        this->tag = 255;
    }
};
struct Tuple0 {
    static_array<Union0,6> v0;
    unsigned int v1;
    Tuple0() = default;
    Tuple0(static_array<Union0,6> t0, unsigned int t1) : v0(t0), v1(t1) {}
};
struct Tuple1 {
    int v0;
    unsigned int v1;
    Tuple1() = default;
    Tuple1(int t0, unsigned int t1) : v0(t0), v1(t1) {}
};
struct Tuple2 {
    Union0 v0;
    unsigned int v1;
    Tuple2() = default;
    Tuple2(Union0 t0, unsigned int t1) : v0(t0), v1(t1) {}
};
inline bool while_method_0(unsigned int v0, unsigned int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
unsigned int find_nth_set_bit_0(int v0, unsigned int v1, unsigned int v2){
    int v4;
    v4 = (int)v1;
    unsigned int v5;
    v5 = v2 >> v4;
    StackMut0 v6{0};
    StackMut1 v7{4294967295u};
    unsigned int v8;
    v8 = 32u - v1;
    unsigned int v9;
    v9 = 0u;
    while (while_method_0(v8, v9)){
        int v11;
        v11 = (int)v9;
        unsigned int v12;
        v12 = v5 >> v11;
        unsigned int v13;
        v13 = v12 & 1u;
        bool v14;
        v14 = v13 == 0u;
        bool v15;
        v15 = v14 != true;
        if (v15){
            int v16 = v6.v0;
            int v17;
            v17 = v16 + 1;
            v6.v0 = v17;
            bool v18;
            v18 = v17 == v0;
            if (v18){
                unsigned int v19;
                v19 = v1 + v9;
                v7.v0 = v19;
                break;
            } else {
            }
        } else {
        }
        v9 += 1u ;
    }
    unsigned int v20 = v7.v0;
    return v20;
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
inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 6;
    return v1;
}
unsigned int loop_4(unsigned int v0, xso::rng & v1){
    unsigned int v2;
    v2 = v1();
    unsigned int v3;
    v3 = v2 % v0;
    unsigned int v4;
    v4 = v2 - v3;
    unsigned int v5;
    v5 = 0u - v0;
    bool v6;
    v6 = v4 <= v5;
    if (v6){
        return v3;
    } else {
        return loop_4(v0, v1);
    }
}
Tuple2 draw_card_3(xso::rng & v0, unsigned int v1){
    int v3;
    v3 = __builtin_popcount(v1);
    unsigned int v4;
    v4 = (unsigned int)v3;
    bool v5;
    v5 = 0u < v4;
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("The range has to be greater than 0." && v5);
    } else {
    }
    unsigned int v8;
    v8 = loop_4(v4, v0);
    int v9;
    v9 = (int)v8;
    int v11;
    v11 = __builtin_popcount(v1);
    bool v12;
    v12 = v9 < v11;
    unsigned int v18;
    if (v12){
        int v13;
        v13 = v9 + 1;
        unsigned int v14;
        v14 = 0u;
        v18 = find_nth_set_bit_0(v13, v14, v1);
    } else {
        int v16;
        v16 = v9 - v11;
        printf("%s\n", "Cannot find the n-th set bit.");
        exit(-1);
    }
    bool v19;
    v19 = 0u == v18;
    Union0 v37;
    if (v19){
        v37 = Union0{Union0_1{}};
    } else {
        bool v21;
        v21 = 1u == v18;
        if (v21){
            v37 = Union0{Union0_1{}};
        } else {
            bool v23;
            v23 = 2u == v18;
            if (v23){
                v37 = Union0{Union0_2{}};
            } else {
                bool v25;
                v25 = 3u == v18;
                if (v25){
                    v37 = Union0{Union0_2{}};
                } else {
                    bool v27;
                    v27 = 4u == v18;
                    if (v27){
                        v37 = Union0{Union0_0{}};
                    } else {
                        bool v29;
                        v29 = 5u == v18;
                        if (v29){
                            v37 = Union0{Union0_0{}};
                        } else {
                            printf("%s\n", "Invalid int in int_to_card.");
                            exit(-1);
                        }
                    }
                }
            }
        }
    }
    int v38;
    v38 = (int)v18;
    unsigned int v39;
    v39 = 1u << v38;
    unsigned int v40;
    v40 = v1 ^ v39;
    return Tuple2{v37, v40};
}
Tuple0 draw_cards_2(xso::rng & v0, unsigned int v1){
    static_array<Union0,6> v3;
    int v5; unsigned int v6;
    Tuple1 tmp0 = Tuple1{0, v1};
    v5 = tmp0.v0; v6 = tmp0.v1;
    while (while_method_1(v5)){
        Union0 v8; unsigned int v9;
        Tuple2 tmp1 = draw_card_3(v0, v6);
        v8 = tmp1.v0; v9 = tmp1.v1;
        v3[v5] = v8;
        v6 = v9;
        v5 += 1 ;
    }
    return Tuple0{v3, v6};
}
void method_5(Union0 v0){
    switch (v0.tag) {
        case 0: { // Jack
            printf("%s","Jack");
            return ;
            break;
        }
        case 1: { // King
            printf("%s","King");
            return ;
            break;
        }
        case 2: { // Queen
            printf("%s","Queen");
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
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
    v9 = find_nth_set_bit_0(v8, v7, v2);
    int v10;
    v10 = v9;
    printf("%d\n",v10);
    run_cuda_host_1(v2);
    gpuErrchk(cudaDeviceSynchronize());
    static_array<Union0,3> v15;
    unsigned int v17;
    v17 = 63u;
    xso::rng v18;
    static_array<Union0,6> v19; unsigned int v20;
    Tuple0 tmp2 = draw_cards_2(v18, v17);
    v19 = tmp2.v0; v20 = tmp2.v1;
    printf("%s","[");
    int v30;
    v30 = 0;
    while (while_method_1(v30)){
        Union0 v33;
        v33 = v19[v30];
        printf("");
        method_5(v33);
        printf("");
        int v35;
        v35 = v30 + 1;
        bool v36;
        v36 = v35 < 6;
        if (v36){
            printf("%s","; ");
        } else {
        }
        v30 += 1 ;
    }
    printf("%s","]");
    printf(", %u",v20);
    printf("\n");
    gpuErrchk(cudaDeviceSynchronize());
    return 0;
}
