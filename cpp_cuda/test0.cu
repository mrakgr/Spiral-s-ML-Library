using default_int = int;
using default_uint = unsigned int;
// #include <new>
#include <assert.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError error, const char *file, int line, bool abort=true) {
    if (error != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(error), file, line);
        if (abort) exit(error);
    }
}

namespace Device {
    struct ClosureBase0 { int refc{0}; __device__ virtual int operator()(int, int) = 0; __device__ virtual ~ClosureBase0() {}; };
    struct Closure0 : ClosureBase0 {
        __device__ int operator()(int tup0, int tup1) override {
            int v0 = tup0; int v1 = tup1;
            int v2;
            v2 = v0 + v1;
            return v2;
        }
        __device__ ~Closure0() override {
            printf("Hello\n");
        };
    };
    extern "C" __global__ void entry0() {
        auto x = new Closure0{};
        delete x;
    }
}
int main() {
    printf("launching kernel\n");
    Device::entry0<<<1,1>>>();
    printf("done with kernel\n");
    gpuErrchk(cudaDeviceSynchronize());
    printf("done with sync\n");
    return 0;
}