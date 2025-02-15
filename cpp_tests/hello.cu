#include <new>
#include <assert.h>
#include <stdio.h>

// #include "kittens.cuh"
// using namespace kittens;

__global__ void test_print(){
    printf("Hello from inside the kernel.\n");
    return;
}

int main() {
    int i = 0;
    printf("Hello. %i\n", i);
    test_print<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}