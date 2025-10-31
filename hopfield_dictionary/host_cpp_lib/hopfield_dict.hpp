#pragma once

#include <stdio.h>
// #include <Eigen/Dense>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

struct Foo {
    int i = 5;
    __host__ __device__ void hi() {
        printf("Hi. %i\n", i);
    }
};

void bar(Foo & x);

template <int i>
void qwer();