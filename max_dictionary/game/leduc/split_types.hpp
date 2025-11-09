#include <stdio.h>

#ifdef __CUDACC__ 
struct Qwe {
    int i;
    float b;
};
void foo(Qwe & x);
#endif
struct Qwe {
    float t;
    int i;
    float b;
};
void foo(Qwe & x);

