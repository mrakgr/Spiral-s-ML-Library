kernel = r"""
template <typename el, int dim> struct static_array { el v[dim]; };
template <typename el, int dim, typename default_int> struct static_array_list { el v[dim]; default_int length; };
#include <assert.h>
#include <stdio.h>
typedef struct UH0 UH0;
__device__ void UHDecref0(UH0 * x);
typedef struct {
    int refc;
    unsigned long len;
    char ptr[];
} Array0;
typedef Array0 String;
typedef struct {
    int refc;
    String * v2;
    long v0;
    bool v1;
} Heap0;
typedef struct {
    int refc;
    unsigned long len;
    float ptr[];
} Array1;
typedef struct Fun0 Fun0;
struct Fun0{
    int refc;
    void (*decref_fptr)(Fun0 *);
    long (*fptr)(Fun0 *, long, long);
};
typedef struct Closure0 Closure0;
struct Closure0 {
    int refc;
    void (*decref_fptr)(Closure0 *);
    long (*fptr)(Closure0 *, long, long);
};
struct UH0 {
    int refc;
    int tag;
    union {
        struct {
            UH0 * v1;
            long v0;
        } case0; // Cons
    };
};
__device__ static inline void ArrayDecrefBody0(Array0 * x){
}
__device__ void ArrayDecref0(Array0 * x){
    if (x != NULL && --(x->refc) == 0) { ArrayDecrefBody0(x); free(x); }
}
__device__ Array0 * ArrayCreate0(unsigned long len, bool init_at_zero){
    unsigned long size = sizeof(Array0) + sizeof(char) * len;
    Array0 * x = (Array0 *) malloc(size);
    if (init_at_zero) { memset(x,0,size); }
    x->refc = 1;
    x->len = len;
    return x;
}
__device__ Array0 * ArrayLit0(unsigned long len, char * ptr){
    Array0 * x = ArrayCreate0(len, false);
    memcpy(x->ptr, ptr, sizeof(char) * len);
    return x;
}
__device__ static inline void StringDecref(String * x){
    return ArrayDecref0(x);
}
__device__ static inline String * StringLit(unsigned long len, char * ptr){
    return ArrayLit0(len, ptr);
}
__device__ static inline void HeapDecrefBody0(Heap0 * x){
    StringDecref(x->v2);
}
__device__ void HeapDecref0(Heap0 * x){
    if (x != NULL && --(x->refc) == 0) { HeapDecrefBody0(x); free(x); }
}
__device__ Heap0 * HeapCreate0(long v0, bool v1, String * v2){
    Heap0 * x = (Heap0 *) malloc(sizeof(Heap0));
    x->refc = 1;
    x->v0 = v0; x->v1 = v1; x->v2 = v2;
    return x;
}
__device__ static inline void ArrayDecrefBody1(Array1 * x){
}
__device__ void ArrayDecref1(Array1 * x){
    if (x != NULL && --(x->refc) == 0) { ArrayDecrefBody1(x); free(x); }
}
__device__ Array1 * ArrayCreate1(unsigned long len, bool init_at_zero){
    unsigned long size = sizeof(Array1) + sizeof(float) * len;
    Array1 * x = (Array1 *) malloc(size);
    if (init_at_zero) { memset(x,0,size); }
    x->refc = 1;
    x->len = len;
    return x;
}
__device__ Array1 * ArrayLit1(unsigned long len, float * ptr){
    Array1 * x = ArrayCreate1(len, false);
    memcpy(x->ptr, ptr, sizeof(float) * len);
    return x;
}
__device__ static inline void ClosureDecrefBody0(Closure0 * x){
}
__device__ void ClosureDecref0(Closure0 * x){
    if (x != NULL && --(x->refc) == 0) { ClosureDecrefBody0(x); free(x); }
}
__device__ long ClosureMethod0(Closure0 * x, long v0, long v1){
    ClosureDecref0(x);
    long v2;
    v2 = v0 + v1;
    return v2;
}
__device__ Fun0 * ClosureCreate0(){
    Closure0 * x = (Closure0 *) malloc(sizeof(Closure0));
    x->refc = 1;
    x->decref_fptr = ClosureDecref0;
    x->fptr = ClosureMethod0;
    return (Fun0 *) x;
}
__device__ static inline void UHDecrefBody0(UH0 * x){
    switch (x->tag) {
        case 0: {
            UHDecref0(x->case0.v1);
            break;
        }
    }
}
__device__ void UHDecref0(UH0 * x){
    if (x != NULL && --(x->refc) == 0) { UHDecrefBody0(x); free(x); }
}
__device__ UH0 * UH0_0(long v0, UH0 * v1) { // Cons
    UH0 * x = (UH0 *) malloc(sizeof(UH0));
    x->tag = 0;
    x->refc = 1;
    x->case0.v0 = v0; x->case0.v1 = v1;
    return x;
}
__device__ UH0 * UH0_1() { // Nil
    UH0 * x = (UH0 *) malloc(sizeof(UH0));
    x->tag = 1;
    x->refc = 1;
    return x;
}
extern "C" __global__ void entry0() {
    String * v0;
    v0 = StringLit(5, "qwer");
    v0->refc++;
    Heap0 * v1;
    v1 = HeapCreate0(1l, true, v0);
    StringDecref(v0); HeapDecref0(v1);
    Array1 * v2;
    v2 = ArrayCreate1(10l, false);
    ArrayDecref1(v2);
    Fun0 * v3;
    v3 = ClosureCreate0();
    v3->decref_fptr(v3);
    long v4;
    v4 = 1l;
    long v5;
    v5 = 2l;
    long v6;
    v6 = 3l;
    long v7;
    v7 = 4l;
    long v8;
    v8 = 5l;
    UH0 * v9;
    v9 = UH0_1();
    v9->refc++;
    UH0 * v10;
    v10 = UH0_0(v8, v9);
    v10->refc++;
    UHDecref0(v9);
    UH0 * v11;
    v11 = UH0_0(v7, v10);
    v11->refc++;
    UHDecref0(v10);
    UH0 * v12;
    v12 = UH0_0(v6, v11);
    v12->refc++;
    UHDecref0(v11);
    UH0 * v13;
    v13 = UH0_0(v5, v12);
    v13->refc++;
    UHDecref0(v12);
    UH0 * v14;
    v14 = UH0_0(v4, v13);
    UHDecref0(v13); UHDecref0(v14);
    return ;
}
"""
class static_array(list):
    def __init__(self, length):
        for _ in range(length):
            self.append(None)

class static_array_list(static_array):
    def __init__(self, length):
        super().__init__(length)
        self.length = 0
import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

options = []
options.append('--diag-suppress=550,20012,2464')
options.append('--dopt=on')
options.append('--restrict')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def main():
    v0 = 0
    v1 = raw_module.get_function(f"entry{v0}")
    del v0
    v1.max_dynamic_shared_size_bytes = 0 
    v1((1,),(32,),(),shared_mem=0)
    del v1
    return 0

if __name__ == '__main__': print(main())
