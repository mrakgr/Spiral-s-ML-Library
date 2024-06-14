kernel = r"""
#include "reference_counting.h"
struct ClosureBase0 { virtual long operator()(long) = 0; void dispose() { call_destructor(*this); } };
typedef sptr<ClosureBase0> Fun0;
struct Closure0 : public ClosureBase0 {
    long v0;
    __device__ long operator() (long tup0) override {
        long & v0 = this->v0;
        long v1 = tup0;
        long v2;
        v2 = v1 + v0;
        return v2;
    }
    Closure0(long _v0) : v0(_v0) { }
};
extern "C" __global__ void entry0() {
    long v0;
    v0 = 2l;
    Fun0 v1{Closure0{v0}};
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
options.append('--diag-suppress=550,20012')
options.append('--dopt=on')
options.append('--restrict')
options.append('-I C:/Spiral\'s ML Library/cpplib')
raw_module = cp.RawModule(code=kernel, backend='nvrtc', enable_cooperative_groups=True, options=tuple(options))
def main():
    v0 = 0
    v1 = raw_module.get_function(f"entry{v0}")
    del v0
    v1.max_dynamic_shared_size_bytes = 0 
    v1((1,),(32,),(),shared_mem=0)
    del v1
    return 

if __name__ == '__main__': print(main())
