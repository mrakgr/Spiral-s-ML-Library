kernel = r"""
template <typename el, int dim> struct static_array { el v[dim]; };
template <typename el, int dim, typename default_int> struct static_array_list { el v[dim]; default_int length; };
#include <assert.h>
#include <stdio.h>

extern "C" __global__ void entry0() {
    malloc(16);
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
    with cp.cuda.Device(0):
        v0 = 0
        v1 = raw_module.get_function(f"entry{v0}")
        del v0
        v1.max_dynamic_shared_size_bytes = 0 
        v1((1,),(32,),(),shared_mem=0)
        del v1
        return 0

if __name__ == '__main__': print(main())
