kernel = r"""
template <typename el, int dim> struct static_array { el v[dim]; };
template <typename el, int dim, typename default_int> struct static_array_list { el v[dim]; default_int length; };
"""
class static_array(list):
    def __init__(self, length):
        for _ in range(length):
            self.append(None)

class static_array_list(static_array):
    def __init__(self, length):
        super().__init__(length)
        self.length = 0
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

import cupy as cp
def main():
    v0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    v1 = cp.array(v0,dtype=cp.float32)
    del v0
    v2 = v1.size
    v3 = 12 == v2
    del v2
    v4 = v3 == False
    if v4:
        v5 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v3, v5
        del v5
    else:
        pass
    del v3, v4
    v6 = v1[6].item()
    v7 = v1[3].item()
    del v1
    return v6, v7

if __name__ == '__main__': print(main())
