kernels_main = r"""
"""
from deckm_auto import *
kernels = kernels_aux + kernels_main
import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = int; i16 = int; i32 = int; i64 = int; u8 = int; u16 = int; u32 = int; u64 = int; f32 = float; f64 = float; char = str; string = str

import random
@dataclass
class Mut0:
    v0 : i32
def method1(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method0(v0 : i32, v1 : i32, v2 : i32) -> i32:
    v3 = v0 >> v1
    del v0
    v4 = Mut0(0)
    v5 = Mut0(-2147483648)
    v6 = 32 - v1
    v7 = 0
    while method1(v6, v7):
        v9 = 1 << v7
        v10 = v3 & v9
        del v9
        v11 = v10 == 0
        del v10
        v12 = v11 != True
        del v11
        if v12:
            v13 = v4.v0
            v14 = v13 + 1
            v4.v0 = v14
            del v14
            v15 = v13 == v2
            del v13
            if v15:
                del v15
                v16 = v1 + v7
                v5.v0 = v16
                del v16
                break
            else:
                del v15
        else:
            pass
        del v12
        v7 += 1 
    del v1, v2, v3, v4, v6, v7
    v17 = v5.v0
    del v5
    return v17
def main():
    v0 = 63
    v1 = v0 ^ 1
    del v0
    v2 = v1 ^ 4
    del v1
    v3 = v2.bit_count()
    v6 = "{}\n"
    print(v6.format(v3),end="")
    del v3
    v7 = bin(v2)
    print(v6.format(v7),end="")
    del v7
    v10 = i32(v2)
    del v2
    v11 = 0
    v12 = 0
    v13 = method0(v10, v11, v12)
    del v10, v11, v12
    v14 = v13
    del v13
    print(v6.format(v14),end="")
    del v6, v14
    return 

if __name__ == '__main__': print(main())
