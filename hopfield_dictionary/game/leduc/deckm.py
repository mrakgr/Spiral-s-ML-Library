kernels_main = r"""
"""
from deckm_auto import *
kernels = kernels_aux + kernels_main
import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = int; i16 = int; i32 = int; i64 = int; u8 = int; u16 = int; u32 = int; u64 = int; f32 = float; f64 = float; char = str; string = str

def main():
    v0 = 63
    v3 = "{}"
    print(v3.format(v0),end="")
    v4 = "\n"
    print(v4.format(),end="")
    v5 = v0 ^ 1
    print(v3.format(v5),end="")
    del v3, v5
    print(v4.format(),end="")
    del v4
    v8 = v0 & 1
    del v0
    v9 = v8 == 0
    del v8
    v10 = v9 != True
    del v9
    if v10:
        v19 = "true"
        v21 = v19
    else:
        v20 = "false"
        v21 = v20
    del v10
    v22 = "{}\n"
    print(v22.format(v21),end="")
    del v21, v22
    return 

if __name__ == '__main__': print(main())
