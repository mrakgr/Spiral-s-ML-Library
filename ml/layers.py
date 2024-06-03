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
import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

def method0(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method2(v0 : char) -> None:
    print(v0, end="")
    del v0
    return 
def method3(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method4(v0 : f32) -> None:
    print("{:.6f}".format(v0), end="")
    del v0
    return 
def method1(v0 : cp.ndarray, v1 : i32, v2 : i32, v3 : i32, v4 : i32, v5 : i32) -> None:
    v6 = 0
    v7 = '['
    method2(v7)
    del v7
    v8 = 0
    while method3(v4, v8):
        v10 = v6
        v11 = v10 >= 100
        del v10
        if v11:
            v12 = " ..."
            method0(v12)
            del v12
            break
        else:
            pass
        del v11
        v13 = v8 == 0
        v14 = v13 != True
        del v13
        if v14:
            v15 = "; "
            method0(v15)
        else:
            pass
        del v14
        v16 = '['
        method2(v16)
        del v16
        v17 = 0
        while method3(v5, v17):
            v19 = v6
            v20 = v19 >= 100
            del v19
            if v20:
                v21 = " ..."
                method0(v21)
                del v21
                break
            else:
                pass
            del v20
            v22 = v17 == 0
            v23 = v22 != True
            del v22
            if v23:
                v24 = "; "
                method0(v24)
            else:
                pass
            del v23
            v25 = v6 + 1
            v6 = v25
            del v25
            v26 = v8 * v2
            v27 = v1 + v26
            del v26
            v28 = v17 * v3
            v29 = v27 + v28
            del v27, v28
            v30 = v0[v29].item()
            del v29
            method4(v30)
            del v30
            v17 += 1 
        del v17
        v31 = ']'
        method2(v31)
        del v31
        v8 += 1 
    del v0, v1, v2, v3, v4, v5, v6, v8
    v32 = ']'
    return method2(v32)
def main():
    v0 = cp.empty(128,dtype=cp.uint8)
    v1 = cp.empty(104,dtype=cp.uint8)
    v2 = "---"
    method0(v2)
    del v2
    print()
    v3 = v0[0:0+4*8].view(cp.float32)
    v4 = 0
    v5 = 4
    v6 = 1
    v7 = 2
    v8 = 4
    method1(v3, v4, v5, v6, v7, v8)
    del v3, v4, v5, v6, v7, v8
    print()
    v9 = v0[32:32+4*16].view(cp.float32)
    v10 = 0
    v11 = 4
    v12 = 1
    v13 = 4
    v14 = 4
    method1(v9, v10, v11, v12, v13, v14)
    del v9, v10, v11, v12, v13, v14
    print()
    v15 = v0[96:96+4*8].view(cp.float32)
    v16 = 0
    v17 = 2
    v18 = 1
    v19 = 4
    v20 = 2
    method1(v15, v16, v17, v18, v19, v20)
    del v15, v16, v17, v18, v19, v20
    print()
    v21 = v0[0:0+4*8].view(cp.float32)
    v22 = cp.random.normal(0.0,1.0,8,dtype=cp.float32) # type: ignore
    cp.copyto(v21[0:0+8],v22[0:0+8])
    del v21, v22
    v23 = v0[32:32+4*16].view(cp.float32)
    v24 = cp.random.normal(0.0,1.0,16,dtype=cp.float32) # type: ignore
    cp.copyto(v23[0:0+16],v24[0:0+16])
    del v23, v24
    v25 = v0[96:96+4*8].view(cp.float32)
    v26 = cp.random.normal(0.0,1.0,8,dtype=cp.float32) # type: ignore
    cp.copyto(v25[0:0+8],v26[0:0+8])
    del v25, v26
    v27 = "Done initing."
    method0(v27)
    del v27
    print()
    v28 = v0[0:0+4*8].view(cp.float32)
    v29 = 0
    v30 = 4
    v31 = 1
    v32 = 2
    v33 = 4
    method1(v28, v29, v30, v31, v32, v33)
    del v28, v29, v30, v31, v32, v33
    print()
    v34 = v0[32:32+4*16].view(cp.float32)
    v35 = 0
    v36 = 4
    v37 = 1
    v38 = 4
    v39 = 4
    method1(v34, v35, v36, v37, v38, v39)
    del v34, v35, v36, v37, v38, v39
    print()
    v40 = v0[96:96+4*8].view(cp.float32)
    del v0
    v41 = 0
    v42 = 2
    v43 = 1
    v44 = 4
    v45 = 2
    method1(v40, v41, v42, v43, v44, v45)
    del v40, v41, v42, v43, v44, v45
    print()
    v46 = "Here is the input tensor."
    method0(v46)
    del v46
    print()
    v47 = v1[0:0+4*2].view(cp.float32)
    del v1
    v48 = 0
    v49 = 2
    v50 = 1
    v51 = 1
    v52 = 2
    method1(v47, v48, v49, v50, v51, v52)
    del v47, v48, v49, v50, v51, v52
    print()
    return 

if __name__ == '__main__': print(main())
