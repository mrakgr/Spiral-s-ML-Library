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
    v0 = cp.empty(112,dtype=cp.uint8)
    v1 = "---"
    method0(v1)
    del v1
    print()
    v2 = v0[0:0+4*4].view(cp.float32)
    v3 = 0
    v4 = 4
    v5 = 1
    v6 = 1
    v7 = 4
    method1(v2, v3, v4, v5, v6, v7)
    del v2, v3, v4, v5, v6, v7
    print()
    v8 = v0[16:16+4*16].view(cp.float32)
    v9 = 0
    v10 = 4
    v11 = 1
    v12 = 4
    v13 = 4
    method1(v8, v9, v10, v11, v12, v13)
    del v8, v9, v10, v11, v12, v13
    print()
    v14 = v0[80:80+4*8].view(cp.float32)
    v15 = 0
    v16 = 2
    v17 = 1
    v18 = 4
    v19 = 2
    method1(v14, v15, v16, v17, v18, v19)
    del v14, v15, v16, v17, v18, v19
    print()
    v20 = v0[0:0+4*4].view(cp.float32)
    v21 = cp.random.normal(0.0,1.0,4,dtype=cp.float32) # type: ignore
    cp.copyto(v20[0:0+4],v21[0:0+4])
    del v20, v21
    v22 = v0[16:16+4*16].view(cp.float32)
    v23 = cp.random.normal(0.0,1.0,16,dtype=cp.float32) # type: ignore
    cp.copyto(v22[0:0+16],v23[0:0+16])
    del v22, v23
    v24 = v0[80:80+4*8].view(cp.float32)
    v25 = cp.random.normal(0.0,1.0,8,dtype=cp.float32) # type: ignore
    cp.copyto(v24[0:0+8],v25[0:0+8])
    del v24, v25
    v26 = "Done initing."
    method0(v26)
    del v26
    print()
    v27 = v0[0:0+4*4].view(cp.float32)
    v28 = 0
    v29 = 4
    v30 = 1
    v31 = 1
    v32 = 4
    method1(v27, v28, v29, v30, v31, v32)
    del v27, v28, v29, v30, v31, v32
    print()
    v33 = v0[16:16+4*16].view(cp.float32)
    v34 = 0
    v35 = 4
    v36 = 1
    v37 = 4
    v38 = 4
    method1(v33, v34, v35, v36, v37, v38)
    del v33, v34, v35, v36, v37, v38
    print()
    v39 = v0[80:80+4*8].view(cp.float32)
    del v0
    v40 = 0
    v41 = 2
    v42 = 1
    v43 = 4
    v44 = 2
    method1(v39, v40, v41, v42, v43, v44)
    del v39, v40, v41, v42, v43, v44
    print()
    return 

if __name__ == '__main__': print(main())
