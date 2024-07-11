kernel = r"""
"""
class static_array():
    def __init__(self, length):
        self.ptr = []
        for _ in range(length):
            self.ptr.append(None)

    def __getitem__(self, index):
        assert 0 <= index < len(self.ptr), "The get index needs to be in range."
        return self.ptr[index]
    
    def __setitem__(self, index, value):
        assert 0 <= index < len(self.ptr), "The set index needs to be in range."
        self.ptr[index] = value

class static_array_list(static_array):
    def __init__(self, length):
        super().__init__(length)
        self.length = 0

    def __getitem__(self, index):
        assert 0 <= index < self.length, "The get index needs to be in range."
        return self.ptr[index]
    
    def __setitem__(self, index, value):
        assert 0 <= index < self.length, "The set index needs to be in range."
        self.ptr[index] = value

    def push(self,value):
        assert (self.length < len(self.ptr)), "The length before pushing has to be less than the maximum length of the array."
        self.ptr[self.length] = value
        self.length += 1

    def pop(self):
        assert (0 < self.length), "The length before popping has to be greater than 0."
        self.length -= 1
        return self.ptr[self.length]

    def unsafe_set_length(self,i):
        assert 0 <= i <= len(self.ptr), "The new length has to be in range."
        self.length = i

class dynamic_array(static_array): 
    pass

class dynamic_array_list(static_array_list):
    def length_(self): return self.length

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
    v0 = cp.empty(160,dtype=cp.uint8)
    v1 = cp.empty(104,dtype=cp.uint8)
    del v1
    v2 = "---"
    method0(v2)
    del v2
    print()
    v4 = v0[0:0+4*16].view(cp.float32)
    v5 = 0
    v6 = 4
    v7 = 1
    v8 = 4
    v9 = 4
    method1(v4, v5, v6, v7, v8, v9)
    del v4, v5, v6, v7, v8, v9
    print()
    v11 = v0[64:64+4*16].view(cp.float32)
    v12 = 0
    v13 = 4
    v14 = 1
    v15 = 4
    v16 = 4
    method1(v11, v12, v13, v14, v15, v16)
    del v11, v12, v13, v14, v15, v16
    print()
    v18 = v0[128:128+4*8].view(cp.float32)
    v19 = 0
    v20 = 4
    v21 = 1
    v22 = 2
    v23 = 4
    method1(v18, v19, v20, v21, v22, v23)
    del v18, v19, v20, v21, v22, v23
    print()
    v25 = v0[0:0+4*16].view(cp.float32)
    v26 = cp.random.normal(0.0,1.0,16,dtype=cp.float32) # type: ignore
    cp.copyto(v25[0:0+16],v26[0:0+16])
    del v25, v26
    v28 = v0[64:64+4*16].view(cp.float32)
    v29 = cp.random.normal(0.0,1.0,16,dtype=cp.float32) # type: ignore
    cp.copyto(v28[0:0+16],v29[0:0+16])
    del v28, v29
    v31 = v0[128:128+4*8].view(cp.float32)
    v32 = cp.random.normal(0.0,1.0,8,dtype=cp.float32) # type: ignore
    cp.copyto(v31[0:0+8],v32[0:0+8])
    del v31, v32
    v33 = "Done initing."
    method0(v33)
    del v33
    print()
    v35 = v0[0:0+4*16].view(cp.float32)
    v36 = 0
    v37 = 4
    v38 = 1
    v39 = 4
    v40 = 4
    method1(v35, v36, v37, v38, v39, v40)
    del v35, v36, v37, v38, v39, v40
    print()
    v42 = v0[64:64+4*16].view(cp.float32)
    v43 = 0
    v44 = 4
    v45 = 1
    v46 = 4
    v47 = 4
    method1(v42, v43, v44, v45, v46, v47)
    del v42, v43, v44, v45, v46, v47
    print()
    v49 = v0[128:128+4*8].view(cp.float32)
    del v0
    v50 = 0
    v51 = 4
    v52 = 1
    v53 = 2
    v54 = 4
    method1(v49, v50, v51, v52, v53, v54)
    del v49, v50, v51, v52, v53, v54
    print()
    return 

if __name__ == '__main__': print(main())
