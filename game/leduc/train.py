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

import collections
def Closure0():
    def inner() -> object:
        v0 = 100
        return method0(v0)
    return inner
def method2() -> object:
    v0 = {}
    return v0
def method4(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method3(v0 : i32) -> object:
    v1 = method4(v0)
    del v0
    v2 = {'training_iterations': v1}
    del v1
    return v2
def method1(v0 : i32) -> object:
    v1 = method2()
    v2 = method3(v0)
    del v0
    v3 = {'private_state': v1, 'public_state': v2}
    del v1, v2
    return v3
def method0(v0 : i32) -> object:
    v1 = method1(v0)
    del v0
    return v1
def main():
    v0 = Closure0()
    v1 = collections.namedtuple("Leduc_Train",['init'])(v0)
    del v0
    return v1

if __name__ == '__main__': print(main())
