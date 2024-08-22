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

def method0(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method1(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method2(v0 : i32) -> bool:
    v1 = v0 < 1
    del v0
    return v1
def main_body():
    v0 = cp.empty(128,dtype=cp.uint8)
    v1 = cp.empty(104,dtype=cp.uint8)
    v4 = "{}\n"
    v5 = "---"
    print(v4.format(v5),end="")
    del v5
    v7 = v0[0:0+4*8].view(cp.float32)
    v8 = cp.random.normal(0.0,0.35355338,8,dtype=cp.float32) # type: ignore
    cp.copyto(v7[0:0+8],v8[0:0+8])
    del v7, v8
    v10 = v0[32:32+4*16].view(cp.float32)
    v11 = cp.random.normal(0.0,0.25,16,dtype=cp.float32) # type: ignore
    cp.copyto(v10[0:0+16],v11[0:0+16])
    del v10, v11
    v13 = v0[96:96+4*8].view(cp.float32)
    v14 = cp.random.normal(0.0,0.35355338,8,dtype=cp.float32) # type: ignore
    cp.copyto(v13[0:0+8],v14[0:0+8])
    del v13, v14
    v17 = "Here are the weight matrices."
    print(v4.format(v17),end="")
    del v17
    v19 = v0[0:0+4*8].view(cp.float32)
    v39 = 0
    v40 = "{}"
    print(v40.format('['),end="")
    v41 = 0
    while method0(v41):
        v43 = v39
        v44 = v43 >= 100
        del v43
        if v44:
            v45 = " ..."
            print(v40.format(v45),end="")
            del v45
            break
        else:
            pass
        del v44
        v46 = v41 == 0
        v47 = v46 != True
        del v46
        if v47:
            v48 = "; "
            print(v40.format(v48),end="")
            del v48
        else:
            pass
        del v47
        print(v40.format('['),end="")
        v49 = 0
        while method1(v49):
            v51 = v39
            v52 = v51 >= 100
            del v51
            if v52:
                v53 = " ..."
                print(v40.format(v53),end="")
                del v53
                break
            else:
                pass
            del v52
            v54 = v49 == 0
            v55 = v54 != True
            del v54
            if v55:
                v56 = "; "
                print(v40.format(v56),end="")
                del v56
            else:
                pass
            del v55
            v57 = v39 + 1
            v39 = v57
            del v57
            v58 = v41 * 2
            v59 = v58 + v49
            del v58
            v60 = v19[v59].item()
            del v59
            v61 = "{:.6f}"
            print(v61.format(v60),end="")
            del v60, v61
            v49 += 1 
        del v49
        print(v40.format(']'),end="")
        v41 += 1 
    del v19, v39, v41
    print(v40.format(']'),end="")
    v62 = "\n"
    print(v62,end="")
    v64 = v0[32:32+4*16].view(cp.float32)
    v84 = 0
    print(v40.format('['),end="")
    v85 = 0
    while method0(v85):
        v87 = v84
        v88 = v87 >= 100
        del v87
        if v88:
            v89 = " ..."
            print(v40.format(v89),end="")
            del v89
            break
        else:
            pass
        del v88
        v90 = v85 == 0
        v91 = v90 != True
        del v90
        if v91:
            v92 = "; "
            print(v40.format(v92),end="")
            del v92
        else:
            pass
        del v91
        print(v40.format('['),end="")
        v93 = 0
        while method0(v93):
            v95 = v84
            v96 = v95 >= 100
            del v95
            if v96:
                v97 = " ..."
                print(v40.format(v97),end="")
                del v97
                break
            else:
                pass
            del v96
            v98 = v93 == 0
            v99 = v98 != True
            del v98
            if v99:
                v100 = "; "
                print(v40.format(v100),end="")
                del v100
            else:
                pass
            del v99
            v101 = v84 + 1
            v84 = v101
            del v101
            v102 = v85 * 4
            v103 = v102 + v93
            del v102
            v104 = v64[v103].item()
            del v103
            v105 = "{:.6f}"
            print(v105.format(v104),end="")
            del v104, v105
            v93 += 1 
        del v93
        print(v40.format(']'),end="")
        v85 += 1 
    del v64, v84, v85
    print(v40.format(']'),end="")
    print(v62,end="")
    v107 = v0[96:96+4*8].view(cp.float32)
    del v0
    v127 = 0
    print(v40.format('['),end="")
    v128 = 0
    while method1(v128):
        v130 = v127
        v131 = v130 >= 100
        del v130
        if v131:
            v132 = " ..."
            print(v40.format(v132),end="")
            del v132
            break
        else:
            pass
        del v131
        v133 = v128 == 0
        v134 = v133 != True
        del v133
        if v134:
            v135 = "; "
            print(v40.format(v135),end="")
            del v135
        else:
            pass
        del v134
        print(v40.format('['),end="")
        v136 = 0
        while method0(v136):
            v138 = v127
            v139 = v138 >= 100
            del v138
            if v139:
                v140 = " ..."
                print(v40.format(v140),end="")
                del v140
                break
            else:
                pass
            del v139
            v141 = v136 == 0
            v142 = v141 != True
            del v141
            if v142:
                v143 = "; "
                print(v40.format(v143),end="")
                del v143
            else:
                pass
            del v142
            v144 = v127 + 1
            v127 = v144
            del v144
            v145 = v128 * 4
            v146 = v145 + v136
            del v145
            v147 = v107[v146].item()
            del v146
            v148 = "{:.6f}"
            print(v148.format(v147),end="")
            del v147, v148
            v136 += 1 
        del v136
        print(v40.format(']'),end="")
        v128 += 1 
    del v107, v127, v128
    print(v40.format(']'),end="")
    print(v62,end="")
    v151 = "Here is the input tensor."
    print(v4.format(v151),end="")
    del v4, v151
    v153 = v1[0:0+4*2].view(cp.float32)
    del v1
    v154 = cp.random.normal(0.0,1.0,2,dtype=cp.float32) # type: ignore
    cp.copyto(v153[0:0+2],v154[0:0+2])
    del v154
    v182 = 0
    print(v40.format('['),end="")
    v183 = 0
    while method2(v183):
        v185 = v182
        v186 = v185 >= 100
        del v185
        if v186:
            v187 = " ..."
            print(v40.format(v187),end="")
            del v187
            break
        else:
            pass
        del v186
        v188 = v183 == 0
        v189 = v188 != True
        del v188
        if v189:
            v190 = "; "
            print(v40.format(v190),end="")
            del v190
        else:
            pass
        del v189
        print(v40.format('['),end="")
        v191 = 0
        while method2(v191):
            v193 = v182
            v194 = v193 >= 100
            del v193
            if v194:
                v195 = " ..."
                print(v40.format(v195),end="")
                del v195
                break
            else:
                pass
            del v194
            v196 = v191 == 0
            v197 = v196 != True
            del v196
            if v197:
                v198 = "; "
                print(v40.format(v198),end="")
                del v198
            else:
                pass
            del v197
            print(v40.format('['),end="")
            v199 = 0
            while method1(v199):
                v201 = v182
                v202 = v201 >= 100
                del v201
                if v202:
                    v203 = " ..."
                    print(v40.format(v203),end="")
                    del v203
                    break
                else:
                    pass
                del v202
                v204 = v199 == 0
                v205 = v204 != True
                del v204
                if v205:
                    v206 = "; "
                    print(v40.format(v206),end="")
                    del v206
                else:
                    pass
                del v205
                v207 = v182 + 1
                v182 = v207
                del v207
                v208 = v183 * 2
                v209 = v191 * 2
                v210 = v208 + v209
                del v208, v209
                v211 = v210 + v199
                del v210
                v212 = v153[v211].item()
                del v211
                v213 = "{:.6f}"
                print(v213.format(v212),end="")
                del v212, v213
                v199 += 1 
            del v199
            print(v40.format(']'),end="")
            v191 += 1 
        del v191
        print(v40.format(']'),end="")
        v183 += 1 
    del v153, v182, v183
    print(v40.format(']'),end="")
    del v40
    print(v62,end="")
    del v62
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
