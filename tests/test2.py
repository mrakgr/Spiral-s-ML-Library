kernel = r"""
template <typename T>
void call_destructor(T &x) { x.~T(); }
template <typename el, int dim>
struct static_array
{
    el v[dim];
};
template <typename el, int dim, typename default_int>
struct static_array_list
{
    default_int length;
    el v[dim];
};

template <typename T>
struct sptr // Shared pointer for the Spiral datatypes. They have to have the refc field inside them to work.
{
    T *base;

    __device__ sptr() : base(nullptr) {}
    __device__ sptr(T *v) : base(v) { this->base->refc++; }

    __device__ ~sptr()
    {
        if (this->base != nullptr && --this->base->refc == 0)
        {
            delete this->base;
        }
    }

    __device__ sptr(sptr &x)
    {
        this->base = x.base;
        this->base->refc++;
    }

    __device__ sptr(sptr &&x)
    {
        this->base = x.base;
        x.base = nullptr;
    }

    __device__ sptr &operator=(sptr &x)
    {
        if (this->base != x.base)
        {
            delete this->base;
            this->base = x.base;
            this->base->refc++;
        }
        return *this;
    }

    __device__ sptr &operator=(sptr &&x)
    {
        if (this->base != x.base)
        {
            delete this->base;
            this->base = x.base;
            x.base = nullptr;
        }
        return *this;
    }
};

template <typename T>
struct csptr : public sptr<T>
{ // Shared pointer for closures specifically.
    using sptr<T>::sptr;
    template <typename... Args>
    __device__ auto operator()(Args... args) -> decltype(this->base->operator()(args...))
    {
        return this->base->operator()(args...);
    }
};

struct Union0;
struct Union0_0 { // Cons
    sptr<Union0> v1;
    long v0;
    __device__ Union0_0(long t0, sptr<Union0> t1) : v0(t0), v1(t1) {}
    __device__ Union0_0() = delete;
};
struct Union0_1 { // Nil
};
struct Union0 {
    union {
        Union0_0 case0; // Cons
        Union0_1 case1; // Nil
    };
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // Cons
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Nil
    __device__ Union0() = delete;
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // Cons
            case 1: new (&this->case1) Union0_1(x.case1); break; // Nil
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // Cons
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Nil
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // Cons
            case 1: this->case1 = x.case1; break; // Nil
        }
        return *this;
    }
    __device__ Union0 & operator=(Union0 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // Cons
            case 1: this->case1 = std::move(x.case1); break; // Nil
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // Cons
            case 1: this->case1.~Union0_1(); break; // Nil
        }
    }
    int refc = 0;
    unsigned char tag;
};
extern "C" __global__ void entry0() {
    long v0;
    v0 = threadIdx.x;
    long v1;
    v1 = blockIdx.x;
    long v2;
    v2 = v1 * 32l;
    long v3;
    v3 = v0 + v2;
    bool v4;
    v4 = v3 == 0l;
    if (v4){
        long v5;
        v5 = 1l;
        long v6;
        v6 = 2l;
        long v7;
        v7 = 3l;
        sptr<Union0> v8;
        v8 = sptr<Union0>{new Union0{Union0_1{}}};
        sptr<Union0> v9;
        v9 = sptr<Union0>{new Union0{Union0_0{v7, v8}}};
        sptr<Union0> v10;
        v10 = sptr<Union0>{new Union0{Union0_0{v6, v9}}};
        sptr<Union0> v11;
        v11 = sptr<Union0>{new Union0{Union0_0{v5, v10}}};
        return ;
    } else {
        return ;
    }
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
# it compiles with nvcc
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
