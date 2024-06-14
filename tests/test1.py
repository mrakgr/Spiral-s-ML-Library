kernel = r"""
template <typename T> void call_destructor(T & x) { x.~T(); }
template <typename el, int dim> struct static_array { el v[dim]; };
template <typename el, int dim, typename default_int> struct static_array_list { default_int length; el v[dim]; };

template <typename T>
struct sptr
{
    struct sptr_base {
        int refc;
        T v;

        sptr_base() = delete;
        sptr_base(T & v_) : refc(1), v(v_) {};
        sptr_base(T && v_) : refc(1), v(std::move(v_)) {};

        ~sptr_base(){ this->v.dispose(); }
    } * base;

    sptr() : base(nullptr) {}
    sptr(T & v) : base(new sptr_base(v)) {}
    sptr(T && v) : base(new sptr_base(std::move(v))) {}

    void dispose(){
        if (this->base != nullptr && --this->base->refc == 0)
        {
            delete this->base;
        }
    }

    ~sptr() { this->dispose(); }

    sptr(sptr & x) {
        this->base = x.base;
        this->base->refc++;
    }

    sptr(sptr && x) {
        this->base = x.base;
        x.base = nullptr;
    }

    sptr & operator=(sptr &x)
    {
        if (this->base != x.base){
            this->dispose();
            this->base = x.base;
            this->base->refc++;
        }
        return *this;
    }
    
    sptr & operator=(sptr &&x)
    {
        if (this->base != x.base){
            this->dispose();
            this->base = x.base;
            x.base = nullptr;
        }
        return *this;
    }
};

template <typename T, typename default_int>
struct array {
    default_int length = 0;
    T * ptr = nullptr;

    array(int l) : length(l), ptr(new T[l]) { }
    void dispose(){ delete[] this->ptr; }
};

struct Closure0 {
    long v0;
    __device__ long operator() (long tup0){
        long & v0 = this->v0;
        long v1 = tup0;
        long v2;
        v2 = v1 + v0;
        return v2;
    }
    Closure0(long _v0) : v0(_v0) { }
}
extern "C" __global__ void entry0() {
    long v0;
    v0 = 2l;
    Closure0 v1{v0};
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
options.append('--diag-suppress=550,20012')
options.append('--dopt=on')
options.append('--restrict')
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
