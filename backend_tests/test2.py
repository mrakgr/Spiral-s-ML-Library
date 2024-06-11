kernel = r"""
extern "C" __global__ void entry0() {
    if (threadIdx.x == 0)
        malloc(16);
    return ;
}
"""

import cupy as cp
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True)
def main():
    v1 = raw_module.get_function(f"entry0")
    v1((2,),(32,),(),shared_mem=0)
    cp.cuda.get_current_stream().synchronize()

if __name__ == '__main__': print(main())