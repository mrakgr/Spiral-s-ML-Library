kernel = r"""
extern "C" __global__ void __cluster_dims__(8,1,1) entry0() {
    return ;
}
"""
import cupy as cp

raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True)
def main():
    v9 = raw_module.get_function(f"entry0")
    v9((128,),(256,),())
    return 

if __name__ == '__main__': print(main())