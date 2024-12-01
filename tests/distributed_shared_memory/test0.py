kernel = r"""
extern "C" __global__ void __cluster_dims__(4,1,1) entry0() {
    return ;
}
"""
import cupy as cp

raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True)
def main_body():
    v9 = raw_module.get_function(f"entry0")
    v9.max_dynamic_shared_size_bytes = 229376 
    print(f'Threads per block, blocks per grid: {256}, {128}')
    v9((128,),(256,),(),shared_mem=229376)
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
