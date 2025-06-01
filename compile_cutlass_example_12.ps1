# Saving this for later here.
# I finally managed to compile Cutlass example directly with my own command.
# Since I include the -g and -G flags, that means I should be able to run it in a debugger.
# This is a major accomplishment.
# I hate the Cutlass library so much for being such an overcomplicated mess, but I am making progress on using it.
# Now that I've come to this point, thanks to the debugger, I should be able to properly study it, and thanks
# to this script, I'll be able to use the library itself in my own projects.
nvcc `
    -I"/home/mrakgr/Spiral-s-ML-Library/cpp_libs/cutlass/include" `
    -I"/home/mrakgr/Spiral-s-ML-Library/cpp_libs/cutlass/examples/common" `
    -I"/home/mrakgr/Spiral-s-ML-Library/cpp_libs/cutlass/build/include" `
    -I"/home/mrakgr/Spiral-s-ML-Library/cpp_libs/cutlass/tools/util/include" `
    -isystem /usr/local/cuda/include `
    -isystem /usr/local/cuda/include/cccl `
    -arch=sm_120a `
    -D=NDEBUG `
    -g -G `
    -dopt=on `
    -restrict `
    -expt-relaxed-constexpr `
    -D__CUDA_NO_HALF_CONVERSIONS__ `
    -std=c++20 `
    -o /home/mrakgr/Spiral-s-ML-Library/cpp_libs/cutlass/build/examples/12_gemm_bias_relu/12_gemm_bias_relu `
    /home/mrakgr/Spiral-s-ML-Library/cpp_libs/cutlass/examples/12_gemm_bias_relu/gemm_bias_relu.cu