# 6/1/2025
# Saving this for later here.
# I finally managed to compile Cutlass example directly with my own command.
# Since I include the -g and -G flags, that means I should be able to run it in a debugger.
# This is a major accomplishment.
# I hate the Cutlass library so much for being such an overcomplicated mess, but I am making progress on using it.
# Now that I've come to this point, thanks to the debugger, I should be able to properly study it, and thanks
# to this script, I'll be able to use the library itself in my own projects.

# 6/8/2025
# Tried running the example in a debugger, and it's incredibly unstable. It crashed my OS several times requiring a manual reset, 
# so I give up on this approach. Learning Cutlass by stepping through the examples one by one would have been a great idea
# had it actually worked. I guess this is about what you'd expect from C++ applications.
# Dynamic languages like Python have great debuggers so I've been spoiled by them.

# It's always two steps forward and one step back when it comes to Cutlass.
# I am going to have to give up on this project until my paid job is over, but when I resume
# I might try putting print statements in and debugging the system the poor man's way.
# Now that I can compile the examples manually, I'll be able to use them as references and
# makes some progress on using the device kernels in my own projects.

# I'll probably build my own kernel in the end for the Hopfield Dictionary project.
# Until I have time to do this full time again, I think I'll work on stock trading systems during my Sundays.

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
    -o /home/mrakgr/Spiral-s-ML-Library/cpp_libs/cutlass/examples/12_gemm_bias_relu/gemm_bias_relu `
    /home/mrakgr/Spiral-s-ML-Library/cpp_libs/cutlass/examples/12_gemm_bias_relu/gemm_bias_relu.cu