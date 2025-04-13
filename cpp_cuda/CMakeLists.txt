cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(cuda_project LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_STANDARD_REQUIRED 12.8)
set(CMAKE_CUDA_FLAGS "-D=NDEBUG -g -G -dopt=on -restrict -expt-relaxed-constexpr -D__CUDA_NO_HALF_CONVERSIONS__ -maxrregcount=255 -diag-suppress 550,20012,68,39,177")
set(CMAKE_CUDA_LINKER_FLAGS "${CMAKE_CUDA_LINKER_FLAGS} -L/usr/local/cuda/lib64 -lcudart_static")

# Find the CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Add the executable
add_executable(cuda_project test1.cu)

# Link the CUDA runtime library
target_link_libraries(cuda_project CUDA::cudart)

