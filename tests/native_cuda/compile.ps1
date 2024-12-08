$dir = "$env:HOME/Spiral-s-ML-Library/tests/native_cuda"
$file = "nvcc_test"
nvcc $dir/$file.cu `
    -arch=sm_90a `
    '-gencode=arch=compute_90a,code=sm_90a' `
    -std=c++20 `
    -o $dir/out/$file
Start-Process $dir/out/$file -Wait