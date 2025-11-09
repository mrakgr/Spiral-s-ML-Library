$gpp_args = @(
    "arrayfire_max_dict_example.cpp"
    "-I/home/mrakgr/ArrayFire-3.10.0-Linux/include" # Arrayfire include
    "-L/home/mrakgr/ArrayFire-3.10.0-Linux/lib64", "-lafcpu" # Arrayfire library
    "-o", "arrayfire_max_dict_example"
)

g++ $gpp_args