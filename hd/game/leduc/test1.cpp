#include "test1.hpp"
void run_cuda_host_from_cpp_host_1(){
    auto kernel = cuda_host_entry0;
    kernel();
    return ;
}
int main() {
    run_cuda_host_from_cpp_host_1();
    return 0;
}
