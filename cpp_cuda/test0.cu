#include "test0.auto.cu"
namespace Device {
    extern "C" __global__ void entry0() {        int v0;        v0 = 5;        int v1;        v1 = 3;        int v2;        v2 = v0 + v1;        return ;    }}
int main() {
    int v0;
    v0 = 2;
    int v1;
    v1 = 8;
    int v2;
    v2 = v0 * v1;
    int v3;
    v3 = 0;
    return 0;
}
