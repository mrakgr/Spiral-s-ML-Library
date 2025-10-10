#include "dictm.auto.cu"
#include <thrust/device_vector.h>
#include <unordered_map>
typedef int (* Fun0)(int);
typedef bool (* Fun1)(int, int);
int FunPointerMethod0(int tup0){
    int v0 = tup0;
    int v1;
    v1 = std::hash<int>()(v0);
    return v1;
}
bool FunPointerMethod1(int tup0, int tup1){
    int v0 = tup0; int v1 = tup1;
    bool v2;
    v2 = v0 == v1;
    return v2;
}
inline bool while_method_0(std::unordered_map<int, const char *, Fun0, Fun1> & v0, std::unordered_map<int, const char *, Fun0, Fun1>::iterator & v1){
    bool v2;
    v2 = v1 != v0.end();
    return v2;
}
int main() {
    Fun0 v0 = FunPointerMethod0;
    Fun1 v1 = FunPointerMethod1;
    std::unordered_map<int, const char *, Fun0, Fun1> v2(8, v0, v1);
    const char * v3;
    v3 = "Hello";
    v2[1] = v3;
    const char * v4;
    v4 = "World";
    v2[2] = v4;
    const char * v5;
    v5 = "!!!";
    v2[3] = v5;
    std::unordered_map<int, const char *, Fun0, Fun1> & v6 = v2;
    auto v7 = v6.begin();
    while (while_method_0(v6, v7)){
        int v9;
        v9 = v7->first;
        const char * v10;
        v10 = v7->second;
        printf("%d, %s\n",v9, v10);
        ++v7;
    }
    return 0;
}
