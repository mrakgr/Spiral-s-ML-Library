#include "game.auto.cu"
#include <thrust/device_vector.h>
#include <unordered_map>
namespace Device {
}
typedef int (* Fun0)(int, bool, long long);
struct Tuple0;
typedef bool (* Fun1)(Tuple0, int, bool, long long);
struct Tuple0 {
    long long v2;
    int v0;
    bool v1;
    Tuple0() = default;
    Tuple0(int t0, bool t1, long long t2) : v0(t0), v1(t1), v2(t2) {}
};
int FunPointerMethod0(int tup0, bool tup1, long long tup2){
    int v0 = tup0; bool v1 = tup1; long long v2 = tup2;
    int v3;
    v3 = std::hash<int>()(v0);
    int v4;
    v4 = v3 * 9973;
    int v5;
    v5 = std::hash<bool>()(v1);
    int v6;
    v6 = v5 * 9973;
    int v7;
    v7 = std::hash<long long>()(v2);
    int v8;
    v8 = v6 + v7;
    int v9;
    v9 = v4 + v8;
    return v9;
}
bool FunPointerMethod1(Tuple0 tup0, int tup1, bool tup2, long long tup3){
    int v0 = tup0.v0; bool v1 = tup0.v1; long long v2 = tup0.v2; int v3 = tup1; bool v4 = tup2; long long v5 = tup3;
    bool v6;
    v6 = v0 == v3;
    if (v6){
        bool v7;
        v7 = v1 == v4;
        if (v7){
            bool v8;
            v8 = v2 == v5;
            return v8;
        } else {
            return false;
        }
    } else {
        return false;
    }
}
int main() {
    int v0;
    v0 = 1;
    bool v1;
    v1 = true;
    long long v2;
    v2 = 3ll;
    const char * v3;
    v3 = "asdf";
    Fun0 v4 = FunPointerMethod0;
    Fun1 v5 = FunPointerMethod1;
    std::unordered_map<Tuple0, const char *, Fun0, Fun1> v6(8, v4, v5);
    std::unordered_map<Tuple0, const char *, Fun0, Fun1> v7(16, v4, v5);
    return 0;
}
