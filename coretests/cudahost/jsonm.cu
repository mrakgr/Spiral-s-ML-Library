#include "jsonm.auto.cu"
#include <thrust/device_vector.h>
#include "nlohmann/json.hpp"
namespace Device {
}
nlohmann::json f_3(int v0);
nlohmann::json f_4(unsigned long long v0);
nlohmann::json f_5(bool v0);
nlohmann::json f_2(int v0, unsigned long long v1, bool v2);
nlohmann::json f_6(const char * v0);
nlohmann::json f_7();
nlohmann::json f_1(int v0, unsigned long long v1, bool v2, const char * v3);
nlohmann::json serialize_0(int v0, unsigned long long v1, bool v2, const char * v3);
nlohmann::json f_3(int v0){
    nlohmann::json v1 = v0;
    return v1;
}
nlohmann::json f_4(unsigned long long v0){
    nlohmann::json v1 = v0;
    return v1;
}
nlohmann::json f_5(bool v0){
    nlohmann::json v1 = v0;
    return v1;
}
nlohmann::json f_2(int v0, unsigned long long v1, bool v2){
    nlohmann::json v3;
    nlohmann::json v4;
    v4 = f_3(v0);
    v3.push_back(v4);
    nlohmann::json v5;
    v5 = f_4(v1);
    v3.push_back(v5);
    nlohmann::json v6;
    v6 = f_5(v2);
    v3.push_back(v6);
    return v3;
}
nlohmann::json f_6(const char * v0){
    nlohmann::json v1 = v0;
    return v1;
}
nlohmann::json f_7(){
    nlohmann::json v0 = nlohmann::json::object();
    return v0;
}
nlohmann::json f_1(int v0, unsigned long long v1, bool v2, const char * v3){
    nlohmann::json v4 = nlohmann::json::object();
    const char * v5;
    v5 = "x";
    nlohmann::json v6;
    v6 = f_2(v0, v1, v2);
    v4[v5] = v6;
    const char * v7;
    v7 = "y";
    nlohmann::json v8;
    v8 = f_6(v3);
    v4[v7] = v8;
    const char * v9;
    v9 = "z";
    nlohmann::json v10;
    v10 = f_7();
    v4[v9] = v10;
    return v4;
}
nlohmann::json serialize_0(int v0, unsigned long long v1, bool v2, const char * v3){
    return f_1(v0, v1, v2, v3);
}
int main() {
    int v0;
    v0 = 1;
    unsigned long long v1;
    v1 = 2ull;
    bool v2;
    v2 = false;
    const char * v3;
    v3 = "hello";
    nlohmann::json v4;
    v4 = serialize_0(v0, v1, v2, v3);
    bool v10;
    v10 = true;
    if (v10){
        auto v11 = v4.dump(4);
        const char * v12;
        v12 = v11.c_str();
        printf("%s",v12);
    } else {
    }
    printf("\n");
    return 0;
}
