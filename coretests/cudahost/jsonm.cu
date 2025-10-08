#include "jsonm.auto.cu"
#include <thrust/device_vector.h>
#include "nlohmann/json.hpp"
namespace Device {
}
struct Union0;
struct Union1;
nlohmann::json f_3();
nlohmann::json f_5(int v0);
nlohmann::json f_4(static_array<int,4> v0);
nlohmann::json f_2(Union0 v0);
nlohmann::json f_7(int v0, sptr<Union1> v1);
nlohmann::json f_6(sptr<Union1> v0);
nlohmann::json f_9(unsigned long long v0);
nlohmann::json f_10(bool v0);
nlohmann::json f_8(int v0, unsigned long long v1, bool v2);
nlohmann::json f_11(const char * v0);
nlohmann::json f_12();
nlohmann::json f_1(Union0 v0, sptr<Union1> v1, int v2, unsigned long long v3, bool v4, const char * v5);
nlohmann::json serialize_0(Union0 v0, sptr<Union1> v1, int v2, unsigned long long v3, bool v4, const char * v5);
struct Union0_0 { // None
};
struct Union0_1 { // Some
    static_array<int,4> v0;
    Union0_1(static_array<int,4> t0) : v0(t0) {}
    Union0_1() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // None
        Union0_1 case1; // Some
    };
    unsigned char tag{255};
    Union0() {}
    Union0(Union0_0 t) : tag(0), case0(t) {} // None
    Union0(Union0_1 t) : tag(1), case1(t) {} // Some
    Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // None
            case 1: new (&this->case1) Union0_1(x.case1); break; // Some
        }
    }
    Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Some
        }
    }
    Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{x};
        }
        return *this;
    }
    Union0 & operator=(Union0 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // None
            case 1: this->case1.~Union0_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union1_0 { // Cons
    sptr<Union1> v1;
    int v0;
    Union1_0(int t0, sptr<Union1> t1) : v0(t0), v1(t1) {}
    Union1_0() = delete;
};
struct Union1_1 { // Nil
};
struct Union1 {
    union {
        Union1_0 case0; // Cons
        Union1_1 case1; // Nil
    };
    int refc{0};
    unsigned char tag{255};
    Union1() {}
    Union1(Union1_0 t) : tag(0), case0(t) {} // Cons
    Union1(Union1_1 t) : tag(1), case1(t) {} // Nil
    Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // Cons
            case 1: new (&this->case1) Union1_1(x.case1); break; // Nil
        }
    }
    Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // Cons
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // Nil
        }
    }
    Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Cons
                case 1: this->case1 = x.case1; break; // Nil
            }
        } else {
            this->~Union1();
            new (this) Union1{x};
        }
        return *this;
    }
    Union1 & operator=(Union1 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Cons
                case 1: this->case1 = std::move(x.case1); break; // Nil
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    ~Union1() {
        printf("in destructor\n");
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // Cons
            case 1: this->case1.~Union1_1(); break; // Nil
        }
        this->tag = 255;
    }
};
inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
nlohmann::json f_3(){
    nlohmann::json v0;
    return v0;
}
nlohmann::json f_5(int v0){
    nlohmann::json v1 = v0;
    return v1;
}
nlohmann::json f_4(static_array<int,4> v0){
    nlohmann::json v1;
    int v2;
    v2 = 0;
    while (while_method_0(v2)){
        int v5;
        v5 = v0[v2];
        nlohmann::json v7;
        v7 = f_5(v5);
        v1.push_back(v7);
        v2 += 1 ;
    }
    return v1;
}
nlohmann::json f_2(Union0 v0){
    switch (v0.tag) {
        case 0: { // None
            nlohmann::json v1;
            v1 = f_3();
            const char * v2;
            v2 = "None";
            nlohmann::json v3{v2, 0, v1};
            return v3;
            break;
        }
        case 1: { // Some
            static_array<int,4> v4 = v0.case1.v0;
            nlohmann::json v5;
            v5 = f_4(v4);
            const char * v6;
            v6 = "Some";
            nlohmann::json v7{v6, 1, v5};
            return v7;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
nlohmann::json f_7(int v0, sptr<Union1> v1){
    nlohmann::json v2;
    nlohmann::json v3;
    v3 = f_5(v0);
    v2.push_back(v3);
    nlohmann::json v4;
    v4 = f_6(v1);
    v2.push_back(v4);
    return v2;
}
nlohmann::json f_6(sptr<Union1> v0){
    switch (v0.base->tag) {
        case 0: { // Cons
            int v1 = v0.base->case0.v0; sptr<Union1> v2 = v0.base->case0.v1;
            nlohmann::json v3;
            v3 = f_7(v1, v2);
            const char * v4;
            v4 = "Cons";
            nlohmann::json v5{v4, 0, v3};
            return v5;
            break;
        }
        case 1: { // Nil
            nlohmann::json v6;
            v6 = f_3();
            const char * v7;
            v7 = "Nil";
            nlohmann::json v8{v7, 1, v6};
            return v8;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
nlohmann::json f_9(unsigned long long v0){
    nlohmann::json v1 = v0;
    return v1;
}
nlohmann::json f_10(bool v0){
    nlohmann::json v1 = v0;
    return v1;
}
nlohmann::json f_8(int v0, unsigned long long v1, bool v2){
    nlohmann::json v3;
    nlohmann::json v4;
    v4 = f_5(v0);
    v3.push_back(v4);
    nlohmann::json v5;
    v5 = f_9(v1);
    v3.push_back(v5);
    nlohmann::json v6;
    v6 = f_10(v2);
    v3.push_back(v6);
    return v3;
}
nlohmann::json f_11(const char * v0){
    nlohmann::json v1 = v0;
    return v1;
}
nlohmann::json f_12(){
    nlohmann::json v0 = nlohmann::json::object();
    return v0;
}
nlohmann::json f_1(Union0 v0, sptr<Union1> v1, int v2, unsigned long long v3, bool v4, const char * v5){
    nlohmann::json v6 = nlohmann::json::object();
    const char * v7;
    v7 = "q";
    nlohmann::json v8;
    v8 = f_2(v0);
    v6[v7] = v8;
    const char * v9;
    v9 = "w";
    nlohmann::json v10;
    v10 = f_6(v1);
    v6[v9] = v10;
    const char * v11;
    v11 = "x";
    nlohmann::json v12;
    v12 = f_8(v2, v3, v4);
    v6[v11] = v12;
    const char * v13;
    v13 = "y";
    nlohmann::json v14;
    v14 = f_11(v5);
    v6[v13] = v14;
    const char * v15;
    v15 = "z";
    nlohmann::json v16;
    v16 = f_12();
    v6[v15] = v16;
    return v6;
}
nlohmann::json serialize_0(Union0 v0, sptr<Union1> v1, int v2, unsigned long long v3, bool v4, const char * v5){
    return f_1(v0, v1, v2, v3, v4, v5);
}
int main() {
    static_array<int,4> v1;
    int v3;
    v3 = 0;
    while (while_method_0(v3)){
        v1[v3] = v3;
        v3 += 1 ;
    }
    Union0 v5;
    v5 = Union0{Union0_1{v1}};
    int v6;
    v6 = 11;
    int v7;
    v7 = 22;
    int v8;
    v8 = 33;
    sptr<Union1> v9;
    v9 = sptr<Union1>{new Union1{Union1_1{}}};
    sptr<Union1> v10;
    v10 = sptr<Union1>{new Union1{Union1_0{v8, v9}}};
    sptr<Union1> v11;
    v11 = sptr<Union1>{new Union1{Union1_0{v7, v10}}};
    sptr<Union1> v12;
    v12 = sptr<Union1>{new Union1{Union1_0{v6, v11}}};
    int v13;
    v13 = 1;
    unsigned long long v14;
    v14 = 2ull;
    bool v15;
    v15 = false;
    const char * v16;
    v16 = "hello";
    nlohmann::json v17;
    v17 = serialize_0(v5, v12, v13, v14, v15, v16);
    bool v23;
    v23 = true;
    if (v23){
        auto v24 = v17.dump(4);
        const char * v25;
        v25 = v24.c_str();
        printf("%s",v25);
    } else {
    }
    printf("\n");
    return 0;
}
