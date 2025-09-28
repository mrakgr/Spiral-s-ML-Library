#include "dictm.auto.cu"
#include <thrust/device_vector.h>
#include <unordered_map>
namespace Device {
}
struct Tuple0;
typedef int (* Fun0)(Tuple0);
typedef bool (* Fun1)(Tuple0, Tuple0);
struct Union0;
void method_0(Union0 v0);
struct Tuple0 {
    int v0;
    bool v1;
    Tuple0() = default;
    Tuple0(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Union0_0 { // None
};
struct Union0_1 { // Some
    const char * v0;
    Union0_1(const char * t0) : v0(t0) {}
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
int FunPointerMethod0(Tuple0 tup0){
    int v0 = tup0.v0; bool v1 = tup0.v1;
    int v2;
    v2 = std::hash<int>()(v0);
    int v3;
    v3 = v2 * 9973;
    int v4;
    v4 = std::hash<bool>()(v1);
    int v5;
    v5 = v3 + v4;
    return v5;
}
bool FunPointerMethod1(Tuple0 tup0, Tuple0 tup1){
    int v0 = tup0.v0; bool v1 = tup0.v1; int v2 = tup1.v0; bool v3 = tup1.v1;
    bool v4;
    v4 = v0 == v2;
    if (v4){
        bool v5;
        v5 = v1 == v3;
        return v5;
    } else {
        return false;
    }
}
void method_0(Union0 v0){
    switch (v0.tag) {
        case 0: { // None
            printf("%s","None");
            return ;
            break;
        }
        case 1: { // Some
            const char * v1 = v0.case1.v0;
            printf("%s(%s)","Some", v1);
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
int main() {
    Fun0 v0 = FunPointerMethod0;
    Fun1 v1 = FunPointerMethod1;
    std::unordered_map<Tuple0, const char *, Fun0, Fun1> v2(8, v0, v1);
    const char * v3;
    v3 = "Hello";
    v2[Tuple0{1, true}] = v3;
    const char * v4;
    v4 = "World";
    v2[Tuple0{2, false}] = v4;
    auto v5 = v2.find(Tuple0{1, true});
    bool v6;
    v6 = v5 != v2.end();
    Union0 v10;
    if (v6){
        const char * v7;
        v7 = v5->second;
        v10 = Union0{Union0_1{v7}};
    } else {
        v10 = Union0{Union0_0{}};
    }
    printf("");
    method_0(v10);
    printf("\n");
    auto v15 = v2.find(Tuple0{2, false});
    bool v16;
    v16 = v15 != v2.end();
    Union0 v20;
    if (v16){
        const char * v17;
        v17 = v15->second;
        v20 = Union0{Union0_1{v17}};
    } else {
        v20 = Union0{Union0_0{}};
    }
    printf("");
    method_0(v20);
    printf("\n");
    auto v25 = v2.find(Tuple0{3, true});
    bool v26;
    v26 = v25 != v2.end();
    Union0 v30;
    if (v26){
        const char * v27;
        v27 = v25->second;
        v30 = Union0{Union0_1{v27}};
    } else {
        v30 = Union0{Union0_0{}};
    }
    printf("");
    method_0(v30);
    printf("\n");
    bool v35;
    v35 = v2.contains(Tuple0{2, false});
    const char * v43;
    if (v35){
        const char * v41;
        v41 = "true";
        v43 = v41;
    } else {
        const char * v42;
        v42 = "false";
        v43 = v42;
    }
    printf("%s\n",v43);
    bool v48;
    v48 = v2.contains(Tuple0{3, true});
    const char * v56;
    if (v48){
        const char * v54;
        v54 = "true";
        v56 = v54;
    } else {
        const char * v55;
        v55 = "false";
        v56 = v55;
    }
    printf("%s\n",v56);
    bool v61;
    v61 = static_cast<bool>(v2.erase(Tuple0{2, false}));
    const char * v69;
    if (v61){
        const char * v67;
        v67 = "true";
        v69 = v67;
    } else {
        const char * v68;
        v68 = "false";
        v69 = v68;
    }
    printf("%s, %s\n","Removing", v69);
    bool v75;
    v75 = v2.contains(Tuple0{2, false});
    const char * v83;
    if (v75){
        const char * v81;
        v81 = "true";
        v83 = v81;
    } else {
        const char * v82;
        v82 = "false";
        v83 = v82;
    }
    printf("%s, %s\n","Contains", v83);
    return 0;
}
