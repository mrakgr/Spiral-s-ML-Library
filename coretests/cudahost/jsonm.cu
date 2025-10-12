#include "jsonm.auto.cu"
#include <thrust/device_vector.h>
#include <unordered_map>
#include "nlohmann/json.hpp"
struct Union1;
struct Union2;
struct Union0;
struct Tuple0;
typedef unsigned long long (* Fun0)(static_array_list<Union0,32>);
typedef bool (* Fun1)(static_array_list<Union0,32>, static_array_list<Union0,32>);
struct Tuple1;
nlohmann::json method_5();
nlohmann::json method_4(Union1 v0);
nlohmann::json method_7(int v0);
nlohmann::json method_8(Union2 v0);
nlohmann::json method_6(int v0, Union2 v1);
nlohmann::json method_9(int v0, Union1 v1);
nlohmann::json method_11(static_array<Union1,2> v0);
nlohmann::json method_10(static_array<Union1,2> v0, int v1, int v2);
nlohmann::json method_3(Union0 v0);
nlohmann::json method_2(static_array_list<Union0,32> v0);
nlohmann::json method_14(float v0);
nlohmann::json method_13(static_array<float,3> v0);
nlohmann::json method_12(static_array<float,3> v0, static_array<float,3> v1, static_array<float,3> v2);
nlohmann::json method_1(std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> v0);
nlohmann::json serialize_0(std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> v0);
void f_20(nlohmann::json v0);
Union1 f_19(nlohmann::json v0);
struct Tuple2;
int f_22(nlohmann::json v0);
Union2 f_23(nlohmann::json v0);
Tuple2 f_21(nlohmann::json v0);
struct Tuple3;
Tuple3 f_24(nlohmann::json v0);
struct Tuple4;
static_array<Union1,2> f_26(nlohmann::json v0);
Tuple4 f_25(nlohmann::json v0);
Union0 f_18(nlohmann::json v0);
static_array_list<Union0,32> f_17(nlohmann::json v0);
float f_29(nlohmann::json v0);
static_array<float,3> f_28(nlohmann::json v0);
Tuple1 f_27(nlohmann::json v0);
std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> f_16(nlohmann::json v0);
std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> deserialize_15(nlohmann::json v0);
void method_31(Union1 v0);
void method_32(Union2 v0);
void method_30(Union0 v0);
struct Union1_0 { // Jack
};
struct Union1_1 { // King
};
struct Union1_2 { // Queen
};
struct Union1 {
    union {
        Union1_0 case0; // Jack
        Union1_1 case1; // King
        Union1_2 case2; // Queen
    };
    unsigned char tag{255};
    __host__ __device__ Union1() {}
    __host__ __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // Jack
    __host__ __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // King
    __host__ __device__ Union1(Union1_2 t) : tag(2), case2(t) {} // Queen
    __host__ __device__ Union1(const Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union1_1(x.case1); break; // King
            case 2: new (&this->case2) Union1_2(x.case2); break; // Queen
        }
    }
    __host__ __device__ Union1(const Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // Queen
        }
    }
    __host__ __device__ Union1 & operator=(const Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Jack
                case 1: this->case1 = x.case1; break; // King
                case 2: this->case2 = x.case2; break; // Queen
            }
        } else {
            this->~Union1();
            new (this) Union1{x};
        }
        return *this;
    }
    __host__ __device__ Union1 & operator=(const Union1 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Jack
                case 1: this->case1 = std::move(x.case1); break; // King
                case 2: this->case2 = std::move(x.case2); break; // Queen
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // Jack
            case 1: this->case1.~Union1_1(); break; // King
            case 2: this->case2.~Union1_2(); break; // Queen
        }
        this->tag = 255;
    }
};
struct Union2_0 { // Call
};
struct Union2_1 { // Fold
};
struct Union2_2 { // Raise
};
struct Union2 {
    union {
        Union2_0 case0; // Call
        Union2_1 case1; // Fold
        Union2_2 case2; // Raise
    };
    unsigned char tag{255};
    __host__ __device__ Union2() {}
    __host__ __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // Call
    __host__ __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // Fold
    __host__ __device__ Union2(Union2_2 t) : tag(2), case2(t) {} // Raise
    __host__ __device__ Union2(const Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // Call
            case 1: new (&this->case1) Union2_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union2_2(x.case2); break; // Raise
        }
    }
    __host__ __device__ Union2(const Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union2_2(std::move(x.case2)); break; // Raise
        }
    }
    __host__ __device__ Union2 & operator=(const Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Call
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // Raise
            }
        } else {
            this->~Union2();
            new (this) Union2{x};
        }
        return *this;
    }
    __host__ __device__ Union2 & operator=(const Union2 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Call
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // Raise
            }
        } else {
            this->~Union2();
            new (this) Union2{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // Call
            case 1: this->case1.~Union2_1(); break; // Fold
            case 2: this->case2.~Union2_2(); break; // Raise
        }
        this->tag = 255;
    }
};
struct Union0_0 { // CommunityCardIs
    Union1 v0;
    __host__ __device__ Union0_0(Union1 t0) : v0(t0) {}
    __host__ __device__ Union0_0() = delete;
};
struct Union0_1 { // PlayerAction
    Union2 v1;
    int v0;
    __host__ __device__ Union0_1(int t0, Union2 t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union0_1() = delete;
};
struct Union0_2 { // PlayerGotCard
    Union1 v1;
    int v0;
    __host__ __device__ Union0_2(int t0, Union1 t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union0_2() = delete;
};
struct Union0_3 { // Showdown
    static_array<Union1,2> v0;
    int v1;
    int v2;
    __host__ __device__ Union0_3(static_array<Union1,2> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    __host__ __device__ Union0_3() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // CommunityCardIs
        Union0_1 case1; // PlayerAction
        Union0_2 case2; // PlayerGotCard
        Union0_3 case3; // Showdown
    };
    unsigned char tag{255};
    __host__ __device__ Union0() {}
    __host__ __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // CommunityCardIs
    __host__ __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // PlayerAction
    __host__ __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // PlayerGotCard
    __host__ __device__ Union0(Union0_3 t) : tag(3), case3(t) {} // Showdown
    __host__ __device__ Union0(const Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // CommunityCardIs
            case 1: new (&this->case1) Union0_1(x.case1); break; // PlayerAction
            case 2: new (&this->case2) Union0_2(x.case2); break; // PlayerGotCard
            case 3: new (&this->case3) Union0_3(x.case3); break; // Showdown
        }
    }
    __host__ __device__ Union0(const Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // CommunityCardIs
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // PlayerAction
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // PlayerGotCard
            case 3: new (&this->case3) Union0_3(std::move(x.case3)); break; // Showdown
        }
    }
    __host__ __device__ Union0 & operator=(const Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CommunityCardIs
                case 1: this->case1 = x.case1; break; // PlayerAction
                case 2: this->case2 = x.case2; break; // PlayerGotCard
                case 3: this->case3 = x.case3; break; // Showdown
            }
        } else {
            this->~Union0();
            new (this) Union0{x};
        }
        return *this;
    }
    __host__ __device__ Union0 & operator=(const Union0 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // CommunityCardIs
                case 1: this->case1 = std::move(x.case1); break; // PlayerAction
                case 2: this->case2 = std::move(x.case2); break; // PlayerGotCard
                case 3: this->case3 = std::move(x.case3); break; // Showdown
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // CommunityCardIs
            case 1: this->case1.~Union0_1(); break; // PlayerAction
            case 2: this->case2.~Union0_2(); break; // PlayerGotCard
            case 3: this->case3.~Union0_3(); break; // Showdown
        }
        this->tag = 255;
    }
};
struct Tuple0 {
    unsigned long long v1;
    unsigned long long v2;
    int v0;
    __host__ __device__ Tuple0() = default;
    __host__ __device__ Tuple0(int t0, unsigned long long t1, unsigned long long t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple1 {
    static_array<float,3> v0;
    static_array<float,3> v1;
    static_array<float,3> v2;
    __host__ __device__ Tuple1() = default;
    __host__ __device__ Tuple1(static_array<float,3> t0, static_array<float,3> t1, static_array<float,3> t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple2 {
    Union2 v1;
    int v0;
    __host__ __device__ Tuple2() = default;
    __host__ __device__ Tuple2(int t0, Union2 t1) : v0(t0), v1(t1) {}
};
struct Tuple3 {
    Union1 v1;
    int v0;
    __host__ __device__ Tuple3() = default;
    __host__ __device__ Tuple3(int t0, Union1 t1) : v0(t0), v1(t1) {}
};
struct Tuple4 {
    static_array<Union1,2> v0;
    int v1;
    int v2;
    __host__ __device__ Tuple4() = default;
    __host__ __device__ Tuple4(static_array<Union1,2> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
inline bool while_method_0(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
unsigned long long FunPointerMethod0(static_array_list<Union0,32> tup0){
    static_array_list<Union0,32> v0 = tup0;
    int v1;
    v1 = v0.length;
    int v2; unsigned long long v3; unsigned long long v4;
    Tuple0 tmp0 = Tuple0{0, 0ull, 1ull};
    v2 = tmp0.v0; v3 = tmp0.v1; v4 = tmp0.v2;
    while (while_method_0(v1, v2)){
        Union0 v7;
        v7 = v0[v2];
        unsigned long long v52;
        switch (v7.tag) {
            case 0: { // CommunityCardIs
                Union1 v9 = v7.case0.v0;
                unsigned long long v10;
                switch (v9.tag) {
                    case 0: { // Jack
                        v10 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v10 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v10 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v11;
                v11 = 9223372036854775807ull + v10;
                unsigned long long v12;
                v12 = v11 * 9973ull;
                v52 = v12;
                break;
            }
            case 1: { // PlayerAction
                int v13 = v7.case1.v0; Union2 v14 = v7.case1.v1;
                unsigned long long v15;
                v15 = std::hash<int>()(v13);
                unsigned long long v16;
                v16 = v15 * 9973ull;
                unsigned long long v17;
                switch (v14.tag) {
                    case 0: { // Call
                        v17 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v17 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v17 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v18;
                v18 = v16 + v17;
                unsigned long long v19;
                v19 = 9223372036854775807ull + v18;
                unsigned long long v20;
                v20 = v19 * 9973ull;
                unsigned long long v21;
                v21 = v20 * 2ull;
                v52 = v21;
                break;
            }
            case 2: { // PlayerGotCard
                int v22 = v7.case2.v0; Union1 v23 = v7.case2.v1;
                unsigned long long v24;
                v24 = std::hash<int>()(v22);
                unsigned long long v25;
                v25 = v24 * 9973ull;
                unsigned long long v26;
                switch (v23.tag) {
                    case 0: { // Jack
                        v26 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v26 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v26 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v27;
                v27 = v25 + v26;
                unsigned long long v28;
                v28 = 9223372036854775807ull + v27;
                unsigned long long v29;
                v29 = v28 * 9973ull;
                unsigned long long v30;
                v30 = v29 * 3ull;
                v52 = v30;
                break;
            }
            case 3: { // Showdown
                static_array<Union1,2> v31 = v7.case3.v0; int v32 = v7.case3.v1; int v33 = v7.case3.v2;
                unsigned long long v34;
                v34 = std::hash<int>()(v33);
                unsigned long long v35;
                v35 = std::hash<int>()(v32);
                unsigned long long v36;
                v36 = v35 * 9973ull;
                unsigned long long v37;
                v37 = v34 + v36;
                int v38; unsigned long long v39; unsigned long long v40;
                Tuple0 tmp1 = Tuple0{0, 0ull, 1ull};
                v38 = tmp1.v0; v39 = tmp1.v1; v40 = tmp1.v2;
                while (while_method_1(v38)){
                    Union1 v43;
                    v43 = v31[v38];
                    unsigned long long v45;
                    switch (v43.tag) {
                        case 0: { // Jack
                            v45 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v45 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v45 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v46;
                    v46 = v45 * v40;
                    unsigned long long v47;
                    v47 = v39 + v46;
                    unsigned long long v48;
                    v48 = v40 * 9973ull;
                    v39 = v47;
                    v40 = v48;
                    v38 += 1 ;
                }
                unsigned long long v49;
                v49 = 9223372036854775807ull + v37;
                unsigned long long v50;
                v50 = v49 * 9973ull;
                unsigned long long v51;
                v51 = v50 * 4ull;
                v52 = v51;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v53;
        v53 = v52 * v4;
        unsigned long long v54;
        v54 = v3 + v53;
        unsigned long long v55;
        v55 = v4 * 9973ull;
        v3 = v54;
        v4 = v55;
        v2 += 1 ;
    }
    return 0ull;
}
bool FunPointerMethod1(static_array_list<Union0,32> tup0, static_array_list<Union0,32> tup1){
    static_array_list<Union0,32> v0 = tup0; static_array_list<Union0,32> v1 = tup1;
    int v2;
    v2 = v0.length;
    int v3;
    v3 = v1.length;
    bool v4;
    v4 = v2 == v3;
    if (v4){
        bool v5;
        v5 = true;
        int v6;
        v6 = v1.length;
        int v7;
        v7 = 0;
        while (while_method_0(v6, v7)){
            Union0 v10;
            v10 = v0[v7];
            Union0 v13;
            v13 = v1[v7];
            bool v54;
            switch (v10.tag == v13.tag ? v10.tag : 255) {
                case 0: { // CommunityCardIs
                    Union1 v15 = v10.case0.v0;
                    Union1 v16 = v13.case0.v0;
                    switch (v15.tag == v16.tag ? v15.tag : 255) {
                        case 0: { // Jack
                            v54 = true;
                            break;
                        }
                        case 1: { // King
                            v54 = true;
                            break;
                        }
                        case 2: { // Queen
                            v54 = true;
                            break;
                        }
                        default: {
                            v54 = false;
                        }
                    }
                    break;
                }
                case 1: { // PlayerAction
                    int v18 = v10.case1.v0; Union2 v19 = v10.case1.v1;
                    int v20 = v13.case1.v0; Union2 v21 = v13.case1.v1;
                    bool v22;
                    v22 = v18 == v20;
                    if (v22){
                        switch (v19.tag == v21.tag ? v19.tag : 255) {
                            case 0: { // Call
                                v54 = true;
                                break;
                            }
                            case 1: { // Fold
                                v54 = true;
                                break;
                            }
                            case 2: { // Raise
                                v54 = true;
                                break;
                            }
                            default: {
                                v54 = false;
                            }
                        }
                    } else {
                        v54 = false;
                    }
                    break;
                }
                case 2: { // PlayerGotCard
                    int v25 = v10.case2.v0; Union1 v26 = v10.case2.v1;
                    int v27 = v13.case2.v0; Union1 v28 = v13.case2.v1;
                    bool v29;
                    v29 = v25 == v27;
                    if (v29){
                        switch (v26.tag == v28.tag ? v26.tag : 255) {
                            case 0: { // Jack
                                v54 = true;
                                break;
                            }
                            case 1: { // King
                                v54 = true;
                                break;
                            }
                            case 2: { // Queen
                                v54 = true;
                                break;
                            }
                            default: {
                                v54 = false;
                            }
                        }
                    } else {
                        v54 = false;
                    }
                    break;
                }
                case 3: { // Showdown
                    static_array<Union1,2> v32 = v10.case3.v0; int v33 = v10.case3.v1; int v34 = v10.case3.v2;
                    static_array<Union1,2> v35 = v13.case3.v0; int v36 = v13.case3.v1; int v37 = v13.case3.v2;
                    bool v38;
                    v38 = true;
                    int v39;
                    v39 = 0;
                    while (while_method_1(v39)){
                        Union1 v42;
                        v42 = v32[v39];
                        Union1 v45;
                        v45 = v35[v39];
                        bool v47;
                        switch (v42.tag == v45.tag ? v42.tag : 255) {
                            case 0: { // Jack
                                v47 = true;
                                break;
                            }
                            case 1: { // King
                                v47 = true;
                                break;
                            }
                            case 2: { // Queen
                                v47 = true;
                                break;
                            }
                            default: {
                                v47 = false;
                            }
                        }
                        bool v48;
                        v48 = v47 != true;
                        if (v48){
                            bool v49;
                            v49 = false;
                            v38 = v49;
                            break;
                        } else {
                        }
                        v39 += 1 ;
                    }
                    if (v38){
                        bool v50;
                        v50 = v33 == v36;
                        if (v50){
                            bool v51;
                            v51 = v34 == v37;
                            v54 = v51;
                        } else {
                            v54 = false;
                        }
                    } else {
                        v54 = false;
                    }
                    break;
                }
                default: {
                    v54 = false;
                }
            }
            bool v55;
            v55 = v54 != true;
            if (v55){
                bool v56;
                v56 = false;
                v5 = v56;
                break;
            } else {
            }
            v7 += 1 ;
        }
        return v5;
    } else {
        return false;
    }
}
inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
inline bool while_method_3(std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v0, std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1>::iterator & v1){
    bool v2;
    v2 = v1 != v0.end();
    return v2;
}
nlohmann::json method_5(){
    nlohmann::json v0;
    return v0;
}
nlohmann::json method_4(Union1 v0){
    switch (v0.tag) {
        case 0: { // Jack
            nlohmann::json v1;
            v1 = method_5();
            const char * v2;
            v2 = "Jack";
            nlohmann::json v3{v2, v1};
            return v3;
            break;
        }
        case 1: { // King
            nlohmann::json v4;
            v4 = method_5();
            const char * v5;
            v5 = "King";
            nlohmann::json v6{v5, v4};
            return v6;
            break;
        }
        case 2: { // Queen
            nlohmann::json v7;
            v7 = method_5();
            const char * v8;
            v8 = "Queen";
            nlohmann::json v9{v8, v7};
            return v9;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
nlohmann::json method_7(int v0){
    nlohmann::json v1 = v0;
    return v1;
}
nlohmann::json method_8(Union2 v0){
    switch (v0.tag) {
        case 0: { // Call
            nlohmann::json v1;
            v1 = method_5();
            const char * v2;
            v2 = "Call";
            nlohmann::json v3{v2, v1};
            return v3;
            break;
        }
        case 1: { // Fold
            nlohmann::json v4;
            v4 = method_5();
            const char * v5;
            v5 = "Fold";
            nlohmann::json v6{v5, v4};
            return v6;
            break;
        }
        case 2: { // Raise
            nlohmann::json v7;
            v7 = method_5();
            const char * v8;
            v8 = "Raise";
            nlohmann::json v9{v8, v7};
            return v9;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
nlohmann::json method_6(int v0, Union2 v1){
    nlohmann::json v2;
    nlohmann::json v3;
    v3 = method_7(v0);
    v2.push_back(v3);
    nlohmann::json v4;
    v4 = method_8(v1);
    v2.push_back(v4);
    return v2;
}
nlohmann::json method_9(int v0, Union1 v1){
    nlohmann::json v2;
    nlohmann::json v3;
    v3 = method_7(v0);
    v2.push_back(v3);
    nlohmann::json v4;
    v4 = method_4(v1);
    v2.push_back(v4);
    return v2;
}
nlohmann::json method_11(static_array<Union1,2> v0){
    nlohmann::json v1;
    int v2;
    v2 = 0;
    while (while_method_1(v2)){
        Union1 v5;
        v5 = v0[v2];
        nlohmann::json v7;
        v7 = method_4(v5);
        v1.push_back(v7);
        v2 += 1 ;
    }
    return v1;
}
nlohmann::json method_10(static_array<Union1,2> v0, int v1, int v2){
    nlohmann::json v3 = nlohmann::json::object();
    const char * v4;
    v4 = "cards_shown";
    nlohmann::json v5;
    v5 = method_11(v0);
    v3[v4] = v5;
    const char * v6;
    v6 = "chips_won";
    nlohmann::json v7;
    v7 = method_7(v1);
    v3[v6] = v7;
    const char * v8;
    v8 = "winner_id";
    nlohmann::json v9;
    v9 = method_7(v2);
    v3[v8] = v9;
    return v3;
}
nlohmann::json method_3(Union0 v0){
    switch (v0.tag) {
        case 0: { // CommunityCardIs
            Union1 v1 = v0.case0.v0;
            nlohmann::json v2;
            v2 = method_4(v1);
            const char * v3;
            v3 = "CommunityCardIs";
            nlohmann::json v4{v3, v2};
            return v4;
            break;
        }
        case 1: { // PlayerAction
            int v5 = v0.case1.v0; Union2 v6 = v0.case1.v1;
            nlohmann::json v7;
            v7 = method_6(v5, v6);
            const char * v8;
            v8 = "PlayerAction";
            nlohmann::json v9{v8, v7};
            return v9;
            break;
        }
        case 2: { // PlayerGotCard
            int v10 = v0.case2.v0; Union1 v11 = v0.case2.v1;
            nlohmann::json v12;
            v12 = method_9(v10, v11);
            const char * v13;
            v13 = "PlayerGotCard";
            nlohmann::json v14{v13, v12};
            return v14;
            break;
        }
        case 3: { // Showdown
            static_array<Union1,2> v15 = v0.case3.v0; int v16 = v0.case3.v1; int v17 = v0.case3.v2;
            nlohmann::json v18;
            v18 = method_10(v15, v16, v17);
            const char * v19;
            v19 = "Showdown";
            nlohmann::json v20{v19, v18};
            return v20;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
nlohmann::json method_2(static_array_list<Union0,32> v0){
    nlohmann::json v1;
    int v2;
    v2 = v0.length;
    int v3;
    v3 = 0;
    while (while_method_0(v2, v3)){
        Union0 v6;
        v6 = v0[v3];
        nlohmann::json v8;
        v8 = method_3(v6);
        v1.push_back(v8);
        v3 += 1 ;
    }
    return v1;
}
nlohmann::json method_14(float v0){
    nlohmann::json v1 = v0;
    return v1;
}
nlohmann::json method_13(static_array<float,3> v0){
    nlohmann::json v1;
    int v2;
    v2 = 0;
    while (while_method_2(v2)){
        float v5;
        v5 = v0[v2];
        nlohmann::json v7;
        v7 = method_14(v5);
        v1.push_back(v7);
        v2 += 1 ;
    }
    return v1;
}
nlohmann::json method_12(static_array<float,3> v0, static_array<float,3> v1, static_array<float,3> v2){
    nlohmann::json v3 = nlohmann::json::object();
    const char * v4;
    v4 = "average_policy";
    nlohmann::json v5;
    v5 = method_13(v0);
    v3[v4] = v5;
    const char * v6;
    v6 = "current_policy";
    nlohmann::json v7;
    v7 = method_13(v1);
    v3[v6] = v7;
    const char * v8;
    v8 = "expected_values";
    nlohmann::json v9;
    v9 = method_13(v2);
    v3[v8] = v9;
    return v3;
}
nlohmann::json method_1(std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> v0){
    nlohmann::json v1;
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v2 = v0;
    auto v3 = v2.begin();
    while (while_method_3(v2, v3)){
        static_array_list<Union0,32> v5;
        v5 = v3->first;
        static_array<float,3> v6; static_array<float,3> v7; static_array<float,3> v8;
        Tuple1 tmp2 = v3->second;
        v6 = tmp2.v0; v7 = tmp2.v1; v8 = tmp2.v2;
        nlohmann::json v9;
        v9 = method_2(v5);
        nlohmann::json v10;
        v10 = method_12(v6, v7, v8);
        v1.push_back(nlohmann::json({v9, v10}));
        ++v3;
    }
    return v1;
}
nlohmann::json serialize_0(std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> v0){
    return method_1(v0);
}
void f_20(nlohmann::json v0){
    bool v1;
    v1 = v0.is_null();
    bool v2;
    v2 = v1 == false;
    if (v2){
        assert("Expected an unit type" && v1);
        return ;
    } else {
        return ;
    }
}
Union1 f_19(nlohmann::json v0){
    std::string v1;
    v1 = v0[0].get<std::string>();
    const char * v2;
    v2 = "Jack";
    std::string v3 = v2;
    bool v4;
    v4 = v1 == v3;
    if (v4){
        nlohmann::json v5;
        v5 = v0[1];
        f_20(v5);
        return Union1{Union1_0{}};
    } else {
        const char * v7;
        v7 = "King";
        std::string v8 = v7;
        bool v9;
        v9 = v1 == v8;
        if (v9){
            nlohmann::json v10;
            v10 = v0[1];
            f_20(v10);
            return Union1{Union1_1{}};
        } else {
            const char * v12;
            v12 = "Queen";
            std::string v13 = v12;
            bool v14;
            v14 = v1 == v13;
            if (v14){
                nlohmann::json v15;
                v15 = v0[1];
                f_20(v15);
                return Union1{Union1_2{}};
            } else {
                printf("%s\n", "Cannot convert the Python object into a Spiral union type.");
                exit(-1);
            }
        }
    }
}
int f_22(nlohmann::json v0){
    int v1;
    v1 = v0.get<int>();
    return v1;
}
Union2 f_23(nlohmann::json v0){
    std::string v1;
    v1 = v0[0].get<std::string>();
    const char * v2;
    v2 = "Call";
    std::string v3 = v2;
    bool v4;
    v4 = v1 == v3;
    if (v4){
        nlohmann::json v5;
        v5 = v0[1];
        f_20(v5);
        return Union2{Union2_0{}};
    } else {
        const char * v7;
        v7 = "Fold";
        std::string v8 = v7;
        bool v9;
        v9 = v1 == v8;
        if (v9){
            nlohmann::json v10;
            v10 = v0[1];
            f_20(v10);
            return Union2{Union2_1{}};
        } else {
            const char * v12;
            v12 = "Raise";
            std::string v13 = v12;
            bool v14;
            v14 = v1 == v13;
            if (v14){
                nlohmann::json v15;
                v15 = v0[1];
                f_20(v15);
                return Union2{Union2_2{}};
            } else {
                printf("%s\n", "Cannot convert the Python object into a Spiral union type.");
                exit(-1);
            }
        }
    }
}
Tuple2 f_21(nlohmann::json v0){
    nlohmann::json v1;
    v1 = v0[0];
    int v2;
    v2 = f_22(v1);
    nlohmann::json v3;
    v3 = v0[1];
    Union2 v4;
    v4 = f_23(v3);
    return Tuple2{v2, v4};
}
Tuple3 f_24(nlohmann::json v0){
    nlohmann::json v1;
    v1 = v0[0];
    int v2;
    v2 = f_22(v1);
    nlohmann::json v3;
    v3 = v0[1];
    Union1 v4;
    v4 = f_19(v3);
    return Tuple3{v2, v4};
}
static_array<Union1,2> f_26(nlohmann::json v0){
    bool v1;
    v1 = v0.is_array();
    bool v2;
    v2 = v1 == false;
    if (v2){
        assert("The json object must be an array." && v1);
    } else {
    }
    int v4;
    v4 = v0.size();
    bool v5;
    v5 = 2 == v4;
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("The type level dimension has to equal the value passed at runtime into create." && v5);
    } else {
    }
    static_array<Union1,2> v9;
    int v11;
    v11 = 0;
    while (while_method_0(v4, v11)){
        nlohmann::json v13;
        v13 = v0[v11];
        Union1 v14;
        v14 = f_19(v13);
        v9[v11] = v14;
        v11 += 1 ;
    }
    return v9;
}
Tuple4 f_25(nlohmann::json v0){
    nlohmann::json v1;
    v1 = v0["cards_shown"];
    static_array<Union1,2> v2;
    v2 = f_26(v1);
    nlohmann::json v3;
    v3 = v0["chips_won"];
    int v4;
    v4 = f_22(v3);
    nlohmann::json v5;
    v5 = v0["winner_id"];
    int v6;
    v6 = f_22(v5);
    return Tuple4{v2, v4, v6};
}
Union0 f_18(nlohmann::json v0){
    std::string v1;
    v1 = v0[0].get<std::string>();
    const char * v2;
    v2 = "CommunityCardIs";
    std::string v3 = v2;
    bool v4;
    v4 = v1 == v3;
    if (v4){
        nlohmann::json v5;
        v5 = v0[1];
        Union1 v6;
        v6 = f_19(v5);
        return Union0{Union0_0{v6}};
    } else {
        const char * v8;
        v8 = "PlayerAction";
        std::string v9 = v8;
        bool v10;
        v10 = v1 == v9;
        if (v10){
            nlohmann::json v11;
            v11 = v0[1];
            int v12; Union2 v13;
            Tuple2 tmp3 = f_21(v11);
            v12 = tmp3.v0; v13 = tmp3.v1;
            return Union0{Union0_1{v12, v13}};
        } else {
            const char * v15;
            v15 = "PlayerGotCard";
            std::string v16 = v15;
            bool v17;
            v17 = v1 == v16;
            if (v17){
                nlohmann::json v18;
                v18 = v0[1];
                int v19; Union1 v20;
                Tuple3 tmp4 = f_24(v18);
                v19 = tmp4.v0; v20 = tmp4.v1;
                return Union0{Union0_2{v19, v20}};
            } else {
                const char * v22;
                v22 = "Showdown";
                std::string v23 = v22;
                bool v24;
                v24 = v1 == v23;
                if (v24){
                    nlohmann::json v25;
                    v25 = v0[1];
                    static_array<Union1,2> v26; int v27; int v28;
                    Tuple4 tmp5 = f_25(v25);
                    v26 = tmp5.v0; v27 = tmp5.v1; v28 = tmp5.v2;
                    return Union0{Union0_3{v26, v27, v28}};
                } else {
                    printf("%s\n", "Cannot convert the Python object into a Spiral union type.");
                    exit(-1);
                }
            }
        }
    }
}
static_array_list<Union0,32> f_17(nlohmann::json v0){
    int v1;
    v1 = v0.size();
    bool v2;
    v2 = 32 >= v1;
    bool v3;
    v3 = v2 == false;
    if (v3){
        assert("The length of the original object has to be greater than or equal to the static array dimension." && v2);
    } else {
    }
    bool v5;
    v5 = v0.is_array();
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("The json object must be an array." && v5);
    } else {
    }
    int v8;
    v8 = v0.size();
    bool v9;
    v9 = 32 >= v8;
    bool v10;
    v10 = v9 == false;
    if (v10){
        assert("The type level dimension has to equal the value passed at runtime into create." && v9);
    } else {
    }
    static_array_list<Union0,32> v13;
    v13 = static_array_list<Union0,32>{};
    v13.unsafe_set_length(v8);
    int v15;
    v15 = 0;
    while (while_method_0(v8, v15)){
        nlohmann::json v17;
        v17 = v0[v15];
        Union0 v18;
        v18 = f_18(v17);
        v13[v15] = v18;
        v15 += 1 ;
    }
    return v13;
}
float f_29(nlohmann::json v0){
    float v1;
    v1 = v0.get<float>();
    return v1;
}
static_array<float,3> f_28(nlohmann::json v0){
    bool v1;
    v1 = v0.is_array();
    bool v2;
    v2 = v1 == false;
    if (v2){
        assert("The json object must be an array." && v1);
    } else {
    }
    int v4;
    v4 = v0.size();
    bool v5;
    v5 = 3 == v4;
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("The type level dimension has to equal the value passed at runtime into create." && v5);
    } else {
    }
    static_array<float,3> v9;
    int v11;
    v11 = 0;
    while (while_method_0(v4, v11)){
        nlohmann::json v13;
        v13 = v0[v11];
        float v14;
        v14 = f_29(v13);
        v9[v11] = v14;
        v11 += 1 ;
    }
    return v9;
}
Tuple1 f_27(nlohmann::json v0){
    nlohmann::json v1;
    v1 = v0["average_policy"];
    static_array<float,3> v2;
    v2 = f_28(v1);
    nlohmann::json v3;
    v3 = v0["current_policy"];
    static_array<float,3> v4;
    v4 = f_28(v3);
    nlohmann::json v5;
    v5 = v0["expected_values"];
    static_array<float,3> v6;
    v6 = f_28(v5);
    return Tuple1{v2, v4, v6};
}
std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> f_16(nlohmann::json v0){
    bool v1;
    v1 = v0.is_array();
    bool v2;
    v2 = v1 == false;
    if (v2){
        assert("The json object must be an array." && v1);
    } else {
    }
    int v4;
    v4 = v0.size();
    Fun0 v5 = FunPointerMethod0;
    Fun1 v6 = FunPointerMethod1;
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> v7(v4, v5, v6);
    int v8;
    v8 = 0;
    while (while_method_0(v4, v8)){
        nlohmann::json v10;
        v10 = v0[v8];
        bool v11;
        v11 = v10.size() == 2;
        bool v12;
        v12 = v11 == false;
        if (v12){
            assert("The key/value pair of an unordered_map being deserilized has to have two elements in the array." && v11);
        } else {
        }
        nlohmann::json v14;
        v14 = v10[0];
        nlohmann::json v15;
        v15 = v10[1];
        static_array_list<Union0,32> v16;
        v16 = f_17(v14);
        static_array<float,3> v17; static_array<float,3> v18; static_array<float,3> v19;
        Tuple1 tmp6 = f_27(v15);
        v17 = tmp6.v0; v18 = tmp6.v1; v19 = tmp6.v2;
        v7[v16] = Tuple1{v17, v18, v19};
        v8 += 1 ;
    }
    return v7;
}
std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> deserialize_15(nlohmann::json v0){
    return f_16(v0);
}
void method_31(Union1 v0){
    switch (v0.tag) {
        case 0: { // Jack
            printf("%s","Jack");
            return ;
            break;
        }
        case 1: { // King
            printf("%s","King");
            return ;
            break;
        }
        case 2: { // Queen
            printf("%s","Queen");
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
void method_32(Union2 v0){
    switch (v0.tag) {
        case 0: { // Call
            printf("%s","Call");
            return ;
            break;
        }
        case 1: { // Fold
            printf("%s","Fold");
            return ;
            break;
        }
        case 2: { // Raise
            printf("%s","Raise");
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
void method_30(Union0 v0){
    switch (v0.tag) {
        case 0: { // CommunityCardIs
            Union1 v1 = v0.case0.v0;
            printf("%s(","CommunityCardIs");
            method_31(v1);
            printf(")");
            return ;
            break;
        }
        case 1: { // PlayerAction
            int v2 = v0.case1.v0; Union2 v3 = v0.case1.v1;
            printf("%s(%d, ","PlayerAction", v2);
            method_32(v3);
            printf(")");
            return ;
            break;
        }
        case 2: { // PlayerGotCard
            int v4 = v0.case2.v0; Union1 v5 = v0.case2.v1;
            printf("%s(%d, ","PlayerGotCard", v4);
            method_31(v5);
            printf(")");
            return ;
            break;
        }
        case 3: { // Showdown
            static_array<Union1,2> v6 = v0.case3.v0; int v7 = v0.case3.v1; int v8 = v0.case3.v2;
            printf("%s({%s = %s","Showdown", "cards_shown", "[");
            int v9;
            v9 = 0;
            while (while_method_1(v9)){
                Union1 v12;
                v12 = v6[v9];
                printf("");
                method_31(v12);
                printf("");
                int v14;
                v14 = v9 + 1;
                bool v15;
                v15 = v14 < 2;
                if (v15){
                    printf("%s","; ");
                } else {
                }
                v9 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d; %s = %d})","chips_won", v7, "winner_id", v8);
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
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> v2(8, v0, v1);
    static_array<float,3> v4;
    int v6;
    v6 = 0;
    while (while_method_2(v6)){
        v4[v6] = 0.0f;
        v6 += 1 ;
    }
    static_array<float,3> v9;
    int v11;
    v11 = 0;
    while (while_method_2(v11)){
        v9[v11] = 0.0f;
        v11 += 1 ;
    }
    static_array<float,3> v14;
    int v16;
    v16 = 0;
    while (while_method_2(v16)){
        v14[v16] = 0.0f;
        v16 += 1 ;
    }
    static_array_list<Union0,32> v19;
    v19 = static_array_list<Union0,32>{};
    Union1 v21;
    v21 = Union1{Union1_1{}};
    Union0 v22;
    v22 = Union0{Union0_2{0, v21}};
    v19.push(v22);
    Union1 v23;
    v23 = Union1{Union1_0{}};
    Union0 v24;
    v24 = Union0{Union0_2{1, v23}};
    v19.push(v24);
    Union2 v25;
    v25 = Union2{Union2_2{}};
    Union0 v26;
    v26 = Union0{Union0_1{0, v25}};
    v19.push(v26);
    v2[v19] = Tuple1{v4, v9, v14};
    static_array_list<Union0,32> v28;
    v28 = static_array_list<Union0,32>{};
    Union1 v30;
    v30 = Union1{Union1_1{}};
    Union0 v31;
    v31 = Union0{Union0_2{0, v30}};
    v28.push(v31);
    Union1 v32;
    v32 = Union1{Union1_0{}};
    Union0 v33;
    v33 = Union0{Union0_2{1, v32}};
    v28.push(v33);
    Union2 v34;
    v34 = Union2{Union2_2{}};
    Union0 v35;
    v35 = Union0{Union0_1{0, v34}};
    v28.push(v35);
    Union2 v36;
    v36 = Union2{Union2_2{}};
    Union0 v37;
    v37 = Union0{Union0_1{1, v36}};
    v28.push(v37);
    v2[v28] = Tuple1{v4, v9, v14};
    static_array_list<Union0,32> v39;
    v39 = static_array_list<Union0,32>{};
    Union1 v41;
    v41 = Union1{Union1_1{}};
    Union0 v42;
    v42 = Union0{Union0_2{0, v41}};
    v39.push(v42);
    Union1 v43;
    v43 = Union1{Union1_0{}};
    Union0 v44;
    v44 = Union0{Union0_2{1, v43}};
    v39.push(v44);
    Union2 v45;
    v45 = Union2{Union2_2{}};
    Union0 v46;
    v46 = Union0{Union0_1{0, v45}};
    v39.push(v46);
    Union2 v47;
    v47 = Union2{Union2_2{}};
    Union0 v48;
    v48 = Union0{Union0_1{1, v47}};
    v39.push(v48);
    Union2 v49;
    v49 = Union2{Union2_0{}};
    Union0 v50;
    v50 = Union0{Union0_1{0, v49}};
    v39.push(v50);
    v2[v39] = Tuple1{v4, v9, v14};
    nlohmann::json v51;
    v51 = serialize_0(v2);
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> v52;
    v52 = deserialize_15(v51);
    printf("%s\n","{");
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v96 = v52;
    auto v97 = v96.begin();
    while (while_method_3(v96, v97)){
        static_array_list<Union0,32> v99;
        v99 = v97->first;
        static_array<float,3> v100; static_array<float,3> v101; static_array<float,3> v102;
        Tuple1 tmp7 = v97->second;
        v100 = tmp7.v0; v101 = tmp7.v1; v102 = tmp7.v2;
        printf("%s","    ");
        printf("%s","[");
        int v103;
        v103 = v99.length;
        bool v104;
        v104 = 100 < v103;
        int v105;
        if (v104){
            v105 = 100;
        } else {
            v105 = v103;
        }
        int v106;
        v106 = 0;
        while (while_method_0(v105, v106)){
            Union0 v109;
            v109 = v99[v106];
            printf("");
            method_30(v109);
            printf("");
            int v111;
            v111 = v106 + 1;
            int v112;
            v112 = v99.length;
            bool v113;
            v113 = v111 < v112;
            if (v113){
                printf("%s","; ");
            } else {
            }
            v106 += 1 ;
        }
        int v114;
        v114 = v99.length;
        bool v115;
        v115 = v114 > 100;
        if (v115){
            printf("%s","; ...");
        } else {
        }
        printf("%s","]");
        printf("");
        printf("%s"," => ");
        printf("{%s = %s","average_policy", "[");
        int v116;
        v116 = 0;
        while (while_method_2(v116)){
            float v119;
            v119 = v100[v116];
            printf("%f",v119);
            int v121;
            v121 = v116 + 1;
            bool v122;
            v122 = v121 < 3;
            if (v122){
                printf("%s","; ");
            } else {
            }
            v116 += 1 ;
        }
        printf("%s","]");
        printf("; %s = %s","current_policy", "[");
        int v123;
        v123 = 0;
        while (while_method_2(v123)){
            float v126;
            v126 = v101[v123];
            printf("%f",v126);
            int v128;
            v128 = v123 + 1;
            bool v129;
            v129 = v128 < 3;
            if (v129){
                printf("%s","; ");
            } else {
            }
            v123 += 1 ;
        }
        printf("%s","]");
        printf("; %s = %s","expected_values", "[");
        int v130;
        v130 = 0;
        while (while_method_2(v130)){
            float v133;
            v133 = v102[v130];
            printf("%f",v133);
            int v135;
            v135 = v130 + 1;
            bool v136;
            v136 = v135 < 3;
            if (v136){
                printf("%s","; ");
            } else {
            }
            v130 += 1 ;
        }
        printf("%s","]");
        printf("}\n");
        ++v97;
    }
    printf("%s\n","}");
    printf("\n");
    return 0;
}
