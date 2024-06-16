kernel = r"""
using default_int = long;
using default_uint = unsigned long;
#include "reference_counting.cuh"
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
struct Union1;
struct Union2;
struct Union0;
__device__ long f_1(unsigned char * v0);
__device__ void f_3(unsigned char * v0);
__device__ Union1 f_2(unsigned char * v0);
struct Tuple0;
__device__ long f_5(unsigned char * v0);
__device__ Tuple0 f_4(unsigned char * v0);
__device__ Union0 f_0(unsigned char * v0);
struct Union3;
struct Union4;
struct Union5;
struct Tuple1;
__device__ Union3 f_8(unsigned char * v0);
__device__ long f_9(unsigned char * v0);
struct Tuple2;
__device__ Tuple2 f_10(unsigned char * v0);
__device__ long f_11(unsigned char * v0);
__device__ long f_12(unsigned char * v0);
__device__ long f_13(unsigned char * v0);
__device__ Tuple1 f_7(unsigned char * v0);
__device__ Tuple1 f_6(unsigned char * v0);
struct Tuple3;
__device__ long loop_18(static_array<float,3l> v0, float v1, long v2);
__device__ long sample_discrete__17(static_array<float,3l> v0, curandStatePhilox4_32_10_t & v1);
__device__ Union1 sample_discrete_16(static_array<Tuple3,3l> v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple1 method_15(curandStatePhilox4_32_10_t & v0, Union3 v1, static_array<Union3,2l> v2, Union4 v3, Union5 v4, Union2 v5, Union2 v6);
__device__ Tuple1 method_14(curandStatePhilox4_32_10_t & v0, static_array<Union3,2l> v1, Union4 v2, Union5 v3, Union2 v4, Union2 v5, Union0 v6);
__device__ void f_22(unsigned char * v0, long v1);
__device__ void f_23(unsigned char * v0);
__device__ void f_24(unsigned char * v0, Union1 v1);
__device__ void f_21(unsigned char * v0, Union3 v1);
__device__ void f_25(unsigned char * v0, long v1);
__device__ void f_27(unsigned char * v0, long v1);
__device__ void f_26(unsigned char * v0, Union1 v1, Union1 v2);
__device__ void f_28(unsigned char * v0, long v1);
__device__ void f_29(unsigned char * v0, long v1);
__device__ void f_30(unsigned char * v0, long v1);
__device__ void f_20(unsigned char * v0, static_array<Union3,2l> v1, Union4 v2, Union5 v3, Union2 v4, Union2 v5);
__device__ void f_19(unsigned char * v0, static_array<Union3,2l> v1, Union4 v2, Union5 v3, Union2 v4, Union2 v5);
struct Union1_0 { // Paper
};
struct Union1_1 { // Rock
};
struct Union1_2 { // Scissors
};
struct Union1 {
    union {
        Union1_0 case0; // Paper
        Union1_1 case1; // Rock
        Union1_2 case2; // Scissors
    };
    unsigned char tag{-1};
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // Paper
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // Rock
    __device__ Union1(Union1_2 t) : tag(2), case2(t) {} // Scissors
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // Paper
            case 1: new (&this->case1) Union1_1(x.case1); break; // Rock
            case 2: new (&this->case2) Union1_2(x.case2); break; // Scissors
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // Paper
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // Rock
            case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // Scissors
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Paper
                case 1: this->case1 = x.case1; break; // Rock
                case 2: this->case2 = x.case2; break; // Scissors
            }
        } else {
            this->~Union1();
            new (this) Union1{x};
        }
        return *this;
    }
    __device__ Union1 & operator=(Union1 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Paper
                case 1: this->case1 = std::move(x.case1); break; // Rock
                case 2: this->case2 = std::move(x.case2); break; // Scissors
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // Paper
            case 1: this->case1.~Union1_1(); break; // Rock
            case 2: this->case2.~Union1_2(); break; // Scissors
        }
        this->tag = -1;
    }
};
struct Union2_0 { // Computer
};
struct Union2_1 { // Human
};
struct Union2 {
    union {
        Union2_0 case0; // Computer
        Union2_1 case1; // Human
    };
    unsigned char tag{-1};
    __device__ Union2() {}
    __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // Computer
    __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // Human
    __device__ Union2(Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // Computer
            case 1: new (&this->case1) Union2_1(x.case1); break; // Human
        }
    }
    __device__ Union2(Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // Computer
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // Human
        }
    }
    __device__ Union2 & operator=(Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Computer
                case 1: this->case1 = x.case1; break; // Human
            }
        } else {
            this->~Union2();
            new (this) Union2{x};
        }
        return *this;
    }
    __device__ Union2 & operator=(Union2 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Computer
                case 1: this->case1 = std::move(x.case1); break; // Human
            }
        } else {
            this->~Union2();
            new (this) Union2{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // Computer
            case 1: this->case1.~Union2_1(); break; // Human
        }
        this->tag = -1;
    }
};
struct Union0_0 { // ActionSelected
    Union1 v0;
    __device__ Union0_0(Union1 t0) : v0(t0) {}
    __device__ Union0_0() = delete;
};
struct Union0_1 { // PlayerChanged
    Union2 v0;
    Union2 v1;
    __device__ Union0_1(Union2 t0, Union2 t1) : v0(t0), v1(t1) {}
    __device__ Union0_1() = delete;
};
struct Union0_2 { // StartGame
};
struct Union0 {
    union {
        Union0_0 case0; // ActionSelected
        Union0_1 case1; // PlayerChanged
        Union0_2 case2; // StartGame
    };
    unsigned char tag{-1};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // ActionSelected
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // PlayerChanged
    __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // StartGame
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // ActionSelected
            case 1: new (&this->case1) Union0_1(x.case1); break; // PlayerChanged
            case 2: new (&this->case2) Union0_2(x.case2); break; // StartGame
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // ActionSelected
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // PlayerChanged
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // StartGame
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // ActionSelected
                case 1: this->case1 = x.case1; break; // PlayerChanged
                case 2: this->case2 = x.case2; break; // StartGame
            }
        } else {
            this->~Union0();
            new (this) Union0{x};
        }
        return *this;
    }
    __device__ Union0 & operator=(Union0 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // ActionSelected
                case 1: this->case1 = std::move(x.case1); break; // PlayerChanged
                case 2: this->case2 = std::move(x.case2); break; // StartGame
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // ActionSelected
            case 1: this->case1.~Union0_1(); break; // PlayerChanged
            case 2: this->case2.~Union0_2(); break; // StartGame
        }
        this->tag = -1;
    }
};
struct Tuple0 {
    Union2 v0;
    Union2 v1;
    __device__ Tuple0(Union2 t0, Union2 t1) : v0(t0), v1(t1) {}
};
struct Union3_0 { // None
};
struct Union3_1 { // Some
    Union1 v0;
    __device__ Union3_1(Union1 t0) : v0(t0) {}
    __device__ Union3_1() = delete;
};
struct Union3 {
    union {
        Union3_0 case0; // None
        Union3_1 case1; // Some
    };
    unsigned char tag{-1};
    __device__ Union3() {}
    __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // None
    __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // Some
    __device__ Union3(Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // None
            case 1: new (&this->case1) Union3_1(x.case1); break; // Some
        }
    }
    __device__ Union3(Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union3 & operator=(Union3 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union3();
            new (this) Union3{x};
        }
        return *this;
    }
    __device__ Union3 & operator=(Union3 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union3();
            new (this) Union3{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // None
            case 1: this->case1.~Union3_1(); break; // Some
        }
        this->tag = -1;
    }
};
struct Union4_0 { // GameNotStarted
};
struct Union4_1 { // GameOver
    Union1 v0;
    Union1 v1;
    __device__ Union4_1(Union1 t0, Union1 t1) : v0(t0), v1(t1) {}
    __device__ Union4_1() = delete;
};
struct Union4_2 { // WaitingForActionFromPlayerId
    long v0;
    __device__ Union4_2(long t0) : v0(t0) {}
    __device__ Union4_2() = delete;
};
struct Union4 {
    union {
        Union4_0 case0; // GameNotStarted
        Union4_1 case1; // GameOver
        Union4_2 case2; // WaitingForActionFromPlayerId
    };
    unsigned char tag{-1};
    __device__ Union4() {}
    __device__ Union4(Union4_0 t) : tag(0), case0(t) {} // GameNotStarted
    __device__ Union4(Union4_1 t) : tag(1), case1(t) {} // GameOver
    __device__ Union4(Union4_2 t) : tag(2), case2(t) {} // WaitingForActionFromPlayerId
    __device__ Union4(Union4 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(x.case0); break; // GameNotStarted
            case 1: new (&this->case1) Union4_1(x.case1); break; // GameOver
            case 2: new (&this->case2) Union4_2(x.case2); break; // WaitingForActionFromPlayerId
        }
    }
    __device__ Union4(Union4 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(std::move(x.case0)); break; // GameNotStarted
            case 1: new (&this->case1) Union4_1(std::move(x.case1)); break; // GameOver
            case 2: new (&this->case2) Union4_2(std::move(x.case2)); break; // WaitingForActionFromPlayerId
        }
    }
    __device__ Union4 & operator=(Union4 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // GameNotStarted
                case 1: this->case1 = x.case1; break; // GameOver
                case 2: this->case2 = x.case2; break; // WaitingForActionFromPlayerId
            }
        } else {
            this->~Union4();
            new (this) Union4{x};
        }
        return *this;
    }
    __device__ Union4 & operator=(Union4 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // GameNotStarted
                case 1: this->case1 = std::move(x.case1); break; // GameOver
                case 2: this->case2 = std::move(x.case2); break; // WaitingForActionFromPlayerId
            }
        } else {
            this->~Union4();
            new (this) Union4{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union4() {
        switch(this->tag){
            case 0: this->case0.~Union4_0(); break; // GameNotStarted
            case 1: this->case1.~Union4_1(); break; // GameOver
            case 2: this->case2.~Union4_2(); break; // WaitingForActionFromPlayerId
        }
        this->tag = -1;
    }
};
struct Union5_0 { // GameStarted
};
struct Union5_1 { // ShowdownResult
    Union1 v0;
    Union1 v1;
    __device__ Union5_1(Union1 t0, Union1 t1) : v0(t0), v1(t1) {}
    __device__ Union5_1() = delete;
};
struct Union5_2 { // WaitingToStart
};
struct Union5 {
    union {
        Union5_0 case0; // GameStarted
        Union5_1 case1; // ShowdownResult
        Union5_2 case2; // WaitingToStart
    };
    unsigned char tag{-1};
    __device__ Union5() {}
    __device__ Union5(Union5_0 t) : tag(0), case0(t) {} // GameStarted
    __device__ Union5(Union5_1 t) : tag(1), case1(t) {} // ShowdownResult
    __device__ Union5(Union5_2 t) : tag(2), case2(t) {} // WaitingToStart
    __device__ Union5(Union5 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(x.case0); break; // GameStarted
            case 1: new (&this->case1) Union5_1(x.case1); break; // ShowdownResult
            case 2: new (&this->case2) Union5_2(x.case2); break; // WaitingToStart
        }
    }
    __device__ Union5(Union5 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(std::move(x.case0)); break; // GameStarted
            case 1: new (&this->case1) Union5_1(std::move(x.case1)); break; // ShowdownResult
            case 2: new (&this->case2) Union5_2(std::move(x.case2)); break; // WaitingToStart
        }
    }
    __device__ Union5 & operator=(Union5 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // GameStarted
                case 1: this->case1 = x.case1; break; // ShowdownResult
                case 2: this->case2 = x.case2; break; // WaitingToStart
            }
        } else {
            this->~Union5();
            new (this) Union5{x};
        }
        return *this;
    }
    __device__ Union5 & operator=(Union5 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // GameStarted
                case 1: this->case1 = std::move(x.case1); break; // ShowdownResult
                case 2: this->case2 = std::move(x.case2); break; // WaitingToStart
            }
        } else {
            this->~Union5();
            new (this) Union5{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union5() {
        switch(this->tag){
            case 0: this->case0.~Union5_0(); break; // GameStarted
            case 1: this->case1.~Union5_1(); break; // ShowdownResult
            case 2: this->case2.~Union5_2(); break; // WaitingToStart
        }
        this->tag = -1;
    }
};
struct Tuple1 {
    static_array<Union3,2l> v0;
    Union4 v1;
    Union5 v2;
    Union2 v3;
    Union2 v4;
    __device__ Tuple1(static_array<Union3,2l> t0, Union4 t1, Union5 t2, Union2 t3, Union2 t4) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4) {}
};
struct Tuple2 {
    Union1 v0;
    Union1 v1;
    __device__ Tuple2(Union1 t0, Union1 t1) : v0(t0), v1(t1) {}
};
struct Tuple3 {
    Union1 v0;
    float v1;
    __device__ Tuple3(Union1 t0, float t1) : v0(t0), v1(t1) {}
};
__device__ long f_1(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+0ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ void f_3(unsigned char * v0){
    return ;
}
__device__ Union1 f_2(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union1{Union1_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ long f_5(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+4ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ Tuple0 f_4(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    Union2 v6;
    switch (v1) {
        case 0: {
            f_3(v2);
            v6 = Union2{Union2_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            v6 = Union2{Union2_1{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    long v7;
    v7 = f_5(v0);
    unsigned char * v8;
    v8 = (unsigned char *)(v0+8ull);
    Union2 v12;
    switch (v7) {
        case 0: {
            f_3(v8);
            v12 = Union2{Union2_0{}};
            break;
        }
        case 1: {
            f_3(v8);
            v12 = Union2{Union2_1{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple0{v6, v12};
}
__device__ Union0 f_0(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+8ull);
    switch (v1) {
        case 0: {
            Union1 v4;
            v4 = f_2(v2);
            return Union0{Union0_0{v4}};
            break;
        }
        case 1: {
            Union2 v6; Union2 v7;
            Tuple0 tmp0 = f_4(v2);
            v6 = tmp0.v0; v7 = tmp0.v1;
            return Union0{Union0_1{v6, v7}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union0{Union0_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ inline bool while_method_0(long v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ Union3 f_8(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union3{Union3_0{}};
            break;
        }
        case 1: {
            Union1 v5;
            v5 = f_2(v2);
            return Union3{Union3_1{v5}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ long f_9(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+16ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ Tuple2 f_10(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    Union1 v7;
    switch (v1) {
        case 0: {
            f_3(v2);
            v7 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            v7 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v2);
            v7 = Union1{Union1_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    long v8;
    v8 = f_5(v0);
    unsigned char * v9;
    v9 = (unsigned char *)(v0+8ull);
    Union1 v14;
    switch (v8) {
        case 0: {
            f_3(v9);
            v14 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v9);
            v14 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v9);
            v14 = Union1{Union1_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple2{v7, v14};
}
__device__ long f_11(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+32ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ long f_12(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+48ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ long f_13(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+52ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ Tuple1 f_7(unsigned char * v0){
    static_array<Union3,2l> v1;
    long v2;
    v2 = 0l;
    while (while_method_0(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned long long v5;
        v5 = v4 * 8ull;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        Union3 v7;
        v7 = f_8(v6);
        v1[v2] = v7;
        v2 += 1l ;
    }
    long v8;
    v8 = f_9(v0);
    unsigned char * v9;
    v9 = (unsigned char *)(v0+24ull);
    Union4 v17;
    switch (v8) {
        case 0: {
            f_3(v9);
            v17 = Union4{Union4_0{}};
            break;
        }
        case 1: {
            Union1 v12; Union1 v13;
            Tuple2 tmp1 = f_10(v9);
            v12 = tmp1.v0; v13 = tmp1.v1;
            v17 = Union4{Union4_1{v12, v13}};
            break;
        }
        case 2: {
            long v15;
            v15 = f_1(v9);
            v17 = Union4{Union4_2{v15}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    long v18;
    v18 = f_11(v0);
    unsigned char * v19;
    v19 = (unsigned char *)(v0+40ull);
    Union5 v26;
    switch (v18) {
        case 0: {
            f_3(v19);
            v26 = Union5{Union5_0{}};
            break;
        }
        case 1: {
            Union1 v22; Union1 v23;
            Tuple2 tmp2 = f_10(v19);
            v22 = tmp2.v0; v23 = tmp2.v1;
            v26 = Union5{Union5_1{v22, v23}};
            break;
        }
        case 2: {
            f_3(v19);
            v26 = Union5{Union5_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    long v27;
    v27 = f_12(v0);
    unsigned char * v28;
    v28 = (unsigned char *)(v0+52ull);
    Union2 v32;
    switch (v27) {
        case 0: {
            f_3(v28);
            v32 = Union2{Union2_0{}};
            break;
        }
        case 1: {
            f_3(v28);
            v32 = Union2{Union2_1{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    long v33;
    v33 = f_13(v0);
    unsigned char * v34;
    v34 = (unsigned char *)(v0+56ull);
    Union2 v38;
    switch (v33) {
        case 0: {
            f_3(v34);
            v38 = Union2{Union2_0{}};
            break;
        }
        case 1: {
            f_3(v34);
            v38 = Union2{Union2_1{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple1{v1, v17, v26, v32, v38};
}
__device__ Tuple1 f_6(unsigned char * v0){
    static_array<Union3,2l> v1; Union4 v2; Union5 v3; Union2 v4; Union2 v5;
    Tuple1 tmp3 = f_7(v0);
    v1 = tmp3.v0; v2 = tmp3.v1; v3 = tmp3.v2; v4 = tmp3.v3; v5 = tmp3.v4;
    return Tuple1{v1, v2, v3, v4, v5};
}
__device__ inline bool while_method_1(long v0){
    bool v1;
    v1 = v0 < 3l;
    return v1;
}
__device__ inline bool while_method_2(static_array<float,3l> v0, long v1){
    bool v2;
    v2 = v1 < 3l;
    return v2;
}
__device__ inline bool while_method_3(long v0, long v1){
    bool v2;
    v2 = v1 > v0;
    return v2;
}
__device__ long loop_18(static_array<float,3l> v0, float v1, long v2){
    bool v3;
    v3 = v2 < 3l;
    if (v3){
        float v4;
        v4 = v0[v2];
        bool v5;
        v5 = v1 <= v4;
        if (v5){
            return v2;
        } else {
            long v6;
            v6 = v2 + 1l;
            return loop_18(v0, v1, v6);
        }
    } else {
        return 2l;
    }
}
__device__ long sample_discrete__17(static_array<float,3l> v0, curandStatePhilox4_32_10_t & v1){
    static_array<float,3l> v2;
    long v3;
    v3 = 0l;
    while (while_method_1(v3)){
        float v5;
        v5 = v0[v3];
        v2[v3] = v5;
        v3 += 1l ;
    }
    long v6;
    v6 = 1l;
    while (while_method_2(v2, v6)){
        long v8;
        v8 = 3l;
        while (while_method_3(v6, v8)){
            v8 -= 1l ;
            long v10;
            v10 = v8 - v6;
            float v11;
            v11 = v2[v10];
            float v12;
            v12 = v2[v8];
            float v13;
            v13 = v11 + v12;
            v2[v8] = v13;
        }
        long v14;
        v14 = v6 * 2l;
        v6 = v14;
    }
    float v15;
    v15 = v2[2l];
    float v16;
    v16 = curand_uniform(&v1);
    float v17;
    v17 = v16 * v15;
    long v18;
    v18 = 0l;
    return loop_18(v2, v17, v18);
}
__device__ Union1 sample_discrete_16(static_array<Tuple3,3l> v0, curandStatePhilox4_32_10_t & v1){
    static_array<float,3l> v2;
    long v3;
    v3 = 0l;
    while (while_method_1(v3)){
        Union1 v5; float v6;
        Tuple3 tmp5 = v0[v3];
        v5 = tmp5.v0; v6 = tmp5.v1;
        v2[v3] = v6;
        v3 += 1l ;
    }
    long v7;
    v7 = sample_discrete__17(v2, v1);
    Union1 v8; float v9;
    Tuple3 tmp6 = v0[v7];
    v8 = tmp6.v0; v9 = tmp6.v1;
    return v8;
}
__device__ Tuple1 method_15(curandStatePhilox4_32_10_t & v0, Union3 v1, static_array<Union3,2l> v2, Union4 v3, Union5 v4, Union2 v5, Union2 v6){
    switch (v3.tag) {
        case 0: { // GameNotStarted
            return Tuple1{v2, v3, v4, v5, v6};
            break;
        }
        case 1: { // GameOver
            Union1 v7 = v3.case1.v0; Union1 v8 = v3.case1.v1;
            return Tuple1{v2, v3, v4, v5, v6};
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            long v9 = v3.case2.v0;
            bool v10;
            v10 = v9 < 2l;
            if (v10){
                bool v11;
                v11 = v9 == 0l;
                Union2 v12;
                if (v11){
                    v12 = v5;
                } else {
                    v12 = v6;
                }
                switch (v12.tag) {
                    case 0: { // Computer
                        bool v14;
                        switch (v1.tag) {
                            case 0: { // None
                                v14 = true;
                                break;
                            }
                            default: {
                                v14 = false;
                            }
                        }
                        bool v15;
                        v15 = v14 == false;
                        if (v15){
                            assert("The computer player should never be receiving an action." && v14);
                        } else {
                        }
                        static_array<Tuple3,3l> v16;
                        Union1 v17;
                        v17 = Union1{Union1_1{}};
                        v16[0l] = Tuple3{v17, 1.0f};
                        Union1 v18;
                        v18 = Union1{Union1_0{}};
                        v16[1l] = Tuple3{v18, 1.0f};
                        Union1 v19;
                        v19 = Union1{Union1_2{}};
                        v16[2l] = Tuple3{v19, 1.0f};
                        Union1 v20;
                        v20 = sample_discrete_16(v16, v0);
                        long v21;
                        v21 = v9 + 1l;
                        static_array<Union3,2l> v22;
                        long v23;
                        v23 = 0l;
                        while (while_method_0(v23)){
                            Union3 v25;
                            v25 = v2[v23];
                            bool v26;
                            v26 = v9 == v23;
                            Union3 v28;
                            if (v26){
                                v28 = Union3{Union3_1{v20}};
                            } else {
                                v28 = v25;
                            }
                            v22[v23] = v28;
                            v23 += 1l ;
                        }
                        Union3 v29;
                        v29 = Union3{Union3_0{}};
                        Union4 v30;
                        v30 = Union4{Union4_2{v21}};
                        return method_15(v0, v29, v22, v30, v4, v5, v6);
                        break;
                    }
                    case 1: { // Human
                        switch (v1.tag) {
                            case 0: { // None
                                return Tuple1{v2, v3, v4, v5, v6};
                                break;
                            }
                            case 1: { // Some
                                Union1 v36 = v1.case1.v0;
                                long v37;
                                v37 = v9 + 1l;
                                static_array<Union3,2l> v38;
                                long v39;
                                v39 = 0l;
                                while (while_method_0(v39)){
                                    Union3 v41;
                                    v41 = v2[v39];
                                    bool v42;
                                    v42 = v9 == v39;
                                    Union3 v44;
                                    if (v42){
                                        v44 = Union3{Union3_1{v36}};
                                    } else {
                                        v44 = v41;
                                    }
                                    v38[v39] = v44;
                                    v39 += 1l ;
                                }
                                Union3 v45;
                                v45 = Union3{Union3_0{}};
                                Union4 v46;
                                v46 = Union4{Union4_2{v37}};
                                return method_15(v0, v45, v38, v46, v4, v5, v6);
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
            } else {
                Union3 v72;
                v72 = v2[0l];
                Union3 v73;
                v73 = v2[1l];
                switch (v72.tag) {
                    case 1: { // Some
                        Union1 v74 = v72.case1.v0;
                        switch (v73.tag) {
                            case 1: { // Some
                                Union1 v75 = v73.case1.v0;
                                static_array<Union3,2l> v76;
                                long v77;
                                v77 = 0l;
                                while (while_method_0(v77)){
                                    Union3 v79;
                                    v79 = Union3{Union3_0{}};
                                    v76[v77] = v79;
                                    v77 += 1l ;
                                }
                                Union4 v80;
                                v80 = Union4{Union4_1{v74, v75}};
                                Union5 v81;
                                v81 = Union5{Union5_1{v74, v75}};
                                return Tuple1{v76, v80, v81, v5, v6};
                                break;
                            }
                            default: {
                                printf("%s\n", "At showdown all the actions have to be selected.");
                                asm("exit;");
                            }
                        }
                        break;
                    }
                    default: {
                        printf("%s\n", "At showdown all the actions have to be selected.");
                        asm("exit;");
                    }
                }
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ Tuple1 method_14(curandStatePhilox4_32_10_t & v0, static_array<Union3,2l> v1, Union4 v2, Union5 v3, Union2 v4, Union2 v5, Union0 v6){
    static_array<Union3,2l> v45; Union4 v46; Union5 v47; Union2 v48; Union2 v49;
    switch (v6.tag) {
        case 0: { // ActionSelected
            Union1 v28 = v6.case0.v0;
            Union3 v29;
            v29 = Union3{Union3_1{v28}};
            Tuple1 tmp7 = method_15(v0, v29, v1, v2, v3, v4, v5);
            v45 = tmp7.v0; v46 = tmp7.v1; v47 = tmp7.v2; v48 = tmp7.v3; v49 = tmp7.v4;
            break;
        }
        case 1: { // PlayerChanged
            Union2 v20 = v6.case1.v0; Union2 v21 = v6.case1.v1;
            Union3 v22;
            v22 = Union3{Union3_0{}};
            Tuple1 tmp8 = method_15(v0, v22, v1, v2, v3, v20, v21);
            v45 = tmp8.v0; v46 = tmp8.v1; v47 = tmp8.v2; v48 = tmp8.v3; v49 = tmp8.v4;
            break;
        }
        case 2: { // StartGame
            static_array<Union3,2l> v7;
            long v8;
            v8 = 0l;
            while (while_method_0(v8)){
                Union3 v10;
                v10 = Union3{Union3_0{}};
                v7[v8] = v10;
                v8 += 1l ;
            }
            Union3 v11;
            v11 = Union3{Union3_0{}};
            long v12;
            v12 = 0l;
            Union4 v13;
            v13 = Union4{Union4_2{v12}};
            Union5 v14;
            v14 = Union5{Union5_0{}};
            Tuple1 tmp9 = method_15(v0, v11, v7, v13, v14, v4, v5);
            v45 = tmp9.v0; v46 = tmp9.v1; v47 = tmp9.v2; v48 = tmp9.v3; v49 = tmp9.v4;
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    return Tuple1{v45, v46, v47, v48, v49};
}
__device__ void f_22(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_23(unsigned char * v0){
    return ;
}
__device__ void f_24(unsigned char * v0, Union1 v1){
    long v2;
    v2 = v1.tag;
    f_22(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Paper
            return f_23(v3);
            break;
        }
        case 1: { // Rock
            return f_23(v3);
            break;
        }
        case 2: { // Scissors
            return f_23(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_21(unsigned char * v0, Union3 v1){
    long v2;
    v2 = v1.tag;
    f_22(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            return f_23(v3);
            break;
        }
        case 1: { // Some
            Union1 v4 = v1.case1.v0;
            return f_24(v3, v4);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_25(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+16ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_27(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+4ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_26(unsigned char * v0, Union1 v1, Union1 v2){
    long v3;
    v3 = v1.tag;
    f_22(v0, v3);
    unsigned char * v4;
    v4 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Paper
            f_23(v4);
            break;
        }
        case 1: { // Rock
            f_23(v4);
            break;
        }
        case 2: { // Scissors
            f_23(v4);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    long v5;
    v5 = v2.tag;
    f_27(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Paper
            return f_23(v6);
            break;
        }
        case 1: { // Rock
            return f_23(v6);
            break;
        }
        case 2: { // Scissors
            return f_23(v6);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_28(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+32ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_29(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+48ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_30(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+52ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_20(unsigned char * v0, static_array<Union3,2l> v1, Union4 v2, Union5 v3, Union2 v4, Union2 v5){
    long v6;
    v6 = 0l;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 8ull;
        unsigned char * v10;
        v10 = (unsigned char *)(v0+v9);
        Union3 v11;
        v11 = v1[v6];
        f_21(v10, v11);
        v6 += 1l ;
    }
    long v12;
    v12 = v2.tag;
    f_25(v0, v12);
    unsigned char * v13;
    v13 = (unsigned char *)(v0+24ull);
    switch (v2.tag) {
        case 0: { // GameNotStarted
            f_23(v13);
            break;
        }
        case 1: { // GameOver
            Union1 v14 = v2.case1.v0; Union1 v15 = v2.case1.v1;
            f_26(v13, v14, v15);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            long v16 = v2.case2.v0;
            f_22(v13, v16);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    long v17;
    v17 = v3.tag;
    f_28(v0, v17);
    unsigned char * v18;
    v18 = (unsigned char *)(v0+40ull);
    switch (v3.tag) {
        case 0: { // GameStarted
            f_23(v18);
            break;
        }
        case 1: { // ShowdownResult
            Union1 v19 = v3.case1.v0; Union1 v20 = v3.case1.v1;
            f_26(v18, v19, v20);
            break;
        }
        case 2: { // WaitingToStart
            f_23(v18);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    long v21;
    v21 = v4.tag;
    f_29(v0, v21);
    unsigned char * v22;
    v22 = (unsigned char *)(v0+52ull);
    switch (v4.tag) {
        case 0: { // Computer
            f_23(v22);
            break;
        }
        case 1: { // Human
            f_23(v22);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    long v23;
    v23 = v5.tag;
    f_30(v0, v23);
    unsigned char * v24;
    v24 = (unsigned char *)(v0+56ull);
    switch (v5.tag) {
        case 0: { // Computer
            return f_23(v24);
            break;
        }
        case 1: { // Human
            return f_23(v24);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_19(unsigned char * v0, static_array<Union3,2l> v1, Union4 v2, Union5 v3, Union2 v4, Union2 v5){
    return f_20(v0, v1, v2, v3, v4, v5);
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    long v2;
    v2 = threadIdx.x;
    long v3;
    v3 = blockIdx.x;
    long v4;
    v4 = v3 * 32l;
    long v5;
    v5 = v2 + v4;
    bool v6;
    v6 = v5 == 0l;
    if (v6){
        unsigned long long v7;
        v7 = clock64();
        unsigned long long v8;
        v8 = (unsigned long long)v5;
        curandStatePhilox4_32_10_t v9;
        curand_init(v7,v8,0ull,&v9);
        Union0 v10;
        v10 = f_0(v0);
        static_array<Union3,2l> v11; Union4 v12; Union5 v13; Union2 v14; Union2 v15;
        Tuple1 tmp4 = f_6(v1);
        v11 = tmp4.v0; v12 = tmp4.v1; v13 = tmp4.v2; v14 = tmp4.v3; v15 = tmp4.v4;
        static_array<Union3,2l> v16; Union4 v17; Union5 v18; Union2 v19; Union2 v20;
        Tuple1 tmp10 = method_14(v9, v11, v12, v13, v14, v15, v10);
        v16 = tmp10.v0; v17 = tmp10.v1; v18 = tmp10.v2; v19 = tmp10.v3; v20 = tmp10.v4;
        return f_19(v1, v16, v17, v18, v19, v20);
    } else {
        return ;
    }
}
"""
class static_array():
    def __init__(self, length):
        self.ptr = []
        for _ in range(length):
            self.ptr.append(None)

    def __getitem__(self, index):
        assert 0 <= index < len(self.ptr), "The get index needs to be in range."
        return self.ptr[index]
    
    def __setitem__(self, index, value):
        assert 0 <= index < len(self.ptr), "The set index needs to be in range."
        self.ptr[index] = value

class static_array_list(static_array):
    def __init__(self, length):
        super().__init__(length)
        self.length = 0

    def __getitem__(self, index):
        assert 0 <= index < self.length, "The get index needs to be in range."
        return self.ptr[index]
    
    def __setitem__(self, index, value):
        assert 0 <= index < self.length, "The set index needs to be in range."
        self.ptr[index] = value

    def push(self,value):
        assert (self.length < len(self.ptr)), "The length before pushing has to be less than the maximum length of the array."
        self.ptr[self.length] = value
        self.length += 1

    def pop(self):
        assert (0 < self.length), "The length before popping has to be greater than 0."
        self.length -= 1
        return self.ptr[self.length]

    def unsafe_set_length(self,i):
        assert 0 <= i <= len(self.ptr), "The new length has to be in range."
        self.length = i
import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

import random
options = []
options.append('--diag-suppress=550,20012,68')
options.append('--dopt=on')
options.append('--restrict')
options.append('-I C:/Spiral_s_ML_Library/cpplib')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
import collections
class US1_0(NamedTuple): # Paper
    tag = 0
class US1_1(NamedTuple): # Rock
    tag = 1
class US1_2(NamedTuple): # Scissors
    tag = 2
US1 = Union[US1_0, US1_1, US1_2]
class US2_0(NamedTuple): # Computer
    tag = 0
class US2_1(NamedTuple): # Human
    tag = 1
US2 = Union[US2_0, US2_1]
class US0_0(NamedTuple): # ActionSelected
    v0 : US1
    tag = 0
class US0_1(NamedTuple): # PlayerChanged
    v0 : US2
    v1 : US2
    tag = 1
class US0_2(NamedTuple): # StartGame
    tag = 2
US0 = Union[US0_0, US0_1, US0_2]
class US3_0(NamedTuple): # GameNotStarted
    tag = 0
class US3_1(NamedTuple): # GameOver
    v0 : US1
    v1 : US1
    tag = 1
class US3_2(NamedTuple): # WaitingForActionFromPlayerId
    v0 : i32
    tag = 2
US3 = Union[US3_0, US3_1, US3_2]
class US4_0(NamedTuple): # GameStarted
    tag = 0
class US4_1(NamedTuple): # ShowdownResult
    v0 : US1
    v1 : US1
    tag = 1
class US4_2(NamedTuple): # WaitingToStart
    tag = 2
US4 = Union[US4_0, US4_1, US4_2]
class US5_0(NamedTuple): # None
    tag = 0
class US5_1(NamedTuple): # Some
    v0 : US1
    tag = 1
US5 = Union[US5_0, US5_1]
def Closure0():
    def inner(v0 : object, v1 : object) -> object:
        v2 = method0(v0)
        v3, v4, v5, v6, v7 = method6(v1)
        v8, v9, v10, v11, v12 = method18(v3, v4, v5, v6, v7, v2)
        del v2, v3, v4, v5, v6, v7
        return method21(v8, v9, v10, v11, v12)
    return inner
def Closure1():
    def inner(v0 : object, v1 : object) -> object:
        v2 = cp.empty(16,dtype=cp.uint8)
        v3 = cp.empty(64,dtype=cp.uint8)
        v4 = method0(v0)
        v5, v6, v7, v8, v9 = method6(v1)
        method36(v2, v4)
        del v4
        method42(v3, v5, v6, v7, v8, v9)
        del v5, v6, v7, v8, v9
        v10 = 0
        v11 = raw_module.get_function(f"entry{v10}")
        del v10
        v11.max_dynamic_shared_size_bytes = 0 
        v11((1,),(32,),(v2, v3),shared_mem=0)
        del v2, v11
        v12, v13, v14, v15, v16 = method50(v3)
        del v3
        return method21(v12, v13, v14, v15, v16)
    return inner
def Closure2():
    def inner() -> object:
        v0 = static_array(2)
        v1 = 0
        while method20(v1):
            v3 = US5_0()
            v0[v1] = v3
            del v3
            v1 += 1 
        del v1
        v4 = US3_0()
        v5 = US4_2()
        v6 = US2_0()
        v7 = US2_1()
        return method62(v0, v4, v5, v6, v7)
    return inner
def method3(v0 : object) -> None:
    assert v0 == [], f'Expected an unit type. Got: {v0}'
    del v0
    return 
def method2(v0 : object) -> US1:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "Paper" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US1_0()
    else:
        del v4
        v7 = "Rock" == v1
        if v7:
            del v1, v7
            method3(v2)
            del v2
            return US1_1()
        else:
            del v7
            v10 = "Scissors" == v1
            if v10:
                del v1, v10
                method3(v2)
                del v2
                return US1_2()
            else:
                del v2, v10
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method5(v0 : object) -> US2:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "Computer" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US2_0()
    else:
        del v4
        v7 = "Human" == v1
        if v7:
            del v1, v7
            method3(v2)
            del v2
            return US2_1()
        else:
            del v2, v7
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method4(v0 : object) -> Tuple[US2, US2]:
    v1 = v0[0] # type: ignore
    v2 = method5(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method5(v3)
    del v3
    return v2, v4
def method1(v0 : object) -> US0:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "ActionSelected" == v1
    if v4:
        del v1, v4
        v5 = method2(v2)
        del v2
        return US0_0(v5)
    else:
        del v4
        v8 = "PlayerChanged" == v1
        if v8:
            del v1, v8
            v9, v10 = method4(v2)
            del v2
            return US0_1(v9, v10)
        else:
            del v8
            v13 = "StartGame" == v1
            if v13:
                del v1, v13
                method3(v2)
                del v2
                return US0_2()
            else:
                del v2, v13
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method0(v0 : object) -> US0:
    return method1(v0)
def method11(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method12(v0 : object) -> US5:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "None" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US5_0()
    else:
        del v4
        v7 = "Some" == v1
        if v7:
            del v1, v7
            v8 = method2(v2)
            del v2
            return US5_1(v8)
        else:
            del v2, v7
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method10(v0 : object) -> static_array:
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v1 = len(v0) # type: ignore
    v2 = 2 == v1
    v3 = v2 == False
    if v3:
        v4 = "The type level dimension has to equal the value passed at runtime into create."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = static_array(2)
    v6 = 0
    while method11(v1, v6):
        v8 = v0[v6]
        v9 = method12(v8)
        del v8
        v5[v6] = v9
        del v9
        v6 += 1 
    del v0, v1, v6
    return v5
def method9(v0 : object) -> static_array:
    v1 = v0["past_actions"] # type: ignore
    del v0
    v2 = method10(v1)
    del v1
    return v2
def method15(v0 : object) -> Tuple[US1, US1]:
    v1 = v0[0] # type: ignore
    v2 = method2(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method2(v3)
    del v3
    return v2, v4
def method16(v0 : object) -> i32:
    assert isinstance(v0,i32), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method14(v0 : object) -> US3:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "GameNotStarted" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US3_0()
    else:
        del v4
        v7 = "GameOver" == v1
        if v7:
            del v1, v7
            v8, v9 = method15(v2)
            del v2
            return US3_1(v8, v9)
        else:
            del v7
            v12 = "WaitingForActionFromPlayerId" == v1
            if v12:
                del v1, v12
                v13 = method16(v2)
                del v2
                return US3_2(v13)
            else:
                del v2, v12
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method17(v0 : object) -> US4:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "GameStarted" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US4_0()
    else:
        del v4
        v7 = "ShowdownResult" == v1
        if v7:
            del v1, v7
            v8, v9 = method15(v2)
            del v2
            return US4_1(v8, v9)
        else:
            del v7
            v12 = "WaitingToStart" == v1
            if v12:
                del v1, v12
                method3(v2)
                del v2
                return US4_2()
            else:
                del v2, v12
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method13(v0 : object) -> Tuple[US3, US4, US2, US2]:
    v1 = v0["game_state"] # type: ignore
    v2 = method14(v1)
    del v1
    v3 = v0["messages"] # type: ignore
    v4 = method17(v3)
    del v3
    v5 = v0["pl_type"] # type: ignore
    del v0
    v6, v7 = method4(v5)
    del v5
    return v2, v4, v6, v7
def method8(v0 : object) -> Tuple[static_array, US3, US4, US2, US2]:
    v1 = v0["game_state"] # type: ignore
    v2 = method9(v1)
    del v1
    v3 = v0["ui_state"] # type: ignore
    del v0
    v4, v5, v6, v7 = method13(v3)
    del v3
    return v2, v4, v5, v6, v7
def method7(v0 : object) -> Tuple[static_array, US3, US4, US2, US2]:
    v1, v2, v3, v4, v5 = method8(v0)
    del v0
    return v1, v2, v3, v4, v5
def method6(v0 : object) -> Tuple[static_array, US3, US4, US2, US2]:
    return method7(v0)
def method20(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method19(v0 : US5, v1 : static_array, v2 : US3, v3 : US4, v4 : US2, v5 : US2) -> Tuple[static_array, US3, US4, US2, US2]:
    match v2:
        case US3_0(): # GameNotStarted
            del v0
            return v1, v2, v3, v4, v5
        case US3_1(_, _): # GameOver
            del v0
            return v1, v2, v3, v4, v5
        case US3_2(v8): # WaitingForActionFromPlayerId
            v9 = v8 < 2
            if v9:
                del v9
                v10 = v8 == 0
                if v10:
                    v11 = v4
                else:
                    v11 = v5
                del v10
                match v11:
                    case US2_0(): # Computer
                        del v2, v11
                        match v0:
                            case US5_0(): # None
                                v13 = True
                            case t:
                                v13 = False
                        del v0
                        v14 = v13 == False
                        if v14:
                            v15 = "The computer player should never be receiving an action."
                            assert v13, v15
                            del v15
                        else:
                            pass
                        del v13, v14
                        v16 = US1_1()
                        v17 = US1_0()
                        v18 = US1_2()
                        v19 = random.choice([v16, v17, v18])
                        del v16, v17, v18
                        v20 = v8 + 1
                        v21 = static_array(2)
                        v22 = 0
                        while method20(v22):
                            v24 = v1[v22]
                            v25 = v8 == v22
                            if v25:
                                v27 = US5_1(v19)
                            else:
                                v27 = v24
                            del v24, v25
                            v21[v22] = v27
                            del v27
                            v22 += 1 
                        del v1, v8, v19, v22
                        v28 = US5_0()
                        v29 = US3_2(v20)
                        del v20
                        return method19(v28, v21, v29, v3, v4, v5)
                    case US2_1(): # Human
                        del v11
                        match v0:
                            case US5_0(): # None
                                del v0, v8
                                return v1, v2, v3, v4, v5
                            case US5_1(v35): # Some
                                del v0, v2
                                v36 = v8 + 1
                                v37 = static_array(2)
                                v38 = 0
                                while method20(v38):
                                    v40 = v1[v38]
                                    v41 = v8 == v38
                                    if v41:
                                        v43 = US5_1(v35)
                                    else:
                                        v43 = v40
                                    del v40, v41
                                    v37[v38] = v43
                                    del v43
                                    v38 += 1 
                                del v1, v8, v35, v38
                                v44 = US5_0()
                                v45 = US3_2(v36)
                                del v36
                                return method19(v44, v37, v45, v3, v4, v5)
                            case t:
                                raise Exception(f'Pattern matching miss. Got: {t}')
                    case t:
                        raise Exception(f'Pattern matching miss. Got: {t}')
            else:
                del v0, v2, v3, v8, v9
                v71 = v1[0]
                v72 = v1[1]
                del v1
                match v71:
                    case US5_1(v73): # Some
                        del v71
                        match v72:
                            case US5_1(v74): # Some
                                del v72
                                v75 = static_array(2)
                                v76 = 0
                                while method20(v76):
                                    v78 = US5_0()
                                    v75[v76] = v78
                                    del v78
                                    v76 += 1 
                                del v76
                                v79 = US3_1(v73, v74)
                                v80 = US4_1(v73, v74)
                                del v73, v74
                                return v75, v79, v80, v4, v5
                            case t:
                                del v4, v5, v72, v73
                                raise Exception("At showdown all the actions have to be selected.")
                    case t:
                        del v4, v5, v71, v72
                        raise Exception("At showdown all the actions have to be selected.")
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method18(v0 : static_array, v1 : US3, v2 : US4, v3 : US2, v4 : US2, v5 : US0) -> Tuple[static_array, US3, US4, US2, US2]:
    match v5:
        case US0_0(v27): # ActionSelected
            v28 = US5_1(v27)
            del v27
            v44, v45, v46, v47, v48 = method19(v28, v0, v1, v2, v3, v4)
        case US0_1(v19, v20): # PlayerChanged
            v21 = US5_0()
            v44, v45, v46, v47, v48 = method19(v21, v0, v1, v2, v19, v20)
        case US0_2(): # StartGame
            v6 = static_array(2)
            v7 = 0
            while method20(v7):
                v9 = US5_0()
                v6[v7] = v9
                del v9
                v7 += 1 
            del v7
            v10 = US5_0()
            v11 = 0
            v12 = US3_2(v11)
            del v11
            v13 = US4_0()
            v44, v45, v46, v47, v48 = method19(v10, v6, v12, v13, v3, v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v0, v1, v2, v3, v4, v5
    return v44, v45, v46, v47, v48
def method27() -> object:
    v0 = []
    return v0
def method28(v0 : US1) -> object:
    match v0:
        case US1_0(): # Paper
            del v0
            v1 = method27()
            v2 = "Paper"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # Rock
            del v0
            v4 = method27()
            v5 = "Rock"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # Scissors
            del v0
            v7 = method27()
            v8 = "Scissors"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method26(v0 : US5) -> object:
    match v0:
        case US5_0(): # None
            del v0
            v1 = method27()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US5_1(v4): # Some
            del v0
            v5 = method28(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method25(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method20(v2):
        v4 = v0[v2]
        v5 = method26(v4)
        del v4
        v1.append(v5)
        del v5
        v2 += 1 
    del v0, v2
    return v1
def method24(v0 : static_array) -> object:
    v1 = method25(v0)
    del v0
    v2 = {'past_actions': v1}
    del v1
    return v2
def method31(v0 : US1, v1 : US1) -> object:
    v2 = []
    v3 = method28(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method28(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method32(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method30(v0 : US3) -> object:
    match v0:
        case US3_0(): # GameNotStarted
            del v0
            v1 = method27()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US3_1(v4, v5): # GameOver
            del v0
            v6 = method31(v4, v5)
            del v4, v5
            v7 = "GameOver"
            v8 = [v7,v6]
            del v6, v7
            return v8
        case US3_2(v9): # WaitingForActionFromPlayerId
            del v0
            v10 = method32(v9)
            del v9
            v11 = "WaitingForActionFromPlayerId"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method33(v0 : US4) -> object:
    match v0:
        case US4_0(): # GameStarted
            del v0
            v1 = method27()
            v2 = "GameStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US4_1(v4, v5): # ShowdownResult
            del v0
            v6 = method31(v4, v5)
            del v4, v5
            v7 = "ShowdownResult"
            v8 = [v7,v6]
            del v6, v7
            return v8
        case US4_2(): # WaitingToStart
            del v0
            v9 = method27()
            v10 = "WaitingToStart"
            v11 = [v10,v9]
            del v9, v10
            return v11
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method35(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method27()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method27()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method34(v0 : US2, v1 : US2) -> object:
    v2 = []
    v3 = method35(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method35(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method29(v0 : US3, v1 : US4, v2 : US2, v3 : US2) -> object:
    v4 = method30(v0)
    del v0
    v5 = method33(v1)
    del v1
    v6 = method34(v2, v3)
    del v2, v3
    v7 = {'game_state': v4, 'messages': v5, 'pl_type': v6}
    del v4, v5, v6
    return v7
def method23(v0 : static_array, v1 : US3, v2 : US4, v3 : US2, v4 : US2) -> object:
    v5 = method24(v0)
    del v0
    v6 = method29(v1, v2, v3, v4)
    del v1, v2, v3, v4
    v7 = {'game_state': v5, 'ui_state': v6}
    del v5, v6
    return v7
def method22(v0 : static_array, v1 : US3, v2 : US4, v3 : US2, v4 : US2) -> object:
    return method23(v0, v1, v2, v3, v4)
def method21(v0 : static_array, v1 : US3, v2 : US4, v3 : US2, v4 : US2) -> object:
    v5 = method22(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    return v5
def method37(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[0:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method39(v0 : cp.ndarray) -> None:
    del v0
    return 
def method38(v0 : cp.ndarray, v1 : US1) -> None:
    v2 = v1.tag
    method37(v0, v2)
    del v2
    v3 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US1_0(): # Paper
            del v1
            return method39(v3)
        case US1_1(): # Rock
            del v1
            return method39(v3)
        case US1_2(): # Scissors
            del v1
            return method39(v3)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method41(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[4:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method40(v0 : cp.ndarray, v1 : US2, v2 : US2) -> None:
    v3 = v1.tag
    method37(v0, v3)
    del v3
    v4 = v0[4:].view(cp.uint8)
    match v1:
        case US2_0(): # Computer
            method39(v4)
        case US2_1(): # Human
            method39(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v4
    v5 = v2.tag
    method41(v0, v5)
    del v5
    v6 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US2_0(): # Computer
            del v2
            return method39(v6)
        case US2_1(): # Human
            del v2
            return method39(v6)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method36(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method37(v0, v2)
    del v2
    v3 = v0[8:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(v4): # ActionSelected
            del v1
            return method38(v3, v4)
        case US0_1(v5, v6): # PlayerChanged
            del v1
            return method40(v3, v5, v6)
        case US0_2(): # StartGame
            del v1
            return method39(v3)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method44(v0 : cp.ndarray, v1 : US5) -> None:
    v2 = v1.tag
    method37(v0, v2)
    del v2
    v3 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US5_0(): # None
            del v1
            return method39(v3)
        case US5_1(v4): # Some
            del v1
            return method38(v3, v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method45(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[16:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method46(v0 : cp.ndarray, v1 : US1, v2 : US1) -> None:
    v3 = v1.tag
    method37(v0, v3)
    del v3
    v4 = v0[4:].view(cp.uint8)
    match v1:
        case US1_0(): # Paper
            method39(v4)
        case US1_1(): # Rock
            method39(v4)
        case US1_2(): # Scissors
            method39(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v4
    v5 = v2.tag
    method41(v0, v5)
    del v5
    v6 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US1_0(): # Paper
            del v2
            return method39(v6)
        case US1_1(): # Rock
            del v2
            return method39(v6)
        case US1_2(): # Scissors
            del v2
            return method39(v6)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method47(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[32:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method48(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[48:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method49(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[52:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method43(v0 : cp.ndarray, v1 : static_array, v2 : US3, v3 : US4, v4 : US2, v5 : US2) -> None:
    v6 = 0
    while method20(v6):
        v8 = u64(v6)
        v9 = v8 * 8
        del v8
        v10 = v0[v9:].view(cp.uint8)
        del v9
        v11 = v1[v6]
        method44(v10, v11)
        del v10, v11
        v6 += 1 
    del v1, v6
    v12 = v2.tag
    method45(v0, v12)
    del v12
    v13 = v0[24:].view(cp.uint8)
    match v2:
        case US3_0(): # GameNotStarted
            method39(v13)
        case US3_1(v14, v15): # GameOver
            method46(v13, v14, v15)
        case US3_2(v16): # WaitingForActionFromPlayerId
            method37(v13, v16)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v2, v13
    v17 = v3.tag
    method47(v0, v17)
    del v17
    v18 = v0[40:].view(cp.uint8)
    match v3:
        case US4_0(): # GameStarted
            method39(v18)
        case US4_1(v19, v20): # ShowdownResult
            method46(v18, v19, v20)
        case US4_2(): # WaitingToStart
            method39(v18)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v3, v18
    v21 = v4.tag
    method48(v0, v21)
    del v21
    v22 = v0[52:].view(cp.uint8)
    match v4:
        case US2_0(): # Computer
            method39(v22)
        case US2_1(): # Human
            method39(v22)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v4, v22
    v23 = v5.tag
    method49(v0, v23)
    del v23
    v24 = v0[56:].view(cp.uint8)
    del v0
    match v5:
        case US2_0(): # Computer
            del v5
            return method39(v24)
        case US2_1(): # Human
            del v5
            return method39(v24)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method42(v0 : cp.ndarray, v1 : static_array, v2 : US3, v3 : US4, v4 : US2, v5 : US2) -> None:
    return method43(v0, v1, v2, v3, v4, v5)
def method53(v0 : cp.ndarray) -> i32:
    v1 = v0[0:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method54(v0 : cp.ndarray) -> None:
    del v0
    return 
def method55(v0 : cp.ndarray) -> US1:
    v1 = method53(v0)
    v2 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method54(v2)
        del v2
        return US1_0()
    elif v1 == 1:
        del v1
        method54(v2)
        del v2
        return US1_1()
    elif v1 == 2:
        del v1
        method54(v2)
        del v2
        return US1_2()
    else:
        del v1, v2
        raise Exception("Invalid tag.")
def method52(v0 : cp.ndarray) -> US5:
    v1 = method53(v0)
    v2 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method54(v2)
        del v2
        return US5_0()
    elif v1 == 1:
        del v1
        v5 = method55(v2)
        del v2
        return US5_1(v5)
    else:
        del v1, v2
        raise Exception("Invalid tag.")
def method56(v0 : cp.ndarray) -> i32:
    v1 = v0[16:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method58(v0 : cp.ndarray) -> i32:
    v1 = v0[4:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method57(v0 : cp.ndarray) -> Tuple[US1, US1]:
    v1 = method53(v0)
    v2 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method54(v2)
        v7 = US1_0()
    elif v1 == 1:
        method54(v2)
        v7 = US1_1()
    elif v1 == 2:
        method54(v2)
        v7 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v1, v2
    v8 = method58(v0)
    v9 = v0[8:].view(cp.uint8)
    del v0
    if v8 == 0:
        method54(v9)
        v14 = US1_0()
    elif v8 == 1:
        method54(v9)
        v14 = US1_1()
    elif v8 == 2:
        method54(v9)
        v14 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v8, v9
    return v7, v14
def method59(v0 : cp.ndarray) -> i32:
    v1 = v0[32:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method60(v0 : cp.ndarray) -> i32:
    v1 = v0[48:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method61(v0 : cp.ndarray) -> i32:
    v1 = v0[52:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method51(v0 : cp.ndarray) -> Tuple[static_array, US3, US4, US2, US2]:
    v1 = static_array(2)
    v2 = 0
    while method20(v2):
        v4 = u64(v2)
        v5 = v4 * 8
        del v4
        v6 = v0[v5:].view(cp.uint8)
        del v5
        v7 = method52(v6)
        del v6
        v1[v2] = v7
        del v7
        v2 += 1 
    del v2
    v8 = method56(v0)
    v9 = v0[24:].view(cp.uint8)
    if v8 == 0:
        method54(v9)
        v17 = US3_0()
    elif v8 == 1:
        v12, v13 = method57(v9)
        v17 = US3_1(v12, v13)
    elif v8 == 2:
        v15 = method53(v9)
        v17 = US3_2(v15)
    else:
        raise Exception("Invalid tag.")
    del v8, v9
    v18 = method59(v0)
    v19 = v0[40:].view(cp.uint8)
    if v18 == 0:
        method54(v19)
        v26 = US4_0()
    elif v18 == 1:
        v22, v23 = method57(v19)
        v26 = US4_1(v22, v23)
    elif v18 == 2:
        method54(v19)
        v26 = US4_2()
    else:
        raise Exception("Invalid tag.")
    del v18, v19
    v27 = method60(v0)
    v28 = v0[52:].view(cp.uint8)
    if v27 == 0:
        method54(v28)
        v32 = US2_0()
    elif v27 == 1:
        method54(v28)
        v32 = US2_1()
    else:
        raise Exception("Invalid tag.")
    del v27, v28
    v33 = method61(v0)
    v34 = v0[56:].view(cp.uint8)
    del v0
    if v33 == 0:
        method54(v34)
        v38 = US2_0()
    elif v33 == 1:
        method54(v34)
        v38 = US2_1()
    else:
        raise Exception("Invalid tag.")
    del v33, v34
    return v1, v17, v26, v32, v38
def method50(v0 : cp.ndarray) -> Tuple[static_array, US3, US4, US2, US2]:
    v1, v2, v3, v4, v5 = method51(v0)
    del v0
    return v1, v2, v3, v4, v5
def method62(v0 : static_array, v1 : US3, v2 : US4, v3 : US2, v4 : US2) -> object:
    v5 = method23(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    return v5
def main():
    v0 = Closure0()
    v1 = Closure1()
    v2 = Closure2()
    v3 = collections.namedtuple("RPS_Game",['event_loop_cpu', 'event_loop_gpu', 'init'])(v0, v1, v2)
    del v0, v1, v2
    return v3

if __name__ == '__main__': print(main())
