#include "test1.auto.cu"
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
struct Union0;
struct Union1;
struct StackMut0;
__device__ void method_1(unsigned int * v0, static_array_list<Union0,5> v1, Union1 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6);
struct Tuple0;
struct Tuple1;
__device__ int method_3(unsigned int * v0, StackMut0 & v1, int v2);
struct Union2;
struct StackMut1;
__device__ void method_4(unsigned int * v0, StackMut0 & v1, int v2);
struct Union3;
struct StackMut2;
__device__ void method_5(unsigned int * v0, StackMut0 & v1, int v2);
__device__ Tuple0 method_2(unsigned int * v0);
__device__ void method_6(Union0 v0);
__device__ void method_7(Union1 v0);
void run_cuda_host_0();
struct Union0_0 { // Call
};
struct Union0_1 { // Fold
};
struct Union0_2 { // Raise
    int v0;
    __host__ __device__ Union0_2(int t0) : v0(t0) {}
    __host__ __device__ Union0_2() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // Call
        Union0_1 case1; // Fold
        Union0_2 case2; // Raise
    };
    unsigned char tag{255};
    __host__ __device__ Union0() {}
    __host__ __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // Call
    __host__ __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Fold
    __host__ __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // Raise
    __host__ __device__ Union0(const Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // Call
            case 1: new (&this->case1) Union0_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union0_2(x.case2); break; // Raise
        }
    }
    __host__ __device__ Union0(const Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // Raise
        }
    }
    __host__ __device__ Union0 & operator=(const Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Call
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // Raise
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
                case 0: this->case0 = std::move(x.case0); break; // Call
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // Raise
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // Call
            case 1: this->case1.~Union0_1(); break; // Fold
            case 2: this->case2.~Union0_2(); break; // Raise
        }
        this->tag = 255;
    }
};
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
struct StackMut0 {
    int v0;
    __host__ __device__ StackMut0() = default;
    __host__ __device__ StackMut0(int t0) : v0(t0) {}
};
struct Tuple0 {
    static_array_list<Union0,5> v0;
    Union1 v1;
    static_array<int,3> v3;
    int v4;
    unsigned int v5;
    bool v2;
    __host__ __device__ Tuple0() = default;
    __host__ __device__ Tuple0(static_array_list<Union0,5> t0, Union1 t1, bool t2, static_array<int,3> t3, int t4, unsigned int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple1 {
    int v0;
    int v1;
    int v2;
    __host__ __device__ Tuple1() = default;
    __host__ __device__ Tuple1(int t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union2_0 { // None
};
struct Union2_1 { // Some
    Union1 v0;
    __host__ __device__ Union2_1(Union1 t0) : v0(t0) {}
    __host__ __device__ Union2_1() = delete;
};
struct Union2 {
    union {
        Union2_0 case0; // None
        Union2_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union2() {}
    __host__ __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union2(const Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // None
            case 1: new (&this->case1) Union2_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union2(const Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union2 & operator=(const Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union2();
            new (this) Union2{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // None
            case 1: this->case1.~Union2_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct StackMut1 {
    Union2 v0;
    __host__ __device__ StackMut1() = default;
    __host__ __device__ StackMut1(Union2 t0) : v0(t0) {}
};
struct Union3_0 { // None
};
struct Union3_1 { // Some
    Union0 v0;
    __host__ __device__ Union3_1(Union0 t0) : v0(t0) {}
    __host__ __device__ Union3_1() = delete;
};
struct Union3 {
    union {
        Union3_0 case0; // None
        Union3_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union3() {}
    __host__ __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union3(const Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // None
            case 1: new (&this->case1) Union3_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union3(const Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union3 & operator=(const Union3 & x) {
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
    __host__ __device__ Union3 & operator=(const Union3 && x) {
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
    __host__ __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // None
            case 1: this->case1.~Union3_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct StackMut2 {
    Union3 v0;
    __host__ __device__ StackMut2() = default;
    __host__ __device__ StackMut2(Union3 t0) : v0(t0) {}
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 115;
    return v1;
}
__device__ inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 5;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
__device__ void method_1(unsigned int * v0, static_array_list<Union0,5> v1, Union1 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6){
    StackMut0 v7{0};
    int v8;
    v8 = (int)v6;
    bool v9;
    v9 = v8 < 10;
    bool v10;
    v10 = v9 == false;
    if (v10){
        assert("The input to the pickler must be 0 or positive." && v9);
    } else {
    }
    int v12 = v7.v0;
    int v13;
    v13 = v12 + v8;
    int v14;
    v14 = v13 / 32;
    unsigned int v15;
    v15 = v0[v14];
    int v16;
    v16 = v13 % 32;
    unsigned int v17;
    v17 = 1u << v16;
    unsigned int v18;
    v18 = v15 | v17;
    v0[v14] = v18;
    int v19 = v7.v0;
    int v20;
    v20 = v19 + 10;
    v7.v0 = v20;
    bool v21;
    v21 = v5 < 10;
    bool v22;
    v22 = v21 == false;
    if (v22){
        assert("The input to the pickler must be 0 or positive." && v21);
    } else {
    }
    bool v24;
    v24 = 0 <= v5;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The input to the pickler must be less than the specified length." && v24);
    } else {
    }
    int v27 = v7.v0;
    int v28;
    v28 = v27 + v5;
    int v29;
    v29 = v28 / 32;
    unsigned int v30;
    v30 = v0[v29];
    int v31;
    v31 = v28 % 32;
    unsigned int v32;
    v32 = 1u << v31;
    unsigned int v33;
    v33 = v30 | v32;
    v0[v29] = v33;
    int v34 = v7.v0;
    int v35;
    v35 = v34 + 10;
    v7.v0 = v35;
    int v36;
    if (v3){
        v36 = 1;
    } else {
        v36 = 0;
    }
    int v37 = v7.v0;
    int v38;
    v38 = v37 + v36;
    int v39;
    v39 = v38 / 32;
    unsigned int v40;
    v40 = v0[v39];
    int v41;
    v41 = v38 % 32;
    unsigned int v42;
    v42 = 1u << v41;
    unsigned int v43;
    v43 = v40 | v42;
    v0[v39] = v43;
    int v44 = v7.v0;
    int v45;
    v45 = v44 + 2;
    v7.v0 = v45;
    int v46 = v7.v0;
    int v47;
    v47 = v46 + 3;
    int v48;
    v48 = v2.tag;
    int v49 = v7.v0;
    int v50;
    v50 = v49 + v48;
    int v51;
    v51 = v50 / 32;
    unsigned int v52;
    v52 = v0[v51];
    int v53;
    v53 = v50 % 32;
    unsigned int v54;
    v54 = 1u << v53;
    unsigned int v55;
    v55 = v52 | v54;
    v0[v51] = v55;
    int v56 = v7.v0;
    int v57;
    v57 = v56 + 3;
    v7.v0 = v57;
    switch (v2.tag) {
        case 0: { // Jack
            int v58 = v7.v0;
            v7.v0 = v58;
            break;
        }
        case 1: { // King
            int v59 = v7.v0;
            v7.v0 = v59;
            break;
        }
        case 2: { // Queen
            int v60 = v7.v0;
            v7.v0 = v60;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            __trap();
        }
    }
    v7.v0 = v47;
    int v61;
    v61 = v1.length;
    int v62 = v7.v0;
    int v63;
    v63 = v62 + v61;
    int v64;
    v64 = v63 / 32;
    unsigned int v65;
    v65 = v0[v64];
    int v66;
    v66 = v63 % 32;
    unsigned int v67;
    v67 = 1u << v66;
    unsigned int v68;
    v68 = v65 | v67;
    v0[v64] = v68;
    int v69 = v7.v0;
    int v70;
    v70 = v69 + 5;
    v7.v0 = v70;
    int v71;
    v71 = v1.length;
    int v72;
    v72 = 0;
    while (while_method_1(v71, v72)){
        Union0 v74;
        v74 = v1[v72];
        int v77 = v7.v0;
        int v78;
        v78 = v77 + 13;
        int v79;
        v79 = v74.tag;
        int v80 = v7.v0;
        int v81;
        v81 = v80 + v79;
        int v82;
        v82 = v81 / 32;
        unsigned int v83;
        v83 = v0[v82];
        int v84;
        v84 = v81 % 32;
        unsigned int v85;
        v85 = 1u << v84;
        unsigned int v86;
        v86 = v83 | v85;
        v0[v82] = v86;
        int v87 = v7.v0;
        int v88;
        v88 = v87 + 3;
        v7.v0 = v88;
        switch (v74.tag) {
            case 0: { // Call
                int v89 = v7.v0;
                v7.v0 = v89;
                break;
            }
            case 1: { // Fold
                int v90 = v7.v0;
                v7.v0 = v90;
                break;
            }
            case 2: { // Raise
                int v91 = v74.case2.v0;
                int v92 = v7.v0;
                v7.v0 = v92;
                bool v93;
                v93 = v91 < 10;
                bool v94;
                v94 = v93 == false;
                if (v94){
                    assert("The input to the pickler must be 0 or positive." && v93);
                } else {
                }
                bool v96;
                v96 = 0 <= v91;
                bool v97;
                v97 = v96 == false;
                if (v97){
                    assert("The input to the pickler must be less than the specified length." && v96);
                } else {
                }
                int v99 = v7.v0;
                int v100;
                v100 = v99 + v91;
                int v101;
                v101 = v100 / 32;
                unsigned int v102;
                v102 = v0[v101];
                int v103;
                v103 = v100 % 32;
                unsigned int v104;
                v104 = 1u << v103;
                unsigned int v105;
                v105 = v102 | v104;
                v0[v101] = v105;
                int v106 = v7.v0;
                int v107;
                v107 = v106 + 10;
                v7.v0 = v107;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                __trap();
            }
        }
        v7.v0 = v78;
        int v108 = v7.v0;
        int v109;
        v109 = v108 + 1;
        v7.v0 = v109;
        v72 += 1 ;
    }
    int v110;
    v110 = v61;
    while (while_method_2(v110)){
        int v112 = v7.v0;
        int v113;
        v113 = v112 + 13;
        v7.v0 = v113;
        int v114 = v7.v0;
        int v115;
        v115 = v114 / 32;
        unsigned int v116;
        v116 = v0[v115];
        int v117;
        v117 = v114 % 32;
        unsigned int v118;
        v118 = 1u << v117;
        unsigned int v119;
        v119 = v116 | v118;
        v0[v115] = v119;
        int v120 = v7.v0;
        int v121;
        v121 = v120 + 1;
        v7.v0 = v121;
        v110 += 1 ;
    }
    int v122;
    v122 = 0;
    while (while_method_3(v122)){
        int v124;
        v124 = v4[v122];
        bool v127;
        v127 = v124 < 5;
        bool v128;
        v128 = v127 == false;
        if (v128){
            assert("The input to the pickler must be 0 or positive." && v127);
        } else {
        }
        bool v130;
        v130 = 0 <= v124;
        bool v131;
        v131 = v130 == false;
        if (v131){
            assert("The input to the pickler must be less than the specified length." && v130);
        } else {
        }
        int v133 = v7.v0;
        int v134;
        v134 = v133 + v124;
        int v135;
        v135 = v134 / 32;
        unsigned int v136;
        v136 = v0[v135];
        int v137;
        v137 = v134 % 32;
        unsigned int v138;
        v138 = 1u << v137;
        unsigned int v139;
        v139 = v136 | v138;
        v0[v135] = v139;
        int v140 = v7.v0;
        int v141;
        v141 = v140 + 5;
        v7.v0 = v141;
        v122 += 1 ;
    }
    return ;
}
__device__ int method_3(unsigned int * v0, StackMut0 & v1, int v2){
    int v3; int v4; int v5;
    Tuple1 tmp0 = Tuple1{0, 0, 0};
    v3 = tmp0.v0; v4 = tmp0.v1; v5 = tmp0.v2;
    while (while_method_1(v2, v3)){
        int v7 = v1.v0;
        int v8;
        v8 = v7 / 32;
        unsigned int v9;
        v9 = v0[v8];
        int v10;
        v10 = v7 % 32;
        unsigned int v11;
        v11 = 1u << v10;
        unsigned int v12;
        v12 = v9 & v11;
        bool v13;
        v13 = v12 == 0u;
        bool v14;
        v14 = v13 != true;
        int v15 = v1.v0;
        int v16;
        v16 = v15 + 1;
        v1.v0 = v16;
        int v18; int v19;
        if (v14){
            int v17;
            v17 = v5 + 1;
            v18 = v3; v19 = v17;
        } else {
            v18 = v4; v19 = v5;
        }
        v4 = v18;
        v5 = v19;
        v3 += 1 ;
    }
    bool v20;
    v20 = v5 == 1;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("The number of integers being scanned in unpickle needs to be exactly 1." && v20);
    } else {
    }
    return v4;
}
__device__ void method_4(unsigned int * v0, StackMut0 & v1, int v2){
    int v3; int v4; int v5;
    Tuple1 tmp1 = Tuple1{0, 0, 0};
    v3 = tmp1.v0; v4 = tmp1.v1; v5 = tmp1.v2;
    while (while_method_1(v2, v3)){
        int v7 = v1.v0;
        int v8;
        v8 = v7 / 32;
        unsigned int v9;
        v9 = v0[v8];
        int v10;
        v10 = v7 % 32;
        unsigned int v11;
        v11 = 1u << v10;
        unsigned int v12;
        v12 = v9 & v11;
        bool v13;
        v13 = v12 == 0u;
        bool v14;
        v14 = v13 != true;
        int v15 = v1.v0;
        int v16;
        v16 = v15 + 1;
        v1.v0 = v16;
        int v18; int v19;
        if (v14){
            int v17;
            v17 = v5 + 1;
            v18 = v3; v19 = v17;
        } else {
            v18 = v4; v19 = v5;
        }
        v4 = v18;
        v5 = v19;
        v3 += 1 ;
    }
    bool v20;
    v20 = v5 == 0;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("Expected that the elements of this particular element in the unpickler would all be 0. `ensure` check failed." && v20);
        return ;
    } else {
        return ;
    }
}
__device__ void method_5(unsigned int * v0, StackMut0 & v1, int v2){
    int v3; int v4; int v5;
    Tuple1 tmp2 = Tuple1{0, 0, 0};
    v3 = tmp2.v0; v4 = tmp2.v1; v5 = tmp2.v2;
    while (while_method_1(v2, v3)){
        int v7 = v1.v0;
        int v8;
        v8 = v7 / 32;
        unsigned int v9;
        v9 = v0[v8];
        int v10;
        v10 = v7 % 32;
        unsigned int v11;
        v11 = 1u << v10;
        unsigned int v12;
        v12 = v9 & v11;
        bool v13;
        v13 = v12 == 0u;
        bool v14;
        v14 = v13 != true;
        int v15 = v1.v0;
        int v16;
        v16 = v15 + 1;
        v1.v0 = v16;
        int v18; int v19;
        if (v14){
            int v17;
            v17 = v5 + 1;
            v18 = v3; v19 = v17;
        } else {
            v18 = v4; v19 = v5;
        }
        v4 = v18;
        v5 = v19;
        v3 += 1 ;
    }
    bool v20;
    v20 = v5 == v2;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("Expected that the elements of this particular element in the unpickler would all be 1. `ensure` check failed." && v20);
        return ;
    } else {
        return ;
    }
}
__device__ Tuple0 method_2(unsigned int * v0){
    StackMut0 v1{0};
    int v2;
    v2 = 10;
    int v3;
    v3 = method_3(v0, v1, v2);
    unsigned int v4;
    v4 = (unsigned int)v3;
    int v5;
    v5 = 10;
    int v6;
    v6 = method_3(v0, v1, v5);
    int v7;
    v7 = 2;
    int v8;
    v8 = method_3(v0, v1, v7);
    bool v9;
    v9 = v8 == 1;
    Union2 v10;
    v10 = Union2{Union2_0{}};
    StackMut1 v11{v10};
    int v12;
    v12 = 3;
    int v13;
    v13 = method_3(v0, v1, v12);
    bool v14;
    v14 = 0 == v13;
    if (v14){
        Union2 v15 = v11.v0;
        switch (v15.tag) {
            case 0: { // None
                Union1 v16;
                v16 = Union1{Union1_0{}};
                Union2 v17;
                v17 = Union2{Union2_1{v16}};
                v11.v0 = v17;
                break;
            }
            case 1: { // Some
                Union1 v18 = v15.case1.v0;
                bool v19;
                v19 = false;
                bool v20;
                v20 = v19 == false;
                if (v20){
                    assert("Duplicate union type instances in the unpickle Alt case." && v19);
                } else {
                }
                break;
            }
            default: {
                assert("Invalid tag." && false);
                __trap();
            }
        }
    } else {
        int v22;
        v22 = 0;
        method_4(v0, v1, v22);
    }
    bool v23;
    v23 = 1 == v13;
    if (v23){
        Union2 v24 = v11.v0;
        switch (v24.tag) {
            case 0: { // None
                Union1 v25;
                v25 = Union1{Union1_1{}};
                Union2 v26;
                v26 = Union2{Union2_1{v25}};
                v11.v0 = v26;
                break;
            }
            case 1: { // Some
                Union1 v27 = v24.case1.v0;
                bool v28;
                v28 = false;
                bool v29;
                v29 = v28 == false;
                if (v29){
                    assert("Duplicate union type instances in the unpickle Alt case." && v28);
                } else {
                }
                break;
            }
            default: {
                assert("Invalid tag." && false);
                __trap();
            }
        }
    } else {
        int v31;
        v31 = 0;
        method_4(v0, v1, v31);
    }
    bool v32;
    v32 = 2 == v13;
    if (v32){
        Union2 v33 = v11.v0;
        switch (v33.tag) {
            case 0: { // None
                Union1 v34;
                v34 = Union1{Union1_2{}};
                Union2 v35;
                v35 = Union2{Union2_1{v34}};
                v11.v0 = v35;
                break;
            }
            case 1: { // Some
                Union1 v36 = v33.case1.v0;
                bool v37;
                v37 = false;
                bool v38;
                v38 = v37 == false;
                if (v38){
                    assert("Duplicate union type instances in the unpickle Alt case." && v37);
                } else {
                }
                break;
            }
            default: {
                assert("Invalid tag." && false);
                __trap();
            }
        }
    } else {
        int v40;
        v40 = 0;
        method_4(v0, v1, v40);
    }
    Union2 v41 = v11.v0;
    Union1 v45;
    switch (v41.tag) {
        case 0: { // None
            printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
            __trap();
            break;
        }
        case 1: { // Some
            Union1 v42 = v41.case1.v0;
            v45 = v42;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            __trap();
        }
    }
    int v46;
    v46 = 5;
    int v47;
    v47 = method_3(v0, v1, v46);
    static_array_list<Union0,5> v48;
    v48 = static_array_list<Union0,5>{};
    int v51;
    v51 = 0;
    while (while_method_1(v47, v51)){
        Union3 v53;
        v53 = Union3{Union3_0{}};
        StackMut2 v54{v53};
        int v55;
        v55 = 3;
        int v56;
        v56 = method_3(v0, v1, v55);
        bool v57;
        v57 = 0 == v56;
        if (v57){
            Union3 v58 = v54.v0;
            switch (v58.tag) {
                case 0: { // None
                    Union0 v59;
                    v59 = Union0{Union0_0{}};
                    Union3 v60;
                    v60 = Union3{Union3_1{v59}};
                    v54.v0 = v60;
                    break;
                }
                case 1: { // Some
                    Union0 v61 = v58.case1.v0;
                    bool v62;
                    v62 = false;
                    bool v63;
                    v63 = v62 == false;
                    if (v63){
                        assert("Duplicate union type instances in the unpickle Alt case." && v62);
                    } else {
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    __trap();
                }
            }
        } else {
            int v65;
            v65 = 0;
            method_4(v0, v1, v65);
        }
        bool v66;
        v66 = 1 == v56;
        if (v66){
            Union3 v67 = v54.v0;
            switch (v67.tag) {
                case 0: { // None
                    Union0 v68;
                    v68 = Union0{Union0_1{}};
                    Union3 v69;
                    v69 = Union3{Union3_1{v68}};
                    v54.v0 = v69;
                    break;
                }
                case 1: { // Some
                    Union0 v70 = v67.case1.v0;
                    bool v71;
                    v71 = false;
                    bool v72;
                    v72 = v71 == false;
                    if (v72){
                        assert("Duplicate union type instances in the unpickle Alt case." && v71);
                    } else {
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    __trap();
                }
            }
        } else {
            int v74;
            v74 = 0;
            method_4(v0, v1, v74);
        }
        bool v75;
        v75 = 2 == v56;
        if (v75){
            Union3 v76 = v54.v0;
            switch (v76.tag) {
                case 0: { // None
                    int v77;
                    v77 = 10;
                    int v78;
                    v78 = method_3(v0, v1, v77);
                    Union0 v79;
                    v79 = Union0{Union0_2{v78}};
                    Union3 v80;
                    v80 = Union3{Union3_1{v79}};
                    v54.v0 = v80;
                    break;
                }
                case 1: { // Some
                    Union0 v81 = v76.case1.v0;
                    bool v82;
                    v82 = false;
                    bool v83;
                    v83 = v82 == false;
                    if (v83){
                        assert("Duplicate union type instances in the unpickle Alt case." && v82);
                    } else {
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    __trap();
                }
            }
        } else {
            int v85;
            v85 = 10;
            method_4(v0, v1, v85);
        }
        Union3 v86 = v54.v0;
        Union0 v90;
        switch (v86.tag) {
            case 0: { // None
                printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                __trap();
                break;
            }
            case 1: { // Some
                Union0 v87 = v86.case1.v0;
                v90 = v87;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                __trap();
            }
        }
        v48.push(v90);
        int v91;
        v91 = 1;
        method_4(v0, v1, v91);
        v51 += 1 ;
    }
    int v92;
    v92 = v47;
    while (while_method_2(v92)){
        int v94;
        v94 = 13;
        method_4(v0, v1, v94);
        int v95;
        v95 = 1;
        method_5(v0, v1, v95);
        v92 += 1 ;
    }
    static_array<int,3> v96;
    int v99;
    v99 = 0;
    while (while_method_3(v99)){
        int v101;
        v101 = 5;
        int v102;
        v102 = method_3(v0, v1, v101);
        v96[v99] = v102;
        v99 += 1 ;
    }
    return Tuple0{v48, v45, v9, v96, v6, v4};
}
__device__ void method_6(Union0 v0){
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
            int v1 = v0.case2.v0;
            printf("%s(%d)","Raise", v1);
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            __trap();
        }
    }
}
__device__ void method_7(Union1 v0){
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
            __trap();
        }
    }
}
extern "C" __global__ void __cluster_dims__(12,1,1) global_entry0() {
    int v0;
    v0 = threadIdx.x;
    int v1;
    v1 = blockIdx.x;
    int v2;
    v2 = v1 * 256;
    int v3;
    v3 = v0 + v2;
    bool v4;
    v4 = v3 == 0;
    if (v4){
        static_array_list<Union0,5> v5;
        v5 = static_array_list<Union0,5>{};
        v5.unsafe_set_length(4);
        Union0 v8;
        v8 = Union0{Union0_2{3}};
        v5[0] = v8;
        Union0 v11;
        v11 = Union0{Union0_2{4}};
        v5[1] = v11;
        Union0 v14;
        v14 = Union0{Union0_0{}};
        v5[2] = v14;
        Union0 v17;
        v17 = Union0{Union0_1{}};
        v5[3] = v17;
        static_array<int,3> v20;
        v20[0] = 1;
        v20[1] = 2;
        v20[2] = 3;
        unsigned int v23[4];
        int v24;
        v24 = 0;
        while (while_method_0(v24)){
            int v26;
            v26 = v24 / 32;
            unsigned int v27;
            v27 = v23[v26];
            int v28;
            v28 = v24 % 32;
            unsigned int v29;
            v29 = 1u << v28;
            unsigned int v30;
            v30 = ~v29;
            unsigned int v31;
            v31 = v27 & v30;
            v23[v26] = v31;
            v24 += 1 ;
        }
        Union1 v32;
        v32 = Union1{Union1_1{}};
        bool v33;
        v33 = false;
        int v34;
        v34 = 8;
        unsigned int v35;
        v35 = 5u;
        method_1(v23, v5, v32, v33, v20, v34, v35);
        static_array_list<Union0,5> v36; Union1 v37; bool v38; static_array<int,3> v39; int v40; unsigned int v41;
        Tuple0 tmp3 = method_2(v23);
        v36 = tmp3.v0; v37 = tmp3.v1; v38 = tmp3.v2; v39 = tmp3.v3; v40 = tmp3.v4; v41 = tmp3.v5;
        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v42 = console_lock;
        auto v43 = cooperative_groups::coalesced_threads();
        v42.acquire();
        printf("{%s = %s","action_history", "[");
        int v44;
        v44 = v36.length;
        bool v45;
        v45 = 100 < v44;
        int v46;
        if (v45){
            v46 = 100;
        } else {
            v46 = v44;
        }
        int v47;
        v47 = 0;
        while (while_method_1(v46, v47)){
            Union0 v49;
            v49 = v36[v47];
            printf("");
            method_6(v49);
            printf("");
            int v52;
            v52 = v47 + 1;
            int v53;
            v53 = v36.length;
            bool v54;
            v54 = v52 < v53;
            if (v54){
                printf("%s","; ");
            } else {
            }
            v47 += 1 ;
        }
        int v55;
        v55 = v36.length;
        bool v56;
        v56 = v55 > 100;
        if (v56){
            printf("%s","; ...");
        } else {
        }
        printf("%s","]");
        printf("; %s = ","card");
        method_7(v37);
        const char * v59;
        if (v38){
            const char * v57;
            v57 = "true";
            v59 = v57;
        } else {
            const char * v58;
            v58 = "false";
            v59 = v58;
        }
        printf("; %s = %s; %s = %s","is_first", v59, "l", "[");
        int v60;
        v60 = 0;
        while (while_method_3(v60)){
            int v62;
            v62 = v39[v60];
            printf("%d",v62);
            int v65;
            v65 = v60 + 1;
            bool v66;
            v66 = v65 < 3;
            if (v66){
                printf("%s","; ");
            } else {
            }
            v60 += 1 ;
        }
        printf("%s","]");
        printf("; %s = %d; %s = %u}\n","pot", v40, "stack", v41);
        v42.release();
        v43.sync() ;
    } else {
    }
    return ;
}
void run_cuda_host_0(){
    auto kernel = global_entry0;
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 12));
    cudaLaunchConfig_t v0 = {0};
    v0.gridDim = 84;
    v0.blockDim = 256;
    v0.dynamicSmemBytes = 98304;
    cudaLaunchAttribute v1;
    v1.id = cudaLaunchAttributeCooperative;
    v1.val.cooperative = 1;
    v0.numAttrs = 1;
    cudaLaunchAttribute v2[] = { v1 };
    v0.attrs = v2;
    int v3;
    v3 = 0;
    cudaOccupancyMaxPotentialClusterSize(&v3, (void *)kernel, &v0);
    bool v4;
    v4 = v3 >= 12;
    bool v5;
    v5 = v4 == false;
    if (v5){
        assert("Max potential cluster size must be greater than or equal to the given cluster size." && v4);
    } else {
    }
    gpuErrchk(cudaLaunchKernelEx(&v0, kernel));
    return ;
}
int main() {
    run_cuda_host_0();
    gpuErrchk(cudaDeviceSynchronize());
    return 0;
}
