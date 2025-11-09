kernels_main = r"""
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
struct Union0;
struct Union1;
struct StackMut0;
struct Tuple0;
struct Tuple1;
struct Union2;
struct StackMut1;
struct Union3;
struct StackMut2;
__device__ void method_0(unsigned int * v0, static_array_list<Union0,5> v1, Union1 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6);
__device__ int method_2(unsigned int * v0, StackMut0 & v1, int v2);
__device__ void method_3(unsigned int * v0, StackMut0 & v1, int v2);
__device__ void method_4(unsigned int * v0, StackMut0 & v1, int v2);
__device__ Tuple0 method_1(unsigned int * v0);
__device__ void method_5(Union0 v0);
__device__ void method_6(Union1 v0);
extern "C" __global__ void entry0();
struct Union0_0 { // Call
};
struct Union0_1 { // Fold
};
struct Union0_2 { // Raise
    int v0;
    __device__ Union0_2(int t0) : v0(t0) {}
    __device__ Union0_2() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // Call
        Union0_1 case1; // Fold
        Union0_2 case2; // Raise
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // Call
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Fold
    __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // Raise
    __device__ Union0(const Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // Call
            case 1: new (&this->case1) Union0_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union0_2(x.case2); break; // Raise
        }
    }
    __device__ Union0(const Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // Raise
        }
    }
    __device__ Union0 & operator=(const Union0 & x) {
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
    __device__ Union0 & operator=(const Union0 && x) {
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
    __device__ ~Union0() {
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
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // Jack
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // King
    __device__ Union1(Union1_2 t) : tag(2), case2(t) {} // Queen
    __device__ Union1(const Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union1_1(x.case1); break; // King
            case 2: new (&this->case2) Union1_2(x.case2); break; // Queen
        }
    }
    __device__ Union1(const Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // Queen
        }
    }
    __device__ Union1 & operator=(const Union1 & x) {
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
    __device__ Union1 & operator=(const Union1 && x) {
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
    __device__ ~Union1() {
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
    __device__ StackMut0() = default;
    __device__ StackMut0(int t0) : v0(t0) {}
};
struct Tuple0 {
    static_array_list<Union0,5> v0;
    Union1 v1;
    static_array<int,3> v3;
    int v4;
    unsigned int v5;
    bool v2;
    __device__ Tuple0() = default;
    __device__ Tuple0(static_array_list<Union0,5> t0, Union1 t1, bool t2, static_array<int,3> t3, int t4, unsigned int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple1 {
    int v0;
    int v1;
    int v2;
    __device__ Tuple1() = default;
    __device__ Tuple1(int t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union2_0 { // None
};
struct Union2_1 { // Some
    Union1 v0;
    __device__ Union2_1(Union1 t0) : v0(t0) {}
    __device__ Union2_1() = delete;
};
struct Union2 {
    union {
        Union2_0 case0; // None
        Union2_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union2() {}
    __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // None
    __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // Some
    __device__ Union2(const Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // None
            case 1: new (&this->case1) Union2_1(x.case1); break; // Some
        }
    }
    __device__ Union2(const Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union2 & operator=(const Union2 & x) {
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
    __device__ Union2 & operator=(const Union2 && x) {
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
    __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // None
            case 1: this->case1.~Union2_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct StackMut1 {
    Union2 v0;
    __device__ StackMut1() = default;
    __device__ StackMut1(Union2 t0) : v0(t0) {}
};
struct Union3_0 { // None
};
struct Union3_1 { // Some
    Union0 v0;
    __device__ Union3_1(Union0 t0) : v0(t0) {}
    __device__ Union3_1() = delete;
};
struct Union3 {
    union {
        Union3_0 case0; // None
        Union3_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union3() {}
    __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // None
    __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // Some
    __device__ Union3(const Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // None
            case 1: new (&this->case1) Union3_1(x.case1); break; // Some
        }
    }
    __device__ Union3(const Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union3 & operator=(const Union3 & x) {
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
    __device__ Union3 & operator=(const Union3 && x) {
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
        this->tag = 255;
    }
};
struct StackMut2 {
    Union3 v0;
    __device__ StackMut2() = default;
    __device__ StackMut2(Union3 t0) : v0(t0) {}
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
__device__ void method_0(unsigned int * v0, static_array_list<Union0,5> v1, Union1 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6){
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
        Union0 v75;
        v75 = v1[v72];
        int v78 = v7.v0;
        int v79;
        v79 = v78 + 13;
        int v80;
        v80 = v75.tag;
        int v81 = v7.v0;
        int v82;
        v82 = v81 + v80;
        int v83;
        v83 = v82 / 32;
        unsigned int v84;
        v84 = v0[v83];
        int v85;
        v85 = v82 % 32;
        unsigned int v86;
        v86 = 1u << v85;
        unsigned int v87;
        v87 = v84 | v86;
        v0[v83] = v87;
        int v88 = v7.v0;
        int v89;
        v89 = v88 + 3;
        v7.v0 = v89;
        switch (v75.tag) {
            case 0: { // Call
                int v90 = v7.v0;
                v7.v0 = v90;
                break;
            }
            case 1: { // Fold
                int v91 = v7.v0;
                v7.v0 = v91;
                break;
            }
            case 2: { // Raise
                int v92 = v75.case2.v0;
                int v93 = v7.v0;
                v7.v0 = v93;
                bool v94;
                v94 = v92 < 10;
                bool v95;
                v95 = v94 == false;
                if (v95){
                    assert("The input to the pickler must be 0 or positive." && v94);
                } else {
                }
                bool v97;
                v97 = 0 <= v92;
                bool v98;
                v98 = v97 == false;
                if (v98){
                    assert("The input to the pickler must be less than the specified length." && v97);
                } else {
                }
                int v100 = v7.v0;
                int v101;
                v101 = v100 + v92;
                int v102;
                v102 = v101 / 32;
                unsigned int v103;
                v103 = v0[v102];
                int v104;
                v104 = v101 % 32;
                unsigned int v105;
                v105 = 1u << v104;
                unsigned int v106;
                v106 = v103 | v105;
                v0[v102] = v106;
                int v107 = v7.v0;
                int v108;
                v108 = v107 + 10;
                v7.v0 = v108;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                __trap();
            }
        }
        v7.v0 = v79;
        int v109 = v7.v0;
        int v110;
        v110 = v109 + 1;
        v7.v0 = v110;
        v72 += 1 ;
    }
    int v111;
    v111 = v61;
    while (while_method_2(v111)){
        int v113 = v7.v0;
        int v114;
        v114 = v113 + 13;
        v7.v0 = v114;
        int v115 = v7.v0;
        int v116;
        v116 = v115 / 32;
        unsigned int v117;
        v117 = v0[v116];
        int v118;
        v118 = v115 % 32;
        unsigned int v119;
        v119 = 1u << v118;
        unsigned int v120;
        v120 = v117 | v119;
        v0[v116] = v120;
        int v121 = v7.v0;
        int v122;
        v122 = v121 + 1;
        v7.v0 = v122;
        v111 += 1 ;
    }
    int v123;
    v123 = 0;
    while (while_method_3(v123)){
        int v126;
        v126 = v4[v123];
        bool v129;
        v129 = v126 < 5;
        bool v130;
        v130 = v129 == false;
        if (v130){
            assert("The input to the pickler must be 0 or positive." && v129);
        } else {
        }
        bool v132;
        v132 = 0 <= v126;
        bool v133;
        v133 = v132 == false;
        if (v133){
            assert("The input to the pickler must be less than the specified length." && v132);
        } else {
        }
        int v135 = v7.v0;
        int v136;
        v136 = v135 + v126;
        int v137;
        v137 = v136 / 32;
        unsigned int v138;
        v138 = v0[v137];
        int v139;
        v139 = v136 % 32;
        unsigned int v140;
        v140 = 1u << v139;
        unsigned int v141;
        v141 = v138 | v140;
        v0[v137] = v141;
        int v142 = v7.v0;
        int v143;
        v143 = v142 + 5;
        v7.v0 = v143;
        v123 += 1 ;
    }
    return ;
}
__device__ int method_2(unsigned int * v0, StackMut0 & v1, int v2){
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
__device__ void method_3(unsigned int * v0, StackMut0 & v1, int v2){
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
__device__ void method_4(unsigned int * v0, StackMut0 & v1, int v2){
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
__device__ Tuple0 method_1(unsigned int * v0){
    StackMut0 v1{0};
    int v2;
    v2 = 10;
    int v3;
    v3 = method_2(v0, v1, v2);
    unsigned int v4;
    v4 = (unsigned int)v3;
    int v5;
    v5 = 10;
    int v6;
    v6 = method_2(v0, v1, v5);
    int v7;
    v7 = 2;
    int v8;
    v8 = method_2(v0, v1, v7);
    bool v9;
    v9 = v8 == 1;
    Union2 v10;
    v10 = Union2{Union2_0{}};
    StackMut1 v11{v10};
    int v12;
    v12 = 3;
    int v13;
    v13 = method_2(v0, v1, v12);
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
        method_3(v0, v1, v22);
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
        method_3(v0, v1, v31);
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
        method_3(v0, v1, v40);
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
    v47 = method_2(v0, v1, v46);
    static_array_list<Union0,5> v49;
    v49 = static_array_list<Union0,5>{};
    int v52;
    v52 = 0;
    while (while_method_1(v47, v52)){
        Union3 v54;
        v54 = Union3{Union3_0{}};
        StackMut2 v55{v54};
        int v56;
        v56 = 3;
        int v57;
        v57 = method_2(v0, v1, v56);
        bool v58;
        v58 = 0 == v57;
        if (v58){
            Union3 v59 = v55.v0;
            switch (v59.tag) {
                case 0: { // None
                    Union0 v60;
                    v60 = Union0{Union0_0{}};
                    Union3 v61;
                    v61 = Union3{Union3_1{v60}};
                    v55.v0 = v61;
                    break;
                }
                case 1: { // Some
                    Union0 v62 = v59.case1.v0;
                    bool v63;
                    v63 = false;
                    bool v64;
                    v64 = v63 == false;
                    if (v64){
                        assert("Duplicate union type instances in the unpickle Alt case." && v63);
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
            int v66;
            v66 = 0;
            method_3(v0, v1, v66);
        }
        bool v67;
        v67 = 1 == v57;
        if (v67){
            Union3 v68 = v55.v0;
            switch (v68.tag) {
                case 0: { // None
                    Union0 v69;
                    v69 = Union0{Union0_1{}};
                    Union3 v70;
                    v70 = Union3{Union3_1{v69}};
                    v55.v0 = v70;
                    break;
                }
                case 1: { // Some
                    Union0 v71 = v68.case1.v0;
                    bool v72;
                    v72 = false;
                    bool v73;
                    v73 = v72 == false;
                    if (v73){
                        assert("Duplicate union type instances in the unpickle Alt case." && v72);
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
            int v75;
            v75 = 0;
            method_3(v0, v1, v75);
        }
        bool v76;
        v76 = 2 == v57;
        if (v76){
            Union3 v77 = v55.v0;
            switch (v77.tag) {
                case 0: { // None
                    int v78;
                    v78 = 10;
                    int v79;
                    v79 = method_2(v0, v1, v78);
                    Union0 v80;
                    v80 = Union0{Union0_2{v79}};
                    Union3 v81;
                    v81 = Union3{Union3_1{v80}};
                    v55.v0 = v81;
                    break;
                }
                case 1: { // Some
                    Union0 v82 = v77.case1.v0;
                    bool v83;
                    v83 = false;
                    bool v84;
                    v84 = v83 == false;
                    if (v84){
                        assert("Duplicate union type instances in the unpickle Alt case." && v83);
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
            int v86;
            v86 = 10;
            method_3(v0, v1, v86);
        }
        Union3 v87 = v55.v0;
        Union0 v91;
        switch (v87.tag) {
            case 0: { // None
                printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                __trap();
                break;
            }
            case 1: { // Some
                Union0 v88 = v87.case1.v0;
                v91 = v88;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                __trap();
            }
        }
        v49.push(v91);
        int v92;
        v92 = 1;
        method_3(v0, v1, v92);
        v52 += 1 ;
    }
    int v93;
    v93 = v47;
    while (while_method_2(v93)){
        int v95;
        v95 = 13;
        method_3(v0, v1, v95);
        int v96;
        v96 = 1;
        method_4(v0, v1, v96);
        v93 += 1 ;
    }
    static_array<int,3> v98;
    int v101;
    v101 = 0;
    while (while_method_3(v101)){
        int v103;
        v103 = 5;
        int v104;
        v104 = method_2(v0, v1, v103);
        v98[v101] = v104;
        v101 += 1 ;
    }
    return Tuple0{v49, v45, v9, v98, v6, v4};
}
__device__ void method_5(Union0 v0){
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
__device__ void method_6(Union1 v0){
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
extern "C" __global__ void entry0() {
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
        static_array_list<Union0,5> v6;
        v6 = static_array_list<Union0,5>{};
        v6.unsafe_set_length(4);
        Union0 v10;
        v10 = Union0{Union0_2{3}};
        v6[0] = v10;
        Union0 v14;
        v14 = Union0{Union0_2{4}};
        v6[1] = v14;
        Union0 v18;
        v18 = Union0{Union0_0{}};
        v6[2] = v18;
        Union0 v22;
        v22 = Union0{Union0_1{}};
        v6[3] = v22;
        static_array<int,3> v26;
        v26[0] = 1;
        v26[1] = 2;
        v26[2] = 3;
        unsigned int v29[4];
        int v30;
        v30 = 0;
        while (while_method_0(v30)){
            int v32;
            v32 = v30 / 32;
            unsigned int v33;
            v33 = v29[v32];
            int v34;
            v34 = v30 % 32;
            unsigned int v35;
            v35 = 1u << v34;
            unsigned int v36;
            v36 = ~v35;
            unsigned int v37;
            v37 = v33 & v36;
            v29[v32] = v37;
            v30 += 1 ;
        }
        Union1 v38;
        v38 = Union1{Union1_1{}};
        bool v39;
        v39 = false;
        int v40;
        v40 = 8;
        unsigned int v41;
        v41 = 5u;
        method_0(v29, v6, v38, v39, v26, v40, v41);
        static_array_list<Union0,5> v42; Union1 v43; bool v44; static_array<int,3> v45; int v46; unsigned int v47;
        Tuple0 tmp3 = method_1(v29);
        v42 = tmp3.v0; v43 = tmp3.v1; v44 = tmp3.v2; v45 = tmp3.v3; v46 = tmp3.v4; v47 = tmp3.v5;
        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v73 = console_lock;
        auto v74 = cooperative_groups::coalesced_threads();
        v73.acquire();
        printf("{%s = %s","action_history", "[");
        int v75;
        v75 = v42.length;
        bool v76;
        v76 = 100 < v75;
        int v77;
        if (v76){
            v77 = 100;
        } else {
            v77 = v75;
        }
        int v78;
        v78 = 0;
        while (while_method_1(v77, v78)){
            Union0 v81;
            v81 = v42[v78];
            printf("");
            method_5(v81);
            printf("");
            int v84;
            v84 = v78 + 1;
            int v85;
            v85 = v42.length;
            bool v86;
            v86 = v84 < v85;
            if (v86){
                printf("%s","; ");
            } else {
            }
            v78 += 1 ;
        }
        int v87;
        v87 = v42.length;
        bool v88;
        v88 = v87 > 100;
        if (v88){
            printf("%s","; ...");
        } else {
        }
        printf("%s","]");
        printf("; %s = ","card");
        method_6(v43);
        const char * v91;
        if (v44){
            const char * v89;
            v89 = "true";
            v91 = v89;
        } else {
            const char * v90;
            v90 = "false";
            v91 = v90;
        }
        printf("; %s = %s; %s = %s","is_first", v91, "l", "[");
        int v92;
        v92 = 0;
        while (while_method_3(v92)){
            int v95;
            v95 = v45[v92];
            printf("%d",v95);
            int v98;
            v98 = v92 + 1;
            bool v99;
            v99 = v98 < 3;
            if (v99){
                printf("%s","; ");
            } else {
            }
            v92 += 1 ;
        }
        printf("%s","]");
        printf("; %s = %d; %s = %u}\n","pot", v46, "stack", v47);
        v73.release();
        v74.sync() ;
        return ;
    } else {
        return ;
    }
}
"""
from test1_auto import *
kernels = kernels_aux + kernels_main
import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = int; i16 = int; i32 = int; i64 = int; u8 = int; u16 = int; u32 = int; u64 = int; f32 = float; f64 = float; char = str; string = str

options = []
options.append('--define-macro=NDEBUG')
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
import os
home = os.getenv('HOME')
options.append(f'-I={home}/ThunderKittens/include')
options.append('--maxrregcount=255')
options.append('--std=c++20')
options.append('--expt-relaxed-constexpr')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernels, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method0() -> None:
    kernel = "entry0"
    v0 = cp.cuda.Device().attributes['MultiProcessorCount']
    v1 = v0 >= 24
    del v0
    v2 = v1 == False
    if v2:
        v3 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v1, v3
        del v3
    else:
        pass
    del v1, v2
    v4 = raw_module.get_function(kernel)
    v4.max_dynamic_shared_size_bytes = 98304 
    print(f'Threads per block, blocks per grid: {256}, {24}')
    v4((24,),(256,),(),shared_mem=98304)
    del v4
    return 
def main():
    method0()
    cp.cuda.get_current_stream().synchronize()
    return 0

if __name__ == '__main__': print(main())
