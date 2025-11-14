#include "test1c.hpp"
inline bool while_method_0(unsigned int v0){
    bool v1;
    v1 = v0 < 20340400u;
    return v1;
}
inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 > 0;
    return v1;
}
unsigned int f_2(unsigned int v0, unsigned int v1, unsigned int v2, unsigned int v3){
    bool v4;
    v4 = v2 > 0u;
    if (v4){
        unsigned int v5;
        v5 = v1 + v3;
        unsigned int v6;
        v6 = v2 - 1u;
        unsigned int v7;
        v7 = v3 * v0;
        return f_2(v0, v5, v6, v7);
    } else {
        return v1;
    }
}
unsigned int sum_pow_1(unsigned int v0, unsigned int v1){
    unsigned int v2;
    v2 = 0u;
    unsigned int v3;
    v3 = 1u;
    return f_2(v0, v2, v1, v3);
}
Tuple1 method_0(unsigned int v0){
    bool v1;
    v1 = 0u <= v0;
    bool v2;
    v2 = v1 == false;
    if (v2){
        assert("The input to unpickle must be >= 0." && v1);
    } else {
    }
    StackMut0 v4{v0};
    static_array<int,2> v5;
    int v9;
    v9 = 2;
    while (while_method_1(v9)){
        v9 -= 1 ;
        unsigned int v11 = v4.v0;
        unsigned int v12;
        v12 = v11 % 5u;
        unsigned int v13 = v4.v0;
        unsigned int v14;
        v14 = v13 / 5u;
        v4.v0 = v14;
        int v15;
        v15 = (int)v12;
        v5[v9] = v15;
    }
    unsigned int v16 = v4.v0;
    unsigned int v17;
    v17 = v16 % 813616u;
    v4.v0 = v17;
    static_array_list<Tuple0,5> v18;
    v18 = static_array_list<Tuple0,5>{};
    int v22;
    v22 = 6;
    while (while_method_1(v22)){
        v22 -= 1 ;
        unsigned int v24;
        v24 = 15u;
        unsigned int v25;
        v25 = (unsigned int)v22;
        unsigned int v26;
        v26 = sum_pow_1(v24, v25);
        unsigned int v27 = v4.v0;
        bool v28;
        v28 = v26 <= v27;
        if (v28){
            unsigned int v29 = v4.v0;
            unsigned int v30;
            v30 = v29 - v26;
            v4.v0 = v30;
            v18.unsafe_set_length(v22);
            int v31;
            v31 = v22;
            while (while_method_1(v31)){
                v31 -= 1 ;
                unsigned int v33 = v4.v0;
                unsigned int v34;
                v34 = v33 % 3u;
                v4.v0 = v34;
                Union2 v35;
                v35 = Union2{Union2_0{}};
                StackMut1 v36{v35};
                Union2 v37 = v36.v0;
                switch (v37.tag) {
                    case 0: { // None
                        unsigned int v39 = v4.v0;
                        bool v40;
                        v40 = v39 < 1u;
                        if (v40){
                            Union1 v41;
                            v41 = Union1{Union1_0{}};
                            Union2 v42;
                            v42 = Union2{Union2_1{v41}};
                            v36.v0 = v42;
                        } else {
                            unsigned int v43 = v4.v0;
                            unsigned int v44;
                            v44 = v43 - 1u;
                            v4.v0 = v44;
                        }
                        break;
                    }
                    case 1: { // Some
                        Union1 v38 = v37.case1.v0;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                Union2 v45 = v36.v0;
                switch (v45.tag) {
                    case 0: { // None
                        unsigned int v47 = v4.v0;
                        bool v48;
                        v48 = v47 < 1u;
                        if (v48){
                            Union1 v49;
                            v49 = Union1{Union1_1{}};
                            Union2 v50;
                            v50 = Union2{Union2_1{v49}};
                            v36.v0 = v50;
                        } else {
                            unsigned int v51 = v4.v0;
                            unsigned int v52;
                            v52 = v51 - 1u;
                            v4.v0 = v52;
                        }
                        break;
                    }
                    case 1: { // Some
                        Union1 v46 = v45.case1.v0;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                Union2 v53 = v36.v0;
                switch (v53.tag) {
                    case 0: { // None
                        unsigned int v55 = v4.v0;
                        bool v56;
                        v56 = v55 < 1u;
                        if (v56){
                            Union1 v57;
                            v57 = Union1{Union1_2{}};
                            Union2 v58;
                            v58 = Union2{Union2_1{v57}};
                            v36.v0 = v58;
                        } else {
                            unsigned int v59 = v4.v0;
                            unsigned int v60;
                            v60 = v59 - 1u;
                            v4.v0 = v60;
                        }
                        break;
                    }
                    case 1: { // Some
                        Union1 v54 = v53.case1.v0;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned int v61;
                v61 = v33 / 3u;
                v4.v0 = v61;
                Union2 v62 = v36.v0;
                Union1 v66;
                switch (v62.tag) {
                    case 0: { // None
                        printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                        exit(-1);
                        break;
                    }
                    case 1: { // Some
                        Union1 v63 = v62.case1.v0;
                        v66 = v63;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned int v67 = v4.v0;
                unsigned int v68;
                v68 = v67 % 5u;
                v4.v0 = v68;
                Union3 v69;
                v69 = Union3{Union3_0{}};
                StackMut2 v70{v69};
                Union3 v71 = v70.v0;
                switch (v71.tag) {
                    case 0: { // None
                        unsigned int v73 = v4.v0;
                        bool v74;
                        v74 = v73 < 1u;
                        if (v74){
                            Union0 v75;
                            v75 = Union0{Union0_0{}};
                            Union3 v76;
                            v76 = Union3{Union3_1{v75}};
                            v70.v0 = v76;
                        } else {
                            unsigned int v77 = v4.v0;
                            unsigned int v78;
                            v78 = v77 - 1u;
                            v4.v0 = v78;
                        }
                        break;
                    }
                    case 1: { // Some
                        Union0 v72 = v71.case1.v0;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                Union3 v79 = v70.v0;
                switch (v79.tag) {
                    case 0: { // None
                        unsigned int v81 = v4.v0;
                        bool v82;
                        v82 = v81 < 1u;
                        if (v82){
                            Union0 v83;
                            v83 = Union0{Union0_1{}};
                            Union3 v84;
                            v84 = Union3{Union3_1{v83}};
                            v70.v0 = v84;
                        } else {
                            unsigned int v85 = v4.v0;
                            unsigned int v86;
                            v86 = v85 - 1u;
                            v4.v0 = v86;
                        }
                        break;
                    }
                    case 1: { // Some
                        Union0 v80 = v79.case1.v0;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                Union3 v87 = v70.v0;
                switch (v87.tag) {
                    case 0: { // None
                        unsigned int v89 = v4.v0;
                        bool v90;
                        v90 = v89 < 3u;
                        if (v90){
                            unsigned int v91 = v4.v0;
                            unsigned int v92;
                            v92 = v91 % 3u;
                            unsigned int v93 = v4.v0;
                            unsigned int v94;
                            v94 = v93 / 3u;
                            v4.v0 = v94;
                            int v95;
                            v95 = (int)v92;
                            Union0 v96;
                            v96 = Union0{Union0_2{v95}};
                            Union3 v97;
                            v97 = Union3{Union3_1{v96}};
                            v70.v0 = v97;
                        } else {
                            unsigned int v98 = v4.v0;
                            unsigned int v99;
                            v99 = v98 - 3u;
                            v4.v0 = v99;
                        }
                        break;
                    }
                    case 1: { // Some
                        Union0 v88 = v87.case1.v0;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned int v100;
                v100 = v67 / 5u;
                v4.v0 = v100;
                Union3 v101 = v70.v0;
                Union0 v105;
                switch (v101.tag) {
                    case 0: { // None
                        printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                        exit(-1);
                        break;
                    }
                    case 1: { // Some
                        Union0 v102 = v101.case1.v0;
                        v105 = v102;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                v18[v31] = Tuple0{v105, v66};
            }
            break;
        } else {
        }
    }
    unsigned int v106;
    v106 = v16 / 813616u;
    v4.v0 = v106;
    return Tuple1{v18, v5};
}
unsigned int method_3(static_array_list<Tuple0,5> v0, static_array<int,2> v1){
    int v2;
    v2 = v0.length;
    int v3; unsigned int v4; unsigned int v5;
    Tuple2 tmp1 = Tuple2{v2, 1u, 0u};
    v3 = tmp1.v0; v4 = tmp1.v1; v5 = tmp1.v2;
    while (while_method_1(v3)){
        v3 -= 1 ;
        Union0 v7; Union1 v8;
        Tuple0 tmp2 = v0[v3];
        v7 = tmp2.v0; v8 = tmp2.v1;
        unsigned int v15;
        v15 = v4 * 15u;
        unsigned int v16;
        switch (v8.tag) {
            case 0: { // Jack
                v16 = 0u;
                break;
            }
            case 1: { // King
                v16 = 1u;
                break;
            }
            case 2: { // Queen
                v16 = 2u;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned int v26;
        switch (v7.tag) {
            case 0: { // Call
                v26 = 0u;
                break;
            }
            case 1: { // Fold
                v26 = 1u;
                break;
            }
            case 2: { // Raise
                int v17 = v7.case2.v0;
                bool v18;
                v18 = 0 <= v17;
                bool v19;
                v19 = v18 == false;
                if (v19){
                    assert("The input to the pickler must be 0 or positive." && v18);
                } else {
                }
                bool v21;
                v21 = v17 < 3;
                bool v22;
                v22 = v21 == false;
                if (v22){
                    assert("The input to the pickler must be less than the specified length." && v21);
                } else {
                }
                unsigned int v24;
                v24 = (unsigned int)v17;
                unsigned int v25;
                v25 = 2u + v24;
                v26 = v25;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned int v27;
        v27 = v26 * 3u;
        unsigned int v28;
        v28 = v16 + v27;
        unsigned int v29;
        v29 = v28 * v4;
        unsigned int v30;
        v30 = v5 + v29;
        v4 = v15;
        v5 = v30;
    }
    unsigned int v31;
    v31 = 15u;
    int v32;
    v32 = v0.length;
    unsigned int v33;
    v33 = (unsigned int)v32;
    unsigned int v34;
    v34 = sum_pow_1(v31, v33);
    unsigned int v35;
    v35 = v34 + v5;
    unsigned int v36;
    v36 = v35 * 25u;
    int v37; unsigned int v38; unsigned int v39;
    Tuple2 tmp3 = Tuple2{2, 1u, 0u};
    v37 = tmp3.v0; v38 = tmp3.v1; v39 = tmp3.v2;
    while (while_method_1(v37)){
        v37 -= 1 ;
        int v41;
        v41 = v1[v37];
        unsigned int v45;
        v45 = v38 * 5u;
        bool v46;
        v46 = 0 <= v41;
        bool v47;
        v47 = v46 == false;
        if (v47){
            assert("The input to the pickler must be 0 or positive." && v46);
        } else {
        }
        bool v49;
        v49 = v41 < 5;
        bool v50;
        v50 = v49 == false;
        if (v50){
            assert("The input to the pickler must be less than the specified length." && v49);
        } else {
        }
        unsigned int v52;
        v52 = (unsigned int)v41;
        unsigned int v53;
        v53 = v52 * v38;
        unsigned int v54;
        v54 = v39 + v53;
        v38 = v45;
        v39 = v54;
    }
    unsigned int v55;
    v55 = v36 + v39;
    return v55;
}
int main() {
    printf("{%s = %u}\n","size", 20340400u);
    fflush(stdout);
    unsigned int v4;
    v4 = 0u;
    while (while_method_0(v4)){
        static_array_list<Tuple0,5> v6; static_array<int,2> v7;
        Tuple1 tmp0 = method_0(v4);
        v6 = tmp0.v0; v7 = tmp0.v1;
        unsigned int v8;
        v8 = method_3(v6, v7);
        bool v9;
        v9 = v8 == v4;
        bool v10;
        v10 = v9 == false;
        if (v10){
            assert("The round trip has to be equal to the original." && v9);
        } else {
        }
        v4 += 1u ;
    }
    printf("%s\n","The test passes.");
    fflush(stdout);
    return 0;
}
