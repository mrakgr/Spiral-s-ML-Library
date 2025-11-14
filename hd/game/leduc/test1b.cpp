#include "test1b.hpp"
inline bool while_method_0(int v0){
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
unsigned int method_0(static_array_list<Tuple0,5> v0){
    int v1;
    v1 = v0.length;
    int v2; unsigned int v3; unsigned int v4;
    Tuple1 tmp0 = Tuple1{v1, 1u, 0u};
    v2 = tmp0.v0; v3 = tmp0.v1; v4 = tmp0.v2;
    while (while_method_0(v2)){
        v2 -= 1 ;
        Union0 v6; Union1 v7;
        Tuple0 tmp1 = v0[v2];
        v6 = tmp1.v0; v7 = tmp1.v1;
        unsigned int v14;
        v14 = v3 * 15u;
        unsigned int v15;
        switch (v7.tag) {
            case 0: { // Jack
                v15 = 0u;
                break;
            }
            case 1: { // King
                v15 = 1u;
                break;
            }
            case 2: { // Queen
                v15 = 2u;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned int v25;
        switch (v6.tag) {
            case 0: { // Call
                v25 = 0u;
                break;
            }
            case 1: { // Fold
                v25 = 1u;
                break;
            }
            case 2: { // Raise
                int v16 = v6.case2.v0;
                bool v17;
                v17 = 0 <= v16;
                bool v18;
                v18 = v17 == false;
                if (v18){
                    assert("The input to the pickler must be 0 or positive." && v17);
                } else {
                }
                bool v20;
                v20 = v16 < 3;
                bool v21;
                v21 = v20 == false;
                if (v21){
                    assert("The input to the pickler must be less than the specified length." && v20);
                } else {
                }
                unsigned int v23;
                v23 = (unsigned int)v16;
                unsigned int v24;
                v24 = 2u + v23;
                v25 = v24;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned int v26;
        v26 = v25 * 3u;
        unsigned int v27;
        v27 = v15 + v26;
        unsigned int v28;
        v28 = v27 * v3;
        unsigned int v29;
        v29 = v4 + v28;
        v3 = v14;
        v4 = v29;
    }
    unsigned int v30;
    v30 = 15u;
    int v31;
    v31 = v0.length;
    unsigned int v32;
    v32 = (unsigned int)v31;
    unsigned int v33;
    v33 = sum_pow_1(v30, v32);
    unsigned int v34;
    v34 = v33 + v4;
    return v34;
}
static_array_list<Tuple0,5> method_3(unsigned int v0){
    bool v1;
    v1 = 0u <= v0;
    bool v2;
    v2 = v1 == false;
    if (v2){
        assert("The input to unpickle must be >= 0." && v1);
    } else {
    }
    StackMut0 v4{v0};
    unsigned int v5 = v4.v0;
    unsigned int v6;
    v6 = v5 % 813616u;
    v4.v0 = v6;
    static_array_list<Tuple0,5> v7;
    v7 = static_array_list<Tuple0,5>{};
    int v11;
    v11 = 5;
    while (while_method_0(v11)){
        v11 -= 1 ;
        unsigned int v13;
        v13 = 15u;
        unsigned int v14;
        v14 = (unsigned int)v11;
        unsigned int v15;
        v15 = sum_pow_1(v13, v14);
        std::cout << v15 << std:endl;
        unsigned int v16 = v4.v0;
        bool v17;
        v17 = v15 <= v16;
        if (v17){
            unsigned int v18 = v4.v0;
            unsigned int v19;
            v19 = v18 - v15;
            v4.v0 = v19;
            v7.unsafe_set_length(v11);
            int v20;
            v20 = v11;
            while (while_method_0(v20)){
                v20 -= 1 ;
                unsigned int v22 = v4.v0;
                unsigned int v23;
                v23 = v22 % 3u;
                v4.v0 = v23;
                Union2 v24;
                v24 = Union2{Union2_0{}};
                StackMut1 v25{v24};
                Union2 v26 = v25.v0;
                switch (v26.tag) {
                    case 0: { // None
                        unsigned int v28 = v4.v0;
                        bool v29;
                        v29 = v28 < 1u;
                        if (v29){
                            Union1 v30;
                            v30 = Union1{Union1_0{}};
                            Union2 v31;
                            v31 = Union2{Union2_1{v30}};
                            v25.v0 = v31;
                        } else {
                            unsigned int v32 = v4.v0;
                            unsigned int v33;
                            v33 = v32 - 1u;
                            v4.v0 = v33;
                        }
                        break;
                    }
                    case 1: { // Some
                        Union1 v27 = v26.case1.v0;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                Union2 v34 = v25.v0;
                switch (v34.tag) {
                    case 0: { // None
                        unsigned int v36 = v4.v0;
                        bool v37;
                        v37 = v36 < 1u;
                        if (v37){
                            Union1 v38;
                            v38 = Union1{Union1_1{}};
                            Union2 v39;
                            v39 = Union2{Union2_1{v38}};
                            v25.v0 = v39;
                        } else {
                            unsigned int v40 = v4.v0;
                            unsigned int v41;
                            v41 = v40 - 1u;
                            v4.v0 = v41;
                        }
                        break;
                    }
                    case 1: { // Some
                        Union1 v35 = v34.case1.v0;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                Union2 v42 = v25.v0;
                switch (v42.tag) {
                    case 0: { // None
                        unsigned int v44 = v4.v0;
                        bool v45;
                        v45 = v44 < 1u;
                        if (v45){
                            Union1 v46;
                            v46 = Union1{Union1_2{}};
                            Union2 v47;
                            v47 = Union2{Union2_1{v46}};
                            v25.v0 = v47;
                        } else {
                            unsigned int v48 = v4.v0;
                            unsigned int v49;
                            v49 = v48 - 1u;
                            v4.v0 = v49;
                        }
                        break;
                    }
                    case 1: { // Some
                        Union1 v43 = v42.case1.v0;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned int v50;
                v50 = v22 / 3u;
                v4.v0 = v50;
                Union2 v51 = v25.v0;
                Union1 v55;
                switch (v51.tag) {
                    case 0: { // None
                        printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                        exit(-1);
                        break;
                    }
                    case 1: { // Some
                        Union1 v52 = v51.case1.v0;
                        v55 = v52;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned int v56 = v4.v0;
                unsigned int v57;
                v57 = v56 % 5u;
                v4.v0 = v57;
                Union3 v58;
                v58 = Union3{Union3_0{}};
                StackMut2 v59{v58};
                Union3 v60 = v59.v0;
                switch (v60.tag) {
                    case 0: { // None
                        unsigned int v62 = v4.v0;
                        bool v63;
                        v63 = v62 < 1u;
                        if (v63){
                            Union0 v64;
                            v64 = Union0{Union0_0{}};
                            Union3 v65;
                            v65 = Union3{Union3_1{v64}};
                            v59.v0 = v65;
                        } else {
                            unsigned int v66 = v4.v0;
                            unsigned int v67;
                            v67 = v66 - 1u;
                            v4.v0 = v67;
                        }
                        break;
                    }
                    case 1: { // Some
                        Union0 v61 = v60.case1.v0;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                Union3 v68 = v59.v0;
                switch (v68.tag) {
                    case 0: { // None
                        unsigned int v70 = v4.v0;
                        bool v71;
                        v71 = v70 < 1u;
                        if (v71){
                            Union0 v72;
                            v72 = Union0{Union0_1{}};
                            Union3 v73;
                            v73 = Union3{Union3_1{v72}};
                            v59.v0 = v73;
                        } else {
                            unsigned int v74 = v4.v0;
                            unsigned int v75;
                            v75 = v74 - 1u;
                            v4.v0 = v75;
                        }
                        break;
                    }
                    case 1: { // Some
                        Union0 v69 = v68.case1.v0;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                Union3 v76 = v59.v0;
                switch (v76.tag) {
                    case 0: { // None
                        unsigned int v78 = v4.v0;
                        bool v79;
                        v79 = v78 < 3u;
                        if (v79){
                            unsigned int v80 = v4.v0;
                            unsigned int v81;
                            v81 = v80 % 3u;
                            unsigned int v82 = v4.v0;
                            unsigned int v83;
                            v83 = v82 / 3u;
                            v4.v0 = v83;
                            int v84;
                            v84 = (int)v81;
                            Union0 v85;
                            v85 = Union0{Union0_2{v84}};
                            Union3 v86;
                            v86 = Union3{Union3_1{v85}};
                            v59.v0 = v86;
                        } else {
                            unsigned int v87 = v4.v0;
                            unsigned int v88;
                            v88 = v87 - 3u;
                            v4.v0 = v88;
                        }
                        break;
                    }
                    case 1: { // Some
                        Union0 v77 = v76.case1.v0;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned int v89;
                v89 = v56 / 5u;
                v4.v0 = v89;
                Union3 v90 = v59.v0;
                Union0 v94;
                switch (v90.tag) {
                    case 0: { // None
                        printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                        exit(-1);
                        break;
                    }
                    case 1: { // Some
                        Union0 v91 = v90.case1.v0;
                        v94 = v91;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                v7[v20] = Tuple0{v94, v55};
            }
            break;
        } else {
        }
    }
    unsigned int v95;
    v95 = v5 / 813616u;
    v4.v0 = v95;
    return v7;
}
inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
void method_4(Union0 v0){
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
            exit(-1);
        }
    }
}
void method_5(Union1 v0){
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
int main() {
    printf("{%s = %u}\n","size", 813616u);
    fflush(stdout);
    static_array_list<Tuple0,5> v4;
    v4 = static_array_list<Tuple0,5>{};
    v4.unsafe_set_length(3);
    Union0 v8;
    v8 = Union0{Union0_2{1}};
    Union1 v9;
    v9 = Union1{Union1_1{}};
    v4[0] = Tuple0{v8, v9};
    Union0 v16;
    v16 = Union0{Union0_0{}};
    Union1 v17;
    v17 = Union1{Union1_2{}};
    v4[1] = Tuple0{v16, v17};
    Union0 v24;
    v24 = Union0{Union0_1{}};
    Union1 v25;
    v25 = Union1{Union1_0{}};
    v4[2] = Tuple0{v24, v25};
    unsigned int v32;
    v32 = method_0(v4);
    static_array_list<Tuple0,5> v33;
    v33 = method_3(v32);
    printf("%s","[");
    int v34;
    v34 = v33.length;
    bool v35;
    v35 = 100 < v34;
    int v36;
    if (v35){
        v36 = 100;
    } else {
        v36 = v34;
    }
    int v37;
    v37 = 0;
    while (while_method_1(v36, v37)){
        Union0 v39; Union1 v40;
        Tuple0 tmp2 = v33[v37];
        v39 = tmp2.v0; v40 = tmp2.v1;
        printf("{%s = ","action");
        method_4(v39);
        printf("; %s = ","card");
        method_5(v40);
        printf("}");
        int v47;
        v47 = v37 + 1;
        int v48;
        v48 = v33.length;
        bool v49;
        v49 = v47 < v48;
        if (v49){
            printf("%s","; ");
        } else {
        }
        v37 += 1 ;
    }
    int v50;
    v50 = v33.length;
    bool v51;
    v51 = v50 > 100;
    if (v51){
        printf("%s","; ...");
    } else {
    }
    printf("%s","]");
    printf("\n");
    fflush(stdout);
    return 0;
}
