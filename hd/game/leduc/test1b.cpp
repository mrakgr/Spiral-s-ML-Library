#include "test1b.hpp"
inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
int method_0(static_array<Tuple0,3> v0){
    int v1; int v2; int v3;
    Tuple1 tmp0 = Tuple1{0, 144703125, 0};
    v1 = tmp0.v0; v2 = tmp0.v1; v3 = tmp0.v2;
    while (while_method_0(v1)){
        Union0 v5; Union1 v6; unsigned int v7; unsigned int v8;
        Tuple0 tmp1 = v0[v1];
        v5 = tmp1.v0; v6 = tmp1.v1; v7 = tmp1.v2; v8 = tmp1.v3;
        int v21;
        v21 = v2 / 525;
        int v22;
        v22 = (int)v7;
        bool v23;
        v23 = v22 < 5;
        bool v24;
        v24 = v23 == false;
        if (v24){
            assert("The input to the pickler must be less than the specified length." && v23);
        } else {
        }
        int v26;
        v26 = v22 * 5;
        int v27;
        v27 = (int)v8;
        bool v28;
        v28 = v27 < 5;
        bool v29;
        v29 = v28 == false;
        if (v29){
            assert("The input to the pickler must be less than the specified length." && v28);
        } else {
        }
        int v31;
        v31 = v26 + v27;
        int v32;
        switch (v6.tag) {
            case 0: { // Jack
                v32 = 0;
                break;
            }
            case 1: { // King
                v32 = 1;
                break;
            }
            case 2: { // Queen
                v32 = 2;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        int v33;
        v33 = v32 * 25;
        int v34;
        v34 = v31 + v33;
        int v43;
        switch (v5.tag) {
            case 0: { // Call
                v43 = 0;
                break;
            }
            case 1: { // Fold
                v43 = 1;
                break;
            }
            case 2: { // Raise
                int v35 = v5.case2.v0;
                bool v36;
                v36 = 0 <= v35;
                bool v37;
                v37 = v36 == false;
                if (v37){
                    assert("The input to the pickler must be 0 or positive." && v36);
                } else {
                }
                bool v39;
                v39 = v35 < 5;
                bool v40;
                v40 = v39 == false;
                if (v40){
                    assert("The input to the pickler must be less than the specified length." && v39);
                } else {
                }
                int v42;
                v42 = 2 + v35;
                v43 = v42;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        int v44;
        v44 = v43 * 75;
        int v45;
        v45 = v34 + v44;
        int v46;
        v46 = v45 * v21;
        int v47;
        v47 = v3 + v46;
        v2 = v21;
        v3 = v47;
        v1 += 1 ;
    }
    return v3;
}
inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 > 0;
    return v1;
}
static_array<Tuple0,3> method_1(int v0){
    bool v1;
    v1 = 0 <= v0;
    bool v2;
    v2 = v1 == false;
    if (v2){
        assert("The input to unpickle must be >= 0." && v1);
    } else {
    }
    StackMut0 v4{v0};
    static_array<Tuple0,3> v5;
    int v9;
    v9 = 3;
    while (while_method_1(v9)){
        v9 -= 1 ;
        int v11 = v4.v0;
        int v12;
        v12 = v11 % 5;
        int v13 = v4.v0;
        int v14;
        v14 = v13 / 5;
        v4.v0 = v14;
        unsigned int v15;
        v15 = (unsigned int)v12;
        int v16 = v4.v0;
        int v17;
        v17 = v16 % 5;
        int v18 = v4.v0;
        int v19;
        v19 = v18 / 5;
        v4.v0 = v19;
        unsigned int v20;
        v20 = (unsigned int)v17;
        int v21 = v4.v0;
        int v22;
        v22 = v21 % 3;
        v4.v0 = v22;
        Union2 v23;
        v23 = Union2{Union2_0{}};
        StackMut1 v24{v23};
        Union2 v25 = v24.v0;
        switch (v25.tag) {
            case 0: { // None
                int v27 = v4.v0;
                bool v28;
                v28 = v27 < 1;
                if (v28){
                    Union1 v29;
                    v29 = Union1{Union1_0{}};
                    Union2 v30;
                    v30 = Union2{Union2_1{v29}};
                    v24.v0 = v30;
                } else {
                    int v31 = v4.v0;
                    int v32;
                    v32 = v31 - 1;
                    v4.v0 = v32;
                }
                break;
            }
            case 1: { // Some
                Union1 v26 = v25.case1.v0;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        Union2 v33 = v24.v0;
        switch (v33.tag) {
            case 0: { // None
                int v35 = v4.v0;
                bool v36;
                v36 = v35 < 1;
                if (v36){
                    Union1 v37;
                    v37 = Union1{Union1_1{}};
                    Union2 v38;
                    v38 = Union2{Union2_1{v37}};
                    v24.v0 = v38;
                } else {
                    int v39 = v4.v0;
                    int v40;
                    v40 = v39 - 1;
                    v4.v0 = v40;
                }
                break;
            }
            case 1: { // Some
                Union1 v34 = v33.case1.v0;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        Union2 v41 = v24.v0;
        switch (v41.tag) {
            case 0: { // None
                int v43 = v4.v0;
                bool v44;
                v44 = v43 < 1;
                if (v44){
                    Union1 v45;
                    v45 = Union1{Union1_2{}};
                    Union2 v46;
                    v46 = Union2{Union2_1{v45}};
                    v24.v0 = v46;
                } else {
                    int v47 = v4.v0;
                    int v48;
                    v48 = v47 - 1;
                    v4.v0 = v48;
                }
                break;
            }
            case 1: { // Some
                Union1 v42 = v41.case1.v0;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        int v49;
        v49 = v21 / 3;
        v4.v0 = v49;
        Union2 v50 = v24.v0;
        Union1 v54;
        switch (v50.tag) {
            case 0: { // None
                printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                exit(-1);
                break;
            }
            case 1: { // Some
                Union1 v51 = v50.case1.v0;
                v54 = v51;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        int v55 = v4.v0;
        int v56;
        v56 = v55 % 7;
        v4.v0 = v56;
        Union3 v57;
        v57 = Union3{Union3_0{}};
        StackMut2 v58{v57};
        Union3 v59 = v58.v0;
        switch (v59.tag) {
            case 0: { // None
                int v61 = v4.v0;
                bool v62;
                v62 = v61 < 1;
                if (v62){
                    Union0 v63;
                    v63 = Union0{Union0_0{}};
                    Union3 v64;
                    v64 = Union3{Union3_1{v63}};
                    v58.v0 = v64;
                } else {
                    int v65 = v4.v0;
                    int v66;
                    v66 = v65 - 1;
                    v4.v0 = v66;
                }
                break;
            }
            case 1: { // Some
                Union0 v60 = v59.case1.v0;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        Union3 v67 = v58.v0;
        switch (v67.tag) {
            case 0: { // None
                int v69 = v4.v0;
                bool v70;
                v70 = v69 < 1;
                if (v70){
                    Union0 v71;
                    v71 = Union0{Union0_1{}};
                    Union3 v72;
                    v72 = Union3{Union3_1{v71}};
                    v58.v0 = v72;
                } else {
                    int v73 = v4.v0;
                    int v74;
                    v74 = v73 - 1;
                    v4.v0 = v74;
                }
                break;
            }
            case 1: { // Some
                Union0 v68 = v67.case1.v0;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        Union3 v75 = v58.v0;
        switch (v75.tag) {
            case 0: { // None
                int v77 = v4.v0;
                bool v78;
                v78 = v77 < 5;
                if (v78){
                    int v79 = v4.v0;
                    int v80;
                    v80 = v79 % 5;
                    int v81 = v4.v0;
                    int v82;
                    v82 = v81 / 5;
                    v4.v0 = v82;
                    Union0 v83;
                    v83 = Union0{Union0_2{v80}};
                    Union3 v84;
                    v84 = Union3{Union3_1{v83}};
                    v58.v0 = v84;
                } else {
                    int v85 = v4.v0;
                    int v86;
                    v86 = v85 - 5;
                    v4.v0 = v86;
                }
                break;
            }
            case 1: { // Some
                Union0 v76 = v75.case1.v0;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        int v87;
        v87 = v55 / 7;
        v4.v0 = v87;
        Union3 v88 = v58.v0;
        Union0 v92;
        switch (v88.tag) {
            case 0: { // None
                printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                exit(-1);
                break;
            }
            case 1: { // Some
                Union0 v89 = v88.case1.v0;
                v92 = v89;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        v5[v9] = Tuple0{v92, v54, v20, v15};
    }
    return v5;
}
void method_2(Union0 v0){
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
void method_3(Union1 v0){
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
    printf("{%s = %d}\n","size", 144703125);
    fflush(stdout);
    static_array<Tuple0,3> v4;
    Union0 v8;
    v8 = Union0{Union0_2{2}};
    Union1 v9;
    v9 = Union1{Union1_1{}};
    v4[0] = Tuple0{v8, v9, 3u, 3u};
    Union0 v16;
    v16 = Union0{Union0_0{}};
    Union1 v17;
    v17 = Union1{Union1_2{}};
    v4[1] = Tuple0{v16, v17, 2u, 2u};
    Union0 v24;
    v24 = Union0{Union0_1{}};
    Union1 v25;
    v25 = Union1{Union1_0{}};
    v4[2] = Tuple0{v24, v25, 1u, 1u};
    int v32;
    v32 = method_0(v4);
    static_array<Tuple0,3> v33;
    v33 = method_1(v32);
    printf("%s","[");
    int v34;
    v34 = 0;
    while (while_method_0(v34)){
        Union0 v36; Union1 v37; unsigned int v38; unsigned int v39;
        Tuple0 tmp2 = v33[v34];
        v36 = tmp2.v0; v37 = tmp2.v1; v38 = tmp2.v2; v39 = tmp2.v3;
        printf("{%s = ","action");
        method_2(v36);
        printf("; %s = ","card");
        method_3(v37);
        printf("; %s = %u, %u}","pot_stack", v38, v39);
        int v52;
        v52 = v34 + 1;
        bool v53;
        v53 = v52 < 3;
        if (v53){
            printf("%s","; ");
        } else {
        }
        v34 += 1 ;
    }
    printf("%s","]");
    printf("\n");
    fflush(stdout);
    return 0;
}
