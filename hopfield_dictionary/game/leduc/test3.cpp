#include "test3.hpp"
inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 12;
    return v1;
}
inline bool while_method_8(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 5;
    return v1;
}
inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
void method_6(float * v0, static_array_list<Union3,5> v1, Union5 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6){
    StackMut7 v7{0};
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
    v0[v13] = 1.0f;
    int v14 = v7.v0;
    int v15;
    v15 = v14 + 10;
    v7.v0 = v15;
    bool v16;
    v16 = v5 < 10;
    bool v17;
    v17 = v16 == false;
    if (v17){
        assert("The input to the pickler must be 0 or positive." && v16);
    } else {
    }
    bool v19;
    v19 = 0 <= v5;
    bool v20;
    v20 = v19 == false;
    if (v20){
        assert("The input to the pickler must be less than the specified length." && v19);
    } else {
    }
    int v22 = v7.v0;
    int v23;
    v23 = v22 + v5;
    v0[v23] = 1.0f;
    int v24 = v7.v0;
    int v25;
    v25 = v24 + 10;
    v7.v0 = v25;
    int v26;
    if (v3){
        v26 = 1;
    } else {
        v26 = 0;
    }
    int v27 = v7.v0;
    int v28;
    v28 = v27 + v26;
    v0[v28] = 1.0f;
    int v29 = v7.v0;
    int v30;
    v30 = v29 + 2;
    v7.v0 = v30;
    int v31 = v7.v0;
    int v32;
    v32 = v31 + 3;
    int v33;
    v33 = v2.tag;
    int v34 = v7.v0;
    int v35;
    v35 = v34 + v33;
    v0[v35] = 1.0f;
    int v36 = v7.v0;
    int v37;
    v37 = v36 + 3;
    v7.v0 = v37;
    switch (v2.tag) {
        case 0: { // Jack
            int v38 = v7.v0;
            v7.v0 = v38;
            break;
        }
        case 1: { // King
            int v39 = v7.v0;
            v7.v0 = v39;
            break;
        }
        case 2: { // Queen
            int v40 = v7.v0;
            v7.v0 = v40;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    v7.v0 = v32;
    int v41;
    v41 = v1.length;
    int v42 = v7.v0;
    int v43;
    v43 = v42 + v41;
    v0[v43] = 1.0f;
    int v44 = v7.v0;
    int v45;
    v45 = v44 + 5;
    v7.v0 = v45;
    int v46;
    v46 = v1.length;
    int v47;
    v47 = 0;
    while (while_method_8(v46, v47)){
        Union3 v51;
        v51 = v1[v47];
        int v53 = v7.v0;
        int v54;
        v54 = v53 + 13;
        int v55;
        v55 = v51.tag;
        int v56 = v7.v0;
        int v57;
        v57 = v56 + v55;
        v0[v57] = 1.0f;
        int v58 = v7.v0;
        int v59;
        v59 = v58 + 3;
        v7.v0 = v59;
        switch (v51.tag) {
            case 0: { // Call
                int v60 = v7.v0;
                v7.v0 = v60;
                break;
            }
            case 1: { // Fold
                int v61 = v7.v0;
                v7.v0 = v61;
                break;
            }
            case 2: { // Raise
                int v62 = v51.case2.v0;
                int v63 = v7.v0;
                v7.v0 = v63;
                bool v64;
                v64 = v62 < 10;
                bool v65;
                v65 = v64 == false;
                if (v65){
                    assert("The input to the pickler must be 0 or positive." && v64);
                } else {
                }
                bool v67;
                v67 = 0 <= v62;
                bool v68;
                v68 = v67 == false;
                if (v68){
                    assert("The input to the pickler must be less than the specified length." && v67);
                } else {
                }
                int v70 = v7.v0;
                int v71;
                v71 = v70 + v62;
                v0[v71] = 1.0f;
                int v72 = v7.v0;
                int v73;
                v73 = v72 + 10;
                v7.v0 = v73;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        v7.v0 = v54;
        int v74 = v7.v0;
        int v75;
        v75 = v74 + 1;
        v7.v0 = v75;
        v47 += 1 ;
    }
    int v76;
    v76 = v41;
    while (while_method_9(v76)){
        int v78 = v7.v0;
        int v79;
        v79 = v78 + 13;
        v7.v0 = v79;
        int v80 = v7.v0;
        v0[v80] = 1.0f;
        int v81 = v7.v0;
        int v82;
        v82 = v81 + 1;
        v7.v0 = v82;
        v76 += 1 ;
    }
    int v83;
    v83 = 0;
    while (while_method_10(v83)){
        int v87;
        v87 = v4[v83];
        bool v89;
        v89 = v87 < 5;
        bool v90;
        v90 = v89 == false;
        if (v90){
            assert("The input to the pickler must be 0 or positive." && v89);
        } else {
        }
        bool v92;
        v92 = 0 <= v87;
        bool v93;
        v93 = v92 == false;
        if (v93){
            assert("The input to the pickler must be less than the specified length." && v92);
        } else {
        }
        int v95 = v7.v0;
        int v96;
        v96 = v95 + v87;
        v0[v96] = 1.0f;
        int v97 = v7.v0;
        int v98;
        v98 = v97 + 5;
        v7.v0 = v98;
        v83 += 1 ;
    }
    return ;
}
int method_14(float * v0, StackMut7 & v1, int v2){
    int v3; int v4; int v5;
    Tuple15 tmp0 = Tuple15{0, 0, 0};
    v3 = tmp0.v0; v4 = tmp0.v1; v5 = tmp0.v2;
    while (while_method_8(v2, v3)){
        int v7 = v1.v0;
        float v8;
        v8 = v0[v7];
        bool v9;
        v9 = v8 == 1.0f;
        bool v14;
        if (v9){
            v14 = true;
        } else {
            float v10;
            v10 = v0[v7];
            bool v11;
            v11 = v10 == 0.0f;
            if (v11){
                v14 = false;
            } else {
                printf("%s\n", "Pickler index expected to get 1 or 0.");
                exit(-1);
            }
        }
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
void method_18(float * v0, StackMut7 & v1, int v2){
    int v3; int v4; int v5;
    Tuple15 tmp1 = Tuple15{0, 0, 0};
    v3 = tmp1.v0; v4 = tmp1.v1; v5 = tmp1.v2;
    while (while_method_8(v2, v3)){
        int v7 = v1.v0;
        float v8;
        v8 = v0[v7];
        bool v9;
        v9 = v8 == 1.0f;
        bool v14;
        if (v9){
            v14 = true;
        } else {
            float v10;
            v10 = v0[v7];
            bool v11;
            v11 = v10 == 0.0f;
            if (v11){
                v14 = false;
            } else {
                printf("%s\n", "Pickler index expected to get 1 or 0.");
                exit(-1);
            }
        }
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
void method_21(float * v0, StackMut7 & v1, int v2){
    int v3; int v4; int v5;
    Tuple15 tmp2 = Tuple15{0, 0, 0};
    v3 = tmp2.v0; v4 = tmp2.v1; v5 = tmp2.v2;
    while (while_method_8(v2, v3)){
        int v7 = v1.v0;
        float v8;
        v8 = v0[v7];
        bool v9;
        v9 = v8 == 1.0f;
        bool v14;
        if (v9){
            v14 = true;
        } else {
            float v10;
            v10 = v0[v7];
            bool v11;
            v11 = v10 == 0.0f;
            if (v11){
                v14 = false;
            } else {
                printf("%s\n", "Pickler index expected to get 1 or 0.");
                exit(-1);
            }
        }
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
Tuple13 method_12(float * v0){
    StackMut7 v1{0};
    int v2;
    v2 = 10;
    int v3;
    v3 = method_14(v0, v1, v2);
    unsigned int v4;
    v4 = (unsigned int)v3;
    int v5;
    v5 = 10;
    int v6;
    v6 = method_14(v0, v1, v5);
    int v7;
    v7 = 2;
    int v8;
    v8 = method_14(v0, v1, v7);
    bool v9;
    v9 = v8 == 1;
    Union16 v10;
    v10 = Union16{Union16_0{}};
    StackMut17 v11{v10};
    int v12;
    v12 = 3;
    int v13;
    v13 = method_14(v0, v1, v12);
    bool v14;
    v14 = 0 == v13;
    if (v14){
        Union16 v15 = v11.v0;
        switch (v15.tag) {
            case 0: { // None
                Union5 v16;
                v16 = Union5{Union5_0{}};
                Union16 v17;
                v17 = Union16{Union16_1{v16}};
                v11.v0 = v17;
                break;
            }
            case 1: { // Some
                Union5 v18 = v15.case1.v0;
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
                exit(-1);
            }
        }
    } else {
        int v22;
        v22 = 0;
        method_18(v0, v1, v22);
    }
    bool v23;
    v23 = 1 == v13;
    if (v23){
        Union16 v24 = v11.v0;
        switch (v24.tag) {
            case 0: { // None
                Union5 v25;
                v25 = Union5{Union5_1{}};
                Union16 v26;
                v26 = Union16{Union16_1{v25}};
                v11.v0 = v26;
                break;
            }
            case 1: { // Some
                Union5 v27 = v24.case1.v0;
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
                exit(-1);
            }
        }
    } else {
        int v31;
        v31 = 0;
        method_18(v0, v1, v31);
    }
    bool v32;
    v32 = 2 == v13;
    if (v32){
        Union16 v33 = v11.v0;
        switch (v33.tag) {
            case 0: { // None
                Union5 v34;
                v34 = Union5{Union5_2{}};
                Union16 v35;
                v35 = Union16{Union16_1{v34}};
                v11.v0 = v35;
                break;
            }
            case 1: { // Some
                Union5 v36 = v33.case1.v0;
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
                exit(-1);
            }
        }
    } else {
        int v40;
        v40 = 0;
        method_18(v0, v1, v40);
    }
    Union16 v41 = v11.v0;
    Union5 v45;
    switch (v41.tag) {
        case 0: { // None
            printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
            exit(-1);
            break;
        }
        case 1: { // Some
            Union5 v42 = v41.case1.v0;
            v45 = v42;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    int v46;
    v46 = 5;
    int v47;
    v47 = method_14(v0, v1, v46);
    static_array_list<Union3,5> v50;
    v50 = static_array_list<Union3,5>{};
    int v52;
    v52 = 0;
    while (while_method_8(v47, v52)){
        Union19 v54;
        v54 = Union19{Union19_0{}};
        StackMut20 v55{v54};
        int v56;
        v56 = 3;
        int v57;
        v57 = method_14(v0, v1, v56);
        bool v58;
        v58 = 0 == v57;
        if (v58){
            Union19 v59 = v55.v0;
            switch (v59.tag) {
                case 0: { // None
                    Union3 v60;
                    v60 = Union3{Union3_0{}};
                    Union19 v61;
                    v61 = Union19{Union19_1{v60}};
                    v55.v0 = v61;
                    break;
                }
                case 1: { // Some
                    Union3 v62 = v59.case1.v0;
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
                    exit(-1);
                }
            }
        } else {
            int v66;
            v66 = 0;
            method_18(v0, v1, v66);
        }
        bool v67;
        v67 = 1 == v57;
        if (v67){
            Union19 v68 = v55.v0;
            switch (v68.tag) {
                case 0: { // None
                    Union3 v69;
                    v69 = Union3{Union3_1{}};
                    Union19 v70;
                    v70 = Union19{Union19_1{v69}};
                    v55.v0 = v70;
                    break;
                }
                case 1: { // Some
                    Union3 v71 = v68.case1.v0;
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
                    exit(-1);
                }
            }
        } else {
            int v75;
            v75 = 0;
            method_18(v0, v1, v75);
        }
        bool v76;
        v76 = 2 == v57;
        if (v76){
            Union19 v77 = v55.v0;
            switch (v77.tag) {
                case 0: { // None
                    int v78;
                    v78 = 10;
                    int v79;
                    v79 = method_14(v0, v1, v78);
                    Union3 v80;
                    v80 = Union3{Union3_2{v79}};
                    Union19 v81;
                    v81 = Union19{Union19_1{v80}};
                    v55.v0 = v81;
                    break;
                }
                case 1: { // Some
                    Union3 v82 = v77.case1.v0;
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
                    exit(-1);
                }
            }
        } else {
            int v86;
            v86 = 10;
            method_18(v0, v1, v86);
        }
        Union19 v87 = v55.v0;
        Union3 v91;
        switch (v87.tag) {
            case 0: { // None
                printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                exit(-1);
                break;
            }
            case 1: { // Some
                Union3 v88 = v87.case1.v0;
                v91 = v88;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        v50.push(v91);
        int v92;
        v92 = 1;
        method_18(v0, v1, v92);
        v52 += 1 ;
    }
    int v93;
    v93 = v47;
    while (while_method_9(v93)){
        int v95;
        v95 = 13;
        method_18(v0, v1, v95);
        int v96;
        v96 = 1;
        method_21(v0, v1, v96);
        v93 += 1 ;
    }
    static_array<int,3> v99;
    int v101;
    v101 = 0;
    while (while_method_10(v101)){
        int v103;
        v103 = 5;
        int v104;
        v104 = method_14(v0, v1, v103);
        v99[v101] = v104;
        v101 += 1 ;
    }
    return Tuple13{v50, v45, v9, v99, v6, v4};
}
void method_22(Union3 v0){
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
void method_23(Union5 v0){
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
    Eigen::Matrix<float,3,115> v0;
    Eigen::Matrix<float,3,48> v1;
    int v2;
    v2 = 0;
    StackRefs2 v3{v2, v0, v1};
    Eigen::Matrix<float,3,115> & v4 = v3.v1;
    v4.setZero();
    Eigen::Matrix<float,3,48> & v5 = v3.v2;
    v5.setZero();
    v3.v0 = 0;
    static_array_list<Union3,5> v8;
    v8 = static_array_list<Union3,5>{};
    v8.unsafe_set_length(4);
    Union3 v12;
    v12 = Union3{Union3_2{3}};
    v8[0] = v12;
    Union3 v16;
    v16 = Union3{Union3_2{4}};
    v8[1] = v16;
    Union3 v20;
    v20 = Union3{Union3_0{}};
    v8[2] = v20;
    Union3 v24;
    v24 = Union3{Union3_1{}};
    v8[3] = v24;
    static_array<int,3> v28;
    v28[0] = 1;
    v28[1] = 2;
    v28[2] = 3;
    static_array<float,12> v32;
    int v34;
    v34 = 0;
    while (while_method_4(v34)){
        v32[v34] = 1.0f;
        v34 += 1 ;
    }
    static_array<float,12> v38;
    int v40;
    v40 = 0;
    while (while_method_4(v40)){
        v38[v40] = 1.0f;
        v40 += 1 ;
    }
    static_array<float,12> v44;
    int v46;
    v46 = 0;
    while (while_method_4(v46)){
        v44[v46] = 1.0f;
        v46 += 1 ;
    }
    static_array<float,12> v50;
    int v52;
    v52 = 0;
    while (while_method_4(v52)){
        v50[v52] = 1.0f;
        v52 += 1 ;
    }
    Eigen::Matrix<float,3,115> & v54 = v3.v1;
    Eigen::Matrix<float,3,48> & v55 = v3.v2;
    int & v56 = v3.v0;
    Eigen::Matrix<float,1,115> v57;
    v57.setZero();
    float * v58;
    v58 = &v57(0,0);
    Union5 v59;
    v59 = Union5{Union5_1{}};
    bool v60;
    v60 = false;
    int v61;
    v61 = 8;
    unsigned int v62;
    v62 = 5u;
    method_6(v58, v8, v59, v60, v28, v61, v62);
    v54.row(v56) = v57;
    int & v63 = v3.v0;
    Eigen::Matrix<float,1,48> v64;
    int v65;
    v65 = 0;
    while (while_method_4(v65)){
        float v69;
        v69 = v32[v65];
        v64(0,v65) = v69;
        v65 += 1 ;
    }
    int v71;
    v71 = 0;
    while (while_method_4(v71)){
        int v73;
        v73 = v71 + 12;
        float v76;
        v76 = v38[v71];
        v64(0,v73) = v76;
        v71 += 1 ;
    }
    int v78;
    v78 = 0;
    while (while_method_4(v78)){
        int v80;
        v80 = v78 + 24;
        float v83;
        v83 = v44[v78];
        v64(0,v80) = v83;
        v78 += 1 ;
    }
    int v85;
    v85 = 0;
    while (while_method_4(v85)){
        int v87;
        v87 = v85 + 36;
        float v90;
        v90 = v50[v85];
        v64(0,v87) = v90;
        v85 += 1 ;
    }
    v55.row(v63) = v64;
    int & v92 = v3.v0; Eigen::Matrix<float,3,115> & v93 = v3.v1; Eigen::Matrix<float,3,48> & v94 = v3.v2;
    int v95;
    v95 = v92 + 1;
    int v96;
    v96 = v95 % 3;
    v3.v0 = v96;
    static_array<float,12> v99;
    int v101;
    v101 = 0;
    while (while_method_4(v101)){
        v99[v101] = 2.0f;
        v101 += 1 ;
    }
    static_array<float,12> v105;
    int v107;
    v107 = 0;
    while (while_method_4(v107)){
        v105[v107] = 2.0f;
        v107 += 1 ;
    }
    static_array<float,12> v111;
    int v113;
    v113 = 0;
    while (while_method_4(v113)){
        v111[v113] = 2.0f;
        v113 += 1 ;
    }
    static_array<float,12> v117;
    int v119;
    v119 = 0;
    while (while_method_4(v119)){
        v117[v119] = 2.0f;
        v119 += 1 ;
    }
    Eigen::Matrix<float,3,115> & v121 = v3.v1;
    Eigen::Matrix<float,3,48> & v122 = v3.v2;
    int & v123 = v3.v0;
    Eigen::Matrix<float,1,115> v124;
    v124.setZero();
    float * v125;
    v125 = &v124(0,0);
    Union5 v126;
    v126 = Union5{Union5_2{}};
    bool v127;
    v127 = false;
    int v128;
    v128 = 8;
    unsigned int v129;
    v129 = 5u;
    method_6(v125, v8, v126, v127, v28, v128, v129);
    v121.row(v123) = v124;
    int & v130 = v3.v0;
    Eigen::Matrix<float,1,48> v131;
    int v132;
    v132 = 0;
    while (while_method_4(v132)){
        float v136;
        v136 = v99[v132];
        v131(0,v132) = v136;
        v132 += 1 ;
    }
    int v138;
    v138 = 0;
    while (while_method_4(v138)){
        int v140;
        v140 = v138 + 12;
        float v143;
        v143 = v105[v138];
        v131(0,v140) = v143;
        v138 += 1 ;
    }
    int v145;
    v145 = 0;
    while (while_method_4(v145)){
        int v147;
        v147 = v145 + 24;
        float v150;
        v150 = v111[v145];
        v131(0,v147) = v150;
        v145 += 1 ;
    }
    int v152;
    v152 = 0;
    while (while_method_4(v152)){
        int v154;
        v154 = v152 + 36;
        float v157;
        v157 = v117[v152];
        v131(0,v154) = v157;
        v152 += 1 ;
    }
    v122.row(v130) = v131;
    int & v159 = v3.v0; Eigen::Matrix<float,3,115> & v160 = v3.v1; Eigen::Matrix<float,3,48> & v161 = v3.v2;
    int v162;
    v162 = v159 + 1;
    int v163;
    v163 = v162 % 3;
    v3.v0 = v163;
    static_array<float,12> v166;
    int v168;
    v168 = 0;
    while (while_method_4(v168)){
        v166[v168] = 3.0f;
        v168 += 1 ;
    }
    static_array<float,12> v172;
    int v174;
    v174 = 0;
    while (while_method_4(v174)){
        v172[v174] = 3.0f;
        v174 += 1 ;
    }
    static_array<float,12> v178;
    int v180;
    v180 = 0;
    while (while_method_4(v180)){
        v178[v180] = 3.0f;
        v180 += 1 ;
    }
    static_array<float,12> v184;
    int v186;
    v186 = 0;
    while (while_method_4(v186)){
        v184[v186] = 3.0f;
        v186 += 1 ;
    }
    Eigen::Matrix<float,3,115> & v188 = v3.v1;
    Eigen::Matrix<float,3,48> & v189 = v3.v2;
    int & v190 = v3.v0;
    Eigen::Matrix<float,1,115> v191;
    v191.setZero();
    float * v192;
    v192 = &v191(0,0);
    Union5 v193;
    v193 = Union5{Union5_0{}};
    bool v194;
    v194 = true;
    int v195;
    v195 = 3;
    unsigned int v196;
    v196 = 5u;
    method_6(v192, v8, v193, v194, v28, v195, v196);
    v188.row(v190) = v191;
    int & v197 = v3.v0;
    Eigen::Matrix<float,1,48> v198;
    int v199;
    v199 = 0;
    while (while_method_4(v199)){
        float v203;
        v203 = v166[v199];
        v198(0,v199) = v203;
        v199 += 1 ;
    }
    int v205;
    v205 = 0;
    while (while_method_4(v205)){
        int v207;
        v207 = v205 + 12;
        float v210;
        v210 = v172[v205];
        v198(0,v207) = v210;
        v205 += 1 ;
    }
    int v212;
    v212 = 0;
    while (while_method_4(v212)){
        int v214;
        v214 = v212 + 24;
        float v217;
        v217 = v178[v212];
        v198(0,v214) = v217;
        v212 += 1 ;
    }
    int v219;
    v219 = 0;
    while (while_method_4(v219)){
        int v221;
        v221 = v219 + 36;
        float v224;
        v224 = v184[v219];
        v198(0,v221) = v224;
        v219 += 1 ;
    }
    v189.row(v197) = v198;
    int & v226 = v3.v0; Eigen::Matrix<float,3,115> & v227 = v3.v1; Eigen::Matrix<float,3,48> & v228 = v3.v2;
    int v229;
    v229 = v226 + 1;
    int v230;
    v230 = v229 % 3;
    v3.v0 = v230;
    printf("%s\n","{");
    Eigen::Matrix<float,3,115> & v455 = v3.v1;
    Eigen::Matrix<float,3,48> & v456 = v3.v2;
    int v457;
    v457 = 0;
    while (while_method_10(v457)){
        auto v459 = v455.row(v457);
        bool v460;
        v460 = (v459.array() == 0).all();
        Union11 v470;
        if (v460){
            float * v461;
            v461 = &v459(0,0);
            static_array_list<Union3,5> v462; Union5 v463; bool v464; static_array<int,3> v465; int v466; unsigned int v467;
            Tuple13 tmp3 = method_12(v461);
            v462 = tmp3.v0; v463 = tmp3.v1; v464 = tmp3.v2; v465 = tmp3.v3; v466 = tmp3.v4; v467 = tmp3.v5;
            v470 = Union11{Union11_1{v462, v463, v464, v465, v466, v467}};
        } else {
            v470 = Union11{Union11_0{}};
        }
        auto v471 = v456.row(v457);
        static_array<float,12> v474;
        int v476;
        v476 = 0;
        while (while_method_4(v476)){
            float v478;
            v478 = v471(0,v476);
            v474[v476] = v478;
            v476 += 1 ;
        }
        static_array<float,12> v481;
        int v483;
        v483 = 0;
        while (while_method_4(v483)){
            int v485;
            v485 = v483 + 12;
            float v486;
            v486 = v471(0,v485);
            v481[v483] = v486;
            v483 += 1 ;
        }
        static_array<float,12> v489;
        int v491;
        v491 = 0;
        while (while_method_4(v491)){
            int v493;
            v493 = v491 + 24;
            float v494;
            v494 = v471(0,v493);
            v489[v491] = v494;
            v491 += 1 ;
        }
        static_array<float,12> v497;
        int v499;
        v499 = 0;
        while (while_method_4(v499)){
            int v501;
            v501 = v499 + 36;
            float v502;
            v502 = v471(0,v501);
            v497[v499] = v502;
            v499 += 1 ;
        }
        switch (v470.tag) {
            case 0: { // None
                break;
            }
            case 1: { // Some
                static_array_list<Union3,5> v503 = v470.case1.v0; Union5 v504 = v470.case1.v1; bool v505 = v470.case1.v2; static_array<int,3> v506 = v470.case1.v3; int v507 = v470.case1.v4; unsigned int v508 = v470.case1.v5;
                printf("{%s = %s","action_history", "[");
                int v509;
                v509 = v503.length;
                bool v510;
                v510 = 100 < v509;
                int v511;
                if (v510){
                    v511 = 100;
                } else {
                    v511 = v509;
                }
                int v512;
                v512 = 0;
                while (while_method_8(v511, v512)){
                    Union3 v516;
                    v516 = v503[v512];
                    printf("");
                    method_22(v516);
                    printf("");
                    int v518;
                    v518 = v512 + 1;
                    int v519;
                    v519 = v503.length;
                    bool v520;
                    v520 = v518 < v519;
                    if (v520){
                        printf("%s","; ");
                    } else {
                    }
                    v512 += 1 ;
                }
                int v521;
                v521 = v503.length;
                bool v522;
                v522 = v521 > 100;
                if (v522){
                    printf("%s","; ...");
                } else {
                }
                printf("%s","]");
                printf("; %s = ","card");
                method_23(v504);
                const char * v525;
                if (v505){
                    const char * v523;
                    v523 = "true";
                    v525 = v523;
                } else {
                    const char * v524;
                    v524 = "false";
                    v525 = v524;
                }
                printf("; %s = %s; %s = %s","is_first", v525, "l", "[");
                int v526;
                v526 = 0;
                while (while_method_10(v526)){
                    int v530;
                    v530 = v506[v526];
                    printf("%d",v530);
                    int v532;
                    v532 = v526 + 1;
                    bool v533;
                    v533 = v532 < 3;
                    if (v533){
                        printf("%s","; ");
                    } else {
                    }
                    v526 += 1 ;
                }
                printf("%s","]");
                printf("; %s = %d; %s = %u}","pot", v507, "stack", v508);
                printf("%s"," => ");
                printf("{%s = %s","average_policy", "[");
                int v534;
                v534 = 0;
                while (while_method_4(v534)){
                    float v538;
                    v538 = v474[v534];
                    printf("%f",v538);
                    int v540;
                    v540 = v534 + 1;
                    bool v541;
                    v541 = v540 < 12;
                    if (v541){
                        printf("%s","; ");
                    } else {
                    }
                    v534 += 1 ;
                }
                printf("%s","]");
                printf("; %s = %s","current_policy", "[");
                int v542;
                v542 = 0;
                while (while_method_4(v542)){
                    float v546;
                    v546 = v481[v542];
                    printf("%f",v546);
                    int v548;
                    v548 = v542 + 1;
                    bool v549;
                    v549 = v548 < 12;
                    if (v549){
                        printf("%s","; ");
                    } else {
                    }
                    v542 += 1 ;
                }
                printf("%s","]");
                printf("; %s = %s","ev_values", "[");
                int v550;
                v550 = 0;
                while (while_method_4(v550)){
                    float v554;
                    v554 = v489[v550];
                    printf("%f",v554);
                    int v556;
                    v556 = v550 + 1;
                    bool v557;
                    v557 = v556 < 12;
                    if (v557){
                        printf("%s","; ");
                    } else {
                    }
                    v550 += 1 ;
                }
                printf("%s","]");
                printf("; %s = %s","ev_weights", "[");
                int v558;
                v558 = 0;
                while (while_method_4(v558)){
                    float v562;
                    v562 = v497[v558];
                    printf("%f",v562);
                    int v564;
                    v564 = v558 + 1;
                    bool v565;
                    v565 = v564 < 12;
                    if (v565){
                        printf("%s","; ");
                    } else {
                    }
                    v558 += 1 ;
                }
                printf("%s","]");
                printf("}\n");
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        v457 += 1 ;
    }
    printf("%s\n","}");
    printf("\n");
    fflush(stdout);
    return 0;
}
