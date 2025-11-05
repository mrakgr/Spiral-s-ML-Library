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
void method_24(Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v0, Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v1, Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v2, Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v3){
    v3 = ((v0 * v2.transpose() - v2.rowwise().sum().transpose().replicate(v0.rows(),1)).array() / 0.001f).exp().matrix().transpose() * v1;
    return ;
}
int main() {
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> v0(16384, 115);
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> v1(16384, 48);
    int v2;
    v2 = 0;
    StackRefs2 v3{v2, v0, v1};
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v4 = v3.v1;
    v4.setZero();
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v5 = v3.v2;
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
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v54 = v3.v1;
    int v55;
    v55 = v54.rows();
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v56 = v3.v1;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v57 = v3.v2;
    int & v58 = v3.v0;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> v59(1, 115);
    v59.setZero();
    float * v60;
    v60 = &v59(0,0);
    Union5 v61;
    v61 = Union5{Union5_1{}};
    bool v62;
    v62 = false;
    int v63;
    v63 = 8;
    unsigned int v64;
    v64 = 5u;
    method_6(v60, v8, v61, v62, v28, v63, v64);
    v56.row(v58) = v59;
    int & v65 = v3.v0;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> v66(1, 48);
    int v67;
    v67 = 0;
    while (while_method_4(v67)){
        float v71;
        v71 = v32[v67];
        v66(0,v67) = v71;
        v67 += 1 ;
    }
    int v73;
    v73 = 0;
    while (while_method_4(v73)){
        int v75;
        v75 = v73 + 12;
        float v78;
        v78 = v38[v73];
        v66(0,v75) = v78;
        v73 += 1 ;
    }
    int v80;
    v80 = 0;
    while (while_method_4(v80)){
        int v82;
        v82 = v80 + 24;
        float v85;
        v85 = v44[v80];
        v66(0,v82) = v85;
        v80 += 1 ;
    }
    int v87;
    v87 = 0;
    while (while_method_4(v87)){
        int v89;
        v89 = v87 + 36;
        float v92;
        v92 = v50[v87];
        v66(0,v89) = v92;
        v87 += 1 ;
    }
    v57.row(v65) = v66;
    int & v94 = v3.v0; Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v95 = v3.v1; Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v96 = v3.v2;
    int v97;
    v97 = v94 + 1;
    int v98;
    v98 = v97 % v55;
    v3.v0 = v98;
    static_array<float,12> v101;
    int v103;
    v103 = 0;
    while (while_method_4(v103)){
        v101[v103] = 2.0f;
        v103 += 1 ;
    }
    static_array<float,12> v107;
    int v109;
    v109 = 0;
    while (while_method_4(v109)){
        v107[v109] = 2.0f;
        v109 += 1 ;
    }
    static_array<float,12> v113;
    int v115;
    v115 = 0;
    while (while_method_4(v115)){
        v113[v115] = 2.0f;
        v115 += 1 ;
    }
    static_array<float,12> v119;
    int v121;
    v121 = 0;
    while (while_method_4(v121)){
        v119[v121] = 2.0f;
        v121 += 1 ;
    }
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v123 = v3.v1;
    int v124;
    v124 = v123.rows();
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v125 = v3.v1;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v126 = v3.v2;
    int & v127 = v3.v0;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> v128(1, 115);
    v128.setZero();
    float * v129;
    v129 = &v128(0,0);
    Union5 v130;
    v130 = Union5{Union5_2{}};
    bool v131;
    v131 = false;
    int v132;
    v132 = 8;
    unsigned int v133;
    v133 = 5u;
    method_6(v129, v8, v130, v131, v28, v132, v133);
    v125.row(v127) = v128;
    int & v134 = v3.v0;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> v135(1, 48);
    int v136;
    v136 = 0;
    while (while_method_4(v136)){
        float v140;
        v140 = v101[v136];
        v135(0,v136) = v140;
        v136 += 1 ;
    }
    int v142;
    v142 = 0;
    while (while_method_4(v142)){
        int v144;
        v144 = v142 + 12;
        float v147;
        v147 = v107[v142];
        v135(0,v144) = v147;
        v142 += 1 ;
    }
    int v149;
    v149 = 0;
    while (while_method_4(v149)){
        int v151;
        v151 = v149 + 24;
        float v154;
        v154 = v113[v149];
        v135(0,v151) = v154;
        v149 += 1 ;
    }
    int v156;
    v156 = 0;
    while (while_method_4(v156)){
        int v158;
        v158 = v156 + 36;
        float v161;
        v161 = v119[v156];
        v135(0,v158) = v161;
        v156 += 1 ;
    }
    v126.row(v134) = v135;
    int & v163 = v3.v0; Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v164 = v3.v1; Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v165 = v3.v2;
    int v166;
    v166 = v163 + 1;
    int v167;
    v167 = v166 % v124;
    v3.v0 = v167;
    static_array<float,12> v170;
    int v172;
    v172 = 0;
    while (while_method_4(v172)){
        v170[v172] = 3.0f;
        v172 += 1 ;
    }
    static_array<float,12> v176;
    int v178;
    v178 = 0;
    while (while_method_4(v178)){
        v176[v178] = 3.0f;
        v178 += 1 ;
    }
    static_array<float,12> v182;
    int v184;
    v184 = 0;
    while (while_method_4(v184)){
        v182[v184] = 3.0f;
        v184 += 1 ;
    }
    static_array<float,12> v188;
    int v190;
    v190 = 0;
    while (while_method_4(v190)){
        v188[v190] = 3.0f;
        v190 += 1 ;
    }
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v192 = v3.v1;
    int v193;
    v193 = v192.rows();
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v194 = v3.v1;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v195 = v3.v2;
    int & v196 = v3.v0;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> v197(1, 115);
    v197.setZero();
    float * v198;
    v198 = &v197(0,0);
    Union5 v199;
    v199 = Union5{Union5_0{}};
    bool v200;
    v200 = true;
    int v201;
    v201 = 3;
    unsigned int v202;
    v202 = 5u;
    method_6(v198, v8, v199, v200, v28, v201, v202);
    v194.row(v196) = v197;
    int & v203 = v3.v0;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> v204(1, 48);
    int v205;
    v205 = 0;
    while (while_method_4(v205)){
        float v209;
        v209 = v170[v205];
        v204(0,v205) = v209;
        v205 += 1 ;
    }
    int v211;
    v211 = 0;
    while (while_method_4(v211)){
        int v213;
        v213 = v211 + 12;
        float v216;
        v216 = v176[v211];
        v204(0,v213) = v216;
        v211 += 1 ;
    }
    int v218;
    v218 = 0;
    while (while_method_4(v218)){
        int v220;
        v220 = v218 + 24;
        float v223;
        v223 = v182[v218];
        v204(0,v220) = v223;
        v218 += 1 ;
    }
    int v225;
    v225 = 0;
    while (while_method_4(v225)){
        int v227;
        v227 = v225 + 36;
        float v230;
        v230 = v188[v225];
        v204(0,v227) = v230;
        v225 += 1 ;
    }
    v195.row(v203) = v204;
    int & v232 = v3.v0; Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v233 = v3.v1; Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v234 = v3.v2;
    int v235;
    v235 = v232 + 1;
    int v236;
    v236 = v235 % v193;
    v3.v0 = v236;
    printf("%s\n","{");
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v465 = v3.v1;
    int v466;
    v466 = v465.rows();
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v467 = v3.v1;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v468 = v3.v2;
    int v469;
    v469 = 0;
    while (while_method_8(v466, v469)){
        auto v471 = v467.row(v469);
        bool v472;
        v472 = (v471.array() == 0).all();
        Union11 v482;
        if (v472){
            v482 = Union11{Union11_0{}};
        } else {
            float * v474;
            v474 = &v471(0,0);
            static_array_list<Union3,5> v475; Union5 v476; bool v477; static_array<int,3> v478; int v479; unsigned int v480;
            Tuple13 tmp3 = method_12(v474);
            v475 = tmp3.v0; v476 = tmp3.v1; v477 = tmp3.v2; v478 = tmp3.v3; v479 = tmp3.v4; v480 = tmp3.v5;
            v482 = Union11{Union11_1{v475, v476, v477, v478, v479, v480}};
        }
        auto v483 = v468.row(v469);
        static_array<float,12> v486;
        int v488;
        v488 = 0;
        while (while_method_4(v488)){
            float v490;
            v490 = v483(0,v488);
            v486[v488] = v490;
            v488 += 1 ;
        }
        static_array<float,12> v493;
        int v495;
        v495 = 0;
        while (while_method_4(v495)){
            int v497;
            v497 = v495 + 12;
            float v498;
            v498 = v483(0,v497);
            v493[v495] = v498;
            v495 += 1 ;
        }
        static_array<float,12> v501;
        int v503;
        v503 = 0;
        while (while_method_4(v503)){
            int v505;
            v505 = v503 + 24;
            float v506;
            v506 = v483(0,v505);
            v501[v503] = v506;
            v503 += 1 ;
        }
        static_array<float,12> v509;
        int v511;
        v511 = 0;
        while (while_method_4(v511)){
            int v513;
            v513 = v511 + 36;
            float v514;
            v514 = v483(0,v513);
            v509[v511] = v514;
            v511 += 1 ;
        }
        switch (v482.tag) {
            case 0: { // None
                break;
            }
            case 1: { // Some
                static_array_list<Union3,5> v515 = v482.case1.v0; Union5 v516 = v482.case1.v1; bool v517 = v482.case1.v2; static_array<int,3> v518 = v482.case1.v3; int v519 = v482.case1.v4; unsigned int v520 = v482.case1.v5;
                printf("{%s = %s","action_history", "[");
                int v521;
                v521 = v515.length;
                bool v522;
                v522 = 100 < v521;
                int v523;
                if (v522){
                    v523 = 100;
                } else {
                    v523 = v521;
                }
                int v524;
                v524 = 0;
                while (while_method_8(v523, v524)){
                    Union3 v528;
                    v528 = v515[v524];
                    printf("");
                    method_22(v528);
                    printf("");
                    int v530;
                    v530 = v524 + 1;
                    int v531;
                    v531 = v515.length;
                    bool v532;
                    v532 = v530 < v531;
                    if (v532){
                        printf("%s","; ");
                    } else {
                    }
                    v524 += 1 ;
                }
                int v533;
                v533 = v515.length;
                bool v534;
                v534 = v533 > 100;
                if (v534){
                    printf("%s","; ...");
                } else {
                }
                printf("%s","]");
                printf("; %s = ","card");
                method_23(v516);
                const char * v537;
                if (v517){
                    const char * v535;
                    v535 = "true";
                    v537 = v535;
                } else {
                    const char * v536;
                    v536 = "false";
                    v537 = v536;
                }
                printf("; %s = %s; %s = %s","is_first", v537, "l", "[");
                int v538;
                v538 = 0;
                while (while_method_10(v538)){
                    int v542;
                    v542 = v518[v538];
                    printf("%d",v542);
                    int v544;
                    v544 = v538 + 1;
                    bool v545;
                    v545 = v544 < 3;
                    if (v545){
                        printf("%s","; ");
                    } else {
                    }
                    v538 += 1 ;
                }
                printf("%s","]");
                printf("; %s = %d; %s = %u}","pot", v519, "stack", v520);
                printf("%s"," => ");
                printf("{%s = %s","average_policy", "[");
                int v546;
                v546 = 0;
                while (while_method_4(v546)){
                    float v550;
                    v550 = v486[v546];
                    printf("%f",v550);
                    int v552;
                    v552 = v546 + 1;
                    bool v553;
                    v553 = v552 < 12;
                    if (v553){
                        printf("%s","; ");
                    } else {
                    }
                    v546 += 1 ;
                }
                printf("%s","]");
                printf("; %s = %s","current_policy", "[");
                int v554;
                v554 = 0;
                while (while_method_4(v554)){
                    float v558;
                    v558 = v493[v554];
                    printf("%f",v558);
                    int v560;
                    v560 = v554 + 1;
                    bool v561;
                    v561 = v560 < 12;
                    if (v561){
                        printf("%s","; ");
                    } else {
                    }
                    v554 += 1 ;
                }
                printf("%s","]");
                printf("; %s = %s","ev_values", "[");
                int v562;
                v562 = 0;
                while (while_method_4(v562)){
                    float v566;
                    v566 = v501[v562];
                    printf("%f",v566);
                    int v568;
                    v568 = v562 + 1;
                    bool v569;
                    v569 = v568 < 12;
                    if (v569){
                        printf("%s","; ");
                    } else {
                    }
                    v562 += 1 ;
                }
                printf("%s","]");
                printf("; %s = %s","ev_weights", "[");
                int v570;
                v570 = 0;
                while (while_method_4(v570)){
                    float v574;
                    v574 = v509[v570];
                    printf("%f",v574);
                    int v576;
                    v576 = v570 + 1;
                    bool v577;
                    v577 = v576 < 12;
                    if (v577){
                        printf("%s","; ");
                    } else {
                    }
                    v570 += 1 ;
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
        v469 += 1 ;
    }
    printf("%s\n","}");
    printf("\n");
    fflush(stdout);
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> v729(1, 115);
    v729.setZero();
    float * v730;
    v730 = &v729(0,0);
    Union5 v731;
    v731 = Union5{Union5_1{}};
    bool v732;
    v732 = false;
    int v733;
    v733 = 8;
    unsigned int v734;
    v734 = 5u;
    method_6(v730, v8, v731, v732, v28, v733, v734);
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v735 = v3.v2;
    int v736;
    v736 = v735.cols();
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> v737(1, v736);
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v738 = v3.v1;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v739 = v738;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v740 = v3.v2;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v741 = v740;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v742 = v729;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v743 = v737;
    method_24(v739, v741, v742, v743);
    static_array<float,12> v746;
    int v748;
    v748 = 0;
    while (while_method_4(v748)){
        float v750;
        v750 = v737(0,v748);
        v746[v748] = v750;
        v748 += 1 ;
    }
    static_array<float,12> v753;
    int v755;
    v755 = 0;
    while (while_method_4(v755)){
        int v757;
        v757 = v755 + 12;
        float v758;
        v758 = v737(0,v757);
        v753[v755] = v758;
        v755 += 1 ;
    }
    static_array<float,12> v761;
    int v763;
    v763 = 0;
    while (while_method_4(v763)){
        int v765;
        v765 = v763 + 24;
        float v766;
        v766 = v737(0,v765);
        v761[v763] = v766;
        v763 += 1 ;
    }
    static_array<float,12> v769;
    int v771;
    v771 = 0;
    while (while_method_4(v771)){
        int v773;
        v773 = v771 + 36;
        float v774;
        v774 = v737(0,v773);
        v769[v771] = v774;
        v771 += 1 ;
    }
    printf("{%s = %s","average_policy", "[");
    int v841;
    v841 = 0;
    while (while_method_4(v841)){
        float v845;
        v845 = v746[v841];
        printf("%f",v845);
        int v847;
        v847 = v841 + 1;
        bool v848;
        v848 = v847 < 12;
        if (v848){
            printf("%s","; ");
        } else {
        }
        v841 += 1 ;
    }
    printf("%s","]");
    printf("; %s = %s","current_policy", "[");
    int v849;
    v849 = 0;
    while (while_method_4(v849)){
        float v853;
        v853 = v753[v849];
        printf("%f",v853);
        int v855;
        v855 = v849 + 1;
        bool v856;
        v856 = v855 < 12;
        if (v856){
            printf("%s","; ");
        } else {
        }
        v849 += 1 ;
    }
    printf("%s","]");
    printf("; %s = %s","ev_values", "[");
    int v857;
    v857 = 0;
    while (while_method_4(v857)){
        float v861;
        v861 = v761[v857];
        printf("%f",v861);
        int v863;
        v863 = v857 + 1;
        bool v864;
        v864 = v863 < 12;
        if (v864){
            printf("%s","; ");
        } else {
        }
        v857 += 1 ;
    }
    printf("%s","]");
    printf("; %s = %s","ev_weights", "[");
    int v865;
    v865 = 0;
    while (while_method_4(v865)){
        float v869;
        v869 = v769[v865];
        printf("%f",v869);
        int v871;
        v871 = v865 + 1;
        bool v872;
        v872 = v871 < 12;
        if (v872){
            printf("%s","; ");
        } else {
        }
        v865 += 1 ;
    }
    printf("%s","]");
    printf("}\n");
    fflush(stdout);
    return 0;
}
