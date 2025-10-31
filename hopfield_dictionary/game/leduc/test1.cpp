#include "test1.hpp"
inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 115;
    return v1;
}
inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 5;
    return v1;
}
inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
void method_0(unsigned int * v0, static_array_list<Union0,5> v1, Union1 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6){
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
            exit(-1);
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
        int v78 = v7.v0;
        int v79;
        v79 = v78 + 13;
        int v80;
        v80 = v74.tag;
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
        switch (v74.tag) {
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
                int v92 = v74.case2.v0;
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
                exit(-1);
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
        int v125;
        v125 = v4[v123];
        bool v129;
        v129 = v125 < 5;
        bool v130;
        v130 = v129 == false;
        if (v130){
            assert("The input to the pickler must be 0 or positive." && v129);
        } else {
        }
        bool v132;
        v132 = 0 <= v125;
        bool v133;
        v133 = v132 == false;
        if (v133){
            assert("The input to the pickler must be less than the specified length." && v132);
        } else {
        }
        int v135 = v7.v0;
        int v136;
        v136 = v135 + v125;
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
int method_2(unsigned int * v0, StackMut0 & v1, int v2){
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
void method_3(unsigned int * v0, StackMut0 & v1, int v2){
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
void method_4(unsigned int * v0, StackMut0 & v1, int v2){
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
Tuple0 method_1(unsigned int * v0){
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
                exit(-1);
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
                exit(-1);
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
                exit(-1);
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
            exit(-1);
            break;
        }
        case 1: { // Some
            Union1 v42 = v41.case1.v0;
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
    v47 = method_2(v0, v1, v46);
    static_array_list<Union0,5> v48;
    v48 = static_array_list<Union0,5>{};
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
                    exit(-1);
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
                    exit(-1);
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
                    exit(-1);
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
                exit(-1);
                break;
            }
            case 1: { // Some
                Union0 v88 = v87.case1.v0;
                v91 = v88;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        v48.push(v91);
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
    static_array<int,3> v97;
    int v101;
    v101 = 0;
    while (while_method_3(v101)){
        int v103;
        v103 = 5;
        int v104;
        v104 = method_2(v0, v1, v103);
        v97[v101] = v104;
        v101 += 1 ;
    }
    return Tuple0{v48, v45, v9, v97, v6, v4};
}
void method_5(Union0 v0){
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
void method_6(Union1 v0){
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
    static_array_list<Union0,5> v0;
    v0 = static_array_list<Union0,5>{};
    v0.unsafe_set_length(4);
    Union0 v4;
    v4 = Union0{Union0_2{3}};
    v0[0] = v4;
    Union0 v8;
    v8 = Union0{Union0_2{4}};
    v0[1] = v8;
    Union0 v12;
    v12 = Union0{Union0_0{}};
    v0[2] = v12;
    Union0 v16;
    v16 = Union0{Union0_1{}};
    v0[3] = v16;
    static_array<int,3> v20;
    v20[0] = 1;
    v20[1] = 2;
    v20[2] = 3;
    unsigned int v24[4];
    int v25;
    v25 = 0;
    while (while_method_0(v25)){
        int v27;
        v27 = v25 / 32;
        unsigned int v28;
        v28 = v24[v27];
        int v29;
        v29 = v25 % 32;
        unsigned int v30;
        v30 = 1u << v29;
        unsigned int v31;
        v31 = ~v30;
        unsigned int v32;
        v32 = v28 & v31;
        v24[v27] = v32;
        v25 += 1 ;
    }
    Union1 v33;
    v33 = Union1{Union1_1{}};
    bool v34;
    v34 = false;
    int v35;
    v35 = 8;
    unsigned int v36;
    v36 = 5u;
    method_0(v24, v0, v33, v34, v20, v35, v36);
    static_array_list<Union0,5> v37; Union1 v38; bool v39; static_array<int,3> v40; int v41; unsigned int v42;
    Tuple0 tmp3 = method_1(v24);
    v37 = tmp3.v0; v38 = tmp3.v1; v39 = tmp3.v2; v40 = tmp3.v3; v41 = tmp3.v4; v42 = tmp3.v5;
    printf("{%s = %s","action_history", "[");
    int v43;
    v43 = v37.length;
    bool v44;
    v44 = 100 < v43;
    int v45;
    if (v44){
        v45 = 100;
    } else {
        v45 = v43;
    }
    int v46;
    v46 = 0;
    while (while_method_1(v45, v46)){
        Union0 v48;
        v48 = v37[v46];
        printf("");
        method_5(v48);
        printf("");
        int v52;
        v52 = v46 + 1;
        int v53;
        v53 = v37.length;
        bool v54;
        v54 = v52 < v53;
        if (v54){
            printf("%s","; ");
        } else {
        }
        v46 += 1 ;
    }
    int v55;
    v55 = v37.length;
    bool v56;
    v56 = v55 > 100;
    if (v56){
        printf("%s","; ...");
    } else {
    }
    printf("%s","]");
    printf("; %s = ","card");
    method_6(v38);
    const char * v59;
    if (v39){
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
        v62 = v40[v60];
        printf("%d",v62);
        int v66;
        v66 = v60 + 1;
        bool v67;
        v67 = v66 < 3;
        if (v67){
            printf("%s","; ");
        } else {
        }
        v60 += 1 ;
    }
    printf("%s","]");
    printf("; %s = %d; %s = %u}\n","pot", v41, "stack", v42);
    fflush(stdout);
    return 0;
}
