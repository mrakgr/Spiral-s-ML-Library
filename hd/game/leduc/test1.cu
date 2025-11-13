#include "test1.hpp"
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 100;
    return v1;
}
__device__ inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
__device__ void method_2(unsigned int * v0, static_array_list<Union0,5> v1, Union1 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6){
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
    switch (v2.tag) {
        case 0: { // Jack
            int v48 = v7.v0;
            v7.v0 = v48;
            int v49 = v7.v0;
            int v50;
            v50 = v49 / 32;
            unsigned int v51;
            v51 = v0[v50];
            int v52;
            v52 = v49 % 32;
            unsigned int v53;
            v53 = 1u << v52;
            unsigned int v54;
            v54 = v51 | v53;
            v0[v50] = v54;
            int v55 = v7.v0;
            int v56;
            v56 = v55 + 1;
            v7.v0 = v56;
            break;
        }
        case 1: { // King
            int v57 = v7.v0;
            int v58;
            v58 = v57 + 1;
            v7.v0 = v58;
            int v59 = v7.v0;
            int v60;
            v60 = v59 / 32;
            unsigned int v61;
            v61 = v0[v60];
            int v62;
            v62 = v59 % 32;
            unsigned int v63;
            v63 = 1u << v62;
            unsigned int v64;
            v64 = v61 | v63;
            v0[v60] = v64;
            int v65 = v7.v0;
            int v66;
            v66 = v65 + 1;
            v7.v0 = v66;
            break;
        }
        case 2: { // Queen
            int v67 = v7.v0;
            int v68;
            v68 = v67 + 2;
            v7.v0 = v68;
            int v69 = v7.v0;
            int v70;
            v70 = v69 / 32;
            unsigned int v71;
            v71 = v0[v70];
            int v72;
            v72 = v69 % 32;
            unsigned int v73;
            v73 = 1u << v72;
            unsigned int v74;
            v74 = v71 | v73;
            v0[v70] = v74;
            int v75 = v7.v0;
            int v76;
            v76 = v75 + 1;
            v7.v0 = v76;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            __trap();
        }
    }
    v7.v0 = v47;
    int v77;
    v77 = v1.length;
    int v78;
    v78 = 0;
    while (while_method_1(v77, v78)){
        Union0 v81;
        v81 = v1[v78];
        int v84 = v7.v0;
        int v85;
        v85 = v84 + 12;
        switch (v81.tag) {
            case 0: { // Call
                int v86 = v7.v0;
                v7.v0 = v86;
                int v87 = v7.v0;
                int v88;
                v88 = v87 / 32;
                unsigned int v89;
                v89 = v0[v88];
                int v90;
                v90 = v87 % 32;
                unsigned int v91;
                v91 = 1u << v90;
                unsigned int v92;
                v92 = v89 | v91;
                v0[v88] = v92;
                int v93 = v7.v0;
                int v94;
                v94 = v93 + 1;
                v7.v0 = v94;
                break;
            }
            case 1: { // Fold
                int v95 = v7.v0;
                int v96;
                v96 = v95 + 1;
                v7.v0 = v96;
                int v97 = v7.v0;
                int v98;
                v98 = v97 / 32;
                unsigned int v99;
                v99 = v0[v98];
                int v100;
                v100 = v97 % 32;
                unsigned int v101;
                v101 = 1u << v100;
                unsigned int v102;
                v102 = v99 | v101;
                v0[v98] = v102;
                int v103 = v7.v0;
                int v104;
                v104 = v103 + 1;
                v7.v0 = v104;
                break;
            }
            case 2: { // Raise
                int v105 = v81.case2.v0;
                int v106 = v7.v0;
                int v107;
                v107 = v106 + 2;
                v7.v0 = v107;
                bool v108;
                v108 = v105 < 10;
                bool v109;
                v109 = v108 == false;
                if (v109){
                    assert("The input to the pickler must be 0 or positive." && v108);
                } else {
                }
                bool v111;
                v111 = 0 <= v105;
                bool v112;
                v112 = v111 == false;
                if (v112){
                    assert("The input to the pickler must be less than the specified length." && v111);
                } else {
                }
                int v114 = v7.v0;
                int v115;
                v115 = v114 + v105;
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
                v122 = v121 + 10;
                v7.v0 = v122;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                __trap();
            }
        }
        v7.v0 = v85;
        v78 += 1 ;
    }
    int v123;
    v123 = 0;
    while (while_method_2(v123)){
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
__device__ int method_4(unsigned int * v0, StackMut0 & v1, int v2){
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
__device__ void method_5(unsigned int * v0, StackMut0 & v1, int v2){
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
__device__ bool method_6(StackMut0 & v0, unsigned int * v1, int v2){
    int v3; int v4; int v5;
    Tuple1 tmp2 = Tuple1{0, 0, 0};
    v3 = tmp2.v0; v4 = tmp2.v1; v5 = tmp2.v2;
    while (while_method_1(v2, v3)){
        int v7 = v0.v0;
        int v8;
        v8 = v7 / 32;
        unsigned int v9;
        v9 = v1[v8];
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
        int v15 = v0.v0;
        int v16;
        v16 = v15 + 1;
        v0.v0 = v16;
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
    int v20;
    v20 = -v2;
    int v21 = v0.v0;
    int v22;
    v22 = v21 + v20;
    v0.v0 = v22;
    bool v23;
    v23 = v5 > 0;
    return v23;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 5;
    return v1;
}
__device__ Tuple0 method_3(unsigned int * v0){
    StackMut0 v1{0};
    int v2;
    v2 = 10;
    int v3;
    v3 = method_4(v0, v1, v2);
    unsigned int v4;
    v4 = (unsigned int)v3;
    int v5;
    v5 = 10;
    int v6;
    v6 = method_4(v0, v1, v5);
    int v7;
    v7 = 2;
    int v8;
    v8 = method_4(v0, v1, v7);
    bool v9;
    v9 = v8 == 1;
    Union2 v10;
    v10 = Union2{Union2_0{}};
    StackMut1 v11{v10};
    StackMut2 v12{false};
    bool v13 = v12.v0;
    if (v13){
        int v14;
        v14 = 1;
        method_5(v0, v1, v14);
    } else {
        int v15;
        v15 = 1;
        bool v16;
        v16 = method_6(v1, v0, v15);
        if (v16){
            Union2 v17 = v11.v0;
            switch (v17.tag) {
                case 0: { // None
                    int v18;
                    v18 = 1;
                    int v19;
                    v19 = method_4(v0, v1, v18);
                    bool v20;
                    v20 = v19 == 0;
                    bool v21;
                    v21 = v20 == false;
                    if (v21){
                        assert("Invalid unit value in unpickle." && v20);
                    } else {
                    }
                    Union1 v23;
                    v23 = Union1{Union1_0{}};
                    Union2 v24;
                    v24 = Union2{Union2_1{v23}};
                    v11.v0 = v24;
                    break;
                }
                case 1: { // Some
                    Union1 v25 = v17.case1.v0;
                    bool v26;
                    v26 = false;
                    bool v27;
                    v27 = v26 == false;
                    if (v27){
                        assert("Duplicate union type instances in the unpickle Alt case." && v26);
                    } else {
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    __trap();
                }
            }
            v12.v0 = true;
        } else {
        }
    }
    bool v29 = v12.v0;
    if (v29){
        int v30;
        v30 = 1;
        method_5(v0, v1, v30);
    } else {
        int v31;
        v31 = 1;
        bool v32;
        v32 = method_6(v1, v0, v31);
        if (v32){
            Union2 v33 = v11.v0;
            switch (v33.tag) {
                case 0: { // None
                    int v34;
                    v34 = 1;
                    int v35;
                    v35 = method_4(v0, v1, v34);
                    bool v36;
                    v36 = v35 == 0;
                    bool v37;
                    v37 = v36 == false;
                    if (v37){
                        assert("Invalid unit value in unpickle." && v36);
                    } else {
                    }
                    Union1 v39;
                    v39 = Union1{Union1_1{}};
                    Union2 v40;
                    v40 = Union2{Union2_1{v39}};
                    v11.v0 = v40;
                    break;
                }
                case 1: { // Some
                    Union1 v41 = v33.case1.v0;
                    bool v42;
                    v42 = false;
                    bool v43;
                    v43 = v42 == false;
                    if (v43){
                        assert("Duplicate union type instances in the unpickle Alt case." && v42);
                    } else {
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    __trap();
                }
            }
            v12.v0 = true;
        } else {
        }
    }
    bool v45 = v12.v0;
    if (v45){
        int v46;
        v46 = 1;
        method_5(v0, v1, v46);
    } else {
        int v47;
        v47 = 1;
        bool v48;
        v48 = method_6(v1, v0, v47);
        if (v48){
            Union2 v49 = v11.v0;
            switch (v49.tag) {
                case 0: { // None
                    int v50;
                    v50 = 1;
                    int v51;
                    v51 = method_4(v0, v1, v50);
                    bool v52;
                    v52 = v51 == 0;
                    bool v53;
                    v53 = v52 == false;
                    if (v53){
                        assert("Invalid unit value in unpickle." && v52);
                    } else {
                    }
                    Union1 v55;
                    v55 = Union1{Union1_2{}};
                    Union2 v56;
                    v56 = Union2{Union2_1{v55}};
                    v11.v0 = v56;
                    break;
                }
                case 1: { // Some
                    Union1 v57 = v49.case1.v0;
                    bool v58;
                    v58 = false;
                    bool v59;
                    v59 = v58 == false;
                    if (v59){
                        assert("Duplicate union type instances in the unpickle Alt case." && v58);
                    } else {
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    __trap();
                }
            }
            v12.v0 = true;
        } else {
        }
    }
    Union2 v61 = v11.v0;
    Union1 v65;
    switch (v61.tag) {
        case 0: { // None
            printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
            __trap();
            break;
        }
        case 1: { // Some
            Union1 v62 = v61.case1.v0;
            v65 = v62;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            __trap();
        }
    }
    static_array_list<Union0,5> v67;
    v67 = static_array_list<Union0,5>{};
    StackMut2 v70{false};
    int v71;
    v71 = 0;
    while (while_method_3(v71)){
        bool v73 = v70.v0;
        if (v73){
            int v74;
            v74 = 12;
            method_5(v0, v1, v74);
        } else {
            int v75;
            v75 = 12;
            bool v76;
            v76 = method_6(v1, v0, v75);
            if (v76){
                Union3 v77;
                v77 = Union3{Union3_0{}};
                StackMut3 v78{v77};
                StackMut2 v79{false};
                bool v80 = v79.v0;
                if (v80){
                    int v81;
                    v81 = 1;
                    method_5(v0, v1, v81);
                } else {
                    int v82;
                    v82 = 1;
                    bool v83;
                    v83 = method_6(v1, v0, v82);
                    if (v83){
                        Union3 v84 = v78.v0;
                        switch (v84.tag) {
                            case 0: { // None
                                int v85;
                                v85 = 1;
                                int v86;
                                v86 = method_4(v0, v1, v85);
                                bool v87;
                                v87 = v86 == 0;
                                bool v88;
                                v88 = v87 == false;
                                if (v88){
                                    assert("Invalid unit value in unpickle." && v87);
                                } else {
                                }
                                Union0 v90;
                                v90 = Union0{Union0_0{}};
                                Union3 v91;
                                v91 = Union3{Union3_1{v90}};
                                v78.v0 = v91;
                                break;
                            }
                            case 1: { // Some
                                Union0 v92 = v84.case1.v0;
                                bool v93;
                                v93 = false;
                                bool v94;
                                v94 = v93 == false;
                                if (v94){
                                    assert("Duplicate union type instances in the unpickle Alt case." && v93);
                                } else {
                                }
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                                __trap();
                            }
                        }
                        v79.v0 = true;
                    } else {
                    }
                }
                bool v96 = v79.v0;
                if (v96){
                    int v97;
                    v97 = 1;
                    method_5(v0, v1, v97);
                } else {
                    int v98;
                    v98 = 1;
                    bool v99;
                    v99 = method_6(v1, v0, v98);
                    if (v99){
                        Union3 v100 = v78.v0;
                        switch (v100.tag) {
                            case 0: { // None
                                int v101;
                                v101 = 1;
                                int v102;
                                v102 = method_4(v0, v1, v101);
                                bool v103;
                                v103 = v102 == 0;
                                bool v104;
                                v104 = v103 == false;
                                if (v104){
                                    assert("Invalid unit value in unpickle." && v103);
                                } else {
                                }
                                Union0 v106;
                                v106 = Union0{Union0_1{}};
                                Union3 v107;
                                v107 = Union3{Union3_1{v106}};
                                v78.v0 = v107;
                                break;
                            }
                            case 1: { // Some
                                Union0 v108 = v100.case1.v0;
                                bool v109;
                                v109 = false;
                                bool v110;
                                v110 = v109 == false;
                                if (v110){
                                    assert("Duplicate union type instances in the unpickle Alt case." && v109);
                                } else {
                                }
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                                __trap();
                            }
                        }
                        v79.v0 = true;
                    } else {
                    }
                }
                bool v112 = v79.v0;
                if (v112){
                    int v113;
                    v113 = 10;
                    method_5(v0, v1, v113);
                } else {
                    int v114;
                    v114 = 10;
                    bool v115;
                    v115 = method_6(v1, v0, v114);
                    if (v115){
                        Union3 v116 = v78.v0;
                        switch (v116.tag) {
                            case 0: { // None
                                int v117;
                                v117 = 10;
                                int v118;
                                v118 = method_4(v0, v1, v117);
                                Union0 v119;
                                v119 = Union0{Union0_2{v118}};
                                Union3 v120;
                                v120 = Union3{Union3_1{v119}};
                                v78.v0 = v120;
                                break;
                            }
                            case 1: { // Some
                                Union0 v121 = v116.case1.v0;
                                bool v122;
                                v122 = false;
                                bool v123;
                                v123 = v122 == false;
                                if (v123){
                                    assert("Duplicate union type instances in the unpickle Alt case." && v122);
                                } else {
                                }
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                                __trap();
                            }
                        }
                        v79.v0 = true;
                    } else {
                    }
                }
                Union3 v125 = v78.v0;
                Union0 v129;
                switch (v125.tag) {
                    case 0: { // None
                        printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                        __trap();
                        break;
                    }
                    case 1: { // Some
                        Union0 v126 = v125.case1.v0;
                        v129 = v126;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        __trap();
                    }
                }
                v67.push(v129);
            } else {
                v70.v0 = true;
            }
        }
        v71 += 1 ;
    }
    static_array<int,3> v131;
    int v134;
    v134 = 0;
    while (while_method_2(v134)){
        int v136;
        v136 = 5;
        int v137;
        v137 = method_4(v0, v1, v136);
        v131[v134] = v137;
        v134 += 1 ;
    }
    return Tuple0{v67, v65, v9, v131, v6, v4};
}
__device__ void method_7(Union0 v0){
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
__device__ void method_8(Union1 v0){
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
extern "C" __global__ void __cluster_dims__(12,1,1) cuda_device_entry0() {
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
        method_2(v29, v6, v38, v39, v26, v40, v41);
        static_array_list<Union0,5> v42; Union1 v43; bool v44; static_array<int,3> v45; int v46; unsigned int v47;
        Tuple0 tmp3 = method_3(v29);
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
            method_7(v81);
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
        method_8(v43);
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
        while (while_method_2(v92)){
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
    } else {
    }
    return ;
}
void run_cuda_device_from_cuda_host_1(){
    auto kernel = cuda_device_entry0;
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
void cuda_host_entry0() {
    run_cuda_device_from_cuda_host_1();
    gpuErrchk(cudaDeviceSynchronize());
    return ;
}
