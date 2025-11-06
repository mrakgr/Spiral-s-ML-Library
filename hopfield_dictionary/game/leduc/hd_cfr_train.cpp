#include "hd_cfr_train.hpp"
inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
unsigned long long FunPointerMethod0(Tuple0 tup0){
    unsigned long long v0 = tup0.v0; static_array_list<Union0,32> v1 = tup0.v1;
    return v0;
}
inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
bool FunPointerMethod1(Tuple0 tup0, Tuple0 tup1){
    unsigned long long v0 = tup0.v0; static_array_list<Union0,32> v1 = tup0.v1; unsigned long long v2 = tup1.v0; static_array_list<Union0,32> v3 = tup1.v1;
    bool v4;
    v4 = v0 == v2;
    if (v4){
        int v5;
        v5 = v1.length;
        int v6;
        v6 = v3.length;
        bool v7;
        v7 = v5 == v6;
        if (v7){
            bool v8;
            v8 = true;
            int v9;
            v9 = v3.length;
            int v10;
            v10 = 0;
            while (while_method_1(v9, v10)){
                Union0 v12;
                v12 = v1[v10];
                Union0 v16;
                v16 = v3[v10];
                bool v61;
                switch (v12.tag == v16.tag ? v12.tag : 255) {
                    case 0: { // CommunityCardIs
                        Union1 v20 = v12.case0.v0;
                        Union1 v21 = v16.case0.v0;
                        switch (v20.tag == v21.tag ? v20.tag : 255) {
                            case 0: { // Jack
                                v61 = true;
                                break;
                            }
                            case 1: { // King
                                v61 = true;
                                break;
                            }
                            case 2: { // Queen
                                v61 = true;
                                break;
                            }
                            default: {
                                v61 = false;
                            }
                        }
                        break;
                    }
                    case 1: { // Hidden
                        v61 = true;
                        break;
                    }
                    case 2: { // PlayerAction
                        int v23 = v12.case2.v0; Union2 v24 = v12.case2.v1;
                        int v25 = v16.case2.v0; Union2 v26 = v16.case2.v1;
                        bool v27;
                        v27 = v23 == v25;
                        if (v27){
                            switch (v24.tag == v26.tag ? v24.tag : 255) {
                                case 0: { // Call
                                    v61 = true;
                                    break;
                                }
                                case 1: { // Fold
                                    v61 = true;
                                    break;
                                }
                                case 2: { // Raise
                                    v61 = true;
                                    break;
                                }
                                default: {
                                    v61 = false;
                                }
                            }
                        } else {
                            v61 = false;
                        }
                        break;
                    }
                    case 3: { // PlayerGotCard
                        int v30 = v12.case3.v0; Union1 v31 = v12.case3.v1;
                        int v32 = v16.case3.v0; Union1 v33 = v16.case3.v1;
                        bool v34;
                        v34 = v30 == v32;
                        if (v34){
                            switch (v31.tag == v33.tag ? v31.tag : 255) {
                                case 0: { // Jack
                                    v61 = true;
                                    break;
                                }
                                case 1: { // King
                                    v61 = true;
                                    break;
                                }
                                case 2: { // Queen
                                    v61 = true;
                                    break;
                                }
                                default: {
                                    v61 = false;
                                }
                            }
                        } else {
                            v61 = false;
                        }
                        break;
                    }
                    case 4: { // Showdown
                        static_array<Union1,2> v37 = v12.case4.v0; int v38 = v12.case4.v1; int v39 = v12.case4.v2;
                        static_array<Union1,2> v40 = v16.case4.v0; int v41 = v16.case4.v1; int v42 = v16.case4.v2;
                        bool v43;
                        v43 = true;
                        int v44;
                        v44 = 0;
                        while (while_method_2(v44)){
                            Union1 v46;
                            v46 = v37[v44];
                            Union1 v50;
                            v50 = v40[v44];
                            bool v54;
                            switch (v46.tag == v50.tag ? v46.tag : 255) {
                                case 0: { // Jack
                                    v54 = true;
                                    break;
                                }
                                case 1: { // King
                                    v54 = true;
                                    break;
                                }
                                case 2: { // Queen
                                    v54 = true;
                                    break;
                                }
                                default: {
                                    v54 = false;
                                }
                            }
                            bool v55;
                            v55 = v54 != true;
                            if (v55){
                                bool v56;
                                v56 = false;
                                v43 = v56;
                                break;
                            } else {
                            }
                            v44 += 1 ;
                        }
                        if (v43){
                            bool v57;
                            v57 = v38 == v41;
                            if (v57){
                                bool v58;
                                v58 = v39 == v42;
                                v61 = v58;
                            } else {
                                v61 = false;
                            }
                        } else {
                            v61 = false;
                        }
                        break;
                    }
                    default: {
                        v61 = false;
                    }
                }
                bool v62;
                v62 = v61 != true;
                if (v62){
                    bool v63;
                    v63 = false;
                    v8 = v63;
                    break;
                } else {
                }
                v10 += 1 ;
            }
            return v8;
        } else {
            return false;
        }
    } else {
        return false;
    }
}
inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 1000000;
    return v1;
}
unsigned int loop_2(unsigned int v0, xso::rng & v1){
    unsigned int v2;
    v2 = v1();
    unsigned int v3;
    v3 = v2 % v0;
    unsigned int v4;
    v4 = v2 - v3;
    unsigned int v5;
    v5 = 0u - v0;
    bool v6;
    v6 = v4 <= v5;
    if (v6){
        return v3;
    } else {
        return loop_2(v0, v1);
    }
}
inline bool while_method_4(unsigned int v0, unsigned int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
unsigned int find_nth_set_bit_3(int v0, unsigned int v1, unsigned int v2){
    int v3;
    v3 = (int)v1;
    unsigned int v4;
    v4 = v2 >> v3;
    StackMut2 v5{0};
    StackMut3 v6{4294967295u};
    unsigned int v7;
    v7 = 32u - v1;
    unsigned int v8;
    v8 = 0u;
    while (while_method_4(v7, v8)){
        int v10;
        v10 = (int)v8;
        unsigned int v11;
        v11 = v4 >> v10;
        unsigned int v12;
        v12 = v11 & 1u;
        bool v13;
        v13 = v12 == 0u;
        bool v14;
        v14 = v13 != true;
        if (v14){
            int v15 = v5.v0;
            int v16;
            v16 = v15 + 1;
            v5.v0 = v16;
            bool v17;
            v17 = v16 == v0;
            if (v17){
                unsigned int v18;
                v18 = v1 + v8;
                v6.v0 = v18;
                break;
            } else {
            }
        } else {
        }
        v8 += 1u ;
    }
    unsigned int v19 = v6.v0;
    return v19;
}
Tuple4 draw_card_1(xso::rng & v0, unsigned int v1){
    int v2;
    v2 = __builtin_popcount(v1);
    unsigned int v5;
    v5 = (unsigned int)v2;
    bool v6;
    v6 = 0u < v5;
    bool v7;
    v7 = v6 == false;
    if (v7){
        assert("The range has to be greater than 0." && v6);
    } else {
    }
    unsigned int v9;
    v9 = loop_2(v5, v0);
    int v10;
    v10 = (int)v9;
    int v11;
    v11 = __builtin_popcount(v1);
    bool v14;
    v14 = v10 < v11;
    unsigned int v20;
    if (v14){
        int v15;
        v15 = v10 + 1;
        unsigned int v16;
        v16 = 0u;
        v20 = find_nth_set_bit_3(v15, v16, v1);
    } else {
        int v18;
        v18 = v10 - v11;
        printf("%s\n", "Cannot find the n-th set bit.");
        exit(-1);
    }
    bool v21;
    v21 = 0u == v20;
    Union1 v39;
    if (v21){
        v39 = Union1{Union1_1{}};
    } else {
        bool v23;
        v23 = 1u == v20;
        if (v23){
            v39 = Union1{Union1_1{}};
        } else {
            bool v25;
            v25 = 2u == v20;
            if (v25){
                v39 = Union1{Union1_2{}};
            } else {
                bool v27;
                v27 = 3u == v20;
                if (v27){
                    v39 = Union1{Union1_2{}};
                } else {
                    bool v29;
                    v29 = 4u == v20;
                    if (v29){
                        v39 = Union1{Union1_0{}};
                    } else {
                        bool v31;
                        v31 = 5u == v20;
                        if (v31){
                            v39 = Union1{Union1_0{}};
                        } else {
                            printf("%s\n", "Invalid int in int_to_card.");
                            exit(-1);
                        }
                    }
                }
            }
        }
    }
    int v40;
    v40 = (int)v20;
    unsigned int v41;
    v41 = 1u << v40;
    unsigned int v42;
    v42 = v1 ^ v41;
    return Tuple4{v39, v42};
}
float loop_4(StackRefs3 & v0, StackRefs1 & v1, StackRefs0 & v2, xso::rng & v3, StackMut0 & v4, StackRefs4 & v5, StackMut1 & v6, Union5 v7){
    switch (v7.tag) {
        case 0: { // T_game_chance_community_card
            Union4 v9 = v7.case0.v0; bool v10 = v7.case0.v1; static_array<Union1,2> v11 = v7.case0.v2; int v12 = v7.case0.v3; static_array<int,2> v13 = v7.case0.v4; int v14 = v7.case0.v5; Union1 v15 = v7.case0.v6;
            int v16;
            v16 = 2;
            int v17; int v18;
            Tuple5 tmp2 = Tuple5{0, 0};
            v17 = tmp2.v0; v18 = tmp2.v1;
            while (while_method_2(v17)){
                int v20;
                v20 = v13[v17];
                bool v24;
                v24 = v18 >= v20;
                int v25;
                if (v24){
                    v25 = v18;
                } else {
                    v25 = v20;
                }
                v18 = v25;
                v17 += 1 ;
            }
            static_array<int,2> v26;
            int v30;
            v30 = 0;
            while (while_method_2(v30)){
                v26[v30] = v18;
                v30 += 1 ;
            }
            Union4 v32;
            v32 = Union4{Union4_1{v15}};
            bool v33;
            v33 = true;
            int v34;
            v34 = 0;
            Union3 v35;
            v35 = Union3{Union3_2{v32, v33, v11, v34, v26, v16}};
            return body_0(v0, v1, v2, v3, v4, v5, v35);
            break;
        }
        case 1: { // T_game_chance_init
            Union1 v37 = v7.case1.v0; Union1 v38 = v7.case1.v1;
            int v39;
            v39 = 2;
            static_array<int,2> v40;
            v40[0] = 1;
            v40[1] = 1;
            static_array<Union1,2> v44;
            v44[0] = v37;
            v44[1] = v38;
            Union4 v48;
            v48 = Union4{Union4_0{}};
            bool v49;
            v49 = true;
            int v50;
            v50 = 0;
            Union3 v51;
            v51 = Union3{Union3_2{v48, v49, v44, v50, v40, v39}};
            return body_0(v0, v1, v2, v3, v4, v5, v51);
            break;
        }
        case 2: { // T_game_round
            Union4 v53 = v7.case2.v0; bool v54 = v7.case2.v1; static_array<Union1,2> v55 = v7.case2.v2; int v56 = v7.case2.v3; static_array<int,2> v57 = v7.case2.v4; int v58 = v7.case2.v5; Union2 v59 = v7.case2.v6;
            Union3 v161;
            switch (v53.tag) {
                case 0: { // None
                    switch (v59.tag) {
                        case 0: { // Call
                            if (v54){
                                int v119;
                                v119 = v56 ^ 1;
                                v161 = Union3{Union3_2{v53, false, v55, v119, v57, v58}};
                            } else {
                                v161 = Union3{Union3_0{v53, v54, v55, v56, v57, v58}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v161 = Union3{Union3_5{v53, v54, v55, v56, v57, v58}};
                            break;
                        }
                        case 2: { // Raise
                            bool v123;
                            v123 = v58 > 0;
                            if (v123){
                                int v124;
                                v124 = v56 ^ 1;
                                int v125;
                                v125 = -1 + v58;
                                int v126; int v127;
                                Tuple5 tmp3 = Tuple5{0, 0};
                                v126 = tmp3.v0; v127 = tmp3.v1;
                                while (while_method_2(v126)){
                                    int v129;
                                    v129 = v57[v126];
                                    bool v133;
                                    v133 = v127 >= v129;
                                    int v134;
                                    if (v133){
                                        v134 = v127;
                                    } else {
                                        v134 = v129;
                                    }
                                    v127 = v134;
                                    v126 += 1 ;
                                }
                                static_array<int,2> v135;
                                int v139;
                                v139 = 0;
                                while (while_method_2(v139)){
                                    v135[v139] = v127;
                                    v139 += 1 ;
                                }
                                static_array<int,2> v141;
                                int v145;
                                v145 = 0;
                                while (while_method_2(v145)){
                                    int v147;
                                    v147 = v135[v145];
                                    bool v151;
                                    v151 = v145 == v56;
                                    int v153;
                                    if (v151){
                                        int v152;
                                        v152 = v147 + 2;
                                        v153 = v152;
                                    } else {
                                        v153 = v147;
                                    }
                                    v141[v145] = v153;
                                    v145 += 1 ;
                                }
                                v161 = Union3{Union3_2{v53, false, v55, v124, v141, v125}};
                            } else {
                                printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                exit(-1);
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    break;
                }
                case 1: { // Some
                    Union1 v60 = v53.case1.v0;
                    switch (v59.tag) {
                        case 0: { // Call
                            if (v54){
                                int v62;
                                v62 = v56 ^ 1;
                                v161 = Union3{Union3_2{v53, false, v55, v62, v57, v58}};
                            } else {
                                int v64; int v65;
                                Tuple5 tmp4 = Tuple5{0, 0};
                                v64 = tmp4.v0; v65 = tmp4.v1;
                                while (while_method_2(v64)){
                                    int v67;
                                    v67 = v57[v64];
                                    bool v71;
                                    v71 = v65 >= v67;
                                    int v72;
                                    if (v71){
                                        v72 = v65;
                                    } else {
                                        v72 = v67;
                                    }
                                    v65 = v72;
                                    v64 += 1 ;
                                }
                                static_array<int,2> v73;
                                int v77;
                                v77 = 0;
                                while (while_method_2(v77)){
                                    v73[v77] = v65;
                                    v77 += 1 ;
                                }
                                v161 = Union3{Union3_4{v53, v54, v55, v56, v73, v58}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v161 = Union3{Union3_5{v53, v54, v55, v56, v57, v58}};
                            break;
                        }
                        case 2: { // Raise
                            bool v81;
                            v81 = v58 > 0;
                            if (v81){
                                int v82;
                                v82 = v56 ^ 1;
                                int v83;
                                v83 = -1 + v58;
                                int v84; int v85;
                                Tuple5 tmp5 = Tuple5{0, 0};
                                v84 = tmp5.v0; v85 = tmp5.v1;
                                while (while_method_2(v84)){
                                    int v87;
                                    v87 = v57[v84];
                                    bool v91;
                                    v91 = v85 >= v87;
                                    int v92;
                                    if (v91){
                                        v92 = v85;
                                    } else {
                                        v92 = v87;
                                    }
                                    v85 = v92;
                                    v84 += 1 ;
                                }
                                static_array<int,2> v93;
                                int v97;
                                v97 = 0;
                                while (while_method_2(v97)){
                                    v93[v97] = v85;
                                    v97 += 1 ;
                                }
                                static_array<int,2> v99;
                                int v103;
                                v103 = 0;
                                while (while_method_2(v103)){
                                    int v105;
                                    v105 = v93[v103];
                                    bool v109;
                                    v109 = v103 == v56;
                                    int v111;
                                    if (v109){
                                        int v110;
                                        v110 = v105 + 4;
                                        v111 = v110;
                                    } else {
                                        v111 = v105;
                                    }
                                    v99[v103] = v111;
                                    v103 += 1 ;
                                }
                                v161 = Union3{Union3_2{v53, false, v55, v82, v99, v83}};
                            } else {
                                printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                exit(-1);
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            return body_0(v0, v1, v2, v3, v4, v5, v161);
            break;
        }
        case 3: { // T_none
            float v8 = v6.v0;
            return v8;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
static_array<float,3> relu_7(static_array<float,3> v0){
    static_array<float,3> v1;
    int v5;
    v5 = 0;
    while (while_method_5(v5)){
        float v7;
        v7 = v0[v5];
        bool v11;
        v11 = 0.0f >= v7;
        float v12;
        if (v11){
            v12 = 0.0f;
        } else {
            v12 = v7;
        }
        v1[v5] = v12;
        v5 += 1 ;
    }
    return v1;
}
static_array<float,3> masking_normalize_8(static_array<float,3> v0, static_array<bool,3> v1){
    int v2; float v3;
    Tuple3 tmp14 = Tuple3{0, 0.0f};
    v2 = tmp14.v0; v3 = tmp14.v1;
    while (while_method_5(v2)){
        bool v5;
        v5 = v1[v2];
        float v9;
        if (v5){
            v9 = 1.0f;
        } else {
            v9 = 0.0f;
        }
        float v10;
        v10 = v3 + v9;
        v3 = v10;
        v2 += 1 ;
    }
    float v11;
    v11 = 1.0f / v3;
    static_array<float,3> v12;
    int v16;
    v16 = 0;
    while (while_method_5(v16)){
        float v18;
        v18 = v0[v16];
        bool v22;
        v22 = v1[v16];
        float v26;
        if (v22){
            v26 = v18;
        } else {
            v26 = 0.0f;
        }
        v12[v16] = v26;
        v16 += 1 ;
    }
    int v27; float v28;
    Tuple3 tmp15 = Tuple3{0, 0.0f};
    v27 = tmp15.v0; v28 = tmp15.v1;
    while (while_method_5(v27)){
        float v30;
        v30 = v12[v27];
        float v34;
        v34 = v28 + v30;
        v28 = v34;
        v27 += 1 ;
    }
    static_array<float,3> v35;
    int v39;
    v39 = 0;
    while (while_method_5(v39)){
        float v41;
        v41 = v12[v39];
        bool v45;
        v45 = v1[v39];
        bool v49;
        v49 = v45 == false;
        float v54;
        if (v49){
            v54 = 0.0f;
        } else {
            bool v50;
            v50 = v28 == 0.0f;
            bool v51;
            v51 = v50 != true;
            if (v51){
                float v52;
                v52 = v41 / v28;
                v54 = v52;
            } else {
                v54 = v11;
            }
        }
        v35[v39] = v54;
        v39 += 1 ;
    }
    return v35;
}
static_array<float,3> regret_match_6(static_array<float,3> v0, static_array<bool,3> v1){
    static_array<float,3> v2;
    v2 = relu_7(v0);
    return masking_normalize_8(v2, v1);
}
inline bool while_method_6(static_array<float,3> v0, int v1){
    bool v2;
    v2 = v1 < 3;
    return v2;
}
inline bool while_method_7(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    return v2;
}
int loop_11(static_array<float,3> v0, float v1, int v2){
    bool v3;
    v3 = v2 < 3;
    if (v3){
        float v4;
        v4 = v0[v2];
        bool v8;
        v8 = v1 < v4;
        if (v8){
            return v2;
        } else {
            int v9;
            v9 = v2 + 1;
            return loop_11(v0, v1, v9);
        }
    } else {
        return 2;
    }
}
int pick_discrete__10(static_array<float,3> v0, float v1){
    static_array<float,3> v2;
    int v6;
    v6 = 0;
    while (while_method_5(v6)){
        float v8;
        v8 = v0[v6];
        v2[v6] = v8;
        v6 += 1 ;
    }
    int v12;
    v12 = 1;
    while (while_method_6(v2, v12)){
        int v14;
        v14 = 3;
        while (while_method_7(v12, v14)){
            v14 -= 1 ;
            int v16;
            v16 = v14 - v12;
            float v17;
            v17 = v2[v16];
            float v21;
            v21 = v2[v14];
            float v25;
            v25 = v17 + v21;
            v2[v14] = v25;
        }
        int v26;
        v26 = v12 * 2;
        v12 = v26;
    }
    float v27;
    v27 = v2[2];
    float v31;
    v31 = v1 * v27;
    int v32;
    v32 = 0;
    return loop_11(v2, v31, v32);
}
int sample_discrete__9(static_array<float,3> v0, xso::rng & v1){
    std::uniform_real_distribution<float> v2(0.0, 1.0);
    float v3;
    v3 = v2(v1);
    return pick_discrete__10(v0, v3);
}
float method_5(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs1 & v9, StackRefs0 & v10, StackMut0 & v11, StackMut1 & v12){
    std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> & v13 = v10.v0;
    static_array_list<Union0,32> & v14 = v1.v0;
    static_array_list<Union0,32> & v15 = v1.v0;
    int v16;
    v16 = v15.length;
    bool v17;
    v17 = 32 >= v16;
    bool v18;
    v18 = v17 == false;
    if (v18){
        assert("The type level dimension has to equal the value passed at runtime into create." && v17);
    } else {
    }
    static_array_list<Union0,32> v20;
    v20 = static_array_list<Union0,32>{};
    v20.unsafe_set_length(v16);
    int v24; int v25;
    Tuple5 tmp8 = Tuple5{0, 0};
    v24 = tmp8.v0; v25 = tmp8.v1;
    while (while_method_1(v16, v24)){
        Union0 v27;
        v27 = v15[v24];
        bool v34;
        switch (v27.tag) {
            case 3: { // PlayerGotCard
                int v31 = v27.case3.v0; Union1 v32 = v27.case3.v1;
                bool v33;
                v33 = v31 == v6;
                v34 = v33;
                break;
            }
            default: {
                v34 = true;
            }
        }
        int v36;
        if (v34){
            v20[v25] = v27;
            int v35;
            v35 = v25 + 1;
            v36 = v35;
        } else {
            v36 = v25;
        }
        v25 = v36;
        v24 += 1 ;
    }
    bool v37;
    v37 = 32 >= v25;
    bool v38;
    v38 = v37 == false;
    if (v38){
        assert("The type level dimension has to equal the value passed at runtime into create." && v37);
    } else {
    }
    static_array_list<Union0,32> v40;
    v40 = static_array_list<Union0,32>{};
    v40.unsafe_set_length(v25);
    int v44;
    v44 = 0;
    while (while_method_1(v25, v44)){
        Union0 v46;
        v46 = v20[v44];
        v40[v44] = v46;
        v44 += 1 ;
    }
    int v50;
    v50 = v40.length;
    int v51; unsigned long long v52; unsigned long long v53;
    Tuple6 tmp9 = Tuple6{0, 0ull, 1ull};
    v51 = tmp9.v0; v52 = tmp9.v1; v53 = tmp9.v2;
    while (while_method_1(v50, v51)){
        Union0 v55;
        v55 = v40[v51];
        unsigned long long v103;
        switch (v55.tag) {
            case 0: { // CommunityCardIs
                Union1 v59 = v55.case0.v0;
                unsigned long long v60;
                switch (v59.tag) {
                    case 0: { // Jack
                        v60 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v60 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v60 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v61;
                v61 = 9223372036854775807ull + v60;
                unsigned long long v62;
                v62 = v61 * 9973ull;
                v103 = v62;
                break;
            }
            case 1: { // Hidden
                v103 = 18446744073709531670ull;
                break;
            }
            case 2: { // PlayerAction
                int v63 = v55.case2.v0; Union2 v64 = v55.case2.v1;
                unsigned long long v65;
                v65 = std::hash<int>()(v63);
                unsigned long long v66;
                v66 = v65 * 9973ull;
                unsigned long long v67;
                switch (v64.tag) {
                    case 0: { // Call
                        v67 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v67 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v67 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v68;
                v68 = v66 + v67;
                unsigned long long v69;
                v69 = 9223372036854775807ull + v68;
                unsigned long long v70;
                v70 = v69 * 9973ull;
                unsigned long long v71;
                v71 = v70 * 3ull;
                v103 = v71;
                break;
            }
            case 3: { // PlayerGotCard
                int v72 = v55.case3.v0; Union1 v73 = v55.case3.v1;
                unsigned long long v74;
                v74 = std::hash<int>()(v72);
                unsigned long long v75;
                v75 = v74 * 9973ull;
                unsigned long long v76;
                switch (v73.tag) {
                    case 0: { // Jack
                        v76 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v76 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v76 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v77;
                v77 = v75 + v76;
                unsigned long long v78;
                v78 = 9223372036854775807ull + v77;
                unsigned long long v79;
                v79 = v78 * 9973ull;
                unsigned long long v80;
                v80 = v79 * 4ull;
                v103 = v80;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v81 = v55.case4.v0; int v82 = v55.case4.v1; int v83 = v55.case4.v2;
                unsigned long long v84;
                v84 = std::hash<int>()(v83);
                unsigned long long v85;
                v85 = std::hash<int>()(v82);
                unsigned long long v86;
                v86 = v85 * 9973ull;
                unsigned long long v87;
                v87 = v84 + v86;
                int v88; unsigned long long v89; unsigned long long v90;
                Tuple6 tmp10 = Tuple6{0, 0ull, 1ull};
                v88 = tmp10.v0; v89 = tmp10.v1; v90 = tmp10.v2;
                while (while_method_2(v88)){
                    Union1 v92;
                    v92 = v81[v88];
                    unsigned long long v96;
                    switch (v92.tag) {
                        case 0: { // Jack
                            v96 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v96 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v96 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v97;
                    v97 = v96 * v90;
                    unsigned long long v98;
                    v98 = v89 + v97;
                    unsigned long long v99;
                    v99 = v90 * 9973ull;
                    v89 = v98;
                    v90 = v99;
                    v88 += 1 ;
                }
                unsigned long long v100;
                v100 = 9223372036854775807ull + v87;
                unsigned long long v101;
                v101 = v100 * 9973ull;
                unsigned long long v102;
                v102 = v101 * 5ull;
                v103 = v102;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v104;
        v104 = v103 * v53;
        unsigned long long v105;
        v105 = v52 + v104;
        unsigned long long v106;
        v106 = v53 * 9973ull;
        v52 = v105;
        v53 = v106;
        v51 += 1 ;
    }
    auto v107 = v13.find(Tuple0{0ull, v40});
    bool v108;
    v108 = v107 != v13.end();
    Union6 v113;
    if (v108){
        static_array<float,3> v109; static_array<float,3> v110;
        Tuple1 tmp11 = v107->second;
        v109 = tmp11.v0; v110 = tmp11.v1;
        v113 = Union6{Union6_1{v109, v110}};
    } else {
        v113 = Union6{Union6_0{}};
    }
    static_array<float,3> v130; static_array<float,3> v131;
    switch (v113.tag) {
        case 0: { // None
            static_array<float,3> v116;
            int v120;
            v120 = 0;
            while (while_method_5(v120)){
                v116[v120] = 0.0f;
                v120 += 1 ;
            }
            static_array<float,3> v122;
            int v126;
            v126 = 0;
            while (while_method_5(v126)){
                v122[v126] = 0.0f;
                v126 += 1 ;
            }
            v130 = v116; v131 = v122;
            break;
        }
        case 1: { // Some
            static_array<float,3> v114 = v113.case1.v0; static_array<float,3> v115 = v113.case1.v1;
            v130 = v114; v131 = v115;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    int v132;
    v132 = v7[0];
    int v136;
    v136 = v7[1];
    bool v140;
    v140 = v132 == v136;
    bool v141;
    v141 = v140 != true;
    Union7 v145;
    if (v141){
        Union2 v142;
        v142 = Union2{Union2_1{}};
        v145 = Union7{Union7_1{v142}};
    } else {
        v145 = Union7{Union7_0{}};
    }
    bool v146;
    v146 = v8 > 0;
    Union7 v150;
    if (v146){
        Union2 v147;
        v147 = Union2{Union2_2{}};
        v150 = Union7{Union7_1{v147}};
    } else {
        v150 = Union7{Union7_0{}};
    }
    bool v153;
    switch (v150.tag) {
        case 0: { // None
            v153 = false;
            break;
        }
        case 1: { // Some
            Union2 v151 = v150.case1.v0;
            v153 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    bool v156;
    switch (v145.tag) {
        case 0: { // None
            v156 = false;
            break;
        }
        case 1: { // Some
            Union2 v154 = v145.case1.v0;
            v156 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    static_array<bool,3> v157;
    v157[0] = true;
    v157[1] = v156;
    v157[2] = v153;
    std::unordered_map<Tuple0, static_array<Tuple2,3>, Fun0, Fun1> & v161 = v9.v0;
    int v162;
    v162 = v14.length;
    int v163; unsigned long long v164; unsigned long long v165;
    Tuple6 tmp12 = Tuple6{0, 0ull, 1ull};
    v163 = tmp12.v0; v164 = tmp12.v1; v165 = tmp12.v2;
    while (while_method_1(v162, v163)){
        Union0 v167;
        v167 = v14[v163];
        unsigned long long v215;
        switch (v167.tag) {
            case 0: { // CommunityCardIs
                Union1 v171 = v167.case0.v0;
                unsigned long long v172;
                switch (v171.tag) {
                    case 0: { // Jack
                        v172 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v172 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v172 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v173;
                v173 = 9223372036854775807ull + v172;
                unsigned long long v174;
                v174 = v173 * 9973ull;
                v215 = v174;
                break;
            }
            case 1: { // Hidden
                v215 = 18446744073709531670ull;
                break;
            }
            case 2: { // PlayerAction
                int v175 = v167.case2.v0; Union2 v176 = v167.case2.v1;
                unsigned long long v177;
                v177 = std::hash<int>()(v175);
                unsigned long long v178;
                v178 = v177 * 9973ull;
                unsigned long long v179;
                switch (v176.tag) {
                    case 0: { // Call
                        v179 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v179 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v179 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v180;
                v180 = v178 + v179;
                unsigned long long v181;
                v181 = 9223372036854775807ull + v180;
                unsigned long long v182;
                v182 = v181 * 9973ull;
                unsigned long long v183;
                v183 = v182 * 3ull;
                v215 = v183;
                break;
            }
            case 3: { // PlayerGotCard
                int v184 = v167.case3.v0; Union1 v185 = v167.case3.v1;
                unsigned long long v186;
                v186 = std::hash<int>()(v184);
                unsigned long long v187;
                v187 = v186 * 9973ull;
                unsigned long long v188;
                switch (v185.tag) {
                    case 0: { // Jack
                        v188 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v188 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v188 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v189;
                v189 = v187 + v188;
                unsigned long long v190;
                v190 = 9223372036854775807ull + v189;
                unsigned long long v191;
                v191 = v190 * 9973ull;
                unsigned long long v192;
                v192 = v191 * 4ull;
                v215 = v192;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v193 = v167.case4.v0; int v194 = v167.case4.v1; int v195 = v167.case4.v2;
                unsigned long long v196;
                v196 = std::hash<int>()(v195);
                unsigned long long v197;
                v197 = std::hash<int>()(v194);
                unsigned long long v198;
                v198 = v197 * 9973ull;
                unsigned long long v199;
                v199 = v196 + v198;
                int v200; unsigned long long v201; unsigned long long v202;
                Tuple6 tmp13 = Tuple6{0, 0ull, 1ull};
                v200 = tmp13.v0; v201 = tmp13.v1; v202 = tmp13.v2;
                while (while_method_2(v200)){
                    Union1 v204;
                    v204 = v193[v200];
                    unsigned long long v208;
                    switch (v204.tag) {
                        case 0: { // Jack
                            v208 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v208 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v208 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v209;
                    v209 = v208 * v202;
                    unsigned long long v210;
                    v210 = v201 + v209;
                    unsigned long long v211;
                    v211 = v202 * 9973ull;
                    v201 = v210;
                    v202 = v211;
                    v200 += 1 ;
                }
                unsigned long long v212;
                v212 = 9223372036854775807ull + v199;
                unsigned long long v213;
                v213 = v212 * 9973ull;
                unsigned long long v214;
                v214 = v213 * 5ull;
                v215 = v214;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v216;
        v216 = v215 * v165;
        unsigned long long v217;
        v217 = v164 + v216;
        unsigned long long v218;
        v218 = v165 * 9973ull;
        v164 = v217;
        v165 = v218;
        v163 += 1 ;
    }
    auto v219 = v161.find(Tuple0{0ull, v14});
    bool v220;
    v220 = v219 != v161.end();
    Union8 v224;
    if (v220){
        static_array<Tuple2,3> v221;
        v221 = v219->second;
        v224 = Union8{Union8_1{v221}};
    } else {
        v224 = Union8{Union8_0{}};
    }
    static_array<Tuple2,3> v233;
    switch (v224.tag) {
        case 0: { // None
            static_array<Tuple2,3> v226;
            int v230;
            v230 = 0;
            while (while_method_5(v230)){
                v226[v230] = Tuple2{0.0f, 0.0f};
                v230 += 1 ;
            }
            v233 = v226;
            break;
        }
        case 1: { // Some
            static_array<Tuple2,3> v225 = v224.case1.v0;
            v233 = v225;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    static_array<float,3> v234;
    v234 = regret_match_6(v131, v157);
    static_array<float,3> v235;
    v235 = masking_normalize_8(v130, v157);
    static_array<float,3> v236;
    int v240;
    v240 = 0;
    while (while_method_5(v240)){
        v236[v240] = 0.0f;
        v240 += 1 ;
    }
    static_array<float,3> v242;
    v242 = masking_normalize_8(v236, v157);
    static_array<float,3> v243;
    int v247;
    v247 = 0;
    while (while_method_5(v247)){
        float v249;
        v249 = v235[v247];
        float v253;
        v253 = v242[v247];
        float v257;
        v257 = 0.875f * v249;
        float v258;
        v258 = 0.125f * v253;
        float v259;
        v259 = v257 + v258;
        v243[v247] = v259;
        v247 += 1 ;
    }
    int v260;
    v260 = sample_discrete__9(v243, v0);
    StackMut1 v261{0.0f};
    float v262; float v263;
    Tuple2 tmp16 = v233[2];
    v262 = tmp16.v0; v263 = tmp16.v1;
    bool v270;
    v270 = v263 == 0.0f;
    bool v271;
    v271 = v270 != true;
    float v273;
    if (v271){
        float v272;
        v272 = v262 / v263;
        v273 = v272;
    } else {
        v273 = 0.0f;
    }
    float v313;
    switch (v150.tag) {
        case 1: { // Some
            Union2 v274 = v150.case1.v0;
            bool v275;
            v275 = 2 == v260;
            if (v275){
                float v276;
                v276 = v234[2];
                float v280;
                v280 = v243[2];
                static_array<Tuple2,2> & v284 = v2.v0;
                float v285; float v286;
                Tuple2 tmp17 = v284[v6];
                v285 = tmp17.v0; v286 = tmp17.v1;
                static_array<Tuple2,2> & v293 = v2.v0;
                float v294;
                v294 = log(v276);
                float v295;
                v295 = v294 + v285;
                float v296;
                v296 = log(v280);
                float v297;
                v297 = v296 + v286;
                v293[v6] = Tuple2{v295, v297};
                static_array_list<Union0,32> & v298 = v1.v0;
                Union0 v299;
                v299 = Union0{Union0_2{v6, v274}};
                v298.push(v299);
                Union5 v300;
                v300 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v274}};
                float v301;
                v301 = loop_4(v1, v9, v10, v0, v11, v2, v12, v300);
                static_array_list<Union0,32> & v302 = v1.v0;
                Union0 v303;
                v303 = v302.pop();
                static_array<Tuple2,2> & v304 = v2.v0;
                v304[v6] = Tuple2{v285, v286};
                bool v305;
                v305 = v6 == 0;
                float v307;
                if (v305){
                    v307 = v301;
                } else {
                    float v306;
                    v306 = -v301;
                    v307 = v306;
                }
                v261.v0 = v307;
                float v308 = v261.v0;
                float v309;
                v309 = v308 - v273;
                float v310;
                v310 = v309 / v280;
                float v311;
                v311 = v310 + v273;
                v313 = v311;
            } else {
                v313 = v273;
            }
            break;
        }
        default: {
            v313 = v273;
        }
    }
    float v314; float v315;
    Tuple2 tmp18 = v233[1];
    v314 = tmp18.v0; v315 = tmp18.v1;
    bool v322;
    v322 = v315 == 0.0f;
    bool v323;
    v323 = v322 != true;
    float v325;
    if (v323){
        float v324;
        v324 = v314 / v315;
        v325 = v324;
    } else {
        v325 = 0.0f;
    }
    float v365;
    switch (v145.tag) {
        case 1: { // Some
            Union2 v326 = v145.case1.v0;
            bool v327;
            v327 = 1 == v260;
            if (v327){
                float v328;
                v328 = v234[1];
                float v332;
                v332 = v243[1];
                static_array<Tuple2,2> & v336 = v2.v0;
                float v337; float v338;
                Tuple2 tmp19 = v336[v6];
                v337 = tmp19.v0; v338 = tmp19.v1;
                static_array<Tuple2,2> & v345 = v2.v0;
                float v346;
                v346 = log(v328);
                float v347;
                v347 = v346 + v337;
                float v348;
                v348 = log(v332);
                float v349;
                v349 = v348 + v338;
                v345[v6] = Tuple2{v347, v349};
                static_array_list<Union0,32> & v350 = v1.v0;
                Union0 v351;
                v351 = Union0{Union0_2{v6, v326}};
                v350.push(v351);
                Union5 v352;
                v352 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v326}};
                float v353;
                v353 = loop_4(v1, v9, v10, v0, v11, v2, v12, v352);
                static_array_list<Union0,32> & v354 = v1.v0;
                Union0 v355;
                v355 = v354.pop();
                static_array<Tuple2,2> & v356 = v2.v0;
                v356[v6] = Tuple2{v337, v338};
                bool v357;
                v357 = v6 == 0;
                float v359;
                if (v357){
                    v359 = v353;
                } else {
                    float v358;
                    v358 = -v353;
                    v359 = v358;
                }
                v261.v0 = v359;
                float v360 = v261.v0;
                float v361;
                v361 = v360 - v325;
                float v362;
                v362 = v361 / v332;
                float v363;
                v363 = v362 + v325;
                v365 = v363;
            } else {
                v365 = v325;
            }
            break;
        }
        default: {
            v365 = v325;
        }
    }
    float v366; float v367;
    Tuple2 tmp20 = v233[0];
    v366 = tmp20.v0; v367 = tmp20.v1;
    bool v374;
    v374 = v367 == 0.0f;
    bool v375;
    v375 = v374 != true;
    float v377;
    if (v375){
        float v376;
        v376 = v366 / v367;
        v377 = v376;
    } else {
        v377 = 0.0f;
    }
    bool v378;
    v378 = 0 == v260;
    float v417;
    if (v378){
        float v379;
        v379 = v234[0];
        float v383;
        v383 = v243[0];
        static_array<Tuple2,2> & v387 = v2.v0;
        float v388; float v389;
        Tuple2 tmp21 = v387[v6];
        v388 = tmp21.v0; v389 = tmp21.v1;
        static_array<Tuple2,2> & v396 = v2.v0;
        float v397;
        v397 = log(v379);
        float v398;
        v398 = v397 + v388;
        float v399;
        v399 = log(v383);
        float v400;
        v400 = v399 + v389;
        v396[v6] = Tuple2{v398, v400};
        static_array_list<Union0,32> & v401 = v1.v0;
        Union2 v402;
        v402 = Union2{Union2_0{}};
        Union0 v403;
        v403 = Union0{Union0_2{v6, v402}};
        v401.push(v403);
        Union2 v404;
        v404 = Union2{Union2_0{}};
        Union5 v405;
        v405 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v404}};
        float v406;
        v406 = loop_4(v1, v9, v10, v0, v11, v2, v12, v405);
        static_array_list<Union0,32> & v407 = v1.v0;
        Union0 v408;
        v408 = v407.pop();
        static_array<Tuple2,2> & v409 = v2.v0;
        v409[v6] = Tuple2{v388, v389};
        bool v410;
        v410 = v6 == 0;
        float v412;
        if (v410){
            v412 = v406;
        } else {
            float v411;
            v411 = -v406;
            v412 = v411;
        }
        v261.v0 = v412;
        float v413 = v261.v0;
        float v414;
        v414 = v413 - v377;
        float v415;
        v415 = v414 / v383;
        float v416;
        v416 = v415 + v377;
        v417 = v416;
    } else {
        v417 = v377;
    }
    static_array<float,3> v418;
    v418[0] = v417;
    v418[1] = v365;
    v418[2] = v313;
    int v422; float v423;
    Tuple3 tmp22 = Tuple3{0, 0.0f};
    v422 = tmp22.v0; v423 = tmp22.v1;
    while (while_method_5(v422)){
        float v425;
        v425 = v418[v422];
        float v429;
        v429 = v234[v422];
        float v433;
        v433 = v425 * v429;
        float v434;
        v434 = v423 + v433;
        v423 = v434;
        v422 += 1 ;
    }
    static_array<Tuple2,2> & v435 = v2.v0;
    int v436; float v437;
    Tuple3 tmp23 = Tuple3{0, 0.0f};
    v436 = tmp23.v0; v437 = tmp23.v1;
    while (while_method_2(v436)){
        float v439; float v440;
        Tuple2 tmp24 = v435[v436];
        v439 = tmp24.v0; v440 = tmp24.v1;
        bool v447;
        v447 = v436 == v6;
        float v448;
        if (v447){
            v448 = 0.0f;
        } else {
            v448 = v439;
        }
        float v449;
        v449 = v437 + v448;
        float v450;
        v450 = v449 - v440;
        v437 = v450;
        v436 += 1 ;
    }
    float v451;
    v451 = exp(v437);
    static_array<float,3> v452;
    int v456;
    v456 = 0;
    while (while_method_5(v456)){
        float v458;
        v458 = v131[v456];
        float v462;
        v462 = v418[v456];
        float v466;
        v466 = v462 - v423;
        float v467;
        v467 = v451 * v466;
        float v468;
        v468 = v458 + v467;
        bool v469;
        v469 = 0.0f >= v468;
        float v470;
        if (v469){
            v470 = 0.0f;
        } else {
            v470 = v468;
        }
        v452[v456] = v470;
        v456 += 1 ;
    }
    static_array<float,3> v471;
    v471 = regret_match_6(v452, v157);
    static_array<float,3> v472;
    int v476;
    v476 = 0;
    while (while_method_5(v476)){
        float v478;
        v478 = v130[v476];
        float v482;
        v482 = v471[v476];
        float v486;
        v486 = v478 + v482;
        v472[v476] = v486;
        v476 += 1 ;
    }
    int v487;
    v487 = v40.length;
    int v488; unsigned long long v489; unsigned long long v490;
    Tuple6 tmp25 = Tuple6{0, 0ull, 1ull};
    v488 = tmp25.v0; v489 = tmp25.v1; v490 = tmp25.v2;
    while (while_method_1(v487, v488)){
        Union0 v492;
        v492 = v40[v488];
        unsigned long long v540;
        switch (v492.tag) {
            case 0: { // CommunityCardIs
                Union1 v496 = v492.case0.v0;
                unsigned long long v497;
                switch (v496.tag) {
                    case 0: { // Jack
                        v497 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v497 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v497 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v498;
                v498 = 9223372036854775807ull + v497;
                unsigned long long v499;
                v499 = v498 * 9973ull;
                v540 = v499;
                break;
            }
            case 1: { // Hidden
                v540 = 18446744073709531670ull;
                break;
            }
            case 2: { // PlayerAction
                int v500 = v492.case2.v0; Union2 v501 = v492.case2.v1;
                unsigned long long v502;
                v502 = std::hash<int>()(v500);
                unsigned long long v503;
                v503 = v502 * 9973ull;
                unsigned long long v504;
                switch (v501.tag) {
                    case 0: { // Call
                        v504 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v504 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v504 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v505;
                v505 = v503 + v504;
                unsigned long long v506;
                v506 = 9223372036854775807ull + v505;
                unsigned long long v507;
                v507 = v506 * 9973ull;
                unsigned long long v508;
                v508 = v507 * 3ull;
                v540 = v508;
                break;
            }
            case 3: { // PlayerGotCard
                int v509 = v492.case3.v0; Union1 v510 = v492.case3.v1;
                unsigned long long v511;
                v511 = std::hash<int>()(v509);
                unsigned long long v512;
                v512 = v511 * 9973ull;
                unsigned long long v513;
                switch (v510.tag) {
                    case 0: { // Jack
                        v513 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v513 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v513 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v514;
                v514 = v512 + v513;
                unsigned long long v515;
                v515 = 9223372036854775807ull + v514;
                unsigned long long v516;
                v516 = v515 * 9973ull;
                unsigned long long v517;
                v517 = v516 * 4ull;
                v540 = v517;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v518 = v492.case4.v0; int v519 = v492.case4.v1; int v520 = v492.case4.v2;
                unsigned long long v521;
                v521 = std::hash<int>()(v520);
                unsigned long long v522;
                v522 = std::hash<int>()(v519);
                unsigned long long v523;
                v523 = v522 * 9973ull;
                unsigned long long v524;
                v524 = v521 + v523;
                int v525; unsigned long long v526; unsigned long long v527;
                Tuple6 tmp26 = Tuple6{0, 0ull, 1ull};
                v525 = tmp26.v0; v526 = tmp26.v1; v527 = tmp26.v2;
                while (while_method_2(v525)){
                    Union1 v529;
                    v529 = v518[v525];
                    unsigned long long v533;
                    switch (v529.tag) {
                        case 0: { // Jack
                            v533 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v533 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v533 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v534;
                    v534 = v533 * v527;
                    unsigned long long v535;
                    v535 = v526 + v534;
                    unsigned long long v536;
                    v536 = v527 * 9973ull;
                    v526 = v535;
                    v527 = v536;
                    v525 += 1 ;
                }
                unsigned long long v537;
                v537 = 9223372036854775807ull + v524;
                unsigned long long v538;
                v538 = v537 * 9973ull;
                unsigned long long v539;
                v539 = v538 * 5ull;
                v540 = v539;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v541;
        v541 = v540 * v490;
        unsigned long long v542;
        v542 = v489 + v541;
        unsigned long long v543;
        v543 = v490 * 9973ull;
        v489 = v542;
        v490 = v543;
        v488 += 1 ;
    }
    v13[Tuple0{0ull, v40}] = Tuple1{v472, v452};
    static_array<Tuple2,3> v544;
    int v548;
    v548 = 0;
    while (while_method_5(v548)){
        float v550; float v551;
        Tuple2 tmp27 = v233[v548];
        v550 = tmp27.v0; v551 = tmp27.v1;
        bool v558;
        v558 = v260 == v548;
        float v562; float v563;
        if (v558){
            float v559 = v261.v0;
            float v560;
            v560 = v550 + v559;
            float v561;
            v561 = v551 + 1.0f;
            v562 = v560; v563 = v561;
        } else {
            v562 = v550; v563 = v551;
        }
        v544[v548] = Tuple2{v562, v563};
        v548 += 1 ;
    }
    int v564;
    v564 = v14.length;
    int v565; unsigned long long v566; unsigned long long v567;
    Tuple6 tmp28 = Tuple6{0, 0ull, 1ull};
    v565 = tmp28.v0; v566 = tmp28.v1; v567 = tmp28.v2;
    while (while_method_1(v564, v565)){
        Union0 v569;
        v569 = v14[v565];
        unsigned long long v617;
        switch (v569.tag) {
            case 0: { // CommunityCardIs
                Union1 v573 = v569.case0.v0;
                unsigned long long v574;
                switch (v573.tag) {
                    case 0: { // Jack
                        v574 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v574 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v574 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v575;
                v575 = 9223372036854775807ull + v574;
                unsigned long long v576;
                v576 = v575 * 9973ull;
                v617 = v576;
                break;
            }
            case 1: { // Hidden
                v617 = 18446744073709531670ull;
                break;
            }
            case 2: { // PlayerAction
                int v577 = v569.case2.v0; Union2 v578 = v569.case2.v1;
                unsigned long long v579;
                v579 = std::hash<int>()(v577);
                unsigned long long v580;
                v580 = v579 * 9973ull;
                unsigned long long v581;
                switch (v578.tag) {
                    case 0: { // Call
                        v581 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v581 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v581 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v582;
                v582 = v580 + v581;
                unsigned long long v583;
                v583 = 9223372036854775807ull + v582;
                unsigned long long v584;
                v584 = v583 * 9973ull;
                unsigned long long v585;
                v585 = v584 * 3ull;
                v617 = v585;
                break;
            }
            case 3: { // PlayerGotCard
                int v586 = v569.case3.v0; Union1 v587 = v569.case3.v1;
                unsigned long long v588;
                v588 = std::hash<int>()(v586);
                unsigned long long v589;
                v589 = v588 * 9973ull;
                unsigned long long v590;
                switch (v587.tag) {
                    case 0: { // Jack
                        v590 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v590 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v590 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v591;
                v591 = v589 + v590;
                unsigned long long v592;
                v592 = 9223372036854775807ull + v591;
                unsigned long long v593;
                v593 = v592 * 9973ull;
                unsigned long long v594;
                v594 = v593 * 4ull;
                v617 = v594;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v595 = v569.case4.v0; int v596 = v569.case4.v1; int v597 = v569.case4.v2;
                unsigned long long v598;
                v598 = std::hash<int>()(v597);
                unsigned long long v599;
                v599 = std::hash<int>()(v596);
                unsigned long long v600;
                v600 = v599 * 9973ull;
                unsigned long long v601;
                v601 = v598 + v600;
                int v602; unsigned long long v603; unsigned long long v604;
                Tuple6 tmp29 = Tuple6{0, 0ull, 1ull};
                v602 = tmp29.v0; v603 = tmp29.v1; v604 = tmp29.v2;
                while (while_method_2(v602)){
                    Union1 v606;
                    v606 = v595[v602];
                    unsigned long long v610;
                    switch (v606.tag) {
                        case 0: { // Jack
                            v610 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v610 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v610 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v611;
                    v611 = v610 * v604;
                    unsigned long long v612;
                    v612 = v603 + v611;
                    unsigned long long v613;
                    v613 = v604 * 9973ull;
                    v603 = v612;
                    v604 = v613;
                    v602 += 1 ;
                }
                unsigned long long v614;
                v614 = 9223372036854775807ull + v601;
                unsigned long long v615;
                v615 = v614 * 9973ull;
                unsigned long long v616;
                v616 = v615 * 5ull;
                v617 = v616;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v618;
        v618 = v617 * v567;
        unsigned long long v619;
        v619 = v566 + v618;
        unsigned long long v620;
        v620 = v567 * 9973ull;
        v566 = v619;
        v567 = v620;
        v565 += 1 ;
    }
    v161[Tuple0{0ull, v14}] = v544;
    int v621; float v622;
    Tuple3 tmp30 = Tuple3{0, 0.0f};
    v621 = tmp30.v0; v622 = tmp30.v1;
    while (while_method_5(v621)){
        float v624;
        v624 = v418[v621];
        float v628;
        v628 = v471[v621];
        float v632;
        v632 = v624 * v628;
        float v633;
        v633 = v622 + v632;
        v622 = v633;
        v621 += 1 ;
    }
    return v622;
}
int tag_13(Union1 v0){
    switch (v0.tag) {
        case 0: { // Jack
            return 0;
            break;
        }
        case 1: { // King
            return 2;
            break;
        }
        case 2: { // Queen
            return 1;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
bool is_pair_14(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
Tuple5 order_15(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple5{v1, v0};
    } else {
        return Tuple5{v0, v1};
    }
}
Union9 compare_hands_12(Union4 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            exit(-1);
            break;
        }
        case 1: { // Some
            Union1 v7 = v0.case1.v0;
            int v8;
            v8 = tag_13(v7);
            Union1 v9;
            v9 = v2[0];
            int v13;
            v13 = tag_13(v9);
            Union1 v14;
            v14 = v2[1];
            int v18;
            v18 = tag_13(v14);
            bool v19;
            v19 = is_pair_14(v8, v13);
            bool v20;
            v20 = is_pair_14(v8, v18);
            if (v19){
                if (v20){
                    bool v21;
                    v21 = v13 < v18;
                    if (v21){
                        return Union9{Union9_2{}};
                    } else {
                        bool v23;
                        v23 = v13 > v18;
                        if (v23){
                            return Union9{Union9_1{}};
                        } else {
                            return Union9{Union9_0{}};
                        }
                    }
                } else {
                    return Union9{Union9_1{}};
                }
            } else {
                if (v20){
                    return Union9{Union9_2{}};
                } else {
                    int v31; int v32;
                    Tuple5 tmp31 = order_15(v8, v13);
                    v31 = tmp31.v0; v32 = tmp31.v1;
                    int v33; int v34;
                    Tuple5 tmp32 = order_15(v8, v18);
                    v33 = tmp32.v0; v34 = tmp32.v1;
                    bool v35;
                    v35 = v31 < v33;
                    Union9 v41;
                    if (v35){
                        v41 = Union9{Union9_2{}};
                    } else {
                        bool v37;
                        v37 = v31 > v33;
                        if (v37){
                            v41 = Union9{Union9_1{}};
                        } else {
                            v41 = Union9{Union9_0{}};
                        }
                    }
                    bool v42;
                    switch (v41.tag) {
                        case 0: { // Eq
                            v42 = true;
                            break;
                        }
                        default: {
                            v42 = false;
                        }
                    }
                    if (v42){
                        bool v43;
                        v43 = v32 < v34;
                        if (v43){
                            return Union9{Union9_2{}};
                        } else {
                            bool v45;
                            v45 = v32 > v34;
                            if (v45){
                                return Union9{Union9_1{}};
                            } else {
                                return Union9{Union9_0{}};
                            }
                        }
                    } else {
                        return v41;
                    }
                }
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
float body_0(StackRefs3 & v0, StackRefs1 & v1, StackRefs0 & v2, xso::rng & v3, StackMut0 & v4, StackRefs4 & v5, Union3 v6){
    StackMut1 v7{0.0f};
    switch (v6.tag) {
        case 0: { // ChanceCommunityCard
            Union4 v90 = v6.case0.v0; bool v91 = v6.case0.v1; static_array<Union1,2> v92 = v6.case0.v2; int v93 = v6.case0.v3; static_array<int,2> v94 = v6.case0.v4; int v95 = v6.case0.v5;
            unsigned int v96 = v4.v0;
            Union1 v97; unsigned int v98;
            Tuple4 tmp1 = draw_card_1(v3, v96);
            v97 = tmp1.v0; v98 = tmp1.v1;
            v4.v0 = v98;
            static_array_list<Union0,32> & v99 = v0.v0;
            Union0 v100;
            v100 = Union0{Union0_0{v97}};
            v99.push(v100);
            Union5 v101;
            v101 = Union5{Union5_0{v90, v91, v92, v93, v94, v95, v97}};
            float v102;
            v102 = loop_4(v0, v1, v2, v3, v4, v5, v7, v101);
            static_array_list<Union0,32> & v103 = v0.v0;
            Union0 v104;
            v104 = v103.pop();
            v4.v0 = v96;
            return v102;
            break;
        }
        case 1: { // ChanceInit
            unsigned int v105 = v4.v0;
            Union1 v106; unsigned int v107;
            Tuple4 tmp6 = draw_card_1(v3, v105);
            v106 = tmp6.v0; v107 = tmp6.v1;
            v4.v0 = v107;
            unsigned int v108 = v4.v0;
            Union1 v109; unsigned int v110;
            Tuple4 tmp7 = draw_card_1(v3, v108);
            v109 = tmp7.v0; v110 = tmp7.v1;
            v4.v0 = v110;
            static_array_list<Union0,32> & v111 = v0.v0;
            Union0 v112;
            v112 = Union0{Union0_3{0, v106}};
            v111.push(v112);
            static_array_list<Union0,32> & v113 = v0.v0;
            Union0 v114;
            v114 = Union0{Union0_3{1, v109}};
            v113.push(v114);
            Union5 v115;
            v115 = Union5{Union5_1{v106, v109}};
            float v116;
            v116 = loop_4(v0, v1, v2, v3, v4, v5, v7, v115);
            static_array_list<Union0,32> & v117 = v0.v0;
            Union0 v118;
            v118 = v117.pop();
            static_array_list<Union0,32> & v119 = v0.v0;
            Union0 v120;
            v120 = v119.pop();
            v4.v0 = v108;
            v4.v0 = v105;
            return v116;
            break;
        }
        case 2: { // Round
            Union4 v60 = v6.case2.v0; bool v61 = v6.case2.v1; static_array<Union1,2> v62 = v6.case2.v2; int v63 = v6.case2.v3; static_array<int,2> v64 = v6.case2.v4; int v65 = v6.case2.v5;
            bool v66;
            v66 = v63 == 0;
            float v72;
            if (v66){
                v72 = method_5(v3, v0, v5, v60, v61, v62, v63, v64, v65, v1, v2, v4, v7);
            } else {
                bool v68;
                v68 = v63 == 1;
                if (v68){
                    v72 = method_5(v3, v0, v5, v60, v61, v62, v63, v64, v65, v1, v2, v4, v7);
                } else {
                    printf("%s\n", "Mode index is out of bounds.");
                    exit(-1);
                }
            }
            float v74;
            if (v66){
                v74 = v72;
            } else {
                float v73;
                v73 = -v72;
                v74 = v73;
            }
            v7.v0 = v74;
            Union5 v75;
            v75 = Union5{Union5_3{}};
            return loop_4(v0, v1, v2, v3, v4, v5, v7, v75);
            break;
        }
        case 3: { // RoundWithAction
            Union4 v77 = v6.case3.v0; bool v78 = v6.case3.v1; static_array<Union1,2> v79 = v6.case3.v2; int v80 = v6.case3.v3; static_array<int,2> v81 = v6.case3.v4; int v82 = v6.case3.v5; Union2 v83 = v6.case3.v6;
            static_array_list<Union0,32> & v84 = v0.v0;
            Union0 v85;
            v85 = Union0{Union0_2{v80, v83}};
            v84.push(v85);
            Union5 v86;
            v86 = Union5{Union5_2{v77, v78, v79, v80, v81, v82, v83}};
            float v87;
            v87 = loop_4(v0, v1, v2, v3, v4, v5, v7, v86);
            static_array_list<Union0,32> & v88 = v0.v0;
            Union0 v89;
            v89 = v88.pop();
            return v87;
            break;
        }
        case 4: { // TerminalCall
            Union4 v30 = v6.case4.v0; bool v31 = v6.case4.v1; static_array<Union1,2> v32 = v6.case4.v2; int v33 = v6.case4.v3; static_array<int,2> v34 = v6.case4.v4; int v35 = v6.case4.v5;
            int v36;
            v36 = v34[v33];
            Union9 v40;
            v40 = compare_hands_12(v30, v31, v32, v33, v34, v35);
            int v45; int v46;
            switch (v40.tag) {
                case 0: { // Eq
                    v45 = 0; v46 = -1;
                    break;
                }
                case 1: { // Gt
                    v45 = v36; v46 = 0;
                    break;
                }
                case 2: { // Lt
                    v45 = v36; v46 = 1;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v47;
            v47 = -v46;
            bool v48;
            v48 = v46 >= v47;
            int v49;
            if (v48){
                v49 = v46;
            } else {
                v49 = v47;
            }
            float v50;
            v50 = (float)v45;
            bool v51;
            v51 = v49 == 0;
            float v53;
            if (v51){
                v53 = v50;
            } else {
                float v52;
                v52 = -v50;
                v53 = v52;
            }
            v7.v0 = v53;
            static_array_list<Union0,32> & v54 = v0.v0;
            Union0 v55;
            v55 = Union0{Union0_4{v32, v45, v46}};
            v54.push(v55);
            Union5 v56;
            v56 = Union5{Union5_3{}};
            float v57;
            v57 = loop_4(v0, v1, v2, v3, v4, v5, v7, v56);
            static_array_list<Union0,32> & v58 = v0.v0;
            Union0 v59;
            v59 = v58.pop();
            return v57;
            break;
        }
        case 5: { // TerminalFold
            Union4 v8 = v6.case5.v0; bool v9 = v6.case5.v1; static_array<Union1,2> v10 = v6.case5.v2; int v11 = v6.case5.v3; static_array<int,2> v12 = v6.case5.v4; int v13 = v6.case5.v5;
            int v14;
            v14 = v12[v11];
            int v18;
            v18 = -v14;
            float v19;
            v19 = (float)v18;
            bool v20;
            v20 = v11 == 0;
            float v22;
            if (v20){
                v22 = v19;
            } else {
                float v21;
                v21 = -v19;
                v22 = v21;
            }
            v7.v0 = v22;
            int v23;
            v23 = v11 ^ 1;
            static_array_list<Union0,32> & v24 = v0.v0;
            Union0 v25;
            v25 = Union0{Union0_4{v10, v14, v23}};
            v24.push(v25);
            Union5 v26;
            v26 = Union5{Union5_3{}};
            float v27;
            v27 = loop_4(v0, v1, v2, v3, v4, v5, v7, v26);
            static_array_list<Union0,32> & v28 = v0.v0;
            Union0 v29;
            v29 = v28.pop();
            return v27;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 100;
    return v1;
}
inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 6;
    return v1;
}
float loop_17(StackRefs3 & v0, StackRefs0 & v1, xso::rng & v2, StackMut0 & v3, StackRefs4 & v4, StackMut1 & v5, Union5 v6){
    switch (v6.tag) {
        case 0: { // T_game_chance_community_card
            Union4 v8 = v6.case0.v0; bool v9 = v6.case0.v1; static_array<Union1,2> v10 = v6.case0.v2; int v11 = v6.case0.v3; static_array<int,2> v12 = v6.case0.v4; int v13 = v6.case0.v5; Union1 v14 = v6.case0.v6;
            int v15;
            v15 = 2;
            int v16; int v17;
            Tuple5 tmp35 = Tuple5{0, 0};
            v16 = tmp35.v0; v17 = tmp35.v1;
            while (while_method_2(v16)){
                int v19;
                v19 = v12[v16];
                bool v23;
                v23 = v17 >= v19;
                int v24;
                if (v23){
                    v24 = v17;
                } else {
                    v24 = v19;
                }
                v17 = v24;
                v16 += 1 ;
            }
            static_array<int,2> v25;
            int v29;
            v29 = 0;
            while (while_method_2(v29)){
                v25[v29] = v17;
                v29 += 1 ;
            }
            Union4 v31;
            v31 = Union4{Union4_1{v14}};
            bool v32;
            v32 = true;
            int v33;
            v33 = 0;
            Union3 v34;
            v34 = Union3{Union3_2{v31, v32, v10, v33, v25, v15}};
            return body_16(v0, v1, v2, v3, v4, v34);
            break;
        }
        case 1: { // T_game_chance_init
            Union1 v36 = v6.case1.v0; Union1 v37 = v6.case1.v1;
            int v38;
            v38 = 2;
            static_array<int,2> v39;
            v39[0] = 1;
            v39[1] = 1;
            static_array<Union1,2> v43;
            v43[0] = v36;
            v43[1] = v37;
            Union4 v47;
            v47 = Union4{Union4_0{}};
            bool v48;
            v48 = true;
            int v49;
            v49 = 0;
            Union3 v50;
            v50 = Union3{Union3_2{v47, v48, v43, v49, v39, v38}};
            return body_16(v0, v1, v2, v3, v4, v50);
            break;
        }
        case 2: { // T_game_round
            Union4 v52 = v6.case2.v0; bool v53 = v6.case2.v1; static_array<Union1,2> v54 = v6.case2.v2; int v55 = v6.case2.v3; static_array<int,2> v56 = v6.case2.v4; int v57 = v6.case2.v5; Union2 v58 = v6.case2.v6;
            Union3 v160;
            switch (v52.tag) {
                case 0: { // None
                    switch (v58.tag) {
                        case 0: { // Call
                            if (v53){
                                int v118;
                                v118 = v55 ^ 1;
                                v160 = Union3{Union3_2{v52, false, v54, v118, v56, v57}};
                            } else {
                                v160 = Union3{Union3_0{v52, v53, v54, v55, v56, v57}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v160 = Union3{Union3_5{v52, v53, v54, v55, v56, v57}};
                            break;
                        }
                        case 2: { // Raise
                            bool v122;
                            v122 = v57 > 0;
                            if (v122){
                                int v123;
                                v123 = v55 ^ 1;
                                int v124;
                                v124 = -1 + v57;
                                int v125; int v126;
                                Tuple5 tmp36 = Tuple5{0, 0};
                                v125 = tmp36.v0; v126 = tmp36.v1;
                                while (while_method_2(v125)){
                                    int v128;
                                    v128 = v56[v125];
                                    bool v132;
                                    v132 = v126 >= v128;
                                    int v133;
                                    if (v132){
                                        v133 = v126;
                                    } else {
                                        v133 = v128;
                                    }
                                    v126 = v133;
                                    v125 += 1 ;
                                }
                                static_array<int,2> v134;
                                int v138;
                                v138 = 0;
                                while (while_method_2(v138)){
                                    v134[v138] = v126;
                                    v138 += 1 ;
                                }
                                static_array<int,2> v140;
                                int v144;
                                v144 = 0;
                                while (while_method_2(v144)){
                                    int v146;
                                    v146 = v134[v144];
                                    bool v150;
                                    v150 = v144 == v55;
                                    int v152;
                                    if (v150){
                                        int v151;
                                        v151 = v146 + 2;
                                        v152 = v151;
                                    } else {
                                        v152 = v146;
                                    }
                                    v140[v144] = v152;
                                    v144 += 1 ;
                                }
                                v160 = Union3{Union3_2{v52, false, v54, v123, v140, v124}};
                            } else {
                                printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                exit(-1);
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    break;
                }
                case 1: { // Some
                    Union1 v59 = v52.case1.v0;
                    switch (v58.tag) {
                        case 0: { // Call
                            if (v53){
                                int v61;
                                v61 = v55 ^ 1;
                                v160 = Union3{Union3_2{v52, false, v54, v61, v56, v57}};
                            } else {
                                int v63; int v64;
                                Tuple5 tmp37 = Tuple5{0, 0};
                                v63 = tmp37.v0; v64 = tmp37.v1;
                                while (while_method_2(v63)){
                                    int v66;
                                    v66 = v56[v63];
                                    bool v70;
                                    v70 = v64 >= v66;
                                    int v71;
                                    if (v70){
                                        v71 = v64;
                                    } else {
                                        v71 = v66;
                                    }
                                    v64 = v71;
                                    v63 += 1 ;
                                }
                                static_array<int,2> v72;
                                int v76;
                                v76 = 0;
                                while (while_method_2(v76)){
                                    v72[v76] = v64;
                                    v76 += 1 ;
                                }
                                v160 = Union3{Union3_4{v52, v53, v54, v55, v72, v57}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v160 = Union3{Union3_5{v52, v53, v54, v55, v56, v57}};
                            break;
                        }
                        case 2: { // Raise
                            bool v80;
                            v80 = v57 > 0;
                            if (v80){
                                int v81;
                                v81 = v55 ^ 1;
                                int v82;
                                v82 = -1 + v57;
                                int v83; int v84;
                                Tuple5 tmp38 = Tuple5{0, 0};
                                v83 = tmp38.v0; v84 = tmp38.v1;
                                while (while_method_2(v83)){
                                    int v86;
                                    v86 = v56[v83];
                                    bool v90;
                                    v90 = v84 >= v86;
                                    int v91;
                                    if (v90){
                                        v91 = v84;
                                    } else {
                                        v91 = v86;
                                    }
                                    v84 = v91;
                                    v83 += 1 ;
                                }
                                static_array<int,2> v92;
                                int v96;
                                v96 = 0;
                                while (while_method_2(v96)){
                                    v92[v96] = v84;
                                    v96 += 1 ;
                                }
                                static_array<int,2> v98;
                                int v102;
                                v102 = 0;
                                while (while_method_2(v102)){
                                    int v104;
                                    v104 = v92[v102];
                                    bool v108;
                                    v108 = v102 == v55;
                                    int v110;
                                    if (v108){
                                        int v109;
                                        v109 = v104 + 4;
                                        v110 = v109;
                                    } else {
                                        v110 = v104;
                                    }
                                    v98[v102] = v110;
                                    v102 += 1 ;
                                }
                                v160 = Union3{Union3_2{v52, false, v54, v81, v98, v82}};
                            } else {
                                printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                exit(-1);
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            return body_16(v0, v1, v2, v3, v4, v160);
            break;
        }
        case 3: { // T_none
            float v7 = v5.v0;
            return v7;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
float method_18(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs0 & v9, StackMut0 & v10, StackMut1 & v11){
    std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> & v12 = v9.v0;
    static_array_list<Union0,32> & v13 = v1.v0;
    static_array_list<Union0,32> & v14 = v1.v0;
    int v15;
    v15 = v14.length;
    bool v16;
    v16 = 32 >= v15;
    bool v17;
    v17 = v16 == false;
    if (v17){
        assert("The type level dimension has to equal the value passed at runtime into create." && v16);
    } else {
    }
    static_array_list<Union0,32> v19;
    v19 = static_array_list<Union0,32>{};
    v19.unsafe_set_length(v15);
    int v23; int v24;
    Tuple5 tmp41 = Tuple5{0, 0};
    v23 = tmp41.v0; v24 = tmp41.v1;
    while (while_method_1(v15, v23)){
        Union0 v26;
        v26 = v14[v23];
        bool v33;
        switch (v26.tag) {
            case 3: { // PlayerGotCard
                int v30 = v26.case3.v0; Union1 v31 = v26.case3.v1;
                bool v32;
                v32 = v30 == v6;
                v33 = v32;
                break;
            }
            default: {
                v33 = true;
            }
        }
        int v35;
        if (v33){
            v19[v24] = v26;
            int v34;
            v34 = v24 + 1;
            v35 = v34;
        } else {
            v35 = v24;
        }
        v24 = v35;
        v23 += 1 ;
    }
    bool v36;
    v36 = 32 >= v24;
    bool v37;
    v37 = v36 == false;
    if (v37){
        assert("The type level dimension has to equal the value passed at runtime into create." && v36);
    } else {
    }
    static_array_list<Union0,32> v39;
    v39 = static_array_list<Union0,32>{};
    v39.unsafe_set_length(v24);
    int v43;
    v43 = 0;
    while (while_method_1(v24, v43)){
        Union0 v45;
        v45 = v19[v43];
        v39[v43] = v45;
        v43 += 1 ;
    }
    int v49;
    v49 = v39.length;
    int v50; unsigned long long v51; unsigned long long v52;
    Tuple6 tmp42 = Tuple6{0, 0ull, 1ull};
    v50 = tmp42.v0; v51 = tmp42.v1; v52 = tmp42.v2;
    while (while_method_1(v49, v50)){
        Union0 v54;
        v54 = v39[v50];
        unsigned long long v102;
        switch (v54.tag) {
            case 0: { // CommunityCardIs
                Union1 v58 = v54.case0.v0;
                unsigned long long v59;
                switch (v58.tag) {
                    case 0: { // Jack
                        v59 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v59 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v59 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v60;
                v60 = 9223372036854775807ull + v59;
                unsigned long long v61;
                v61 = v60 * 9973ull;
                v102 = v61;
                break;
            }
            case 1: { // Hidden
                v102 = 18446744073709531670ull;
                break;
            }
            case 2: { // PlayerAction
                int v62 = v54.case2.v0; Union2 v63 = v54.case2.v1;
                unsigned long long v64;
                v64 = std::hash<int>()(v62);
                unsigned long long v65;
                v65 = v64 * 9973ull;
                unsigned long long v66;
                switch (v63.tag) {
                    case 0: { // Call
                        v66 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v66 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v66 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v67;
                v67 = v65 + v66;
                unsigned long long v68;
                v68 = 9223372036854775807ull + v67;
                unsigned long long v69;
                v69 = v68 * 9973ull;
                unsigned long long v70;
                v70 = v69 * 3ull;
                v102 = v70;
                break;
            }
            case 3: { // PlayerGotCard
                int v71 = v54.case3.v0; Union1 v72 = v54.case3.v1;
                unsigned long long v73;
                v73 = std::hash<int>()(v71);
                unsigned long long v74;
                v74 = v73 * 9973ull;
                unsigned long long v75;
                switch (v72.tag) {
                    case 0: { // Jack
                        v75 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v75 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v75 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v76;
                v76 = v74 + v75;
                unsigned long long v77;
                v77 = 9223372036854775807ull + v76;
                unsigned long long v78;
                v78 = v77 * 9973ull;
                unsigned long long v79;
                v79 = v78 * 4ull;
                v102 = v79;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v80 = v54.case4.v0; int v81 = v54.case4.v1; int v82 = v54.case4.v2;
                unsigned long long v83;
                v83 = std::hash<int>()(v82);
                unsigned long long v84;
                v84 = std::hash<int>()(v81);
                unsigned long long v85;
                v85 = v84 * 9973ull;
                unsigned long long v86;
                v86 = v83 + v85;
                int v87; unsigned long long v88; unsigned long long v89;
                Tuple6 tmp43 = Tuple6{0, 0ull, 1ull};
                v87 = tmp43.v0; v88 = tmp43.v1; v89 = tmp43.v2;
                while (while_method_2(v87)){
                    Union1 v91;
                    v91 = v80[v87];
                    unsigned long long v95;
                    switch (v91.tag) {
                        case 0: { // Jack
                            v95 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v95 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v95 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v96;
                    v96 = v95 * v89;
                    unsigned long long v97;
                    v97 = v88 + v96;
                    unsigned long long v98;
                    v98 = v89 * 9973ull;
                    v88 = v97;
                    v89 = v98;
                    v87 += 1 ;
                }
                unsigned long long v99;
                v99 = 9223372036854775807ull + v86;
                unsigned long long v100;
                v100 = v99 * 9973ull;
                unsigned long long v101;
                v101 = v100 * 5ull;
                v102 = v101;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v103;
        v103 = v102 * v52;
        unsigned long long v104;
        v104 = v51 + v103;
        unsigned long long v105;
        v105 = v52 * 9973ull;
        v51 = v104;
        v52 = v105;
        v50 += 1 ;
    }
    auto v106 = v12.find(Tuple0{0ull, v39});
    bool v107;
    v107 = v106 != v12.end();
    Union6 v112;
    if (v107){
        static_array<float,3> v108; static_array<float,3> v109;
        Tuple1 tmp44 = v106->second;
        v108 = tmp44.v0; v109 = tmp44.v1;
        v112 = Union6{Union6_1{v108, v109}};
    } else {
        v112 = Union6{Union6_0{}};
    }
    static_array<float,3> v129; static_array<float,3> v130;
    switch (v112.tag) {
        case 0: { // None
            static_array<float,3> v115;
            int v119;
            v119 = 0;
            while (while_method_5(v119)){
                v115[v119] = 0.0f;
                v119 += 1 ;
            }
            static_array<float,3> v121;
            int v125;
            v125 = 0;
            while (while_method_5(v125)){
                v121[v125] = 0.0f;
                v125 += 1 ;
            }
            v129 = v115; v130 = v121;
            break;
        }
        case 1: { // Some
            static_array<float,3> v113 = v112.case1.v0; static_array<float,3> v114 = v112.case1.v1;
            v129 = v113; v130 = v114;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    int v131;
    v131 = v7[0];
    int v135;
    v135 = v7[1];
    bool v139;
    v139 = v131 == v135;
    bool v140;
    v140 = v139 != true;
    Union7 v144;
    if (v140){
        Union2 v141;
        v141 = Union2{Union2_1{}};
        v144 = Union7{Union7_1{v141}};
    } else {
        v144 = Union7{Union7_0{}};
    }
    bool v145;
    v145 = v8 > 0;
    Union7 v149;
    if (v145){
        Union2 v146;
        v146 = Union2{Union2_2{}};
        v149 = Union7{Union7_1{v146}};
    } else {
        v149 = Union7{Union7_0{}};
    }
    bool v152;
    switch (v149.tag) {
        case 0: { // None
            v152 = false;
            break;
        }
        case 1: { // Some
            Union2 v150 = v149.case1.v0;
            v152 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    bool v155;
    switch (v144.tag) {
        case 0: { // None
            v155 = false;
            break;
        }
        case 1: { // Some
            Union2 v153 = v144.case1.v0;
            v155 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    static_array<bool,3> v156;
    v156[0] = true;
    v156[1] = v155;
    v156[2] = v152;
    static_array<float,3> v160;
    v160 = regret_match_6(v130, v156);
    float v189;
    switch (v149.tag) {
        case 0: { // None
            v189 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v161 = v149.case1.v0;
            float v162;
            v162 = v160[2];
            static_array<Tuple2,2> & v166 = v2.v0;
            float v167; float v168;
            Tuple2 tmp45 = v166[v6];
            v167 = tmp45.v0; v168 = tmp45.v1;
            static_array<Tuple2,2> & v175 = v2.v0;
            float v176;
            v176 = log(v162);
            float v177;
            v177 = v176 + v167;
            v175[v6] = Tuple2{v177, v168};
            static_array_list<Union0,32> & v178 = v1.v0;
            Union0 v179;
            v179 = Union0{Union0_2{v6, v161}};
            v178.push(v179);
            Union5 v180;
            v180 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v161}};
            float v181;
            v181 = loop_17(v1, v9, v0, v10, v2, v11, v180);
            static_array_list<Union0,32> & v182 = v1.v0;
            Union0 v183;
            v183 = v182.pop();
            static_array<Tuple2,2> & v184 = v2.v0;
            v184[v6] = Tuple2{v167, v168};
            bool v185;
            v185 = v6 == 0;
            if (v185){
                v189 = v181;
            } else {
                float v186;
                v186 = -v181;
                v189 = v186;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v218;
    switch (v144.tag) {
        case 0: { // None
            v218 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v190 = v144.case1.v0;
            float v191;
            v191 = v160[1];
            static_array<Tuple2,2> & v195 = v2.v0;
            float v196; float v197;
            Tuple2 tmp46 = v195[v6];
            v196 = tmp46.v0; v197 = tmp46.v1;
            static_array<Tuple2,2> & v204 = v2.v0;
            float v205;
            v205 = log(v191);
            float v206;
            v206 = v205 + v196;
            v204[v6] = Tuple2{v206, v197};
            static_array_list<Union0,32> & v207 = v1.v0;
            Union0 v208;
            v208 = Union0{Union0_2{v6, v190}};
            v207.push(v208);
            Union5 v209;
            v209 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v190}};
            float v210;
            v210 = loop_17(v1, v9, v0, v10, v2, v11, v209);
            static_array_list<Union0,32> & v211 = v1.v0;
            Union0 v212;
            v212 = v211.pop();
            static_array<Tuple2,2> & v213 = v2.v0;
            v213[v6] = Tuple2{v196, v197};
            bool v214;
            v214 = v6 == 0;
            if (v214){
                v218 = v210;
            } else {
                float v215;
                v215 = -v210;
                v218 = v215;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v219;
    v219 = v160[0];
    static_array<Tuple2,2> & v223 = v2.v0;
    float v224; float v225;
    Tuple2 tmp47 = v223[v6];
    v224 = tmp47.v0; v225 = tmp47.v1;
    static_array<Tuple2,2> & v232 = v2.v0;
    float v233;
    v233 = log(v219);
    float v234;
    v234 = v233 + v224;
    v232[v6] = Tuple2{v234, v225};
    static_array_list<Union0,32> & v235 = v1.v0;
    Union2 v236;
    v236 = Union2{Union2_0{}};
    Union0 v237;
    v237 = Union0{Union0_2{v6, v236}};
    v235.push(v237);
    Union2 v238;
    v238 = Union2{Union2_0{}};
    Union5 v239;
    v239 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v238}};
    float v240;
    v240 = loop_17(v1, v9, v0, v10, v2, v11, v239);
    static_array_list<Union0,32> & v241 = v1.v0;
    Union0 v242;
    v242 = v241.pop();
    static_array<Tuple2,2> & v243 = v2.v0;
    v243[v6] = Tuple2{v224, v225};
    bool v244;
    v244 = v6 == 0;
    float v246;
    if (v244){
        v246 = v240;
    } else {
        float v245;
        v245 = -v240;
        v246 = v245;
    }
    static_array<float,3> v247;
    v247[0] = v246;
    v247[1] = v218;
    v247[2] = v189;
    int v251; float v252;
    Tuple3 tmp48 = Tuple3{0, 0.0f};
    v251 = tmp48.v0; v252 = tmp48.v1;
    while (while_method_5(v251)){
        float v254;
        v254 = v247[v251];
        float v258;
        v258 = v160[v251];
        float v262;
        v262 = v254 * v258;
        float v263;
        v263 = v252 + v262;
        v252 = v263;
        v251 += 1 ;
    }
    static_array<float,3> v264;
    int v268;
    v268 = 0;
    while (while_method_5(v268)){
        float v270;
        v270 = v129[v268];
        float v274;
        v274 = v160[v268];
        float v278;
        v278 = 0.99609375f * v270;
        float v279;
        v279 = v278 + v274;
        v264[v268] = v279;
        v268 += 1 ;
    }
    static_array<Tuple2,2> & v280 = v2.v0;
    int v281; float v282;
    Tuple3 tmp49 = Tuple3{0, 0.0f};
    v281 = tmp49.v0; v282 = tmp49.v1;
    while (while_method_2(v281)){
        float v284; float v285;
        Tuple2 tmp50 = v280[v281];
        v284 = tmp50.v0; v285 = tmp50.v1;
        bool v292;
        v292 = v281 == v6;
        float v293;
        if (v292){
            v293 = 0.0f;
        } else {
            v293 = v284;
        }
        float v294;
        v294 = v282 + v293;
        float v295;
        v295 = v294 - v285;
        v282 = v295;
        v281 += 1 ;
    }
    float v296;
    v296 = exp(v282);
    static_array<float,3> v297;
    int v301;
    v301 = 0;
    while (while_method_5(v301)){
        float v303;
        v303 = v130[v301];
        float v307;
        v307 = v247[v301];
        float v311;
        v311 = v307 - v252;
        float v312;
        v312 = v296 * v311;
        float v313;
        v313 = v303 + v312;
        bool v314;
        v314 = 0.0f >= v313;
        float v315;
        if (v314){
            v315 = 0.0f;
        } else {
            v315 = v313;
        }
        v297[v301] = v315;
        v301 += 1 ;
    }
    int v316;
    v316 = v39.length;
    int v317; unsigned long long v318; unsigned long long v319;
    Tuple6 tmp51 = Tuple6{0, 0ull, 1ull};
    v317 = tmp51.v0; v318 = tmp51.v1; v319 = tmp51.v2;
    while (while_method_1(v316, v317)){
        Union0 v321;
        v321 = v39[v317];
        unsigned long long v369;
        switch (v321.tag) {
            case 0: { // CommunityCardIs
                Union1 v325 = v321.case0.v0;
                unsigned long long v326;
                switch (v325.tag) {
                    case 0: { // Jack
                        v326 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v326 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v326 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v327;
                v327 = 9223372036854775807ull + v326;
                unsigned long long v328;
                v328 = v327 * 9973ull;
                v369 = v328;
                break;
            }
            case 1: { // Hidden
                v369 = 18446744073709531670ull;
                break;
            }
            case 2: { // PlayerAction
                int v329 = v321.case2.v0; Union2 v330 = v321.case2.v1;
                unsigned long long v331;
                v331 = std::hash<int>()(v329);
                unsigned long long v332;
                v332 = v331 * 9973ull;
                unsigned long long v333;
                switch (v330.tag) {
                    case 0: { // Call
                        v333 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v333 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v333 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v334;
                v334 = v332 + v333;
                unsigned long long v335;
                v335 = 9223372036854775807ull + v334;
                unsigned long long v336;
                v336 = v335 * 9973ull;
                unsigned long long v337;
                v337 = v336 * 3ull;
                v369 = v337;
                break;
            }
            case 3: { // PlayerGotCard
                int v338 = v321.case3.v0; Union1 v339 = v321.case3.v1;
                unsigned long long v340;
                v340 = std::hash<int>()(v338);
                unsigned long long v341;
                v341 = v340 * 9973ull;
                unsigned long long v342;
                switch (v339.tag) {
                    case 0: { // Jack
                        v342 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v342 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v342 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v343;
                v343 = v341 + v342;
                unsigned long long v344;
                v344 = 9223372036854775807ull + v343;
                unsigned long long v345;
                v345 = v344 * 9973ull;
                unsigned long long v346;
                v346 = v345 * 4ull;
                v369 = v346;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v347 = v321.case4.v0; int v348 = v321.case4.v1; int v349 = v321.case4.v2;
                unsigned long long v350;
                v350 = std::hash<int>()(v349);
                unsigned long long v351;
                v351 = std::hash<int>()(v348);
                unsigned long long v352;
                v352 = v351 * 9973ull;
                unsigned long long v353;
                v353 = v350 + v352;
                int v354; unsigned long long v355; unsigned long long v356;
                Tuple6 tmp52 = Tuple6{0, 0ull, 1ull};
                v354 = tmp52.v0; v355 = tmp52.v1; v356 = tmp52.v2;
                while (while_method_2(v354)){
                    Union1 v358;
                    v358 = v347[v354];
                    unsigned long long v362;
                    switch (v358.tag) {
                        case 0: { // Jack
                            v362 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v362 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v362 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v363;
                    v363 = v362 * v356;
                    unsigned long long v364;
                    v364 = v355 + v363;
                    unsigned long long v365;
                    v365 = v356 * 9973ull;
                    v355 = v364;
                    v356 = v365;
                    v354 += 1 ;
                }
                unsigned long long v366;
                v366 = 9223372036854775807ull + v353;
                unsigned long long v367;
                v367 = v366 * 9973ull;
                unsigned long long v368;
                v368 = v367 * 5ull;
                v369 = v368;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v370;
        v370 = v369 * v319;
        unsigned long long v371;
        v371 = v318 + v370;
        unsigned long long v372;
        v372 = v319 * 9973ull;
        v318 = v371;
        v319 = v372;
        v317 += 1 ;
    }
    v12[Tuple0{0ull, v39}] = Tuple1{v264, v297};
    return v252;
}
float method_19(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs0 & v9, StackMut0 & v10, StackMut1 & v11){
    std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> & v12 = v9.v0;
    static_array_list<Union0,32> & v13 = v1.v0;
    static_array_list<Union0,32> & v14 = v1.v0;
    int v15;
    v15 = v14.length;
    bool v16;
    v16 = 32 >= v15;
    bool v17;
    v17 = v16 == false;
    if (v17){
        assert("The type level dimension has to equal the value passed at runtime into create." && v16);
    } else {
    }
    static_array_list<Union0,32> v19;
    v19 = static_array_list<Union0,32>{};
    v19.unsafe_set_length(v15);
    int v23; int v24;
    Tuple5 tmp53 = Tuple5{0, 0};
    v23 = tmp53.v0; v24 = tmp53.v1;
    while (while_method_1(v15, v23)){
        Union0 v26;
        v26 = v14[v23];
        bool v33;
        switch (v26.tag) {
            case 3: { // PlayerGotCard
                int v30 = v26.case3.v0; Union1 v31 = v26.case3.v1;
                bool v32;
                v32 = v30 == v6;
                v33 = v32;
                break;
            }
            default: {
                v33 = true;
            }
        }
        int v35;
        if (v33){
            v19[v24] = v26;
            int v34;
            v34 = v24 + 1;
            v35 = v34;
        } else {
            v35 = v24;
        }
        v24 = v35;
        v23 += 1 ;
    }
    bool v36;
    v36 = 32 >= v24;
    bool v37;
    v37 = v36 == false;
    if (v37){
        assert("The type level dimension has to equal the value passed at runtime into create." && v36);
    } else {
    }
    static_array_list<Union0,32> v39;
    v39 = static_array_list<Union0,32>{};
    v39.unsafe_set_length(v24);
    int v43;
    v43 = 0;
    while (while_method_1(v24, v43)){
        Union0 v45;
        v45 = v19[v43];
        v39[v43] = v45;
        v43 += 1 ;
    }
    int v49;
    v49 = v39.length;
    int v50; unsigned long long v51; unsigned long long v52;
    Tuple6 tmp54 = Tuple6{0, 0ull, 1ull};
    v50 = tmp54.v0; v51 = tmp54.v1; v52 = tmp54.v2;
    while (while_method_1(v49, v50)){
        Union0 v54;
        v54 = v39[v50];
        unsigned long long v102;
        switch (v54.tag) {
            case 0: { // CommunityCardIs
                Union1 v58 = v54.case0.v0;
                unsigned long long v59;
                switch (v58.tag) {
                    case 0: { // Jack
                        v59 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v59 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v59 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v60;
                v60 = 9223372036854775807ull + v59;
                unsigned long long v61;
                v61 = v60 * 9973ull;
                v102 = v61;
                break;
            }
            case 1: { // Hidden
                v102 = 18446744073709531670ull;
                break;
            }
            case 2: { // PlayerAction
                int v62 = v54.case2.v0; Union2 v63 = v54.case2.v1;
                unsigned long long v64;
                v64 = std::hash<int>()(v62);
                unsigned long long v65;
                v65 = v64 * 9973ull;
                unsigned long long v66;
                switch (v63.tag) {
                    case 0: { // Call
                        v66 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v66 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v66 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v67;
                v67 = v65 + v66;
                unsigned long long v68;
                v68 = 9223372036854775807ull + v67;
                unsigned long long v69;
                v69 = v68 * 9973ull;
                unsigned long long v70;
                v70 = v69 * 3ull;
                v102 = v70;
                break;
            }
            case 3: { // PlayerGotCard
                int v71 = v54.case3.v0; Union1 v72 = v54.case3.v1;
                unsigned long long v73;
                v73 = std::hash<int>()(v71);
                unsigned long long v74;
                v74 = v73 * 9973ull;
                unsigned long long v75;
                switch (v72.tag) {
                    case 0: { // Jack
                        v75 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v75 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v75 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v76;
                v76 = v74 + v75;
                unsigned long long v77;
                v77 = 9223372036854775807ull + v76;
                unsigned long long v78;
                v78 = v77 * 9973ull;
                unsigned long long v79;
                v79 = v78 * 4ull;
                v102 = v79;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v80 = v54.case4.v0; int v81 = v54.case4.v1; int v82 = v54.case4.v2;
                unsigned long long v83;
                v83 = std::hash<int>()(v82);
                unsigned long long v84;
                v84 = std::hash<int>()(v81);
                unsigned long long v85;
                v85 = v84 * 9973ull;
                unsigned long long v86;
                v86 = v83 + v85;
                int v87; unsigned long long v88; unsigned long long v89;
                Tuple6 tmp55 = Tuple6{0, 0ull, 1ull};
                v87 = tmp55.v0; v88 = tmp55.v1; v89 = tmp55.v2;
                while (while_method_2(v87)){
                    Union1 v91;
                    v91 = v80[v87];
                    unsigned long long v95;
                    switch (v91.tag) {
                        case 0: { // Jack
                            v95 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v95 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v95 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v96;
                    v96 = v95 * v89;
                    unsigned long long v97;
                    v97 = v88 + v96;
                    unsigned long long v98;
                    v98 = v89 * 9973ull;
                    v88 = v97;
                    v89 = v98;
                    v87 += 1 ;
                }
                unsigned long long v99;
                v99 = 9223372036854775807ull + v86;
                unsigned long long v100;
                v100 = v99 * 9973ull;
                unsigned long long v101;
                v101 = v100 * 5ull;
                v102 = v101;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v103;
        v103 = v102 * v52;
        unsigned long long v104;
        v104 = v51 + v103;
        unsigned long long v105;
        v105 = v52 * 9973ull;
        v51 = v104;
        v52 = v105;
        v50 += 1 ;
    }
    auto v106 = v12.find(Tuple0{0ull, v39});
    bool v107;
    v107 = v106 != v12.end();
    Union6 v112;
    if (v107){
        static_array<float,3> v108; static_array<float,3> v109;
        Tuple1 tmp56 = v106->second;
        v108 = tmp56.v0; v109 = tmp56.v1;
        v112 = Union6{Union6_1{v108, v109}};
    } else {
        v112 = Union6{Union6_0{}};
    }
    static_array<float,3> v129; static_array<float,3> v130;
    switch (v112.tag) {
        case 0: { // None
            static_array<float,3> v115;
            int v119;
            v119 = 0;
            while (while_method_5(v119)){
                v115[v119] = 0.0f;
                v119 += 1 ;
            }
            static_array<float,3> v121;
            int v125;
            v125 = 0;
            while (while_method_5(v125)){
                v121[v125] = 0.0f;
                v125 += 1 ;
            }
            v129 = v115; v130 = v121;
            break;
        }
        case 1: { // Some
            static_array<float,3> v113 = v112.case1.v0; static_array<float,3> v114 = v112.case1.v1;
            v129 = v113; v130 = v114;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    int v131;
    v131 = v7[0];
    int v135;
    v135 = v7[1];
    bool v139;
    v139 = v131 == v135;
    bool v140;
    v140 = v139 != true;
    Union7 v144;
    if (v140){
        Union2 v141;
        v141 = Union2{Union2_1{}};
        v144 = Union7{Union7_1{v141}};
    } else {
        v144 = Union7{Union7_0{}};
    }
    bool v145;
    v145 = v8 > 0;
    Union7 v149;
    if (v145){
        Union2 v146;
        v146 = Union2{Union2_2{}};
        v149 = Union7{Union7_1{v146}};
    } else {
        v149 = Union7{Union7_0{}};
    }
    bool v152;
    switch (v149.tag) {
        case 0: { // None
            v152 = false;
            break;
        }
        case 1: { // Some
            Union2 v150 = v149.case1.v0;
            v152 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    bool v155;
    switch (v144.tag) {
        case 0: { // None
            v155 = false;
            break;
        }
        case 1: { // Some
            Union2 v153 = v144.case1.v0;
            v155 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    static_array<bool,3> v156;
    v156[0] = true;
    v156[1] = v155;
    v156[2] = v152;
    static_array<float,3> v160;
    v160 = masking_normalize_8(v129, v156);
    float v189;
    switch (v149.tag) {
        case 0: { // None
            v189 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v161 = v149.case1.v0;
            float v162;
            v162 = v160[2];
            static_array<Tuple2,2> & v166 = v2.v0;
            float v167; float v168;
            Tuple2 tmp57 = v166[v6];
            v167 = tmp57.v0; v168 = tmp57.v1;
            static_array<Tuple2,2> & v175 = v2.v0;
            float v176;
            v176 = log(v162);
            float v177;
            v177 = v176 + v167;
            v175[v6] = Tuple2{v177, v168};
            static_array_list<Union0,32> & v178 = v1.v0;
            Union0 v179;
            v179 = Union0{Union0_2{v6, v161}};
            v178.push(v179);
            Union5 v180;
            v180 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v161}};
            float v181;
            v181 = loop_17(v1, v9, v0, v10, v2, v11, v180);
            static_array_list<Union0,32> & v182 = v1.v0;
            Union0 v183;
            v183 = v182.pop();
            static_array<Tuple2,2> & v184 = v2.v0;
            v184[v6] = Tuple2{v167, v168};
            bool v185;
            v185 = v6 == 0;
            if (v185){
                v189 = v181;
            } else {
                float v186;
                v186 = -v181;
                v189 = v186;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v218;
    switch (v144.tag) {
        case 0: { // None
            v218 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v190 = v144.case1.v0;
            float v191;
            v191 = v160[1];
            static_array<Tuple2,2> & v195 = v2.v0;
            float v196; float v197;
            Tuple2 tmp58 = v195[v6];
            v196 = tmp58.v0; v197 = tmp58.v1;
            static_array<Tuple2,2> & v204 = v2.v0;
            float v205;
            v205 = log(v191);
            float v206;
            v206 = v205 + v196;
            v204[v6] = Tuple2{v206, v197};
            static_array_list<Union0,32> & v207 = v1.v0;
            Union0 v208;
            v208 = Union0{Union0_2{v6, v190}};
            v207.push(v208);
            Union5 v209;
            v209 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v190}};
            float v210;
            v210 = loop_17(v1, v9, v0, v10, v2, v11, v209);
            static_array_list<Union0,32> & v211 = v1.v0;
            Union0 v212;
            v212 = v211.pop();
            static_array<Tuple2,2> & v213 = v2.v0;
            v213[v6] = Tuple2{v196, v197};
            bool v214;
            v214 = v6 == 0;
            if (v214){
                v218 = v210;
            } else {
                float v215;
                v215 = -v210;
                v218 = v215;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v219;
    v219 = v160[0];
    static_array<Tuple2,2> & v223 = v2.v0;
    float v224; float v225;
    Tuple2 tmp59 = v223[v6];
    v224 = tmp59.v0; v225 = tmp59.v1;
    static_array<Tuple2,2> & v232 = v2.v0;
    float v233;
    v233 = log(v219);
    float v234;
    v234 = v233 + v224;
    v232[v6] = Tuple2{v234, v225};
    static_array_list<Union0,32> & v235 = v1.v0;
    Union2 v236;
    v236 = Union2{Union2_0{}};
    Union0 v237;
    v237 = Union0{Union0_2{v6, v236}};
    v235.push(v237);
    Union2 v238;
    v238 = Union2{Union2_0{}};
    Union5 v239;
    v239 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v238}};
    float v240;
    v240 = loop_17(v1, v9, v0, v10, v2, v11, v239);
    static_array_list<Union0,32> & v241 = v1.v0;
    Union0 v242;
    v242 = v241.pop();
    static_array<Tuple2,2> & v243 = v2.v0;
    v243[v6] = Tuple2{v224, v225};
    bool v244;
    v244 = v6 == 0;
    float v246;
    if (v244){
        v246 = v240;
    } else {
        float v245;
        v245 = -v240;
        v246 = v245;
    }
    static_array<float,3> v247;
    v247[0] = v246;
    v247[1] = v218;
    v247[2] = v189;
    int v251; float v252;
    Tuple3 tmp60 = Tuple3{0, 0.0f};
    v251 = tmp60.v0; v252 = tmp60.v1;
    while (while_method_5(v251)){
        float v254;
        v254 = v247[v251];
        float v258;
        v258 = v160[v251];
        float v262;
        v262 = v254 * v258;
        float v263;
        v263 = v252 + v262;
        v252 = v263;
        v251 += 1 ;
    }
    return v252;
}
float body_16(StackRefs3 & v0, StackRefs0 & v1, xso::rng & v2, StackMut0 & v3, StackRefs4 & v4, Union3 v5){
    StackMut1 v6{0.0f};
    switch (v5.tag) {
        case 0: { // ChanceCommunityCard
            Union4 v89 = v5.case0.v0; bool v90 = v5.case0.v1; static_array<Union1,2> v91 = v5.case0.v2; int v92 = v5.case0.v3; static_array<int,2> v93 = v5.case0.v4; int v94 = v5.case0.v5;
            int v95; float v96; float v97;
            Tuple7 tmp34 = Tuple7{0, 0.0f, 0.0f};
            v95 = tmp34.v0; v96 = tmp34.v1; v97 = tmp34.v2;
            while (while_method_9(v95)){
                unsigned int v99 = v3.v0;
                unsigned int v100;
                v100 = 1u << v95;
                unsigned int v101;
                v101 = v99 & v100;
                bool v102;
                v102 = v101 == 0u;
                bool v103;
                v103 = v102 != true;
                float v135; float v136;
                if (v103){
                    unsigned int v104 = v3.v0;
                    unsigned int v105;
                    v105 = v104 ^ v100;
                    v3.v0 = v105;
                    bool v106;
                    v106 = 0 == v95;
                    Union1 v124;
                    if (v106){
                        v124 = Union1{Union1_1{}};
                    } else {
                        bool v108;
                        v108 = 1 == v95;
                        if (v108){
                            v124 = Union1{Union1_1{}};
                        } else {
                            bool v110;
                            v110 = 2 == v95;
                            if (v110){
                                v124 = Union1{Union1_2{}};
                            } else {
                                bool v112;
                                v112 = 3 == v95;
                                if (v112){
                                    v124 = Union1{Union1_2{}};
                                } else {
                                    bool v114;
                                    v114 = 4 == v95;
                                    if (v114){
                                        v124 = Union1{Union1_0{}};
                                    } else {
                                        bool v116;
                                        v116 = 5 == v95;
                                        if (v116){
                                            v124 = Union1{Union1_0{}};
                                        } else {
                                            printf("%s\n", "Invalid int in int_to_card.");
                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    static_array_list<Union0,32> & v125 = v0.v0;
                    Union0 v126;
                    v126 = Union0{Union0_0{v124}};
                    v125.push(v126);
                    Union5 v127;
                    v127 = Union5{Union5_0{v89, v90, v91, v92, v93, v94, v124}};
                    float v128;
                    v128 = loop_17(v0, v1, v2, v3, v4, v6, v127);
                    static_array_list<Union0,32> & v129 = v0.v0;
                    Union0 v130;
                    v130 = v129.pop();
                    unsigned int v131 = v3.v0;
                    unsigned int v132;
                    v132 = v131 ^ v100;
                    v3.v0 = v132;
                    float v133;
                    v133 = v96 + v128;
                    float v134;
                    v134 = v97 + 1.0f;
                    v135 = v133; v136 = v134;
                } else {
                    v135 = v96; v136 = v97;
                }
                v96 = v135;
                v97 = v136;
                v95 += 1 ;
            }
            bool v137;
            v137 = v97 == 0.0f;
            bool v138;
            v138 = v137 != true;
            if (v138){
                float v139;
                v139 = v96 / v97;
                return v139;
            } else {
                return 0.0f;
            }
            break;
        }
        case 1: { // ChanceInit
            int v141; float v142; float v143;
            Tuple7 tmp39 = Tuple7{0, 0.0f, 0.0f};
            v141 = tmp39.v0; v142 = tmp39.v1; v143 = tmp39.v2;
            while (while_method_9(v141)){
                unsigned int v145 = v3.v0;
                unsigned int v146;
                v146 = 1u << v141;
                unsigned int v147;
                v147 = v145 & v146;
                bool v148;
                v148 = v147 == 0u;
                bool v149;
                v149 = v148 != true;
                float v225; float v226;
                if (v149){
                    unsigned int v150 = v3.v0;
                    unsigned int v151;
                    v151 = v150 ^ v146;
                    v3.v0 = v151;
                    bool v152;
                    v152 = 0 == v141;
                    Union1 v170;
                    if (v152){
                        v170 = Union1{Union1_1{}};
                    } else {
                        bool v154;
                        v154 = 1 == v141;
                        if (v154){
                            v170 = Union1{Union1_1{}};
                        } else {
                            bool v156;
                            v156 = 2 == v141;
                            if (v156){
                                v170 = Union1{Union1_2{}};
                            } else {
                                bool v158;
                                v158 = 3 == v141;
                                if (v158){
                                    v170 = Union1{Union1_2{}};
                                } else {
                                    bool v160;
                                    v160 = 4 == v141;
                                    if (v160){
                                        v170 = Union1{Union1_0{}};
                                    } else {
                                        bool v162;
                                        v162 = 5 == v141;
                                        if (v162){
                                            v170 = Union1{Union1_0{}};
                                        } else {
                                            printf("%s\n", "Invalid int in int_to_card.");
                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    int v171; float v172; float v173;
                    Tuple7 tmp40 = Tuple7{0, 0.0f, 0.0f};
                    v171 = tmp40.v0; v172 = tmp40.v1; v173 = tmp40.v2;
                    while (while_method_9(v171)){
                        unsigned int v175 = v3.v0;
                        unsigned int v176;
                        v176 = 1u << v171;
                        unsigned int v177;
                        v177 = v175 & v176;
                        bool v178;
                        v178 = v177 == 0u;
                        bool v179;
                        v179 = v178 != true;
                        float v215; float v216;
                        if (v179){
                            unsigned int v180 = v3.v0;
                            unsigned int v181;
                            v181 = v180 ^ v176;
                            v3.v0 = v181;
                            bool v182;
                            v182 = 0 == v171;
                            Union1 v200;
                            if (v182){
                                v200 = Union1{Union1_1{}};
                            } else {
                                bool v184;
                                v184 = 1 == v171;
                                if (v184){
                                    v200 = Union1{Union1_1{}};
                                } else {
                                    bool v186;
                                    v186 = 2 == v171;
                                    if (v186){
                                        v200 = Union1{Union1_2{}};
                                    } else {
                                        bool v188;
                                        v188 = 3 == v171;
                                        if (v188){
                                            v200 = Union1{Union1_2{}};
                                        } else {
                                            bool v190;
                                            v190 = 4 == v171;
                                            if (v190){
                                                v200 = Union1{Union1_0{}};
                                            } else {
                                                bool v192;
                                                v192 = 5 == v171;
                                                if (v192){
                                                    v200 = Union1{Union1_0{}};
                                                } else {
                                                    printf("%s\n", "Invalid int in int_to_card.");
                                                    exit(-1);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            static_array_list<Union0,32> & v201 = v0.v0;
                            Union0 v202;
                            v202 = Union0{Union0_3{0, v170}};
                            v201.push(v202);
                            static_array_list<Union0,32> & v203 = v0.v0;
                            Union0 v204;
                            v204 = Union0{Union0_3{1, v200}};
                            v203.push(v204);
                            Union5 v205;
                            v205 = Union5{Union5_1{v170, v200}};
                            float v206;
                            v206 = loop_17(v0, v1, v2, v3, v4, v6, v205);
                            static_array_list<Union0,32> & v207 = v0.v0;
                            Union0 v208;
                            v208 = v207.pop();
                            static_array_list<Union0,32> & v209 = v0.v0;
                            Union0 v210;
                            v210 = v209.pop();
                            unsigned int v211 = v3.v0;
                            unsigned int v212;
                            v212 = v211 ^ v176;
                            v3.v0 = v212;
                            float v213;
                            v213 = v172 + v206;
                            float v214;
                            v214 = v173 + 1.0f;
                            v215 = v213; v216 = v214;
                        } else {
                            v215 = v172; v216 = v173;
                        }
                        v172 = v215;
                        v173 = v216;
                        v171 += 1 ;
                    }
                    bool v217;
                    v217 = v173 == 0.0f;
                    bool v218;
                    v218 = v217 != true;
                    float v220;
                    if (v218){
                        float v219;
                        v219 = v172 / v173;
                        v220 = v219;
                    } else {
                        v220 = 0.0f;
                    }
                    unsigned int v221 = v3.v0;
                    unsigned int v222;
                    v222 = v221 ^ v146;
                    v3.v0 = v222;
                    float v223;
                    v223 = v142 + v220;
                    float v224;
                    v224 = v143 + 1.0f;
                    v225 = v223; v226 = v224;
                } else {
                    v225 = v142; v226 = v143;
                }
                v142 = v225;
                v143 = v226;
                v141 += 1 ;
            }
            bool v227;
            v227 = v143 == 0.0f;
            bool v228;
            v228 = v227 != true;
            if (v228){
                float v229;
                v229 = v142 / v143;
                return v229;
            } else {
                return 0.0f;
            }
            break;
        }
        case 2: { // Round
            Union4 v59 = v5.case2.v0; bool v60 = v5.case2.v1; static_array<Union1,2> v61 = v5.case2.v2; int v62 = v5.case2.v3; static_array<int,2> v63 = v5.case2.v4; int v64 = v5.case2.v5;
            bool v65;
            v65 = v62 == 0;
            float v71;
            if (v65){
                v71 = method_18(v2, v0, v4, v59, v60, v61, v62, v63, v64, v1, v3, v6);
            } else {
                bool v67;
                v67 = v62 == 1;
                if (v67){
                    v71 = method_19(v2, v0, v4, v59, v60, v61, v62, v63, v64, v1, v3, v6);
                } else {
                    printf("%s\n", "Mode index is out of bounds.");
                    exit(-1);
                }
            }
            float v73;
            if (v65){
                v73 = v71;
            } else {
                float v72;
                v72 = -v71;
                v73 = v72;
            }
            v6.v0 = v73;
            Union5 v74;
            v74 = Union5{Union5_3{}};
            return loop_17(v0, v1, v2, v3, v4, v6, v74);
            break;
        }
        case 3: { // RoundWithAction
            Union4 v76 = v5.case3.v0; bool v77 = v5.case3.v1; static_array<Union1,2> v78 = v5.case3.v2; int v79 = v5.case3.v3; static_array<int,2> v80 = v5.case3.v4; int v81 = v5.case3.v5; Union2 v82 = v5.case3.v6;
            static_array_list<Union0,32> & v83 = v0.v0;
            Union0 v84;
            v84 = Union0{Union0_2{v79, v82}};
            v83.push(v84);
            Union5 v85;
            v85 = Union5{Union5_2{v76, v77, v78, v79, v80, v81, v82}};
            float v86;
            v86 = loop_17(v0, v1, v2, v3, v4, v6, v85);
            static_array_list<Union0,32> & v87 = v0.v0;
            Union0 v88;
            v88 = v87.pop();
            return v86;
            break;
        }
        case 4: { // TerminalCall
            Union4 v29 = v5.case4.v0; bool v30 = v5.case4.v1; static_array<Union1,2> v31 = v5.case4.v2; int v32 = v5.case4.v3; static_array<int,2> v33 = v5.case4.v4; int v34 = v5.case4.v5;
            int v35;
            v35 = v33[v32];
            Union9 v39;
            v39 = compare_hands_12(v29, v30, v31, v32, v33, v34);
            int v44; int v45;
            switch (v39.tag) {
                case 0: { // Eq
                    v44 = 0; v45 = -1;
                    break;
                }
                case 1: { // Gt
                    v44 = v35; v45 = 0;
                    break;
                }
                case 2: { // Lt
                    v44 = v35; v45 = 1;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v46;
            v46 = -v45;
            bool v47;
            v47 = v45 >= v46;
            int v48;
            if (v47){
                v48 = v45;
            } else {
                v48 = v46;
            }
            float v49;
            v49 = (float)v44;
            bool v50;
            v50 = v48 == 0;
            float v52;
            if (v50){
                v52 = v49;
            } else {
                float v51;
                v51 = -v49;
                v52 = v51;
            }
            v6.v0 = v52;
            static_array_list<Union0,32> & v53 = v0.v0;
            Union0 v54;
            v54 = Union0{Union0_4{v31, v44, v45}};
            v53.push(v54);
            Union5 v55;
            v55 = Union5{Union5_3{}};
            float v56;
            v56 = loop_17(v0, v1, v2, v3, v4, v6, v55);
            static_array_list<Union0,32> & v57 = v0.v0;
            Union0 v58;
            v58 = v57.pop();
            return v56;
            break;
        }
        case 5: { // TerminalFold
            Union4 v7 = v5.case5.v0; bool v8 = v5.case5.v1; static_array<Union1,2> v9 = v5.case5.v2; int v10 = v5.case5.v3; static_array<int,2> v11 = v5.case5.v4; int v12 = v5.case5.v5;
            int v13;
            v13 = v11[v10];
            int v17;
            v17 = -v13;
            float v18;
            v18 = (float)v17;
            bool v19;
            v19 = v10 == 0;
            float v21;
            if (v19){
                v21 = v18;
            } else {
                float v20;
                v20 = -v18;
                v21 = v20;
            }
            v6.v0 = v21;
            int v22;
            v22 = v10 ^ 1;
            static_array_list<Union0,32> & v23 = v0.v0;
            Union0 v24;
            v24 = Union0{Union0_4{v9, v13, v22}};
            v23.push(v24);
            Union5 v25;
            v25 = Union5{Union5_3{}};
            float v26;
            v26 = loop_17(v0, v1, v2, v3, v4, v6, v25);
            static_array_list<Union0,32> & v27 = v0.v0;
            Union0 v28;
            v28 = v27.pop();
            return v26;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
int main() {
    int v0;
    v0 = 0;
    while (while_method_0(v0)){
        Fun0 v2 = FunPointerMethod0;
        Fun1 v3 = FunPointerMethod1;
        std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> v4(512, v2, v3);
        StackRefs0 v5{v4};
        std::unordered_map<Tuple0, static_array<Tuple2,3>, Fun0, Fun1> v6(512, v2, v3);
        StackRefs1 v7{v6};
        Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> v8(16384, 1312);
        Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> v9(16384, 12);
        int v10;
        v10 = 0;
        StackRefs2 v11{v10, v8, v9};
        Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v12 = v11.v1;
        v12.setZero();
        Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v13 = v11.v2;
        v13.setZero();
        v11.v0 = 0;
        xso::rng v14;
        StackMut0 v15{63u};
        static_array_list<Union0,32> v16;
        v16 = static_array_list<Union0,32>{};
        StackRefs3 v20{v16};
        static_array<Tuple2,2> v21;
        int v25;
        v25 = 0;
        while (while_method_2(v25)){
            v21[v25] = Tuple2{0.0f, 0.0f};
            v25 += 1 ;
        }
        StackRefs4 v27{v21};
        int v28; float v29;
        Tuple3 tmp0 = Tuple3{0, 0.0f};
        v28 = tmp0.v0; v29 = tmp0.v1;
        while (while_method_3(v28)){
            int v31;
            v31 = v28 % 40000;
            bool v32;
            v32 = v31 == 0;
            if (v32){
                printf("{%s = %d; %s = %d}\n","i", v28, "nearTo", 1000000);
                fflush(stdout);
            } else {
            }
            Union3 v38;
            v38 = Union3{Union3_1{}};
            float v39;
            v39 = body_0(v20, v7, v5, v14, v15, v27, v38);
            v29 = v39;
            v28 += 1 ;
        }
        xso::rng v40;
        StackMut0 v41{63u};
        static_array_list<Union0,32> v42;
        v42 = static_array_list<Union0,32>{};
        StackRefs3 v46{v42};
        static_array<Tuple2,2> v47;
        int v51;
        v51 = 0;
        while (while_method_2(v51)){
            v47[v51] = Tuple2{0.0f, 0.0f};
            v51 += 1 ;
        }
        StackRefs4 v53{v47};
        int v54; float v55;
        Tuple3 tmp33 = Tuple3{0, 0.0f};
        v54 = tmp33.v0; v55 = tmp33.v1;
        while (while_method_8(v54)){
            int v57;
            v57 = v54 % 4;
            bool v58;
            v58 = v57 == 0;
            if (v58){
                printf("{%s = %d; %s = %d}\n","i", v54, "nearTo", 100);
                fflush(stdout);
            } else {
            }
            Union3 v64;
            v64 = Union3{Union3_1{}};
            float v65;
            v65 = body_16(v46, v5, v40, v41, v53, v64);
            v55 = v65;
            v54 += 1 ;
        }
        printf("{%s = %f}\n","reward_for_pl0", v55);
        fflush(stdout);
        v0 += 1 ;
    }
    return 0;
}
