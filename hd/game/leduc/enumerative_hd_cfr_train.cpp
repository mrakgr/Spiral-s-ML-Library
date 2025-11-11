#include "enumerative_hd_cfr_train.hpp"
unsigned long long FunPointerMethod0(Tuple0 tup0){
    unsigned long long v0 = tup0.v0; static_array_list<Union0,32> v1 = tup0.v1;
    return v0;
}
inline bool while_method_0(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
inline bool while_method_1(int v0){
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
            while (while_method_0(v9, v10)){
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
                        while (while_method_1(v44)){
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
inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 50;
    return v1;
}
inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 6;
    return v1;
}
float loop_1(StackRefs3 & v0, StackRefs0 & v1, StackRefs2 & v2, xso::rng & v3, StackMut0 & v4, StackRefs4 & v5, StackMut1 & v6, Union5 v7){
    switch (v7.tag) {
        case 0: { // T_game_chance_community_card
            Union4 v9 = v7.case0.v0; bool v10 = v7.case0.v1; static_array<Union1,2> v11 = v7.case0.v2; int v12 = v7.case0.v3; static_array<int,2> v13 = v7.case0.v4; int v14 = v7.case0.v5; Union1 v15 = v7.case0.v6;
            int v16;
            v16 = 2;
            int v17; int v18;
            Tuple5 tmp2 = Tuple5{0, 0};
            v17 = tmp2.v0; v18 = tmp2.v1;
            while (while_method_1(v17)){
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
            while (while_method_1(v30)){
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
                                while (while_method_1(v126)){
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
                                while (while_method_1(v139)){
                                    v135[v139] = v127;
                                    v139 += 1 ;
                                }
                                static_array<int,2> v141;
                                int v145;
                                v145 = 0;
                                while (while_method_1(v145)){
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
                                while (while_method_1(v64)){
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
                                while (while_method_1(v77)){
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
                                while (while_method_1(v84)){
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
                                while (while_method_1(v97)){
                                    v93[v97] = v85;
                                    v97 += 1 ;
                                }
                                static_array<int,2> v99;
                                int v103;
                                v103 = 0;
                                while (while_method_1(v103)){
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
inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
static_array<float,3> relu_4(static_array<float,3> v0){
    static_array<float,3> v1;
    int v5;
    v5 = 0;
    while (while_method_4(v5)){
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
static_array<float,3> masking_normalize_5(static_array<float,3> v0, static_array<bool,3> v1){
    int v2; float v3;
    Tuple3 tmp12 = Tuple3{0, 0.0f};
    v2 = tmp12.v0; v3 = tmp12.v1;
    while (while_method_4(v2)){
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
    while (while_method_4(v16)){
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
    Tuple3 tmp13 = Tuple3{0, 0.0f};
    v27 = tmp13.v0; v28 = tmp13.v1;
    while (while_method_4(v27)){
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
    while (while_method_4(v39)){
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
static_array<float,3> regret_match_3(static_array<float,3> v0, static_array<bool,3> v1){
    static_array<float,3> v2;
    v2 = relu_4(v0);
    return masking_normalize_5(v2, v1);
}
float method_2(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs0 & v9, StackRefs2 & v10, StackMut0 & v11, StackMut1 & v12){
    int v13;
    v13 = v7[0];
    int v17;
    v17 = v7[1];
    bool v21;
    v21 = v13 == v17;
    bool v22;
    v22 = v21 != true;
    Union6 v26;
    if (v22){
        Union2 v23;
        v23 = Union2{Union2_1{}};
        v26 = Union6{Union6_1{v23}};
    } else {
        v26 = Union6{Union6_0{}};
    }
    bool v27;
    v27 = v8 > 0;
    Union6 v31;
    if (v27){
        Union2 v28;
        v28 = Union2{Union2_2{}};
        v31 = Union6{Union6_1{v28}};
    } else {
        v31 = Union6{Union6_0{}};
    }
    bool v34;
    switch (v31.tag) {
        case 0: { // None
            v34 = false;
            break;
        }
        case 1: { // Some
            Union2 v32 = v31.case1.v0;
            v34 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    bool v37;
    switch (v26.tag) {
        case 0: { // None
            v37 = false;
            break;
        }
        case 1: { // Some
            Union2 v35 = v26.case1.v0;
            v37 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    static_array<bool,3> v38;
    v38[0] = true;
    v38[1] = v37;
    v38[2] = v34;
    std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> & v42 = v9.v0;
    static_array_list<Union0,32> & v43 = v1.v0;
    static_array_list<Union0,32> & v44 = v1.v0;
    int v45;
    v45 = v44.length;
    bool v46;
    v46 = 32 >= v45;
    bool v47;
    v47 = v46 == false;
    if (v47){
        assert("The type level dimension has to equal the value passed at runtime into create." && v46);
    } else {
    }
    static_array_list<Union0,32> v49;
    v49 = static_array_list<Union0,32>{};
    v49.unsafe_set_length(v45);
    int v53; int v54;
    Tuple5 tmp8 = Tuple5{0, 0};
    v53 = tmp8.v0; v54 = tmp8.v1;
    while (while_method_0(v45, v53)){
        Union0 v56;
        v56 = v44[v53];
        bool v63;
        switch (v56.tag) {
            case 3: { // PlayerGotCard
                int v60 = v56.case3.v0; Union1 v61 = v56.case3.v1;
                bool v62;
                v62 = v60 == v6;
                v63 = v62;
                break;
            }
            default: {
                v63 = true;
            }
        }
        int v65;
        if (v63){
            v49[v54] = v56;
            int v64;
            v64 = v54 + 1;
            v65 = v64;
        } else {
            v65 = v54;
        }
        v54 = v65;
        v53 += 1 ;
    }
    bool v66;
    v66 = 32 >= v54;
    bool v67;
    v67 = v66 == false;
    if (v67){
        assert("The type level dimension has to equal the value passed at runtime into create." && v66);
    } else {
    }
    static_array_list<Union0,32> v69;
    v69 = static_array_list<Union0,32>{};
    v69.unsafe_set_length(v54);
    int v73;
    v73 = 0;
    while (while_method_0(v54, v73)){
        Union0 v75;
        v75 = v49[v73];
        v69[v73] = v75;
        v73 += 1 ;
    }
    int v79;
    v79 = v69.length;
    int v80; unsigned long long v81; unsigned long long v82;
    Tuple6 tmp9 = Tuple6{0, 0ull, 1ull};
    v80 = tmp9.v0; v81 = tmp9.v1; v82 = tmp9.v2;
    while (while_method_0(v79, v80)){
        Union0 v84;
        v84 = v69[v80];
        unsigned long long v132;
        switch (v84.tag) {
            case 0: { // CommunityCardIs
                Union1 v88 = v84.case0.v0;
                unsigned long long v89;
                switch (v88.tag) {
                    case 0: { // Jack
                        v89 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v89 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v89 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v90;
                v90 = 9223372036854775807ull + v89;
                unsigned long long v91;
                v91 = v90 * 9973ull;
                v132 = v91;
                break;
            }
            case 1: { // Hidden
                v132 = 18446744073709531670ull;
                break;
            }
            case 2: { // PlayerAction
                int v92 = v84.case2.v0; Union2 v93 = v84.case2.v1;
                unsigned long long v94;
                v94 = std::hash<int>()(v92);
                unsigned long long v95;
                v95 = v94 * 9973ull;
                unsigned long long v96;
                switch (v93.tag) {
                    case 0: { // Call
                        v96 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v96 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v96 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v97;
                v97 = v95 + v96;
                unsigned long long v98;
                v98 = 9223372036854775807ull + v97;
                unsigned long long v99;
                v99 = v98 * 9973ull;
                unsigned long long v100;
                v100 = v99 * 3ull;
                v132 = v100;
                break;
            }
            case 3: { // PlayerGotCard
                int v101 = v84.case3.v0; Union1 v102 = v84.case3.v1;
                unsigned long long v103;
                v103 = std::hash<int>()(v101);
                unsigned long long v104;
                v104 = v103 * 9973ull;
                unsigned long long v105;
                switch (v102.tag) {
                    case 0: { // Jack
                        v105 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v105 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v105 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v106;
                v106 = v104 + v105;
                unsigned long long v107;
                v107 = 9223372036854775807ull + v106;
                unsigned long long v108;
                v108 = v107 * 9973ull;
                unsigned long long v109;
                v109 = v108 * 4ull;
                v132 = v109;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v110 = v84.case4.v0; int v111 = v84.case4.v1; int v112 = v84.case4.v2;
                unsigned long long v113;
                v113 = std::hash<int>()(v112);
                unsigned long long v114;
                v114 = std::hash<int>()(v111);
                unsigned long long v115;
                v115 = v114 * 9973ull;
                unsigned long long v116;
                v116 = v113 + v115;
                int v117; unsigned long long v118; unsigned long long v119;
                Tuple6 tmp10 = Tuple6{0, 0ull, 1ull};
                v117 = tmp10.v0; v118 = tmp10.v1; v119 = tmp10.v2;
                while (while_method_1(v117)){
                    Union1 v121;
                    v121 = v110[v117];
                    unsigned long long v125;
                    switch (v121.tag) {
                        case 0: { // Jack
                            v125 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v125 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v125 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v126;
                    v126 = v125 * v119;
                    unsigned long long v127;
                    v127 = v118 + v126;
                    unsigned long long v128;
                    v128 = v119 * 9973ull;
                    v118 = v127;
                    v119 = v128;
                    v117 += 1 ;
                }
                unsigned long long v129;
                v129 = 9223372036854775807ull + v116;
                unsigned long long v130;
                v130 = v129 * 9973ull;
                unsigned long long v131;
                v131 = v130 * 5ull;
                v132 = v131;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v133;
        v133 = v132 * v82;
        unsigned long long v134;
        v134 = v81 + v133;
        unsigned long long v135;
        v135 = v82 * 9973ull;
        v81 = v134;
        v82 = v135;
        v80 += 1 ;
    }
    auto v136 = v42.find(Tuple0{0ull, v69});
    bool v137;
    v137 = v136 != v42.end();
    Union7 v142;
    if (v137){
        static_array<float,3> v138; static_array<float,3> v139;
        Tuple1 tmp11 = v136->second;
        v138 = tmp11.v0; v139 = tmp11.v1;
        v142 = Union7{Union7_1{v138, v139}};
    } else {
        v142 = Union7{Union7_0{}};
    }
    static_array<float,3> v159; static_array<float,3> v160;
    switch (v142.tag) {
        case 0: { // None
            static_array<float,3> v145;
            int v149;
            v149 = 0;
            while (while_method_4(v149)){
                v145[v149] = 0.0f;
                v149 += 1 ;
            }
            static_array<float,3> v151;
            int v155;
            v155 = 0;
            while (while_method_4(v155)){
                v151[v155] = 0.0f;
                v155 += 1 ;
            }
            v159 = v145; v160 = v151;
            break;
        }
        case 1: { // Some
            static_array<float,3> v143 = v142.case1.v0; static_array<float,3> v144 = v142.case1.v1;
            v159 = v143; v160 = v144;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    static_array<float,3> v161;
    v161 = regret_match_3(v160, v38);
    float v190;
    switch (v31.tag) {
        case 0: { // None
            v190 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v162 = v31.case1.v0;
            float v163;
            v163 = v161[2];
            static_array<Tuple2,2> & v167 = v2.v0;
            float v168; float v169;
            Tuple2 tmp14 = v167[v6];
            v168 = tmp14.v0; v169 = tmp14.v1;
            static_array<Tuple2,2> & v176 = v2.v0;
            float v177;
            v177 = log(v163);
            float v178;
            v178 = v177 + v168;
            v176[v6] = Tuple2{v178, v169};
            static_array_list<Union0,32> & v179 = v1.v0;
            Union0 v180;
            v180 = Union0{Union0_2{v6, v162}};
            v179.push(v180);
            Union5 v181;
            v181 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v162}};
            float v182;
            v182 = loop_1(v1, v9, v10, v0, v11, v2, v12, v181);
            static_array_list<Union0,32> & v183 = v1.v0;
            Union0 v184;
            v184 = v183.pop();
            static_array<Tuple2,2> & v185 = v2.v0;
            v185[v6] = Tuple2{v168, v169};
            bool v186;
            v186 = v6 == 0;
            if (v186){
                v190 = v182;
            } else {
                float v187;
                v187 = -v182;
                v190 = v187;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v219;
    switch (v26.tag) {
        case 0: { // None
            v219 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v191 = v26.case1.v0;
            float v192;
            v192 = v161[1];
            static_array<Tuple2,2> & v196 = v2.v0;
            float v197; float v198;
            Tuple2 tmp15 = v196[v6];
            v197 = tmp15.v0; v198 = tmp15.v1;
            static_array<Tuple2,2> & v205 = v2.v0;
            float v206;
            v206 = log(v192);
            float v207;
            v207 = v206 + v197;
            v205[v6] = Tuple2{v207, v198};
            static_array_list<Union0,32> & v208 = v1.v0;
            Union0 v209;
            v209 = Union0{Union0_2{v6, v191}};
            v208.push(v209);
            Union5 v210;
            v210 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v191}};
            float v211;
            v211 = loop_1(v1, v9, v10, v0, v11, v2, v12, v210);
            static_array_list<Union0,32> & v212 = v1.v0;
            Union0 v213;
            v213 = v212.pop();
            static_array<Tuple2,2> & v214 = v2.v0;
            v214[v6] = Tuple2{v197, v198};
            bool v215;
            v215 = v6 == 0;
            if (v215){
                v219 = v211;
            } else {
                float v216;
                v216 = -v211;
                v219 = v216;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v220;
    v220 = v161[0];
    static_array<Tuple2,2> & v224 = v2.v0;
    float v225; float v226;
    Tuple2 tmp16 = v224[v6];
    v225 = tmp16.v0; v226 = tmp16.v1;
    static_array<Tuple2,2> & v233 = v2.v0;
    float v234;
    v234 = log(v220);
    float v235;
    v235 = v234 + v225;
    v233[v6] = Tuple2{v235, v226};
    static_array_list<Union0,32> & v236 = v1.v0;
    Union2 v237;
    v237 = Union2{Union2_0{}};
    Union0 v238;
    v238 = Union0{Union0_2{v6, v237}};
    v236.push(v238);
    Union2 v239;
    v239 = Union2{Union2_0{}};
    Union5 v240;
    v240 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v239}};
    float v241;
    v241 = loop_1(v1, v9, v10, v0, v11, v2, v12, v240);
    static_array_list<Union0,32> & v242 = v1.v0;
    Union0 v243;
    v243 = v242.pop();
    static_array<Tuple2,2> & v244 = v2.v0;
    v244[v6] = Tuple2{v225, v226};
    bool v245;
    v245 = v6 == 0;
    float v247;
    if (v245){
        v247 = v241;
    } else {
        float v246;
        v246 = -v241;
        v247 = v246;
    }
    static_array<float,3> v248;
    v248[0] = v247;
    v248[1] = v219;
    v248[2] = v190;
    int v252; float v253;
    Tuple3 tmp17 = Tuple3{0, 0.0f};
    v252 = tmp17.v0; v253 = tmp17.v1;
    while (while_method_4(v252)){
        float v255;
        v255 = v248[v252];
        float v259;
        v259 = v161[v252];
        float v263;
        v263 = v255 * v259;
        float v264;
        v264 = v253 + v263;
        v253 = v264;
        v252 += 1 ;
    }
    static_array<float,3> v265;
    int v269;
    v269 = 0;
    while (while_method_4(v269)){
        float v271;
        v271 = v159[v269];
        float v275;
        v275 = v161[v269];
        float v279;
        v279 = 0.99609375f * v271;
        float v280;
        v280 = v279 + v275;
        v265[v269] = v280;
        v269 += 1 ;
    }
    static_array<Tuple2,2> & v281 = v2.v0;
    int v282; float v283;
    Tuple3 tmp18 = Tuple3{0, 0.0f};
    v282 = tmp18.v0; v283 = tmp18.v1;
    while (while_method_1(v282)){
        float v285; float v286;
        Tuple2 tmp19 = v281[v282];
        v285 = tmp19.v0; v286 = tmp19.v1;
        bool v293;
        v293 = v282 == v6;
        float v294;
        if (v293){
            v294 = 0.0f;
        } else {
            v294 = v285;
        }
        float v295;
        v295 = v283 + v294;
        float v296;
        v296 = v295 - v286;
        v283 = v296;
        v282 += 1 ;
    }
    float v297;
    v297 = exp(v283);
    static_array<float,3> v298;
    int v302;
    v302 = 0;
    while (while_method_4(v302)){
        float v304;
        v304 = v160[v302];
        float v308;
        v308 = v248[v302];
        float v312;
        v312 = v308 - v253;
        float v313;
        v313 = v297 * v312;
        float v314;
        v314 = v304 + v313;
        bool v315;
        v315 = 0.0f >= v314;
        float v316;
        if (v315){
            v316 = 0.0f;
        } else {
            v316 = v314;
        }
        v298[v302] = v316;
        v302 += 1 ;
    }
    int v317;
    v317 = v69.length;
    int v318; unsigned long long v319; unsigned long long v320;
    Tuple6 tmp20 = Tuple6{0, 0ull, 1ull};
    v318 = tmp20.v0; v319 = tmp20.v1; v320 = tmp20.v2;
    while (while_method_0(v317, v318)){
        Union0 v322;
        v322 = v69[v318];
        unsigned long long v370;
        switch (v322.tag) {
            case 0: { // CommunityCardIs
                Union1 v326 = v322.case0.v0;
                unsigned long long v327;
                switch (v326.tag) {
                    case 0: { // Jack
                        v327 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v327 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v327 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v328;
                v328 = 9223372036854775807ull + v327;
                unsigned long long v329;
                v329 = v328 * 9973ull;
                v370 = v329;
                break;
            }
            case 1: { // Hidden
                v370 = 18446744073709531670ull;
                break;
            }
            case 2: { // PlayerAction
                int v330 = v322.case2.v0; Union2 v331 = v322.case2.v1;
                unsigned long long v332;
                v332 = std::hash<int>()(v330);
                unsigned long long v333;
                v333 = v332 * 9973ull;
                unsigned long long v334;
                switch (v331.tag) {
                    case 0: { // Call
                        v334 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v334 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v334 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v335;
                v335 = v333 + v334;
                unsigned long long v336;
                v336 = 9223372036854775807ull + v335;
                unsigned long long v337;
                v337 = v336 * 9973ull;
                unsigned long long v338;
                v338 = v337 * 3ull;
                v370 = v338;
                break;
            }
            case 3: { // PlayerGotCard
                int v339 = v322.case3.v0; Union1 v340 = v322.case3.v1;
                unsigned long long v341;
                v341 = std::hash<int>()(v339);
                unsigned long long v342;
                v342 = v341 * 9973ull;
                unsigned long long v343;
                switch (v340.tag) {
                    case 0: { // Jack
                        v343 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v343 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v343 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v344;
                v344 = v342 + v343;
                unsigned long long v345;
                v345 = 9223372036854775807ull + v344;
                unsigned long long v346;
                v346 = v345 * 9973ull;
                unsigned long long v347;
                v347 = v346 * 4ull;
                v370 = v347;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v348 = v322.case4.v0; int v349 = v322.case4.v1; int v350 = v322.case4.v2;
                unsigned long long v351;
                v351 = std::hash<int>()(v350);
                unsigned long long v352;
                v352 = std::hash<int>()(v349);
                unsigned long long v353;
                v353 = v352 * 9973ull;
                unsigned long long v354;
                v354 = v351 + v353;
                int v355; unsigned long long v356; unsigned long long v357;
                Tuple6 tmp21 = Tuple6{0, 0ull, 1ull};
                v355 = tmp21.v0; v356 = tmp21.v1; v357 = tmp21.v2;
                while (while_method_1(v355)){
                    Union1 v359;
                    v359 = v348[v355];
                    unsigned long long v363;
                    switch (v359.tag) {
                        case 0: { // Jack
                            v363 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v363 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v363 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v364;
                    v364 = v363 * v357;
                    unsigned long long v365;
                    v365 = v356 + v364;
                    unsigned long long v366;
                    v366 = v357 * 9973ull;
                    v356 = v365;
                    v357 = v366;
                    v355 += 1 ;
                }
                unsigned long long v367;
                v367 = 9223372036854775807ull + v354;
                unsigned long long v368;
                v368 = v367 * 9973ull;
                unsigned long long v369;
                v369 = v368 * 5ull;
                v370 = v369;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v371;
        v371 = v370 * v320;
        unsigned long long v372;
        v372 = v319 + v371;
        unsigned long long v373;
        v373 = v320 * 9973ull;
        v319 = v372;
        v320 = v373;
        v318 += 1 ;
    }
    v42[Tuple0{0ull, v69}] = Tuple1{v265, v298};
    return v253;
}
inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 1312;
    return v1;
}
inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 32;
    return v1;
}
void method_7(float * v0, static_array_list<Union0,32> v1){
    StackMut2 v2{0};
    int v3;
    v3 = v1.length;
    int v4 = v2.v0;
    int v5;
    v5 = v4 + v3;
    v0[v5] = 1.0f;
    int v6 = v2.v0;
    int v7;
    v7 = v6 + 32;
    v2.v0 = v7;
    int v8;
    v8 = v1.length;
    int v9;
    v9 = 0;
    while (while_method_0(v8, v9)){
        Union0 v11;
        v11 = v1[v9];
        int v15 = v2.v0;
        int v16;
        v16 = v15 + 39;
        int v17;
        v17 = v11.tag;
        int v18 = v2.v0;
        int v19;
        v19 = v18 + v17;
        v0[v19] = 1.0f;
        int v20 = v2.v0;
        int v21;
        v21 = v20 + 5;
        v2.v0 = v21;
        switch (v11.tag) {
            case 0: { // CommunityCardIs
                Union1 v22 = v11.case0.v0;
                int v23 = v2.v0;
                v2.v0 = v23;
                int v24 = v2.v0;
                int v25;
                v25 = v24 + 3;
                int v26;
                v26 = v22.tag;
                int v27 = v2.v0;
                int v28;
                v28 = v27 + v26;
                v0[v28] = 1.0f;
                int v29 = v2.v0;
                int v30;
                v30 = v29 + 3;
                v2.v0 = v30;
                switch (v22.tag) {
                    case 0: { // Jack
                        int v31 = v2.v0;
                        v2.v0 = v31;
                        break;
                    }
                    case 1: { // King
                        int v32 = v2.v0;
                        v2.v0 = v32;
                        break;
                    }
                    case 2: { // Queen
                        int v33 = v2.v0;
                        v2.v0 = v33;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                v2.v0 = v25;
                break;
            }
            case 1: { // Hidden
                int v34 = v2.v0;
                int v35;
                v35 = v34 + 3;
                v2.v0 = v35;
                break;
            }
            case 2: { // PlayerAction
                int v36 = v11.case2.v0; Union2 v37 = v11.case2.v1;
                int v38 = v2.v0;
                int v39;
                v39 = v38 + 3;
                v2.v0 = v39;
                bool v40;
                v40 = v36 < 2;
                bool v41;
                v41 = v40 == false;
                if (v41){
                    assert("The input to the pickler must be 0 or positive." && v40);
                } else {
                }
                bool v43;
                v43 = 0 <= v36;
                bool v44;
                v44 = v43 == false;
                if (v44){
                    assert("The input to the pickler must be less than the specified length." && v43);
                } else {
                }
                int v46 = v2.v0;
                int v47;
                v47 = v46 + v36;
                v0[v47] = 1.0f;
                int v48 = v2.v0;
                int v49;
                v49 = v48 + 2;
                v2.v0 = v49;
                int v50 = v2.v0;
                int v51;
                v51 = v50 + 3;
                int v52;
                v52 = v37.tag;
                int v53 = v2.v0;
                int v54;
                v54 = v53 + v52;
                v0[v54] = 1.0f;
                int v55 = v2.v0;
                int v56;
                v56 = v55 + 3;
                v2.v0 = v56;
                switch (v37.tag) {
                    case 0: { // Call
                        int v57 = v2.v0;
                        v2.v0 = v57;
                        break;
                    }
                    case 1: { // Fold
                        int v58 = v2.v0;
                        v2.v0 = v58;
                        break;
                    }
                    case 2: { // Raise
                        int v59 = v2.v0;
                        v2.v0 = v59;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                v2.v0 = v51;
                break;
            }
            case 3: { // PlayerGotCard
                int v60 = v11.case3.v0; Union1 v61 = v11.case3.v1;
                int v62 = v2.v0;
                int v63;
                v63 = v62 + 8;
                v2.v0 = v63;
                bool v64;
                v64 = v60 < 2;
                bool v65;
                v65 = v64 == false;
                if (v65){
                    assert("The input to the pickler must be 0 or positive." && v64);
                } else {
                }
                bool v67;
                v67 = 0 <= v60;
                bool v68;
                v68 = v67 == false;
                if (v68){
                    assert("The input to the pickler must be less than the specified length." && v67);
                } else {
                }
                int v70 = v2.v0;
                int v71;
                v71 = v70 + v60;
                v0[v71] = 1.0f;
                int v72 = v2.v0;
                int v73;
                v73 = v72 + 2;
                v2.v0 = v73;
                int v74 = v2.v0;
                int v75;
                v75 = v74 + 3;
                int v76;
                v76 = v61.tag;
                int v77 = v2.v0;
                int v78;
                v78 = v77 + v76;
                v0[v78] = 1.0f;
                int v79 = v2.v0;
                int v80;
                v80 = v79 + 3;
                v2.v0 = v80;
                switch (v61.tag) {
                    case 0: { // Jack
                        int v81 = v2.v0;
                        v2.v0 = v81;
                        break;
                    }
                    case 1: { // King
                        int v82 = v2.v0;
                        v2.v0 = v82;
                        break;
                    }
                    case 2: { // Queen
                        int v83 = v2.v0;
                        v2.v0 = v83;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                v2.v0 = v75;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v84 = v11.case4.v0; int v85 = v11.case4.v1; int v86 = v11.case4.v2;
                int v87 = v2.v0;
                int v88;
                v88 = v87 + 13;
                v2.v0 = v88;
                int v89;
                v89 = 0;
                while (while_method_1(v89)){
                    Union1 v91;
                    v91 = v84[v89];
                    int v95 = v2.v0;
                    int v96;
                    v96 = v95 + 3;
                    int v97;
                    v97 = v91.tag;
                    int v98 = v2.v0;
                    int v99;
                    v99 = v98 + v97;
                    v0[v99] = 1.0f;
                    int v100 = v2.v0;
                    int v101;
                    v101 = v100 + 3;
                    v2.v0 = v101;
                    switch (v91.tag) {
                        case 0: { // Jack
                            int v102 = v2.v0;
                            v2.v0 = v102;
                            break;
                        }
                        case 1: { // King
                            int v103 = v2.v0;
                            v2.v0 = v103;
                            break;
                        }
                        case 2: { // Queen
                            int v104 = v2.v0;
                            v2.v0 = v104;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    v2.v0 = v96;
                    v89 += 1 ;
                }
                int v105;
                v105 = -1 + v85;
                bool v106;
                v106 = v105 < 13;
                bool v107;
                v107 = v106 == false;
                if (v107){
                    assert("The input to the pickler must be 0 or positive." && v106);
                } else {
                }
                bool v109;
                v109 = 0 <= v105;
                bool v110;
                v110 = v109 == false;
                if (v110){
                    assert("The input to the pickler must be less than the specified length." && v109);
                } else {
                }
                int v112 = v2.v0;
                int v113;
                v113 = v112 + v105;
                v0[v113] = 1.0f;
                int v114 = v2.v0;
                int v115;
                v115 = v114 + 13;
                v2.v0 = v115;
                bool v116;
                v116 = v86 < 2;
                bool v117;
                v117 = v116 == false;
                if (v117){
                    assert("The input to the pickler must be 0 or positive." && v116);
                } else {
                }
                bool v119;
                v119 = 0 <= v86;
                bool v120;
                v120 = v119 == false;
                if (v120){
                    assert("The input to the pickler must be less than the specified length." && v119);
                } else {
                }
                int v122 = v2.v0;
                int v123;
                v123 = v122 + v86;
                v0[v123] = 1.0f;
                int v124 = v2.v0;
                int v125;
                v125 = v124 + 2;
                v2.v0 = v125;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        v2.v0 = v16;
        int v126 = v2.v0;
        int v127;
        v127 = v126 + 1;
        v2.v0 = v127;
        v9 += 1 ;
    }
    int v128;
    v128 = v3;
    while (while_method_6(v128)){
        int v130 = v2.v0;
        int v131;
        v131 = v130 + 39;
        v2.v0 = v131;
        int v132 = v2.v0;
        v0[v132] = 1.0f;
        int v133 = v2.v0;
        int v134;
        v134 = v133 + 1;
        v2.v0 = v134;
        v128 += 1 ;
    }
    return ;
}
void method_8(StackRefs5 & v0){
    af::array & v1 = v0.v0; af::array & v2 = v0.v1; af::array & v3 = v0.v2; af::array & v4 = v0.v3; af::array & v5 = v0.v4;
    af::array v6;
    v6 = af::matmul(af::transpose(v1), v2);
    af::array v7;
    af::max(v7, v3, v6 + af::randu(v6.dims()), 1);
    v4 = af::lookup(v5, v3, 1);
    return ;
}
float method_6(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs0 & v9, StackRefs2 & v10, StackMut0 & v11, StackMut1 & v12){
    int v13;
    v13 = v7[0];
    int v17;
    v17 = v7[1];
    bool v21;
    v21 = v13 == v17;
    bool v22;
    v22 = v21 != true;
    Union6 v26;
    if (v22){
        Union2 v23;
        v23 = Union2{Union2_1{}};
        v26 = Union6{Union6_1{v23}};
    } else {
        v26 = Union6{Union6_0{}};
    }
    bool v27;
    v27 = v8 > 0;
    Union6 v31;
    if (v27){
        Union2 v28;
        v28 = Union2{Union2_2{}};
        v31 = Union6{Union6_1{v28}};
    } else {
        v31 = Union6{Union6_0{}};
    }
    bool v34;
    switch (v31.tag) {
        case 0: { // None
            v34 = false;
            break;
        }
        case 1: { // Some
            Union2 v32 = v31.case1.v0;
            v34 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    bool v37;
    switch (v26.tag) {
        case 0: { // None
            v37 = false;
            break;
        }
        case 1: { // Some
            Union2 v35 = v26.case1.v0;
            v37 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    static_array<bool,3> v38;
    v38[0] = true;
    v38[1] = v37;
    v38[2] = v34;
    static_array_list<Union0,32> & v42 = v1.v0;
    static_array_list<Union0,32> & v43 = v1.v0;
    int v44;
    v44 = v43.length;
    bool v45;
    v45 = 32 >= v44;
    bool v46;
    v46 = v45 == false;
    if (v46){
        assert("The type level dimension has to equal the value passed at runtime into create." && v45);
    } else {
    }
    static_array_list<Union0,32> v48;
    v48 = static_array_list<Union0,32>{};
    v48.unsafe_set_length(v44);
    int v52;
    v52 = 0;
    while (while_method_0(v44, v52)){
        Union0 v54;
        v54 = v43[v52];
        Union0 v64;
        switch (v54.tag) {
            case 3: { // PlayerGotCard
                int v58 = v54.case3.v0; Union1 v59 = v54.case3.v1;
                bool v60;
                v60 = v58 == v6;
                bool v61;
                v61 = v60 != true;
                if (v61){
                    v64 = Union0{Union0_1{}};
                } else {
                    v64 = v54;
                }
                break;
            }
            default: {
                v64 = v54;
            }
        }
        v48[v52] = v64;
        v52 += 1 ;
    }
    float v65[1312];
    int v66;
    v66 = 0;
    while (while_method_5(v66)){
        v65[v66] = 0.0f;
        v66 += 1 ;
    }
    method_7(v65, v48);
    af::array v68(1312, 1, v65);
    af::array v69;
    af::array v70;
    af::array & v71 = v10.v0;
    af::array & v72 = v10.v1;
    StackRefs5 v73{v68, v71, v70, v69, v72};
    method_8(v73);
    unsigned int v74;
    v74 = v70(0);
    int v75;
    v75 = v69.elements();
    float v76[v75];
    v69.host<float>(v76);;
    static_array<float,3> v77;
    int v81;
    v81 = 0;
    while (while_method_4(v81)){
        float v83;
        v83 = v76[v81];
        v77[v81] = v83;
        v81 += 1 ;
    }
    static_array<float,3> v84;
    int v88;
    v88 = 0;
    while (while_method_4(v88)){
        int v90;
        v90 = v88 + 3;
        float v91;
        v91 = v76[v90];
        v84[v88] = v91;
        v88 += 1 ;
    }
    static_array<float,3> v92;
    v92 = regret_match_3(v84, v38);
    float v121;
    switch (v31.tag) {
        case 0: { // None
            v121 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v93 = v31.case1.v0;
            float v94;
            v94 = v92[2];
            static_array<Tuple2,2> & v98 = v2.v0;
            float v99; float v100;
            Tuple2 tmp22 = v98[v6];
            v99 = tmp22.v0; v100 = tmp22.v1;
            static_array<Tuple2,2> & v107 = v2.v0;
            float v108;
            v108 = log(v94);
            float v109;
            v109 = v108 + v99;
            v107[v6] = Tuple2{v109, v100};
            static_array_list<Union0,32> & v110 = v1.v0;
            Union0 v111;
            v111 = Union0{Union0_2{v6, v93}};
            v110.push(v111);
            Union5 v112;
            v112 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v93}};
            float v113;
            v113 = loop_1(v1, v9, v10, v0, v11, v2, v12, v112);
            static_array_list<Union0,32> & v114 = v1.v0;
            Union0 v115;
            v115 = v114.pop();
            static_array<Tuple2,2> & v116 = v2.v0;
            v116[v6] = Tuple2{v99, v100};
            bool v117;
            v117 = v6 == 0;
            if (v117){
                v121 = v113;
            } else {
                float v118;
                v118 = -v113;
                v121 = v118;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v150;
    switch (v26.tag) {
        case 0: { // None
            v150 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v122 = v26.case1.v0;
            float v123;
            v123 = v92[1];
            static_array<Tuple2,2> & v127 = v2.v0;
            float v128; float v129;
            Tuple2 tmp23 = v127[v6];
            v128 = tmp23.v0; v129 = tmp23.v1;
            static_array<Tuple2,2> & v136 = v2.v0;
            float v137;
            v137 = log(v123);
            float v138;
            v138 = v137 + v128;
            v136[v6] = Tuple2{v138, v129};
            static_array_list<Union0,32> & v139 = v1.v0;
            Union0 v140;
            v140 = Union0{Union0_2{v6, v122}};
            v139.push(v140);
            Union5 v141;
            v141 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v122}};
            float v142;
            v142 = loop_1(v1, v9, v10, v0, v11, v2, v12, v141);
            static_array_list<Union0,32> & v143 = v1.v0;
            Union0 v144;
            v144 = v143.pop();
            static_array<Tuple2,2> & v145 = v2.v0;
            v145[v6] = Tuple2{v128, v129};
            bool v146;
            v146 = v6 == 0;
            if (v146){
                v150 = v142;
            } else {
                float v147;
                v147 = -v142;
                v150 = v147;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v151;
    v151 = v92[0];
    static_array<Tuple2,2> & v155 = v2.v0;
    float v156; float v157;
    Tuple2 tmp24 = v155[v6];
    v156 = tmp24.v0; v157 = tmp24.v1;
    static_array<Tuple2,2> & v164 = v2.v0;
    float v165;
    v165 = log(v151);
    float v166;
    v166 = v165 + v156;
    v164[v6] = Tuple2{v166, v157};
    static_array_list<Union0,32> & v167 = v1.v0;
    Union2 v168;
    v168 = Union2{Union2_0{}};
    Union0 v169;
    v169 = Union0{Union0_2{v6, v168}};
    v167.push(v169);
    Union2 v170;
    v170 = Union2{Union2_0{}};
    Union5 v171;
    v171 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v170}};
    float v172;
    v172 = loop_1(v1, v9, v10, v0, v11, v2, v12, v171);
    static_array_list<Union0,32> & v173 = v1.v0;
    Union0 v174;
    v174 = v173.pop();
    static_array<Tuple2,2> & v175 = v2.v0;
    v175[v6] = Tuple2{v156, v157};
    bool v176;
    v176 = v6 == 0;
    float v178;
    if (v176){
        v178 = v172;
    } else {
        float v177;
        v177 = -v172;
        v178 = v177;
    }
    static_array<float,3> v179;
    v179[0] = v178;
    v179[1] = v150;
    v179[2] = v121;
    int v183; float v184;
    Tuple3 tmp25 = Tuple3{0, 0.0f};
    v183 = tmp25.v0; v184 = tmp25.v1;
    while (while_method_4(v183)){
        float v186;
        v186 = v179[v183];
        float v190;
        v190 = v92[v183];
        float v194;
        v194 = v186 * v190;
        float v195;
        v195 = v184 + v194;
        v184 = v195;
        v183 += 1 ;
    }
    static_array<float,3> v196;
    int v200;
    v200 = 0;
    while (while_method_4(v200)){
        float v202;
        v202 = v77[v200];
        float v206;
        v206 = v92[v200];
        float v210;
        v210 = 0.99609375f * v202;
        float v211;
        v211 = v210 + v206;
        v196[v200] = v211;
        v200 += 1 ;
    }
    static_array<Tuple2,2> & v212 = v2.v0;
    int v213; float v214;
    Tuple3 tmp26 = Tuple3{0, 0.0f};
    v213 = tmp26.v0; v214 = tmp26.v1;
    while (while_method_1(v213)){
        float v216; float v217;
        Tuple2 tmp27 = v212[v213];
        v216 = tmp27.v0; v217 = tmp27.v1;
        bool v224;
        v224 = v213 == v6;
        float v225;
        if (v224){
            v225 = 0.0f;
        } else {
            v225 = v216;
        }
        float v226;
        v226 = v214 + v225;
        float v227;
        v227 = v226 - v217;
        v214 = v227;
        v213 += 1 ;
    }
    float v228;
    v228 = exp(v214);
    static_array<float,3> v229;
    int v233;
    v233 = 0;
    while (while_method_4(v233)){
        float v235;
        v235 = v84[v233];
        float v239;
        v239 = v179[v233];
        float v243;
        v243 = v239 - v184;
        float v244;
        v244 = v228 * v243;
        float v245;
        v245 = v235 + v244;
        bool v246;
        v246 = 0.0f >= v245;
        float v247;
        if (v246){
            v247 = 0.0f;
        } else {
            v247 = v245;
        }
        v229[v233] = v247;
        v233 += 1 ;
    }
    af::array & v248 = v10.v0;
    int v249;
    v249 = v248.dims(1);
    af::array & v250 = v10.v0;
    af::array & v251 = v10.v1;
    float v252[1312];
    int v253;
    v253 = 0;
    while (while_method_5(v253)){
        v252[v253] = 0.0f;
        v253 += 1 ;
    }
    method_7(v252, v48);
    af::array v255(1312, 1, v252);
    v250(af::span, v74) = v255;
    float v256[6];
    int v257;
    v257 = 0;
    while (while_method_3(v257)){
        v256[v257] = 0.0f;
        v257 += 1 ;
    }
    int v259;
    v259 = 0;
    while (while_method_4(v259)){
        float v261;
        v261 = v196[v259];
        v256[v259] = v261;
        v259 += 1 ;
    }
    int v265;
    v265 = 0;
    while (while_method_4(v265)){
        int v267;
        v267 = v265 + 3;
        float v268;
        v268 = v229[v265];
        v256[v267] = v268;
        v265 += 1 ;
    }
    af::array v272(6, 1, v256);
    v251(af::span, v74) = v272;
    return v184;
}
int tag_10(Union1 v0){
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
bool is_pair_11(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
Tuple5 order_12(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple5{v1, v0};
    } else {
        return Tuple5{v0, v1};
    }
}
Union8 compare_hands_9(Union4 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            exit(-1);
            break;
        }
        case 1: { // Some
            Union1 v7 = v0.case1.v0;
            int v8;
            v8 = tag_10(v7);
            Union1 v9;
            v9 = v2[0];
            int v13;
            v13 = tag_10(v9);
            Union1 v14;
            v14 = v2[1];
            int v18;
            v18 = tag_10(v14);
            bool v19;
            v19 = is_pair_11(v8, v13);
            bool v20;
            v20 = is_pair_11(v8, v18);
            if (v19){
                if (v20){
                    bool v21;
                    v21 = v13 < v18;
                    if (v21){
                        return Union8{Union8_2{}};
                    } else {
                        bool v23;
                        v23 = v13 > v18;
                        if (v23){
                            return Union8{Union8_1{}};
                        } else {
                            return Union8{Union8_0{}};
                        }
                    }
                } else {
                    return Union8{Union8_1{}};
                }
            } else {
                if (v20){
                    return Union8{Union8_2{}};
                } else {
                    int v31; int v32;
                    Tuple5 tmp28 = order_12(v8, v13);
                    v31 = tmp28.v0; v32 = tmp28.v1;
                    int v33; int v34;
                    Tuple5 tmp29 = order_12(v8, v18);
                    v33 = tmp29.v0; v34 = tmp29.v1;
                    bool v35;
                    v35 = v31 < v33;
                    Union8 v41;
                    if (v35){
                        v41 = Union8{Union8_2{}};
                    } else {
                        bool v37;
                        v37 = v31 > v33;
                        if (v37){
                            v41 = Union8{Union8_1{}};
                        } else {
                            v41 = Union8{Union8_0{}};
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
                            return Union8{Union8_2{}};
                        } else {
                            bool v45;
                            v45 = v32 > v34;
                            if (v45){
                                return Union8{Union8_1{}};
                            } else {
                                return Union8{Union8_0{}};
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
float body_0(StackRefs3 & v0, StackRefs0 & v1, StackRefs2 & v2, xso::rng & v3, StackMut0 & v4, StackRefs4 & v5, Union3 v6){
    StackMut1 v7{0.0f};
    switch (v6.tag) {
        case 0: { // ChanceCommunityCard
            Union4 v90 = v6.case0.v0; bool v91 = v6.case0.v1; static_array<Union1,2> v92 = v6.case0.v2; int v93 = v6.case0.v3; static_array<int,2> v94 = v6.case0.v4; int v95 = v6.case0.v5;
            int v96; float v97; float v98;
            Tuple4 tmp1 = Tuple4{0, 0.0f, 0.0f};
            v96 = tmp1.v0; v97 = tmp1.v1; v98 = tmp1.v2;
            while (while_method_3(v96)){
                unsigned int v100 = v4.v0;
                unsigned int v101;
                v101 = 1u << v96;
                unsigned int v102;
                v102 = v100 & v101;
                bool v103;
                v103 = v102 == 0u;
                bool v104;
                v104 = v103 != true;
                float v136; float v137;
                if (v104){
                    unsigned int v105 = v4.v0;
                    unsigned int v106;
                    v106 = v105 ^ v101;
                    v4.v0 = v106;
                    bool v107;
                    v107 = 0 == v96;
                    Union1 v125;
                    if (v107){
                        v125 = Union1{Union1_1{}};
                    } else {
                        bool v109;
                        v109 = 1 == v96;
                        if (v109){
                            v125 = Union1{Union1_1{}};
                        } else {
                            bool v111;
                            v111 = 2 == v96;
                            if (v111){
                                v125 = Union1{Union1_2{}};
                            } else {
                                bool v113;
                                v113 = 3 == v96;
                                if (v113){
                                    v125 = Union1{Union1_2{}};
                                } else {
                                    bool v115;
                                    v115 = 4 == v96;
                                    if (v115){
                                        v125 = Union1{Union1_0{}};
                                    } else {
                                        bool v117;
                                        v117 = 5 == v96;
                                        if (v117){
                                            v125 = Union1{Union1_0{}};
                                        } else {
                                            printf("%s\n", "Invalid int in int_to_card.");
                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    static_array_list<Union0,32> & v126 = v0.v0;
                    Union0 v127;
                    v127 = Union0{Union0_0{v125}};
                    v126.push(v127);
                    Union5 v128;
                    v128 = Union5{Union5_0{v90, v91, v92, v93, v94, v95, v125}};
                    float v129;
                    v129 = loop_1(v0, v1, v2, v3, v4, v5, v7, v128);
                    static_array_list<Union0,32> & v130 = v0.v0;
                    Union0 v131;
                    v131 = v130.pop();
                    unsigned int v132 = v4.v0;
                    unsigned int v133;
                    v133 = v132 ^ v101;
                    v4.v0 = v133;
                    float v134;
                    v134 = v97 + v129;
                    float v135;
                    v135 = v98 + 1.0f;
                    v136 = v134; v137 = v135;
                } else {
                    v136 = v97; v137 = v98;
                }
                v97 = v136;
                v98 = v137;
                v96 += 1 ;
            }
            bool v138;
            v138 = v98 == 0.0f;
            bool v139;
            v139 = v138 != true;
            if (v139){
                float v140;
                v140 = v97 / v98;
                return v140;
            } else {
                return 0.0f;
            }
            break;
        }
        case 1: { // ChanceInit
            int v142; float v143; float v144;
            Tuple4 tmp6 = Tuple4{0, 0.0f, 0.0f};
            v142 = tmp6.v0; v143 = tmp6.v1; v144 = tmp6.v2;
            while (while_method_3(v142)){
                unsigned int v146 = v4.v0;
                unsigned int v147;
                v147 = 1u << v142;
                unsigned int v148;
                v148 = v146 & v147;
                bool v149;
                v149 = v148 == 0u;
                bool v150;
                v150 = v149 != true;
                float v226; float v227;
                if (v150){
                    unsigned int v151 = v4.v0;
                    unsigned int v152;
                    v152 = v151 ^ v147;
                    v4.v0 = v152;
                    bool v153;
                    v153 = 0 == v142;
                    Union1 v171;
                    if (v153){
                        v171 = Union1{Union1_1{}};
                    } else {
                        bool v155;
                        v155 = 1 == v142;
                        if (v155){
                            v171 = Union1{Union1_1{}};
                        } else {
                            bool v157;
                            v157 = 2 == v142;
                            if (v157){
                                v171 = Union1{Union1_2{}};
                            } else {
                                bool v159;
                                v159 = 3 == v142;
                                if (v159){
                                    v171 = Union1{Union1_2{}};
                                } else {
                                    bool v161;
                                    v161 = 4 == v142;
                                    if (v161){
                                        v171 = Union1{Union1_0{}};
                                    } else {
                                        bool v163;
                                        v163 = 5 == v142;
                                        if (v163){
                                            v171 = Union1{Union1_0{}};
                                        } else {
                                            printf("%s\n", "Invalid int in int_to_card.");
                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    int v172; float v173; float v174;
                    Tuple4 tmp7 = Tuple4{0, 0.0f, 0.0f};
                    v172 = tmp7.v0; v173 = tmp7.v1; v174 = tmp7.v2;
                    while (while_method_3(v172)){
                        unsigned int v176 = v4.v0;
                        unsigned int v177;
                        v177 = 1u << v172;
                        unsigned int v178;
                        v178 = v176 & v177;
                        bool v179;
                        v179 = v178 == 0u;
                        bool v180;
                        v180 = v179 != true;
                        float v216; float v217;
                        if (v180){
                            unsigned int v181 = v4.v0;
                            unsigned int v182;
                            v182 = v181 ^ v177;
                            v4.v0 = v182;
                            bool v183;
                            v183 = 0 == v172;
                            Union1 v201;
                            if (v183){
                                v201 = Union1{Union1_1{}};
                            } else {
                                bool v185;
                                v185 = 1 == v172;
                                if (v185){
                                    v201 = Union1{Union1_1{}};
                                } else {
                                    bool v187;
                                    v187 = 2 == v172;
                                    if (v187){
                                        v201 = Union1{Union1_2{}};
                                    } else {
                                        bool v189;
                                        v189 = 3 == v172;
                                        if (v189){
                                            v201 = Union1{Union1_2{}};
                                        } else {
                                            bool v191;
                                            v191 = 4 == v172;
                                            if (v191){
                                                v201 = Union1{Union1_0{}};
                                            } else {
                                                bool v193;
                                                v193 = 5 == v172;
                                                if (v193){
                                                    v201 = Union1{Union1_0{}};
                                                } else {
                                                    printf("%s\n", "Invalid int in int_to_card.");
                                                    exit(-1);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            static_array_list<Union0,32> & v202 = v0.v0;
                            Union0 v203;
                            v203 = Union0{Union0_3{0, v171}};
                            v202.push(v203);
                            static_array_list<Union0,32> & v204 = v0.v0;
                            Union0 v205;
                            v205 = Union0{Union0_3{1, v201}};
                            v204.push(v205);
                            Union5 v206;
                            v206 = Union5{Union5_1{v171, v201}};
                            float v207;
                            v207 = loop_1(v0, v1, v2, v3, v4, v5, v7, v206);
                            static_array_list<Union0,32> & v208 = v0.v0;
                            Union0 v209;
                            v209 = v208.pop();
                            static_array_list<Union0,32> & v210 = v0.v0;
                            Union0 v211;
                            v211 = v210.pop();
                            unsigned int v212 = v4.v0;
                            unsigned int v213;
                            v213 = v212 ^ v177;
                            v4.v0 = v213;
                            float v214;
                            v214 = v173 + v207;
                            float v215;
                            v215 = v174 + 1.0f;
                            v216 = v214; v217 = v215;
                        } else {
                            v216 = v173; v217 = v174;
                        }
                        v173 = v216;
                        v174 = v217;
                        v172 += 1 ;
                    }
                    bool v218;
                    v218 = v174 == 0.0f;
                    bool v219;
                    v219 = v218 != true;
                    float v221;
                    if (v219){
                        float v220;
                        v220 = v173 / v174;
                        v221 = v220;
                    } else {
                        v221 = 0.0f;
                    }
                    unsigned int v222 = v4.v0;
                    unsigned int v223;
                    v223 = v222 ^ v147;
                    v4.v0 = v223;
                    float v224;
                    v224 = v143 + v221;
                    float v225;
                    v225 = v144 + 1.0f;
                    v226 = v224; v227 = v225;
                } else {
                    v226 = v143; v227 = v144;
                }
                v143 = v226;
                v144 = v227;
                v142 += 1 ;
            }
            bool v228;
            v228 = v144 == 0.0f;
            bool v229;
            v229 = v228 != true;
            if (v229){
                float v230;
                v230 = v143 / v144;
                return v230;
            } else {
                return 0.0f;
            }
            break;
        }
        case 2: { // Round
            Union4 v60 = v6.case2.v0; bool v61 = v6.case2.v1; static_array<Union1,2> v62 = v6.case2.v2; int v63 = v6.case2.v3; static_array<int,2> v64 = v6.case2.v4; int v65 = v6.case2.v5;
            bool v66;
            v66 = v63 == 0;
            float v72;
            if (v66){
                v72 = method_2(v3, v0, v5, v60, v61, v62, v63, v64, v65, v1, v2, v4, v7);
            } else {
                bool v68;
                v68 = v63 == 1;
                if (v68){
                    v72 = method_6(v3, v0, v5, v60, v61, v62, v63, v64, v65, v1, v2, v4, v7);
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
            return loop_1(v0, v1, v2, v3, v4, v5, v7, v75);
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
            v87 = loop_1(v0, v1, v2, v3, v4, v5, v7, v86);
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
            Union8 v40;
            v40 = compare_hands_9(v30, v31, v32, v33, v34, v35);
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
            v57 = loop_1(v0, v1, v2, v3, v4, v5, v7, v56);
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
            v27 = loop_1(v0, v1, v2, v3, v4, v5, v7, v26);
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
inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 10;
    return v1;
}
float loop_14(StackRefs3 & v0, StackRefs0 & v1, StackRefs2 & v2, xso::rng & v3, StackMut0 & v4, StackRefs4 & v5, StackMut1 & v6, Union5 v7){
    switch (v7.tag) {
        case 0: { // T_game_chance_community_card
            Union4 v9 = v7.case0.v0; bool v10 = v7.case0.v1; static_array<Union1,2> v11 = v7.case0.v2; int v12 = v7.case0.v3; static_array<int,2> v13 = v7.case0.v4; int v14 = v7.case0.v5; Union1 v15 = v7.case0.v6;
            int v16;
            v16 = 2;
            int v17; int v18;
            Tuple5 tmp32 = Tuple5{0, 0};
            v17 = tmp32.v0; v18 = tmp32.v1;
            while (while_method_1(v17)){
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
            while (while_method_1(v30)){
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
            return body_13(v0, v1, v2, v3, v4, v5, v35);
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
            return body_13(v0, v1, v2, v3, v4, v5, v51);
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
                                Tuple5 tmp33 = Tuple5{0, 0};
                                v126 = tmp33.v0; v127 = tmp33.v1;
                                while (while_method_1(v126)){
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
                                while (while_method_1(v139)){
                                    v135[v139] = v127;
                                    v139 += 1 ;
                                }
                                static_array<int,2> v141;
                                int v145;
                                v145 = 0;
                                while (while_method_1(v145)){
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
                                Tuple5 tmp34 = Tuple5{0, 0};
                                v64 = tmp34.v0; v65 = tmp34.v1;
                                while (while_method_1(v64)){
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
                                while (while_method_1(v77)){
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
                                Tuple5 tmp35 = Tuple5{0, 0};
                                v84 = tmp35.v0; v85 = tmp35.v1;
                                while (while_method_1(v84)){
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
                                while (while_method_1(v97)){
                                    v93[v97] = v85;
                                    v97 += 1 ;
                                }
                                static_array<int,2> v99;
                                int v103;
                                v103 = 0;
                                while (while_method_1(v103)){
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
            return body_13(v0, v1, v2, v3, v4, v5, v161);
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
float method_15(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs0 & v9, StackRefs2 & v10, StackMut0 & v11, StackMut1 & v12){
    int v13;
    v13 = v7[0];
    int v17;
    v17 = v7[1];
    bool v21;
    v21 = v13 == v17;
    bool v22;
    v22 = v21 != true;
    Union6 v26;
    if (v22){
        Union2 v23;
        v23 = Union2{Union2_1{}};
        v26 = Union6{Union6_1{v23}};
    } else {
        v26 = Union6{Union6_0{}};
    }
    bool v27;
    v27 = v8 > 0;
    Union6 v31;
    if (v27){
        Union2 v28;
        v28 = Union2{Union2_2{}};
        v31 = Union6{Union6_1{v28}};
    } else {
        v31 = Union6{Union6_0{}};
    }
    bool v34;
    switch (v31.tag) {
        case 0: { // None
            v34 = false;
            break;
        }
        case 1: { // Some
            Union2 v32 = v31.case1.v0;
            v34 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    bool v37;
    switch (v26.tag) {
        case 0: { // None
            v37 = false;
            break;
        }
        case 1: { // Some
            Union2 v35 = v26.case1.v0;
            v37 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    static_array<bool,3> v38;
    v38[0] = true;
    v38[1] = v37;
    v38[2] = v34;
    std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> & v42 = v9.v0;
    static_array_list<Union0,32> & v43 = v1.v0;
    static_array_list<Union0,32> & v44 = v1.v0;
    int v45;
    v45 = v44.length;
    bool v46;
    v46 = 32 >= v45;
    bool v47;
    v47 = v46 == false;
    if (v47){
        assert("The type level dimension has to equal the value passed at runtime into create." && v46);
    } else {
    }
    static_array_list<Union0,32> v49;
    v49 = static_array_list<Union0,32>{};
    v49.unsafe_set_length(v45);
    int v53; int v54;
    Tuple5 tmp38 = Tuple5{0, 0};
    v53 = tmp38.v0; v54 = tmp38.v1;
    while (while_method_0(v45, v53)){
        Union0 v56;
        v56 = v44[v53];
        bool v63;
        switch (v56.tag) {
            case 3: { // PlayerGotCard
                int v60 = v56.case3.v0; Union1 v61 = v56.case3.v1;
                bool v62;
                v62 = v60 == v6;
                v63 = v62;
                break;
            }
            default: {
                v63 = true;
            }
        }
        int v65;
        if (v63){
            v49[v54] = v56;
            int v64;
            v64 = v54 + 1;
            v65 = v64;
        } else {
            v65 = v54;
        }
        v54 = v65;
        v53 += 1 ;
    }
    bool v66;
    v66 = 32 >= v54;
    bool v67;
    v67 = v66 == false;
    if (v67){
        assert("The type level dimension has to equal the value passed at runtime into create." && v66);
    } else {
    }
    static_array_list<Union0,32> v69;
    v69 = static_array_list<Union0,32>{};
    v69.unsafe_set_length(v54);
    int v73;
    v73 = 0;
    while (while_method_0(v54, v73)){
        Union0 v75;
        v75 = v49[v73];
        v69[v73] = v75;
        v73 += 1 ;
    }
    int v79;
    v79 = v69.length;
    int v80; unsigned long long v81; unsigned long long v82;
    Tuple6 tmp39 = Tuple6{0, 0ull, 1ull};
    v80 = tmp39.v0; v81 = tmp39.v1; v82 = tmp39.v2;
    while (while_method_0(v79, v80)){
        Union0 v84;
        v84 = v69[v80];
        unsigned long long v132;
        switch (v84.tag) {
            case 0: { // CommunityCardIs
                Union1 v88 = v84.case0.v0;
                unsigned long long v89;
                switch (v88.tag) {
                    case 0: { // Jack
                        v89 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v89 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v89 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v90;
                v90 = 9223372036854775807ull + v89;
                unsigned long long v91;
                v91 = v90 * 9973ull;
                v132 = v91;
                break;
            }
            case 1: { // Hidden
                v132 = 18446744073709531670ull;
                break;
            }
            case 2: { // PlayerAction
                int v92 = v84.case2.v0; Union2 v93 = v84.case2.v1;
                unsigned long long v94;
                v94 = std::hash<int>()(v92);
                unsigned long long v95;
                v95 = v94 * 9973ull;
                unsigned long long v96;
                switch (v93.tag) {
                    case 0: { // Call
                        v96 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v96 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v96 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v97;
                v97 = v95 + v96;
                unsigned long long v98;
                v98 = 9223372036854775807ull + v97;
                unsigned long long v99;
                v99 = v98 * 9973ull;
                unsigned long long v100;
                v100 = v99 * 3ull;
                v132 = v100;
                break;
            }
            case 3: { // PlayerGotCard
                int v101 = v84.case3.v0; Union1 v102 = v84.case3.v1;
                unsigned long long v103;
                v103 = std::hash<int>()(v101);
                unsigned long long v104;
                v104 = v103 * 9973ull;
                unsigned long long v105;
                switch (v102.tag) {
                    case 0: { // Jack
                        v105 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v105 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v105 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v106;
                v106 = v104 + v105;
                unsigned long long v107;
                v107 = 9223372036854775807ull + v106;
                unsigned long long v108;
                v108 = v107 * 9973ull;
                unsigned long long v109;
                v109 = v108 * 4ull;
                v132 = v109;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v110 = v84.case4.v0; int v111 = v84.case4.v1; int v112 = v84.case4.v2;
                unsigned long long v113;
                v113 = std::hash<int>()(v112);
                unsigned long long v114;
                v114 = std::hash<int>()(v111);
                unsigned long long v115;
                v115 = v114 * 9973ull;
                unsigned long long v116;
                v116 = v113 + v115;
                int v117; unsigned long long v118; unsigned long long v119;
                Tuple6 tmp40 = Tuple6{0, 0ull, 1ull};
                v117 = tmp40.v0; v118 = tmp40.v1; v119 = tmp40.v2;
                while (while_method_1(v117)){
                    Union1 v121;
                    v121 = v110[v117];
                    unsigned long long v125;
                    switch (v121.tag) {
                        case 0: { // Jack
                            v125 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v125 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v125 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v126;
                    v126 = v125 * v119;
                    unsigned long long v127;
                    v127 = v118 + v126;
                    unsigned long long v128;
                    v128 = v119 * 9973ull;
                    v118 = v127;
                    v119 = v128;
                    v117 += 1 ;
                }
                unsigned long long v129;
                v129 = 9223372036854775807ull + v116;
                unsigned long long v130;
                v130 = v129 * 9973ull;
                unsigned long long v131;
                v131 = v130 * 5ull;
                v132 = v131;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v133;
        v133 = v132 * v82;
        unsigned long long v134;
        v134 = v81 + v133;
        unsigned long long v135;
        v135 = v82 * 9973ull;
        v81 = v134;
        v82 = v135;
        v80 += 1 ;
    }
    auto v136 = v42.find(Tuple0{0ull, v69});
    bool v137;
    v137 = v136 != v42.end();
    Union7 v142;
    if (v137){
        static_array<float,3> v138; static_array<float,3> v139;
        Tuple1 tmp41 = v136->second;
        v138 = tmp41.v0; v139 = tmp41.v1;
        v142 = Union7{Union7_1{v138, v139}};
    } else {
        v142 = Union7{Union7_0{}};
    }
    static_array<float,3> v159; static_array<float,3> v160;
    switch (v142.tag) {
        case 0: { // None
            static_array<float,3> v145;
            int v149;
            v149 = 0;
            while (while_method_4(v149)){
                v145[v149] = 0.0f;
                v149 += 1 ;
            }
            static_array<float,3> v151;
            int v155;
            v155 = 0;
            while (while_method_4(v155)){
                v151[v155] = 0.0f;
                v155 += 1 ;
            }
            v159 = v145; v160 = v151;
            break;
        }
        case 1: { // Some
            static_array<float,3> v143 = v142.case1.v0; static_array<float,3> v144 = v142.case1.v1;
            v159 = v143; v160 = v144;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    static_array<float,3> v161;
    v161 = regret_match_3(v160, v38);
    float v190;
    switch (v31.tag) {
        case 0: { // None
            v190 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v162 = v31.case1.v0;
            float v163;
            v163 = v161[2];
            static_array<Tuple2,2> & v167 = v2.v0;
            float v168; float v169;
            Tuple2 tmp42 = v167[v6];
            v168 = tmp42.v0; v169 = tmp42.v1;
            static_array<Tuple2,2> & v176 = v2.v0;
            float v177;
            v177 = log(v163);
            float v178;
            v178 = v177 + v168;
            v176[v6] = Tuple2{v178, v169};
            static_array_list<Union0,32> & v179 = v1.v0;
            Union0 v180;
            v180 = Union0{Union0_2{v6, v162}};
            v179.push(v180);
            Union5 v181;
            v181 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v162}};
            float v182;
            v182 = loop_14(v1, v9, v10, v0, v11, v2, v12, v181);
            static_array_list<Union0,32> & v183 = v1.v0;
            Union0 v184;
            v184 = v183.pop();
            static_array<Tuple2,2> & v185 = v2.v0;
            v185[v6] = Tuple2{v168, v169};
            bool v186;
            v186 = v6 == 0;
            if (v186){
                v190 = v182;
            } else {
                float v187;
                v187 = -v182;
                v190 = v187;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v219;
    switch (v26.tag) {
        case 0: { // None
            v219 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v191 = v26.case1.v0;
            float v192;
            v192 = v161[1];
            static_array<Tuple2,2> & v196 = v2.v0;
            float v197; float v198;
            Tuple2 tmp43 = v196[v6];
            v197 = tmp43.v0; v198 = tmp43.v1;
            static_array<Tuple2,2> & v205 = v2.v0;
            float v206;
            v206 = log(v192);
            float v207;
            v207 = v206 + v197;
            v205[v6] = Tuple2{v207, v198};
            static_array_list<Union0,32> & v208 = v1.v0;
            Union0 v209;
            v209 = Union0{Union0_2{v6, v191}};
            v208.push(v209);
            Union5 v210;
            v210 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v191}};
            float v211;
            v211 = loop_14(v1, v9, v10, v0, v11, v2, v12, v210);
            static_array_list<Union0,32> & v212 = v1.v0;
            Union0 v213;
            v213 = v212.pop();
            static_array<Tuple2,2> & v214 = v2.v0;
            v214[v6] = Tuple2{v197, v198};
            bool v215;
            v215 = v6 == 0;
            if (v215){
                v219 = v211;
            } else {
                float v216;
                v216 = -v211;
                v219 = v216;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v220;
    v220 = v161[0];
    static_array<Tuple2,2> & v224 = v2.v0;
    float v225; float v226;
    Tuple2 tmp44 = v224[v6];
    v225 = tmp44.v0; v226 = tmp44.v1;
    static_array<Tuple2,2> & v233 = v2.v0;
    float v234;
    v234 = log(v220);
    float v235;
    v235 = v234 + v225;
    v233[v6] = Tuple2{v235, v226};
    static_array_list<Union0,32> & v236 = v1.v0;
    Union2 v237;
    v237 = Union2{Union2_0{}};
    Union0 v238;
    v238 = Union0{Union0_2{v6, v237}};
    v236.push(v238);
    Union2 v239;
    v239 = Union2{Union2_0{}};
    Union5 v240;
    v240 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v239}};
    float v241;
    v241 = loop_14(v1, v9, v10, v0, v11, v2, v12, v240);
    static_array_list<Union0,32> & v242 = v1.v0;
    Union0 v243;
    v243 = v242.pop();
    static_array<Tuple2,2> & v244 = v2.v0;
    v244[v6] = Tuple2{v225, v226};
    bool v245;
    v245 = v6 == 0;
    float v247;
    if (v245){
        v247 = v241;
    } else {
        float v246;
        v246 = -v241;
        v247 = v246;
    }
    static_array<float,3> v248;
    v248[0] = v247;
    v248[1] = v219;
    v248[2] = v190;
    int v252; float v253;
    Tuple3 tmp45 = Tuple3{0, 0.0f};
    v252 = tmp45.v0; v253 = tmp45.v1;
    while (while_method_4(v252)){
        float v255;
        v255 = v248[v252];
        float v259;
        v259 = v161[v252];
        float v263;
        v263 = v255 * v259;
        float v264;
        v264 = v253 + v263;
        v253 = v264;
        v252 += 1 ;
    }
    static_array<float,3> v265;
    int v269;
    v269 = 0;
    while (while_method_4(v269)){
        float v271;
        v271 = v159[v269];
        float v275;
        v275 = v161[v269];
        float v279;
        v279 = 0.99609375f * v271;
        float v280;
        v280 = v279 + v275;
        v265[v269] = v280;
        v269 += 1 ;
    }
    static_array<Tuple2,2> & v281 = v2.v0;
    int v282; float v283;
    Tuple3 tmp46 = Tuple3{0, 0.0f};
    v282 = tmp46.v0; v283 = tmp46.v1;
    while (while_method_1(v282)){
        float v285; float v286;
        Tuple2 tmp47 = v281[v282];
        v285 = tmp47.v0; v286 = tmp47.v1;
        bool v293;
        v293 = v282 == v6;
        float v294;
        if (v293){
            v294 = 0.0f;
        } else {
            v294 = v285;
        }
        float v295;
        v295 = v283 + v294;
        float v296;
        v296 = v295 - v286;
        v283 = v296;
        v282 += 1 ;
    }
    float v297;
    v297 = exp(v283);
    static_array<float,3> v298;
    int v302;
    v302 = 0;
    while (while_method_4(v302)){
        float v304;
        v304 = v160[v302];
        float v308;
        v308 = v248[v302];
        float v312;
        v312 = v308 - v253;
        float v313;
        v313 = v297 * v312;
        float v314;
        v314 = v304 + v313;
        bool v315;
        v315 = 0.0f >= v314;
        float v316;
        if (v315){
            v316 = 0.0f;
        } else {
            v316 = v314;
        }
        v298[v302] = v316;
        v302 += 1 ;
    }
    int v317;
    v317 = v69.length;
    int v318; unsigned long long v319; unsigned long long v320;
    Tuple6 tmp48 = Tuple6{0, 0ull, 1ull};
    v318 = tmp48.v0; v319 = tmp48.v1; v320 = tmp48.v2;
    while (while_method_0(v317, v318)){
        Union0 v322;
        v322 = v69[v318];
        unsigned long long v370;
        switch (v322.tag) {
            case 0: { // CommunityCardIs
                Union1 v326 = v322.case0.v0;
                unsigned long long v327;
                switch (v326.tag) {
                    case 0: { // Jack
                        v327 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v327 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v327 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v328;
                v328 = 9223372036854775807ull + v327;
                unsigned long long v329;
                v329 = v328 * 9973ull;
                v370 = v329;
                break;
            }
            case 1: { // Hidden
                v370 = 18446744073709531670ull;
                break;
            }
            case 2: { // PlayerAction
                int v330 = v322.case2.v0; Union2 v331 = v322.case2.v1;
                unsigned long long v332;
                v332 = std::hash<int>()(v330);
                unsigned long long v333;
                v333 = v332 * 9973ull;
                unsigned long long v334;
                switch (v331.tag) {
                    case 0: { // Call
                        v334 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v334 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v334 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v335;
                v335 = v333 + v334;
                unsigned long long v336;
                v336 = 9223372036854775807ull + v335;
                unsigned long long v337;
                v337 = v336 * 9973ull;
                unsigned long long v338;
                v338 = v337 * 3ull;
                v370 = v338;
                break;
            }
            case 3: { // PlayerGotCard
                int v339 = v322.case3.v0; Union1 v340 = v322.case3.v1;
                unsigned long long v341;
                v341 = std::hash<int>()(v339);
                unsigned long long v342;
                v342 = v341 * 9973ull;
                unsigned long long v343;
                switch (v340.tag) {
                    case 0: { // Jack
                        v343 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v343 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v343 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v344;
                v344 = v342 + v343;
                unsigned long long v345;
                v345 = 9223372036854775807ull + v344;
                unsigned long long v346;
                v346 = v345 * 9973ull;
                unsigned long long v347;
                v347 = v346 * 4ull;
                v370 = v347;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v348 = v322.case4.v0; int v349 = v322.case4.v1; int v350 = v322.case4.v2;
                unsigned long long v351;
                v351 = std::hash<int>()(v350);
                unsigned long long v352;
                v352 = std::hash<int>()(v349);
                unsigned long long v353;
                v353 = v352 * 9973ull;
                unsigned long long v354;
                v354 = v351 + v353;
                int v355; unsigned long long v356; unsigned long long v357;
                Tuple6 tmp49 = Tuple6{0, 0ull, 1ull};
                v355 = tmp49.v0; v356 = tmp49.v1; v357 = tmp49.v2;
                while (while_method_1(v355)){
                    Union1 v359;
                    v359 = v348[v355];
                    unsigned long long v363;
                    switch (v359.tag) {
                        case 0: { // Jack
                            v363 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v363 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v363 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v364;
                    v364 = v363 * v357;
                    unsigned long long v365;
                    v365 = v356 + v364;
                    unsigned long long v366;
                    v366 = v357 * 9973ull;
                    v356 = v365;
                    v357 = v366;
                    v355 += 1 ;
                }
                unsigned long long v367;
                v367 = 9223372036854775807ull + v354;
                unsigned long long v368;
                v368 = v367 * 9973ull;
                unsigned long long v369;
                v369 = v368 * 5ull;
                v370 = v369;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v371;
        v371 = v370 * v320;
        unsigned long long v372;
        v372 = v319 + v371;
        unsigned long long v373;
        v373 = v320 * 9973ull;
        v319 = v372;
        v320 = v373;
        v318 += 1 ;
    }
    v42[Tuple0{0ull, v69}] = Tuple1{v265, v298};
    return v253;
}
float method_16(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs0 & v9, StackRefs2 & v10, StackMut0 & v11, StackMut1 & v12){
    int v13;
    v13 = v7[0];
    int v17;
    v17 = v7[1];
    bool v21;
    v21 = v13 == v17;
    bool v22;
    v22 = v21 != true;
    Union6 v26;
    if (v22){
        Union2 v23;
        v23 = Union2{Union2_1{}};
        v26 = Union6{Union6_1{v23}};
    } else {
        v26 = Union6{Union6_0{}};
    }
    bool v27;
    v27 = v8 > 0;
    Union6 v31;
    if (v27){
        Union2 v28;
        v28 = Union2{Union2_2{}};
        v31 = Union6{Union6_1{v28}};
    } else {
        v31 = Union6{Union6_0{}};
    }
    bool v34;
    switch (v31.tag) {
        case 0: { // None
            v34 = false;
            break;
        }
        case 1: { // Some
            Union2 v32 = v31.case1.v0;
            v34 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    bool v37;
    switch (v26.tag) {
        case 0: { // None
            v37 = false;
            break;
        }
        case 1: { // Some
            Union2 v35 = v26.case1.v0;
            v37 = true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    static_array<bool,3> v38;
    v38[0] = true;
    v38[1] = v37;
    v38[2] = v34;
    static_array_list<Union0,32> & v42 = v1.v0;
    static_array_list<Union0,32> & v43 = v1.v0;
    int v44;
    v44 = v43.length;
    bool v45;
    v45 = 32 >= v44;
    bool v46;
    v46 = v45 == false;
    if (v46){
        assert("The type level dimension has to equal the value passed at runtime into create." && v45);
    } else {
    }
    static_array_list<Union0,32> v48;
    v48 = static_array_list<Union0,32>{};
    v48.unsafe_set_length(v44);
    int v52;
    v52 = 0;
    while (while_method_0(v44, v52)){
        Union0 v54;
        v54 = v43[v52];
        Union0 v64;
        switch (v54.tag) {
            case 3: { // PlayerGotCard
                int v58 = v54.case3.v0; Union1 v59 = v54.case3.v1;
                bool v60;
                v60 = v58 == v6;
                bool v61;
                v61 = v60 != true;
                if (v61){
                    v64 = Union0{Union0_1{}};
                } else {
                    v64 = v54;
                }
                break;
            }
            default: {
                v64 = v54;
            }
        }
        v48[v52] = v64;
        v52 += 1 ;
    }
    float v65[1312];
    int v66;
    v66 = 0;
    while (while_method_5(v66)){
        v65[v66] = 0.0f;
        v66 += 1 ;
    }
    method_7(v65, v48);
    af::array v68(1312, 1, v65);
    af::array v69;
    af::array v70;
    af::array & v71 = v10.v0;
    af::array & v72 = v10.v1;
    StackRefs5 v73{v68, v71, v70, v69, v72};
    method_8(v73);
    unsigned int v74;
    v74 = v70(0);
    int v75;
    v75 = v69.elements();
    float v76[v75];
    v69.host<float>(v76);;
    static_array<float,3> v77;
    int v81;
    v81 = 0;
    while (while_method_4(v81)){
        float v83;
        v83 = v76[v81];
        v77[v81] = v83;
        v81 += 1 ;
    }
    static_array<float,3> v84;
    int v88;
    v88 = 0;
    while (while_method_4(v88)){
        int v90;
        v90 = v88 + 3;
        float v91;
        v91 = v76[v90];
        v84[v88] = v91;
        v88 += 1 ;
    }
    static_array<float,3> v92;
    v92 = masking_normalize_5(v77, v38);
    float v121;
    switch (v31.tag) {
        case 0: { // None
            v121 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v93 = v31.case1.v0;
            float v94;
            v94 = v92[2];
            static_array<Tuple2,2> & v98 = v2.v0;
            float v99; float v100;
            Tuple2 tmp50 = v98[v6];
            v99 = tmp50.v0; v100 = tmp50.v1;
            static_array<Tuple2,2> & v107 = v2.v0;
            float v108;
            v108 = log(v94);
            float v109;
            v109 = v108 + v99;
            v107[v6] = Tuple2{v109, v100};
            static_array_list<Union0,32> & v110 = v1.v0;
            Union0 v111;
            v111 = Union0{Union0_2{v6, v93}};
            v110.push(v111);
            Union5 v112;
            v112 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v93}};
            float v113;
            v113 = loop_14(v1, v9, v10, v0, v11, v2, v12, v112);
            static_array_list<Union0,32> & v114 = v1.v0;
            Union0 v115;
            v115 = v114.pop();
            static_array<Tuple2,2> & v116 = v2.v0;
            v116[v6] = Tuple2{v99, v100};
            bool v117;
            v117 = v6 == 0;
            if (v117){
                v121 = v113;
            } else {
                float v118;
                v118 = -v113;
                v121 = v118;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v150;
    switch (v26.tag) {
        case 0: { // None
            v150 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v122 = v26.case1.v0;
            float v123;
            v123 = v92[1];
            static_array<Tuple2,2> & v127 = v2.v0;
            float v128; float v129;
            Tuple2 tmp51 = v127[v6];
            v128 = tmp51.v0; v129 = tmp51.v1;
            static_array<Tuple2,2> & v136 = v2.v0;
            float v137;
            v137 = log(v123);
            float v138;
            v138 = v137 + v128;
            v136[v6] = Tuple2{v138, v129};
            static_array_list<Union0,32> & v139 = v1.v0;
            Union0 v140;
            v140 = Union0{Union0_2{v6, v122}};
            v139.push(v140);
            Union5 v141;
            v141 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v122}};
            float v142;
            v142 = loop_14(v1, v9, v10, v0, v11, v2, v12, v141);
            static_array_list<Union0,32> & v143 = v1.v0;
            Union0 v144;
            v144 = v143.pop();
            static_array<Tuple2,2> & v145 = v2.v0;
            v145[v6] = Tuple2{v128, v129};
            bool v146;
            v146 = v6 == 0;
            if (v146){
                v150 = v142;
            } else {
                float v147;
                v147 = -v142;
                v150 = v147;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v151;
    v151 = v92[0];
    static_array<Tuple2,2> & v155 = v2.v0;
    float v156; float v157;
    Tuple2 tmp52 = v155[v6];
    v156 = tmp52.v0; v157 = tmp52.v1;
    static_array<Tuple2,2> & v164 = v2.v0;
    float v165;
    v165 = log(v151);
    float v166;
    v166 = v165 + v156;
    v164[v6] = Tuple2{v166, v157};
    static_array_list<Union0,32> & v167 = v1.v0;
    Union2 v168;
    v168 = Union2{Union2_0{}};
    Union0 v169;
    v169 = Union0{Union0_2{v6, v168}};
    v167.push(v169);
    Union2 v170;
    v170 = Union2{Union2_0{}};
    Union5 v171;
    v171 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v170}};
    float v172;
    v172 = loop_14(v1, v9, v10, v0, v11, v2, v12, v171);
    static_array_list<Union0,32> & v173 = v1.v0;
    Union0 v174;
    v174 = v173.pop();
    static_array<Tuple2,2> & v175 = v2.v0;
    v175[v6] = Tuple2{v156, v157};
    bool v176;
    v176 = v6 == 0;
    float v178;
    if (v176){
        v178 = v172;
    } else {
        float v177;
        v177 = -v172;
        v178 = v177;
    }
    static_array<float,3> v179;
    v179[0] = v178;
    v179[1] = v150;
    v179[2] = v121;
    int v183; float v184;
    Tuple3 tmp53 = Tuple3{0, 0.0f};
    v183 = tmp53.v0; v184 = tmp53.v1;
    while (while_method_4(v183)){
        float v186;
        v186 = v179[v183];
        float v190;
        v190 = v92[v183];
        float v194;
        v194 = v186 * v190;
        float v195;
        v195 = v184 + v194;
        v184 = v195;
        v183 += 1 ;
    }
    return v184;
}
float body_13(StackRefs3 & v0, StackRefs0 & v1, StackRefs2 & v2, xso::rng & v3, StackMut0 & v4, StackRefs4 & v5, Union3 v6){
    StackMut1 v7{0.0f};
    switch (v6.tag) {
        case 0: { // ChanceCommunityCard
            Union4 v90 = v6.case0.v0; bool v91 = v6.case0.v1; static_array<Union1,2> v92 = v6.case0.v2; int v93 = v6.case0.v3; static_array<int,2> v94 = v6.case0.v4; int v95 = v6.case0.v5;
            int v96; float v97; float v98;
            Tuple4 tmp31 = Tuple4{0, 0.0f, 0.0f};
            v96 = tmp31.v0; v97 = tmp31.v1; v98 = tmp31.v2;
            while (while_method_3(v96)){
                unsigned int v100 = v4.v0;
                unsigned int v101;
                v101 = 1u << v96;
                unsigned int v102;
                v102 = v100 & v101;
                bool v103;
                v103 = v102 == 0u;
                bool v104;
                v104 = v103 != true;
                float v136; float v137;
                if (v104){
                    unsigned int v105 = v4.v0;
                    unsigned int v106;
                    v106 = v105 ^ v101;
                    v4.v0 = v106;
                    bool v107;
                    v107 = 0 == v96;
                    Union1 v125;
                    if (v107){
                        v125 = Union1{Union1_1{}};
                    } else {
                        bool v109;
                        v109 = 1 == v96;
                        if (v109){
                            v125 = Union1{Union1_1{}};
                        } else {
                            bool v111;
                            v111 = 2 == v96;
                            if (v111){
                                v125 = Union1{Union1_2{}};
                            } else {
                                bool v113;
                                v113 = 3 == v96;
                                if (v113){
                                    v125 = Union1{Union1_2{}};
                                } else {
                                    bool v115;
                                    v115 = 4 == v96;
                                    if (v115){
                                        v125 = Union1{Union1_0{}};
                                    } else {
                                        bool v117;
                                        v117 = 5 == v96;
                                        if (v117){
                                            v125 = Union1{Union1_0{}};
                                        } else {
                                            printf("%s\n", "Invalid int in int_to_card.");
                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    static_array_list<Union0,32> & v126 = v0.v0;
                    Union0 v127;
                    v127 = Union0{Union0_0{v125}};
                    v126.push(v127);
                    Union5 v128;
                    v128 = Union5{Union5_0{v90, v91, v92, v93, v94, v95, v125}};
                    float v129;
                    v129 = loop_14(v0, v1, v2, v3, v4, v5, v7, v128);
                    static_array_list<Union0,32> & v130 = v0.v0;
                    Union0 v131;
                    v131 = v130.pop();
                    unsigned int v132 = v4.v0;
                    unsigned int v133;
                    v133 = v132 ^ v101;
                    v4.v0 = v133;
                    float v134;
                    v134 = v97 + v129;
                    float v135;
                    v135 = v98 + 1.0f;
                    v136 = v134; v137 = v135;
                } else {
                    v136 = v97; v137 = v98;
                }
                v97 = v136;
                v98 = v137;
                v96 += 1 ;
            }
            bool v138;
            v138 = v98 == 0.0f;
            bool v139;
            v139 = v138 != true;
            if (v139){
                float v140;
                v140 = v97 / v98;
                return v140;
            } else {
                return 0.0f;
            }
            break;
        }
        case 1: { // ChanceInit
            int v142; float v143; float v144;
            Tuple4 tmp36 = Tuple4{0, 0.0f, 0.0f};
            v142 = tmp36.v0; v143 = tmp36.v1; v144 = tmp36.v2;
            while (while_method_3(v142)){
                unsigned int v146 = v4.v0;
                unsigned int v147;
                v147 = 1u << v142;
                unsigned int v148;
                v148 = v146 & v147;
                bool v149;
                v149 = v148 == 0u;
                bool v150;
                v150 = v149 != true;
                float v226; float v227;
                if (v150){
                    unsigned int v151 = v4.v0;
                    unsigned int v152;
                    v152 = v151 ^ v147;
                    v4.v0 = v152;
                    bool v153;
                    v153 = 0 == v142;
                    Union1 v171;
                    if (v153){
                        v171 = Union1{Union1_1{}};
                    } else {
                        bool v155;
                        v155 = 1 == v142;
                        if (v155){
                            v171 = Union1{Union1_1{}};
                        } else {
                            bool v157;
                            v157 = 2 == v142;
                            if (v157){
                                v171 = Union1{Union1_2{}};
                            } else {
                                bool v159;
                                v159 = 3 == v142;
                                if (v159){
                                    v171 = Union1{Union1_2{}};
                                } else {
                                    bool v161;
                                    v161 = 4 == v142;
                                    if (v161){
                                        v171 = Union1{Union1_0{}};
                                    } else {
                                        bool v163;
                                        v163 = 5 == v142;
                                        if (v163){
                                            v171 = Union1{Union1_0{}};
                                        } else {
                                            printf("%s\n", "Invalid int in int_to_card.");
                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    int v172; float v173; float v174;
                    Tuple4 tmp37 = Tuple4{0, 0.0f, 0.0f};
                    v172 = tmp37.v0; v173 = tmp37.v1; v174 = tmp37.v2;
                    while (while_method_3(v172)){
                        unsigned int v176 = v4.v0;
                        unsigned int v177;
                        v177 = 1u << v172;
                        unsigned int v178;
                        v178 = v176 & v177;
                        bool v179;
                        v179 = v178 == 0u;
                        bool v180;
                        v180 = v179 != true;
                        float v216; float v217;
                        if (v180){
                            unsigned int v181 = v4.v0;
                            unsigned int v182;
                            v182 = v181 ^ v177;
                            v4.v0 = v182;
                            bool v183;
                            v183 = 0 == v172;
                            Union1 v201;
                            if (v183){
                                v201 = Union1{Union1_1{}};
                            } else {
                                bool v185;
                                v185 = 1 == v172;
                                if (v185){
                                    v201 = Union1{Union1_1{}};
                                } else {
                                    bool v187;
                                    v187 = 2 == v172;
                                    if (v187){
                                        v201 = Union1{Union1_2{}};
                                    } else {
                                        bool v189;
                                        v189 = 3 == v172;
                                        if (v189){
                                            v201 = Union1{Union1_2{}};
                                        } else {
                                            bool v191;
                                            v191 = 4 == v172;
                                            if (v191){
                                                v201 = Union1{Union1_0{}};
                                            } else {
                                                bool v193;
                                                v193 = 5 == v172;
                                                if (v193){
                                                    v201 = Union1{Union1_0{}};
                                                } else {
                                                    printf("%s\n", "Invalid int in int_to_card.");
                                                    exit(-1);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            static_array_list<Union0,32> & v202 = v0.v0;
                            Union0 v203;
                            v203 = Union0{Union0_3{0, v171}};
                            v202.push(v203);
                            static_array_list<Union0,32> & v204 = v0.v0;
                            Union0 v205;
                            v205 = Union0{Union0_3{1, v201}};
                            v204.push(v205);
                            Union5 v206;
                            v206 = Union5{Union5_1{v171, v201}};
                            float v207;
                            v207 = loop_14(v0, v1, v2, v3, v4, v5, v7, v206);
                            static_array_list<Union0,32> & v208 = v0.v0;
                            Union0 v209;
                            v209 = v208.pop();
                            static_array_list<Union0,32> & v210 = v0.v0;
                            Union0 v211;
                            v211 = v210.pop();
                            unsigned int v212 = v4.v0;
                            unsigned int v213;
                            v213 = v212 ^ v177;
                            v4.v0 = v213;
                            float v214;
                            v214 = v173 + v207;
                            float v215;
                            v215 = v174 + 1.0f;
                            v216 = v214; v217 = v215;
                        } else {
                            v216 = v173; v217 = v174;
                        }
                        v173 = v216;
                        v174 = v217;
                        v172 += 1 ;
                    }
                    bool v218;
                    v218 = v174 == 0.0f;
                    bool v219;
                    v219 = v218 != true;
                    float v221;
                    if (v219){
                        float v220;
                        v220 = v173 / v174;
                        v221 = v220;
                    } else {
                        v221 = 0.0f;
                    }
                    unsigned int v222 = v4.v0;
                    unsigned int v223;
                    v223 = v222 ^ v147;
                    v4.v0 = v223;
                    float v224;
                    v224 = v143 + v221;
                    float v225;
                    v225 = v144 + 1.0f;
                    v226 = v224; v227 = v225;
                } else {
                    v226 = v143; v227 = v144;
                }
                v143 = v226;
                v144 = v227;
                v142 += 1 ;
            }
            bool v228;
            v228 = v144 == 0.0f;
            bool v229;
            v229 = v228 != true;
            if (v229){
                float v230;
                v230 = v143 / v144;
                return v230;
            } else {
                return 0.0f;
            }
            break;
        }
        case 2: { // Round
            Union4 v60 = v6.case2.v0; bool v61 = v6.case2.v1; static_array<Union1,2> v62 = v6.case2.v2; int v63 = v6.case2.v3; static_array<int,2> v64 = v6.case2.v4; int v65 = v6.case2.v5;
            bool v66;
            v66 = v63 == 0;
            float v72;
            if (v66){
                v72 = method_15(v3, v0, v5, v60, v61, v62, v63, v64, v65, v1, v2, v4, v7);
            } else {
                bool v68;
                v68 = v63 == 1;
                if (v68){
                    v72 = method_16(v3, v0, v5, v60, v61, v62, v63, v64, v65, v1, v2, v4, v7);
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
            return loop_14(v0, v1, v2, v3, v4, v5, v7, v75);
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
            v87 = loop_14(v0, v1, v2, v3, v4, v5, v7, v86);
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
            Union8 v40;
            v40 = compare_hands_9(v30, v31, v32, v33, v34, v35);
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
            v57 = loop_14(v0, v1, v2, v3, v4, v5, v7, v56);
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
            v27 = loop_14(v0, v1, v2, v3, v4, v5, v7, v26);
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
int method_18(float * v0, StackMut2 & v1, int v2){
    int v3; int v4; int v5;
    Tuple7 tmp54 = Tuple7{0, 0, 0};
    v3 = tmp54.v0; v4 = tmp54.v1; v5 = tmp54.v2;
    while (while_method_0(v2, v3)){
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
void method_19(float * v0, StackMut2 & v1, int v2){
    int v3; int v4; int v5;
    Tuple7 tmp55 = Tuple7{0, 0, 0};
    v3 = tmp55.v0; v4 = tmp55.v1; v5 = tmp55.v2;
    while (while_method_0(v2, v3)){
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
void method_20(float * v0, StackMut2 & v1, int v2){
    int v3; int v4; int v5;
    Tuple7 tmp56 = Tuple7{0, 0, 0};
    v3 = tmp56.v0; v4 = tmp56.v1; v5 = tmp56.v2;
    while (while_method_0(v2, v3)){
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
static_array_list<Union0,32> method_17(float * v0){
    StackMut2 v1{0};
    int v2;
    v2 = 32;
    int v3;
    v3 = method_18(v0, v1, v2);
    static_array_list<Union0,32> v4;
    v4 = static_array_list<Union0,32>{};
    int v8;
    v8 = 0;
    while (while_method_0(v3, v8)){
        Union10 v10;
        v10 = Union10{Union10_0{}};
        StackMut3 v11{v10};
        int v12;
        v12 = 5;
        int v13;
        v13 = method_18(v0, v1, v12);
        bool v14;
        v14 = 0 == v13;
        if (v14){
            Union10 v15 = v11.v0;
            switch (v15.tag) {
                case 0: { // None
                    Union4 v16;
                    v16 = Union4{Union4_0{}};
                    StackMut4 v17{v16};
                    int v18;
                    v18 = 3;
                    int v19;
                    v19 = method_18(v0, v1, v18);
                    bool v20;
                    v20 = 0 == v19;
                    if (v20){
                        Union4 v21 = v17.v0;
                        switch (v21.tag) {
                            case 0: { // None
                                Union1 v22;
                                v22 = Union1{Union1_0{}};
                                Union4 v23;
                                v23 = Union4{Union4_1{v22}};
                                v17.v0 = v23;
                                break;
                            }
                            case 1: { // Some
                                Union1 v24 = v21.case1.v0;
                                bool v25;
                                v25 = false;
                                bool v26;
                                v26 = v25 == false;
                                if (v26){
                                    assert("Duplicate union type instances in the unpickle Alt case." && v25);
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
                        int v28;
                        v28 = 0;
                        method_19(v0, v1, v28);
                    }
                    bool v29;
                    v29 = 1 == v19;
                    if (v29){
                        Union4 v30 = v17.v0;
                        switch (v30.tag) {
                            case 0: { // None
                                Union1 v31;
                                v31 = Union1{Union1_1{}};
                                Union4 v32;
                                v32 = Union4{Union4_1{v31}};
                                v17.v0 = v32;
                                break;
                            }
                            case 1: { // Some
                                Union1 v33 = v30.case1.v0;
                                bool v34;
                                v34 = false;
                                bool v35;
                                v35 = v34 == false;
                                if (v35){
                                    assert("Duplicate union type instances in the unpickle Alt case." && v34);
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
                        int v37;
                        v37 = 0;
                        method_19(v0, v1, v37);
                    }
                    bool v38;
                    v38 = 2 == v19;
                    if (v38){
                        Union4 v39 = v17.v0;
                        switch (v39.tag) {
                            case 0: { // None
                                Union1 v40;
                                v40 = Union1{Union1_2{}};
                                Union4 v41;
                                v41 = Union4{Union4_1{v40}};
                                v17.v0 = v41;
                                break;
                            }
                            case 1: { // Some
                                Union1 v42 = v39.case1.v0;
                                bool v43;
                                v43 = false;
                                bool v44;
                                v44 = v43 == false;
                                if (v44){
                                    assert("Duplicate union type instances in the unpickle Alt case." && v43);
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
                        int v46;
                        v46 = 0;
                        method_19(v0, v1, v46);
                    }
                    Union4 v47 = v17.v0;
                    Union1 v51;
                    switch (v47.tag) {
                        case 0: { // None
                            printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                            exit(-1);
                            break;
                        }
                        case 1: { // Some
                            Union1 v48 = v47.case1.v0;
                            v51 = v48;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    Union0 v52;
                    v52 = Union0{Union0_0{v51}};
                    Union10 v53;
                    v53 = Union10{Union10_1{v52}};
                    v11.v0 = v53;
                    break;
                }
                case 1: { // Some
                    Union0 v54 = v15.case1.v0;
                    bool v55;
                    v55 = false;
                    bool v56;
                    v56 = v55 == false;
                    if (v56){
                        assert("Duplicate union type instances in the unpickle Alt case." && v55);
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
            int v58;
            v58 = 3;
            method_19(v0, v1, v58);
        }
        bool v59;
        v59 = 1 == v13;
        if (v59){
            Union10 v60 = v11.v0;
            switch (v60.tag) {
                case 0: { // None
                    Union0 v61;
                    v61 = Union0{Union0_1{}};
                    Union10 v62;
                    v62 = Union10{Union10_1{v61}};
                    v11.v0 = v62;
                    break;
                }
                case 1: { // Some
                    Union0 v63 = v60.case1.v0;
                    bool v64;
                    v64 = false;
                    bool v65;
                    v65 = v64 == false;
                    if (v65){
                        assert("Duplicate union type instances in the unpickle Alt case." && v64);
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
            int v67;
            v67 = 0;
            method_19(v0, v1, v67);
        }
        bool v68;
        v68 = 2 == v13;
        if (v68){
            Union10 v69 = v11.v0;
            switch (v69.tag) {
                case 0: { // None
                    int v70;
                    v70 = 2;
                    int v71;
                    v71 = method_18(v0, v1, v70);
                    Union6 v72;
                    v72 = Union6{Union6_0{}};
                    StackMut5 v73{v72};
                    int v74;
                    v74 = 3;
                    int v75;
                    v75 = method_18(v0, v1, v74);
                    bool v76;
                    v76 = 0 == v75;
                    if (v76){
                        Union6 v77 = v73.v0;
                        switch (v77.tag) {
                            case 0: { // None
                                Union2 v78;
                                v78 = Union2{Union2_0{}};
                                Union6 v79;
                                v79 = Union6{Union6_1{v78}};
                                v73.v0 = v79;
                                break;
                            }
                            case 1: { // Some
                                Union2 v80 = v77.case1.v0;
                                bool v81;
                                v81 = false;
                                bool v82;
                                v82 = v81 == false;
                                if (v82){
                                    assert("Duplicate union type instances in the unpickle Alt case." && v81);
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
                        int v84;
                        v84 = 0;
                        method_19(v0, v1, v84);
                    }
                    bool v85;
                    v85 = 1 == v75;
                    if (v85){
                        Union6 v86 = v73.v0;
                        switch (v86.tag) {
                            case 0: { // None
                                Union2 v87;
                                v87 = Union2{Union2_1{}};
                                Union6 v88;
                                v88 = Union6{Union6_1{v87}};
                                v73.v0 = v88;
                                break;
                            }
                            case 1: { // Some
                                Union2 v89 = v86.case1.v0;
                                bool v90;
                                v90 = false;
                                bool v91;
                                v91 = v90 == false;
                                if (v91){
                                    assert("Duplicate union type instances in the unpickle Alt case." && v90);
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
                        int v93;
                        v93 = 0;
                        method_19(v0, v1, v93);
                    }
                    bool v94;
                    v94 = 2 == v75;
                    if (v94){
                        Union6 v95 = v73.v0;
                        switch (v95.tag) {
                            case 0: { // None
                                Union2 v96;
                                v96 = Union2{Union2_2{}};
                                Union6 v97;
                                v97 = Union6{Union6_1{v96}};
                                v73.v0 = v97;
                                break;
                            }
                            case 1: { // Some
                                Union2 v98 = v95.case1.v0;
                                bool v99;
                                v99 = false;
                                bool v100;
                                v100 = v99 == false;
                                if (v100){
                                    assert("Duplicate union type instances in the unpickle Alt case." && v99);
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
                        int v102;
                        v102 = 0;
                        method_19(v0, v1, v102);
                    }
                    Union6 v103 = v73.v0;
                    Union2 v107;
                    switch (v103.tag) {
                        case 0: { // None
                            printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                            exit(-1);
                            break;
                        }
                        case 1: { // Some
                            Union2 v104 = v103.case1.v0;
                            v107 = v104;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    Union0 v108;
                    v108 = Union0{Union0_2{v71, v107}};
                    Union10 v109;
                    v109 = Union10{Union10_1{v108}};
                    v11.v0 = v109;
                    break;
                }
                case 1: { // Some
                    Union0 v110 = v69.case1.v0;
                    bool v111;
                    v111 = false;
                    bool v112;
                    v112 = v111 == false;
                    if (v112){
                        assert("Duplicate union type instances in the unpickle Alt case." && v111);
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
            int v114;
            v114 = 5;
            method_19(v0, v1, v114);
        }
        bool v115;
        v115 = 3 == v13;
        if (v115){
            Union10 v116 = v11.v0;
            switch (v116.tag) {
                case 0: { // None
                    int v117;
                    v117 = 2;
                    int v118;
                    v118 = method_18(v0, v1, v117);
                    Union4 v119;
                    v119 = Union4{Union4_0{}};
                    StackMut4 v120{v119};
                    int v121;
                    v121 = 3;
                    int v122;
                    v122 = method_18(v0, v1, v121);
                    bool v123;
                    v123 = 0 == v122;
                    if (v123){
                        Union4 v124 = v120.v0;
                        switch (v124.tag) {
                            case 0: { // None
                                Union1 v125;
                                v125 = Union1{Union1_0{}};
                                Union4 v126;
                                v126 = Union4{Union4_1{v125}};
                                v120.v0 = v126;
                                break;
                            }
                            case 1: { // Some
                                Union1 v127 = v124.case1.v0;
                                bool v128;
                                v128 = false;
                                bool v129;
                                v129 = v128 == false;
                                if (v129){
                                    assert("Duplicate union type instances in the unpickle Alt case." && v128);
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
                        int v131;
                        v131 = 0;
                        method_19(v0, v1, v131);
                    }
                    bool v132;
                    v132 = 1 == v122;
                    if (v132){
                        Union4 v133 = v120.v0;
                        switch (v133.tag) {
                            case 0: { // None
                                Union1 v134;
                                v134 = Union1{Union1_1{}};
                                Union4 v135;
                                v135 = Union4{Union4_1{v134}};
                                v120.v0 = v135;
                                break;
                            }
                            case 1: { // Some
                                Union1 v136 = v133.case1.v0;
                                bool v137;
                                v137 = false;
                                bool v138;
                                v138 = v137 == false;
                                if (v138){
                                    assert("Duplicate union type instances in the unpickle Alt case." && v137);
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
                        int v140;
                        v140 = 0;
                        method_19(v0, v1, v140);
                    }
                    bool v141;
                    v141 = 2 == v122;
                    if (v141){
                        Union4 v142 = v120.v0;
                        switch (v142.tag) {
                            case 0: { // None
                                Union1 v143;
                                v143 = Union1{Union1_2{}};
                                Union4 v144;
                                v144 = Union4{Union4_1{v143}};
                                v120.v0 = v144;
                                break;
                            }
                            case 1: { // Some
                                Union1 v145 = v142.case1.v0;
                                bool v146;
                                v146 = false;
                                bool v147;
                                v147 = v146 == false;
                                if (v147){
                                    assert("Duplicate union type instances in the unpickle Alt case." && v146);
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
                        int v149;
                        v149 = 0;
                        method_19(v0, v1, v149);
                    }
                    Union4 v150 = v120.v0;
                    Union1 v154;
                    switch (v150.tag) {
                        case 0: { // None
                            printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                            exit(-1);
                            break;
                        }
                        case 1: { // Some
                            Union1 v151 = v150.case1.v0;
                            v154 = v151;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    Union0 v155;
                    v155 = Union0{Union0_3{v118, v154}};
                    Union10 v156;
                    v156 = Union10{Union10_1{v155}};
                    v11.v0 = v156;
                    break;
                }
                case 1: { // Some
                    Union0 v157 = v116.case1.v0;
                    bool v158;
                    v158 = false;
                    bool v159;
                    v159 = v158 == false;
                    if (v159){
                        assert("Duplicate union type instances in the unpickle Alt case." && v158);
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
            int v161;
            v161 = 5;
            method_19(v0, v1, v161);
        }
        bool v162;
        v162 = 4 == v13;
        if (v162){
            Union10 v163 = v11.v0;
            switch (v163.tag) {
                case 0: { // None
                    static_array<Union1,2> v164;
                    int v168;
                    v168 = 0;
                    while (while_method_1(v168)){
                        Union4 v170;
                        v170 = Union4{Union4_0{}};
                        StackMut4 v171{v170};
                        int v172;
                        v172 = 3;
                        int v173;
                        v173 = method_18(v0, v1, v172);
                        bool v174;
                        v174 = 0 == v173;
                        if (v174){
                            Union4 v175 = v171.v0;
                            switch (v175.tag) {
                                case 0: { // None
                                    Union1 v176;
                                    v176 = Union1{Union1_0{}};
                                    Union4 v177;
                                    v177 = Union4{Union4_1{v176}};
                                    v171.v0 = v177;
                                    break;
                                }
                                case 1: { // Some
                                    Union1 v178 = v175.case1.v0;
                                    bool v179;
                                    v179 = false;
                                    bool v180;
                                    v180 = v179 == false;
                                    if (v180){
                                        assert("Duplicate union type instances in the unpickle Alt case." && v179);
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
                            int v182;
                            v182 = 0;
                            method_19(v0, v1, v182);
                        }
                        bool v183;
                        v183 = 1 == v173;
                        if (v183){
                            Union4 v184 = v171.v0;
                            switch (v184.tag) {
                                case 0: { // None
                                    Union1 v185;
                                    v185 = Union1{Union1_1{}};
                                    Union4 v186;
                                    v186 = Union4{Union4_1{v185}};
                                    v171.v0 = v186;
                                    break;
                                }
                                case 1: { // Some
                                    Union1 v187 = v184.case1.v0;
                                    bool v188;
                                    v188 = false;
                                    bool v189;
                                    v189 = v188 == false;
                                    if (v189){
                                        assert("Duplicate union type instances in the unpickle Alt case." && v188);
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
                            int v191;
                            v191 = 0;
                            method_19(v0, v1, v191);
                        }
                        bool v192;
                        v192 = 2 == v173;
                        if (v192){
                            Union4 v193 = v171.v0;
                            switch (v193.tag) {
                                case 0: { // None
                                    Union1 v194;
                                    v194 = Union1{Union1_2{}};
                                    Union4 v195;
                                    v195 = Union4{Union4_1{v194}};
                                    v171.v0 = v195;
                                    break;
                                }
                                case 1: { // Some
                                    Union1 v196 = v193.case1.v0;
                                    bool v197;
                                    v197 = false;
                                    bool v198;
                                    v198 = v197 == false;
                                    if (v198){
                                        assert("Duplicate union type instances in the unpickle Alt case." && v197);
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
                            int v200;
                            v200 = 0;
                            method_19(v0, v1, v200);
                        }
                        Union4 v201 = v171.v0;
                        Union1 v205;
                        switch (v201.tag) {
                            case 0: { // None
                                printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                                exit(-1);
                                break;
                            }
                            case 1: { // Some
                                Union1 v202 = v201.case1.v0;
                                v205 = v202;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                                exit(-1);
                            }
                        }
                        v164[v168] = v205;
                        v168 += 1 ;
                    }
                    int v206;
                    v206 = 13;
                    int v207;
                    v207 = method_18(v0, v1, v206);
                    int v208;
                    v208 = 1 + v207;
                    int v209;
                    v209 = 2;
                    int v210;
                    v210 = method_18(v0, v1, v209);
                    Union0 v211;
                    v211 = Union0{Union0_4{v164, v208, v210}};
                    Union10 v212;
                    v212 = Union10{Union10_1{v211}};
                    v11.v0 = v212;
                    break;
                }
                case 1: { // Some
                    Union0 v213 = v163.case1.v0;
                    bool v214;
                    v214 = false;
                    bool v215;
                    v215 = v214 == false;
                    if (v215){
                        assert("Duplicate union type instances in the unpickle Alt case." && v214);
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
            int v217;
            v217 = 21;
            method_19(v0, v1, v217);
        }
        Union10 v218 = v11.v0;
        Union0 v222;
        switch (v218.tag) {
            case 0: { // None
                printf("%s\n", "Could not parse the union type in unpickle's Alt case.");
                exit(-1);
                break;
            }
            case 1: { // Some
                Union0 v219 = v218.case1.v0;
                v222 = v219;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        v4.push(v222);
        int v223;
        v223 = 1;
        method_19(v0, v1, v223);
        v8 += 1 ;
    }
    int v224;
    v224 = v3;
    while (while_method_6(v224)){
        int v226;
        v226 = 39;
        method_19(v0, v1, v226);
        int v227;
        v227 = 1;
        method_20(v0, v1, v227);
        v224 += 1 ;
    }
    return v4;
}
void method_22(Union1 v0){
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
void method_23(Union2 v0){
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
            printf("%s","Raise");
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
void method_21(Union0 v0){
    switch (v0.tag) {
        case 0: { // CommunityCardIs
            Union1 v1 = v0.case0.v0;
            printf("%s(","CommunityCardIs");
            method_22(v1);
            printf(")");
            return ;
            break;
        }
        case 1: { // Hidden
            printf("%s","Hidden");
            return ;
            break;
        }
        case 2: { // PlayerAction
            int v2 = v0.case2.v0; Union2 v3 = v0.case2.v1;
            printf("%s(%d, ","PlayerAction", v2);
            method_23(v3);
            printf(")");
            return ;
            break;
        }
        case 3: { // PlayerGotCard
            int v4 = v0.case3.v0; Union1 v5 = v0.case3.v1;
            printf("%s(%d, ","PlayerGotCard", v4);
            method_22(v5);
            printf(")");
            return ;
            break;
        }
        case 4: { // Showdown
            static_array<Union1,2> v6 = v0.case4.v0; int v7 = v0.case4.v1; int v8 = v0.case4.v2;
            printf("%s({%s = %s","Showdown", "cards_shown", "[");
            int v9;
            v9 = 0;
            while (while_method_1(v9)){
                Union1 v11;
                v11 = v6[v9];
                printf("");
                method_22(v11);
                printf("");
                int v15;
                v15 = v9 + 1;
                bool v16;
                v16 = v15 < 2;
                if (v16){
                    printf("%s","; ");
                } else {
                }
                v9 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d; %s = %d})","chips_won", v7, "winner_id", v8);
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
    Fun0 v0 = FunPointerMethod0;
    Fun1 v1 = FunPointerMethod1;
    std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> v2(512, v0, v1);
    StackRefs0 v3{v2};
    std::unordered_map<Tuple0, static_array<Tuple2,3>, Fun0, Fun1> v4(512, v0, v1);
    StackRefs1 v5{v4};
    af::array v6(1312, 512);
    af::array v7(6, 512);
    StackRefs2 v8{v6, v7};
    af::array & v9 = v8.v0;
    v9 = af::constant<float>(1.0f, v9.dims());
    af::array & v10 = v8.v1;
    v10 = af::constant<float>(0, v10.dims());
    xso::rng v11;
    StackMut0 v12{63u};
    static_array_list<Union0,32> v13;
    v13 = static_array_list<Union0,32>{};
    StackRefs3 v17{v13};
    static_array<Tuple2,2> v18;
    int v22;
    v22 = 0;
    while (while_method_1(v22)){
        v18[v22] = Tuple2{0.0f, 0.0f};
        v22 += 1 ;
    }
    StackRefs4 v24{v18};
    int v25; float v26;
    Tuple3 tmp0 = Tuple3{0, 0.0f};
    v25 = tmp0.v0; v26 = tmp0.v1;
    while (while_method_2(v25)){
        int v28;
        v28 = v25 % 2;
        bool v29;
        v29 = v28 == 0;
        if (v29){
            printf("{%s = %d; %s = %d}\n","i", v25, "nearTo", 50);
            fflush(stdout);
        } else {
        }
        Union3 v35;
        v35 = Union3{Union3_1{}};
        float v36;
        v36 = body_0(v17, v3, v8, v11, v12, v24, v35);
        v26 = v36;
        v25 += 1 ;
    }
    xso::rng v37;
    StackMut0 v38{63u};
    static_array_list<Union0,32> v39;
    v39 = static_array_list<Union0,32>{};
    StackRefs3 v43{v39};
    static_array<Tuple2,2> v44;
    int v48;
    v48 = 0;
    while (while_method_1(v48)){
        v44[v48] = Tuple2{0.0f, 0.0f};
        v48 += 1 ;
    }
    StackRefs4 v50{v44};
    int v51; float v52;
    Tuple3 tmp30 = Tuple3{0, 0.0f};
    v51 = tmp30.v0; v52 = tmp30.v1;
    while (while_method_7(v51)){
        int v54;
        v54 = v51 % 1;
        bool v55;
        v55 = v54 == 0;
        if (v55){
            printf("{%s = %d; %s = %d}\n","i", v51, "nearTo", 10);
            fflush(stdout);
        } else {
        }
        Union3 v61;
        v61 = Union3{Union3_1{}};
        float v62;
        v62 = body_13(v43, v3, v8, v37, v38, v50, v61);
        v52 = v62;
        v51 += 1 ;
    }
    printf("{%s = %f}\n","reward_for_pl0", v52);
    fflush(stdout);
    printf("%s\n","{");
    af::array & v67 = v8.v0;
    int v68;
    v68 = v67.dims(1);
    af::array & v69 = v8.v0;
    af::array & v70 = v8.v1;
    int v71;
    v71 = 0;
    while (while_method_0(v68, v71)){
        auto v73 = v69(af::span, v71);
        bool v74;
        v74 = af::allTrue<bool>(v73 == 1);
        Union9 v80;
        if (v74){
            v80 = Union9{Union9_0{}};
        } else {
            int v76;
            v76 = v73.elements();
            float v77[v76];
            v73.host<float>(v77);;
            static_array_list<Union0,32> v78;
            v78 = method_17(v77);
            v80 = Union9{Union9_1{v78}};
        }
        switch (v80.tag) {
            case 0: { // None
                break;
            }
            case 1: { // Some
                static_array_list<Union0,32> v81 = v80.case1.v0;
                auto v82 = v70(af::span, v71);
                int v83;
                v83 = v82.elements();
                float v84[v83];
                v82.host<float>(v84);;
                static_array<float,3> v85;
                int v89;
                v89 = 0;
                while (while_method_4(v89)){
                    float v91;
                    v91 = v84[v89];
                    v85[v89] = v91;
                    v89 += 1 ;
                }
                static_array<float,3> v92;
                int v96;
                v96 = 0;
                while (while_method_4(v96)){
                    int v98;
                    v98 = v96 + 3;
                    float v99;
                    v99 = v84[v98];
                    v92[v96] = v99;
                    v96 += 1 ;
                }
                printf("%s","[");
                int v100;
                v100 = v81.length;
                bool v101;
                v101 = 100 < v100;
                int v102;
                if (v101){
                    v102 = 100;
                } else {
                    v102 = v100;
                }
                int v103;
                v103 = 0;
                while (while_method_0(v102, v103)){
                    Union0 v105;
                    v105 = v81[v103];
                    printf("");
                    method_21(v105);
                    printf("");
                    int v109;
                    v109 = v103 + 1;
                    int v110;
                    v110 = v81.length;
                    bool v111;
                    v111 = v109 < v110;
                    if (v111){
                        printf("%s","; ");
                    } else {
                    }
                    v103 += 1 ;
                }
                int v112;
                v112 = v81.length;
                bool v113;
                v113 = v112 > 100;
                if (v113){
                    printf("%s","; ...");
                } else {
                }
                printf("%s","]");
                printf("");
                printf("%s"," => ");
                printf("{%s = %s","average_policy", "[");
                int v114;
                v114 = 0;
                while (while_method_4(v114)){
                    float v116;
                    v116 = v85[v114];
                    printf("%f",v116);
                    int v120;
                    v120 = v114 + 1;
                    bool v121;
                    v121 = v120 < 3;
                    if (v121){
                        printf("%s","; ");
                    } else {
                    }
                    v114 += 1 ;
                }
                printf("%s","]");
                printf("; %s = %s","current_policy", "[");
                int v122;
                v122 = 0;
                while (while_method_4(v122)){
                    float v124;
                    v124 = v92[v122];
                    printf("%f",v124);
                    int v128;
                    v128 = v122 + 1;
                    bool v129;
                    v129 = v128 < 3;
                    if (v129){
                        printf("%s","; ");
                    } else {
                    }
                    v122 += 1 ;
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
        v71 += 1 ;
    }
    printf("%s\n","}");
    printf("\n");
    fflush(stdout);
    return 0;
}
