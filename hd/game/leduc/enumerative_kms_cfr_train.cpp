#include "enumerative_kms_cfr_train.hpp"
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
    v1 = v0 < 500;
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
    v1 = v0 < 1121;
    return v1;
}
void method_7(float * v0, static_array_list<Union0,32> v1){
    StackMut2 v2{0};
    int v3 = v2.v0;
    v0[v3] = 1.0f;
    int v4 = v2.v0;
    int v5;
    v5 = v4 + 1;
    v2.v0 = v5;
    int v6;
    v6 = v1.length;
    int v7;
    v7 = v1.length;
    int v8;
    v8 = 0;
    while (while_method_0(v7, v8)){
        Union0 v10;
        v10 = v1[v8];
        int v14 = v2.v0;
        int v15;
        v15 = v14 + 35;
        switch (v10.tag) {
            case 0: { // CommunityCardIs
                Union1 v16 = v10.case0.v0;
                int v17 = v2.v0;
                v2.v0 = v17;
                int v18 = v2.v0;
                int v19;
                v19 = v18 + 3;
                switch (v16.tag) {
                    case 0: { // Jack
                        int v20 = v2.v0;
                        v2.v0 = v20;
                        int v21 = v2.v0;
                        v0[v21] = 1.0f;
                        int v22 = v2.v0;
                        int v23;
                        v23 = v22 + 1;
                        v2.v0 = v23;
                        break;
                    }
                    case 1: { // King
                        int v24 = v2.v0;
                        int v25;
                        v25 = v24 + 1;
                        v2.v0 = v25;
                        int v26 = v2.v0;
                        v0[v26] = 1.0f;
                        int v27 = v2.v0;
                        int v28;
                        v28 = v27 + 1;
                        v2.v0 = v28;
                        break;
                    }
                    case 2: { // Queen
                        int v29 = v2.v0;
                        int v30;
                        v30 = v29 + 2;
                        v2.v0 = v30;
                        int v31 = v2.v0;
                        v0[v31] = 1.0f;
                        int v32 = v2.v0;
                        int v33;
                        v33 = v32 + 1;
                        v2.v0 = v33;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                v2.v0 = v19;
                break;
            }
            case 1: { // Hidden
                int v34 = v2.v0;
                int v35;
                v35 = v34 + 3;
                v2.v0 = v35;
                int v36 = v2.v0;
                v0[v36] = 1.0f;
                int v37 = v2.v0;
                int v38;
                v38 = v37 + 1;
                v2.v0 = v38;
                break;
            }
            case 2: { // PlayerAction
                int v39 = v10.case2.v0; Union2 v40 = v10.case2.v1;
                int v41 = v2.v0;
                int v42;
                v42 = v41 + 4;
                v2.v0 = v42;
                bool v43;
                v43 = 0 <= v39;
                bool v44;
                v44 = v43 == false;
                if (v44){
                    assert("The input to the pickler must be 0 or positive." && v43);
                } else {
                }
                bool v46;
                v46 = v39 < 2;
                bool v47;
                v47 = v46 == false;
                if (v47){
                    assert("The input to the pickler must be less than the specified length." && v46);
                } else {
                }
                int v49 = v2.v0;
                int v50;
                v50 = v49 + v39;
                v0[v50] = 1.0f;
                int v51 = v2.v0;
                int v52;
                v52 = v51 + 2;
                v2.v0 = v52;
                int v53 = v2.v0;
                int v54;
                v54 = v53 + 3;
                switch (v40.tag) {
                    case 0: { // Call
                        int v55 = v2.v0;
                        v2.v0 = v55;
                        int v56 = v2.v0;
                        v0[v56] = 1.0f;
                        int v57 = v2.v0;
                        int v58;
                        v58 = v57 + 1;
                        v2.v0 = v58;
                        break;
                    }
                    case 1: { // Fold
                        int v59 = v2.v0;
                        int v60;
                        v60 = v59 + 1;
                        v2.v0 = v60;
                        int v61 = v2.v0;
                        v0[v61] = 1.0f;
                        int v62 = v2.v0;
                        int v63;
                        v63 = v62 + 1;
                        v2.v0 = v63;
                        break;
                    }
                    case 2: { // Raise
                        int v64 = v2.v0;
                        int v65;
                        v65 = v64 + 2;
                        v2.v0 = v65;
                        int v66 = v2.v0;
                        v0[v66] = 1.0f;
                        int v67 = v2.v0;
                        int v68;
                        v68 = v67 + 1;
                        v2.v0 = v68;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                v2.v0 = v54;
                break;
            }
            case 3: { // PlayerGotCard
                int v69 = v10.case3.v0; Union1 v70 = v10.case3.v1;
                int v71 = v2.v0;
                int v72;
                v72 = v71 + 9;
                v2.v0 = v72;
                bool v73;
                v73 = 0 <= v69;
                bool v74;
                v74 = v73 == false;
                if (v74){
                    assert("The input to the pickler must be 0 or positive." && v73);
                } else {
                }
                bool v76;
                v76 = v69 < 2;
                bool v77;
                v77 = v76 == false;
                if (v77){
                    assert("The input to the pickler must be less than the specified length." && v76);
                } else {
                }
                int v79 = v2.v0;
                int v80;
                v80 = v79 + v69;
                v0[v80] = 1.0f;
                int v81 = v2.v0;
                int v82;
                v82 = v81 + 2;
                v2.v0 = v82;
                int v83 = v2.v0;
                int v84;
                v84 = v83 + 3;
                switch (v70.tag) {
                    case 0: { // Jack
                        int v85 = v2.v0;
                        v2.v0 = v85;
                        int v86 = v2.v0;
                        v0[v86] = 1.0f;
                        int v87 = v2.v0;
                        int v88;
                        v88 = v87 + 1;
                        v2.v0 = v88;
                        break;
                    }
                    case 1: { // King
                        int v89 = v2.v0;
                        int v90;
                        v90 = v89 + 1;
                        v2.v0 = v90;
                        int v91 = v2.v0;
                        v0[v91] = 1.0f;
                        int v92 = v2.v0;
                        int v93;
                        v93 = v92 + 1;
                        v2.v0 = v93;
                        break;
                    }
                    case 2: { // Queen
                        int v94 = v2.v0;
                        int v95;
                        v95 = v94 + 2;
                        v2.v0 = v95;
                        int v96 = v2.v0;
                        v0[v96] = 1.0f;
                        int v97 = v2.v0;
                        int v98;
                        v98 = v97 + 1;
                        v2.v0 = v98;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                v2.v0 = v84;
                break;
            }
            case 4: { // Showdown
                static_array<Union1,2> v99 = v10.case4.v0; int v100 = v10.case4.v1; int v101 = v10.case4.v2;
                int v102 = v2.v0;
                int v103;
                v103 = v102 + 14;
                v2.v0 = v103;
                int v104;
                v104 = 0;
                while (while_method_1(v104)){
                    Union1 v106;
                    v106 = v99[v104];
                    int v110 = v2.v0;
                    int v111;
                    v111 = v110 + 3;
                    switch (v106.tag) {
                        case 0: { // Jack
                            int v112 = v2.v0;
                            v2.v0 = v112;
                            int v113 = v2.v0;
                            v0[v113] = 1.0f;
                            int v114 = v2.v0;
                            int v115;
                            v115 = v114 + 1;
                            v2.v0 = v115;
                            break;
                        }
                        case 1: { // King
                            int v116 = v2.v0;
                            int v117;
                            v117 = v116 + 1;
                            v2.v0 = v117;
                            int v118 = v2.v0;
                            v0[v118] = 1.0f;
                            int v119 = v2.v0;
                            int v120;
                            v120 = v119 + 1;
                            v2.v0 = v120;
                            break;
                        }
                        case 2: { // Queen
                            int v121 = v2.v0;
                            int v122;
                            v122 = v121 + 2;
                            v2.v0 = v122;
                            int v123 = v2.v0;
                            v0[v123] = 1.0f;
                            int v124 = v2.v0;
                            int v125;
                            v125 = v124 + 1;
                            v2.v0 = v125;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    v2.v0 = v111;
                    v104 += 1 ;
                }
                int v126;
                v126 = -1 + v100;
                bool v127;
                v127 = 0 <= v126;
                bool v128;
                v128 = v127 == false;
                if (v128){
                    assert("The input to the pickler must be 0 or positive." && v127);
                } else {
                }
                bool v130;
                v130 = v126 < 13;
                bool v131;
                v131 = v130 == false;
                if (v131){
                    assert("The input to the pickler must be less than the specified length." && v130);
                } else {
                }
                int v133 = v2.v0;
                int v134;
                v134 = v133 + v126;
                v0[v134] = 1.0f;
                int v135 = v2.v0;
                int v136;
                v136 = v135 + 13;
                v2.v0 = v136;
                bool v137;
                v137 = 0 <= v101;
                bool v138;
                v138 = v137 == false;
                if (v138){
                    assert("The input to the pickler must be 0 or positive." && v137);
                } else {
                }
                bool v140;
                v140 = v101 < 2;
                bool v141;
                v141 = v140 == false;
                if (v141){
                    assert("The input to the pickler must be less than the specified length." && v140);
                } else {
                }
                int v143 = v2.v0;
                int v144;
                v144 = v143 + v101;
                v0[v144] = 1.0f;
                int v145 = v2.v0;
                int v146;
                v146 = v145 + 2;
                v2.v0 = v146;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        v2.v0 = v15;
        v8 += 1 ;
    }
    int v147;
    v147 = 32 - v6;
    int v148;
    v148 = 35 * v147;
    int v149 = v2.v0;
    int v150;
    v150 = v149 + v148;
    v2.v0 = v150;
    return ;
}
void method_8(StackRefs5 & v0, StackRefs6 & v1){
    af::array & v2 = v0.v0; af::array & v3 = v0.v1; af::array & v4 = v0.v2;
    af::array & v5 = v1.v0; af::array & v6 = v1.v1;
    af::array v7;
    v7 = af::matmul(af::transpose(v2), v3);
    af::array v8;
    af::max(v8, v6, v7, 1);
    v5 = af::lookup(v4, v6, 1);
    return ;
}
void method_9(StackRefs7 & v0, StackRefs8 & v1){
    af::array & v2 = v0.v0; af::array & v3 = v0.v1;
    af::array & v4 = v1.v0;
    af::array v5;
    v5 = v2 / af::tile(af::sqrt(af::sum(v2 * v2, 0)), v2.dims(0));
    int v6;
    v6 = v5.dims(0);
    af::array v7;
    v7 = af::lookup(v4, v3, 1);
    af::array v8;
    v8 = v5 - af::tile(af::sum(v5 * v7, 0), v6) * v7;
    af::array v9;
    v9 = v7 + 1.0f * v8;
    af::array v10;
    v10 = v9 / af::tile(af::sqrt(af::sum(v9 * v9, 0)), v9.dims(0));
    v4(af::span, v3) = v10;
    return ;
}
void method_10(StackRefs9 & v0, StackRefs10 & v1){
    af::array & v2 = v0.v0; af::array & v3 = v0.v1;
    af::array & v4 = v1.v0;
    v4(af::span, v2) = v3;
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
    float v65[1121];
    int v66;
    v66 = 0;
    while (while_method_5(v66)){
        v65[v66] = 0.0f;
        v66 += 1 ;
    }
    method_7(v65, v48);
    af::array v68(1121, 1, v65);
    af::array v69;
    af::array v70;
    af::array & v71 = v10.v0;
    af::array & v72 = v10.v1;
    StackRefs5 v73{v68, v71, v72};
    StackRefs6 v74{v69, v70};
    method_8(v73, v74);
    StackRefs7 v75{v68, v70};
    af::array & v76 = v10.v0;
    StackRefs8 v77{v76};
    method_9(v75, v77);
    int v78;
    v78 = v69.elements();
    float v79[v78];
    v69.host(v79);;
    static_array<float,3> v80;
    int v84;
    v84 = 0;
    while (while_method_4(v84)){
        float v86;
        v86 = v79[v84];
        v80[v84] = v86;
        v84 += 1 ;
    }
    static_array<float,3> v87;
    int v91;
    v91 = 0;
    while (while_method_4(v91)){
        int v93;
        v93 = v91 + 3;
        float v94;
        v94 = v79[v93];
        v87[v91] = v94;
        v91 += 1 ;
    }
    static_array<float,3> v95;
    v95 = regret_match_3(v87, v38);
    float v124;
    switch (v31.tag) {
        case 0: { // None
            v124 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v96 = v31.case1.v0;
            float v97;
            v97 = v95[2];
            static_array<Tuple2,2> & v101 = v2.v0;
            float v102; float v103;
            Tuple2 tmp22 = v101[v6];
            v102 = tmp22.v0; v103 = tmp22.v1;
            static_array<Tuple2,2> & v110 = v2.v0;
            float v111;
            v111 = log(v97);
            float v112;
            v112 = v111 + v102;
            v110[v6] = Tuple2{v112, v103};
            static_array_list<Union0,32> & v113 = v1.v0;
            Union0 v114;
            v114 = Union0{Union0_2{v6, v96}};
            v113.push(v114);
            Union5 v115;
            v115 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v96}};
            float v116;
            v116 = loop_1(v1, v9, v10, v0, v11, v2, v12, v115);
            static_array_list<Union0,32> & v117 = v1.v0;
            Union0 v118;
            v118 = v117.pop();
            static_array<Tuple2,2> & v119 = v2.v0;
            v119[v6] = Tuple2{v102, v103};
            bool v120;
            v120 = v6 == 0;
            if (v120){
                v124 = v116;
            } else {
                float v121;
                v121 = -v116;
                v124 = v121;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v153;
    switch (v26.tag) {
        case 0: { // None
            v153 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v125 = v26.case1.v0;
            float v126;
            v126 = v95[1];
            static_array<Tuple2,2> & v130 = v2.v0;
            float v131; float v132;
            Tuple2 tmp23 = v130[v6];
            v131 = tmp23.v0; v132 = tmp23.v1;
            static_array<Tuple2,2> & v139 = v2.v0;
            float v140;
            v140 = log(v126);
            float v141;
            v141 = v140 + v131;
            v139[v6] = Tuple2{v141, v132};
            static_array_list<Union0,32> & v142 = v1.v0;
            Union0 v143;
            v143 = Union0{Union0_2{v6, v125}};
            v142.push(v143);
            Union5 v144;
            v144 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v125}};
            float v145;
            v145 = loop_1(v1, v9, v10, v0, v11, v2, v12, v144);
            static_array_list<Union0,32> & v146 = v1.v0;
            Union0 v147;
            v147 = v146.pop();
            static_array<Tuple2,2> & v148 = v2.v0;
            v148[v6] = Tuple2{v131, v132};
            bool v149;
            v149 = v6 == 0;
            if (v149){
                v153 = v145;
            } else {
                float v150;
                v150 = -v145;
                v153 = v150;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v154;
    v154 = v95[0];
    static_array<Tuple2,2> & v158 = v2.v0;
    float v159; float v160;
    Tuple2 tmp24 = v158[v6];
    v159 = tmp24.v0; v160 = tmp24.v1;
    static_array<Tuple2,2> & v167 = v2.v0;
    float v168;
    v168 = log(v154);
    float v169;
    v169 = v168 + v159;
    v167[v6] = Tuple2{v169, v160};
    static_array_list<Union0,32> & v170 = v1.v0;
    Union2 v171;
    v171 = Union2{Union2_0{}};
    Union0 v172;
    v172 = Union0{Union0_2{v6, v171}};
    v170.push(v172);
    Union2 v173;
    v173 = Union2{Union2_0{}};
    Union5 v174;
    v174 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v173}};
    float v175;
    v175 = loop_1(v1, v9, v10, v0, v11, v2, v12, v174);
    static_array_list<Union0,32> & v176 = v1.v0;
    Union0 v177;
    v177 = v176.pop();
    static_array<Tuple2,2> & v178 = v2.v0;
    v178[v6] = Tuple2{v159, v160};
    bool v179;
    v179 = v6 == 0;
    float v181;
    if (v179){
        v181 = v175;
    } else {
        float v180;
        v180 = -v175;
        v181 = v180;
    }
    static_array<float,3> v182;
    v182[0] = v181;
    v182[1] = v153;
    v182[2] = v124;
    int v186; float v187;
    Tuple3 tmp25 = Tuple3{0, 0.0f};
    v186 = tmp25.v0; v187 = tmp25.v1;
    while (while_method_4(v186)){
        float v189;
        v189 = v182[v186];
        float v193;
        v193 = v95[v186];
        float v197;
        v197 = v189 * v193;
        float v198;
        v198 = v187 + v197;
        v187 = v198;
        v186 += 1 ;
    }
    static_array<float,3> v199;
    int v203;
    v203 = 0;
    while (while_method_4(v203)){
        float v205;
        v205 = v80[v203];
        float v209;
        v209 = v95[v203];
        float v213;
        v213 = 0.99609375f * v205;
        float v214;
        v214 = v213 + v209;
        v199[v203] = v214;
        v203 += 1 ;
    }
    static_array<Tuple2,2> & v215 = v2.v0;
    int v216; float v217;
    Tuple3 tmp26 = Tuple3{0, 0.0f};
    v216 = tmp26.v0; v217 = tmp26.v1;
    while (while_method_1(v216)){
        float v219; float v220;
        Tuple2 tmp27 = v215[v216];
        v219 = tmp27.v0; v220 = tmp27.v1;
        bool v227;
        v227 = v216 == v6;
        float v228;
        if (v227){
            v228 = 0.0f;
        } else {
            v228 = v219;
        }
        float v229;
        v229 = v217 + v228;
        float v230;
        v230 = v229 - v220;
        v217 = v230;
        v216 += 1 ;
    }
    float v231;
    v231 = exp(v217);
    static_array<float,3> v232;
    int v236;
    v236 = 0;
    while (while_method_4(v236)){
        float v238;
        v238 = v87[v236];
        float v242;
        v242 = v182[v236];
        float v246;
        v246 = v242 - v187;
        float v247;
        v247 = v231 * v246;
        float v248;
        v248 = v238 + v247;
        bool v249;
        v249 = 0.0f >= v248;
        float v250;
        if (v249){
            v250 = 0.0f;
        } else {
            v250 = v248;
        }
        v232[v236] = v250;
        v236 += 1 ;
    }
    af::array & v251 = v10.v0;
    int v252;
    v252 = v251.dims(1);
    af::array & v253 = v10.v0;
    af::array & v254 = v10.v1;
    float v255[6];
    int v256;
    v256 = 0;
    while (while_method_3(v256)){
        v255[v256] = 0.0f;
        v256 += 1 ;
    }
    int v258;
    v258 = 0;
    while (while_method_4(v258)){
        float v260;
        v260 = v199[v258];
        v255[v258] = v260;
        v258 += 1 ;
    }
    int v264;
    v264 = 0;
    while (while_method_4(v264)){
        int v266;
        v266 = v264 + 3;
        float v267;
        v267 = v232[v264];
        v255[v266] = v267;
        v264 += 1 ;
    }
    af::array v271(6, 1, v255);
    af::array v272;
    v272 = v271;
    StackRefs9 v273{v70, v272};
    StackRefs10 v274{v254};
    method_10(v273, v274);
    return v187;
}
int tag_12(Union1 v0){
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
bool is_pair_13(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
Tuple5 order_14(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple5{v1, v0};
    } else {
        return Tuple5{v0, v1};
    }
}
Union8 compare_hands_11(Union4 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            exit(-1);
            break;
        }
        case 1: { // Some
            Union1 v7 = v0.case1.v0;
            int v8;
            v8 = tag_12(v7);
            Union1 v9;
            v9 = v2[0];
            int v13;
            v13 = tag_12(v9);
            Union1 v14;
            v14 = v2[1];
            int v18;
            v18 = tag_12(v14);
            bool v19;
            v19 = is_pair_13(v8, v13);
            bool v20;
            v20 = is_pair_13(v8, v18);
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
                    Tuple5 tmp28 = order_14(v8, v13);
                    v31 = tmp28.v0; v32 = tmp28.v1;
                    int v33; int v34;
                    Tuple5 tmp29 = order_14(v8, v18);
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
            v40 = compare_hands_11(v30, v31, v32, v33, v34, v35);
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
inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 100;
    return v1;
}
float loop_16(StackRefs3 & v0, StackRefs0 & v1, StackRefs2 & v2, xso::rng & v3, StackMut0 & v4, StackRefs4 & v5, StackMut1 & v6, Union5 v7){
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
            return body_15(v0, v1, v2, v3, v4, v5, v35);
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
            return body_15(v0, v1, v2, v3, v4, v5, v51);
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
            return body_15(v0, v1, v2, v3, v4, v5, v161);
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
float method_17(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs0 & v9, StackRefs2 & v10, StackMut0 & v11, StackMut1 & v12){
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
            v182 = loop_16(v1, v9, v10, v0, v11, v2, v12, v181);
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
            v211 = loop_16(v1, v9, v10, v0, v11, v2, v12, v210);
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
    v241 = loop_16(v1, v9, v10, v0, v11, v2, v12, v240);
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
float method_18(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs0 & v9, StackRefs2 & v10, StackMut0 & v11, StackMut1 & v12){
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
    float v65[1121];
    int v66;
    v66 = 0;
    while (while_method_5(v66)){
        v65[v66] = 0.0f;
        v66 += 1 ;
    }
    method_7(v65, v48);
    af::array v68(1121, 1, v65);
    af::array v69;
    af::array v70;
    af::array & v71 = v10.v0;
    af::array & v72 = v10.v1;
    StackRefs5 v73{v68, v71, v72};
    StackRefs6 v74{v69, v70};
    method_8(v73, v74);
    StackRefs7 v75{v68, v70};
    af::array & v76 = v10.v0;
    StackRefs8 v77{v76};
    method_9(v75, v77);
    int v78;
    v78 = v69.elements();
    float v79[v78];
    v69.host(v79);;
    static_array<float,3> v80;
    int v84;
    v84 = 0;
    while (while_method_4(v84)){
        float v86;
        v86 = v79[v84];
        v80[v84] = v86;
        v84 += 1 ;
    }
    static_array<float,3> v87;
    int v91;
    v91 = 0;
    while (while_method_4(v91)){
        int v93;
        v93 = v91 + 3;
        float v94;
        v94 = v79[v93];
        v87[v91] = v94;
        v91 += 1 ;
    }
    static_array<float,3> v95;
    v95 = masking_normalize_5(v80, v38);
    float v124;
    switch (v31.tag) {
        case 0: { // None
            v124 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v96 = v31.case1.v0;
            float v97;
            v97 = v95[2];
            static_array<Tuple2,2> & v101 = v2.v0;
            float v102; float v103;
            Tuple2 tmp50 = v101[v6];
            v102 = tmp50.v0; v103 = tmp50.v1;
            static_array<Tuple2,2> & v110 = v2.v0;
            float v111;
            v111 = log(v97);
            float v112;
            v112 = v111 + v102;
            v110[v6] = Tuple2{v112, v103};
            static_array_list<Union0,32> & v113 = v1.v0;
            Union0 v114;
            v114 = Union0{Union0_2{v6, v96}};
            v113.push(v114);
            Union5 v115;
            v115 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v96}};
            float v116;
            v116 = loop_16(v1, v9, v10, v0, v11, v2, v12, v115);
            static_array_list<Union0,32> & v117 = v1.v0;
            Union0 v118;
            v118 = v117.pop();
            static_array<Tuple2,2> & v119 = v2.v0;
            v119[v6] = Tuple2{v102, v103};
            bool v120;
            v120 = v6 == 0;
            if (v120){
                v124 = v116;
            } else {
                float v121;
                v121 = -v116;
                v124 = v121;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v153;
    switch (v26.tag) {
        case 0: { // None
            v153 = 0.0f;
            break;
        }
        case 1: { // Some
            Union2 v125 = v26.case1.v0;
            float v126;
            v126 = v95[1];
            static_array<Tuple2,2> & v130 = v2.v0;
            float v131; float v132;
            Tuple2 tmp51 = v130[v6];
            v131 = tmp51.v0; v132 = tmp51.v1;
            static_array<Tuple2,2> & v139 = v2.v0;
            float v140;
            v140 = log(v126);
            float v141;
            v141 = v140 + v131;
            v139[v6] = Tuple2{v141, v132};
            static_array_list<Union0,32> & v142 = v1.v0;
            Union0 v143;
            v143 = Union0{Union0_2{v6, v125}};
            v142.push(v143);
            Union5 v144;
            v144 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v125}};
            float v145;
            v145 = loop_16(v1, v9, v10, v0, v11, v2, v12, v144);
            static_array_list<Union0,32> & v146 = v1.v0;
            Union0 v147;
            v147 = v146.pop();
            static_array<Tuple2,2> & v148 = v2.v0;
            v148[v6] = Tuple2{v131, v132};
            bool v149;
            v149 = v6 == 0;
            if (v149){
                v153 = v145;
            } else {
                float v150;
                v150 = -v145;
                v153 = v150;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    float v154;
    v154 = v95[0];
    static_array<Tuple2,2> & v158 = v2.v0;
    float v159; float v160;
    Tuple2 tmp52 = v158[v6];
    v159 = tmp52.v0; v160 = tmp52.v1;
    static_array<Tuple2,2> & v167 = v2.v0;
    float v168;
    v168 = log(v154);
    float v169;
    v169 = v168 + v159;
    v167[v6] = Tuple2{v169, v160};
    static_array_list<Union0,32> & v170 = v1.v0;
    Union2 v171;
    v171 = Union2{Union2_0{}};
    Union0 v172;
    v172 = Union0{Union0_2{v6, v171}};
    v170.push(v172);
    Union2 v173;
    v173 = Union2{Union2_0{}};
    Union5 v174;
    v174 = Union5{Union5_2{v3, v4, v5, v6, v7, v8, v173}};
    float v175;
    v175 = loop_16(v1, v9, v10, v0, v11, v2, v12, v174);
    static_array_list<Union0,32> & v176 = v1.v0;
    Union0 v177;
    v177 = v176.pop();
    static_array<Tuple2,2> & v178 = v2.v0;
    v178[v6] = Tuple2{v159, v160};
    bool v179;
    v179 = v6 == 0;
    float v181;
    if (v179){
        v181 = v175;
    } else {
        float v180;
        v180 = -v175;
        v181 = v180;
    }
    static_array<float,3> v182;
    v182[0] = v181;
    v182[1] = v153;
    v182[2] = v124;
    int v186; float v187;
    Tuple3 tmp53 = Tuple3{0, 0.0f};
    v186 = tmp53.v0; v187 = tmp53.v1;
    while (while_method_4(v186)){
        float v189;
        v189 = v182[v186];
        float v193;
        v193 = v95[v186];
        float v197;
        v197 = v189 * v193;
        float v198;
        v198 = v187 + v197;
        v187 = v198;
        v186 += 1 ;
    }
    return v187;
}
float body_15(StackRefs3 & v0, StackRefs0 & v1, StackRefs2 & v2, xso::rng & v3, StackMut0 & v4, StackRefs4 & v5, Union3 v6){
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
                    v129 = loop_16(v0, v1, v2, v3, v4, v5, v7, v128);
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
                            v207 = loop_16(v0, v1, v2, v3, v4, v5, v7, v206);
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
                v72 = method_17(v3, v0, v5, v60, v61, v62, v63, v64, v65, v1, v2, v4, v7);
            } else {
                bool v68;
                v68 = v63 == 1;
                if (v68){
                    v72 = method_18(v3, v0, v5, v60, v61, v62, v63, v64, v65, v1, v2, v4, v7);
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
            return loop_16(v0, v1, v2, v3, v4, v5, v7, v75);
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
            v87 = loop_16(v0, v1, v2, v3, v4, v5, v7, v86);
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
            v40 = compare_hands_11(v30, v31, v32, v33, v34, v35);
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
            v57 = loop_16(v0, v1, v2, v3, v4, v5, v7, v56);
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
            v27 = loop_16(v0, v1, v2, v3, v4, v5, v7, v26);
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
int main() {
    Fun0 v0 = FunPointerMethod0;
    Fun1 v1 = FunPointerMethod1;
    std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> v2(512, v0, v1);
    StackRefs0 v3{v2};
    std::unordered_map<Tuple0, static_array<Tuple2,3>, Fun0, Fun1> v4(512, v0, v1);
    StackRefs1 v5{v4};
    af::array v6(1121, 288);
    af::array v7(6, 288);
    StackRefs2 v8{v6, v7};
    af::array & v9 = v8.v0;
    int v10;
    v10 = v9.dims(1);
    af::array & v11 = v8.v0;
    int v12;
    v12 = v11.dims(0);
    af::array v13;
    v13 = af::randu(v12, v10);
    af::array v14;
    v14 = v13 / af::tile(af::sqrt(af::sum(v13 * v13, 0)), v13.dims(0));
    af::array & v15 = v8.v0;
    v15 = v14;
    af::array & v16 = v8.v1;
    v16 = af::constant<float>(0, v16.dims());
    xso::rng v17;
    StackMut0 v18{63u};
    static_array_list<Union0,32> v19;
    v19 = static_array_list<Union0,32>{};
    StackRefs3 v23{v19};
    static_array<Tuple2,2> v24;
    int v28;
    v28 = 0;
    while (while_method_1(v28)){
        v24[v28] = Tuple2{0.0f, 0.0f};
        v28 += 1 ;
    }
    StackRefs4 v30{v24};
    int v31; float v32;
    Tuple3 tmp0 = Tuple3{0, 0.0f};
    v31 = tmp0.v0; v32 = tmp0.v1;
    while (while_method_2(v31)){
        int v34;
        v34 = v31 % 20;
        bool v35;
        v35 = v34 == 0;
        if (v35){
            printf("{%s = %d; %s = %d}\n","i", v31, "nearTo", 500);
            fflush(stdout);
        } else {
        }
        Union3 v41;
        v41 = Union3{Union3_1{}};
        float v42;
        v42 = body_0(v23, v3, v8, v17, v18, v30, v41);
        v32 = v42;
        v31 += 1 ;
    }
    xso::rng v43;
    StackMut0 v44{63u};
    static_array_list<Union0,32> v45;
    v45 = static_array_list<Union0,32>{};
    StackRefs3 v49{v45};
    static_array<Tuple2,2> v50;
    int v54;
    v54 = 0;
    while (while_method_1(v54)){
        v50[v54] = Tuple2{0.0f, 0.0f};
        v54 += 1 ;
    }
    StackRefs4 v56{v50};
    int v57; float v58;
    Tuple3 tmp30 = Tuple3{0, 0.0f};
    v57 = tmp30.v0; v58 = tmp30.v1;
    while (while_method_6(v57)){
        int v60;
        v60 = v57 % 4;
        bool v61;
        v61 = v60 == 0;
        if (v61){
            printf("{%s = %d; %s = %d}\n","i", v57, "nearTo", 100);
            fflush(stdout);
        } else {
        }
        Union3 v67;
        v67 = Union3{Union3_1{}};
        float v68;
        v68 = body_15(v49, v3, v8, v43, v44, v56, v67);
        v58 = v68;
        v57 += 1 ;
    }
    printf("{%s = %f}\n","reward_for_pl0", v58);
    fflush(stdout);
    return 0;
}
