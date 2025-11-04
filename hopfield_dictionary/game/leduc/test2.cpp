#include "test2.hpp"
inline bool while_method_7(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 5;
    return v1;
}
inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
void method_5(float * v0, static_array_list<Union3,5> v1, Union4 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6){
    StackMut6 v7{0};
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
    while (while_method_7(v46, v47)){
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
    while (while_method_8(v76)){
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
    while (while_method_9(v83)){
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
void method_10(Eigen::Matrix<float,1,115> & v0, Eigen::Matrix<float,1,48> & v1, Eigen::Matrix<float,1,115> & v2, Eigen::Matrix<float,1,48> & v3){
    v3 = ((v0 * v2.transpose() - v2.rowwise().sum().transpose().replicate(v0.rows(),1)).array() / 0.001f).exp().matrix().transpose() * v1;
    return ;
}
inline bool while_method_11(int v0){
    bool v1;
    v1 = v0 < 12;
    return v1;
}
void method_12(Union3 v0){
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
int main() {
    Eigen::Matrix<float,1,115> v0;
    Eigen::Matrix<float,1,48> v1;
    int v2;
    v2 = 0;
    StackRefs2 v3{v2, v0, v1};
    Eigen::Matrix<float,1,115> & v4 = v3.v1;
    v4.setZero();
    Eigen::Matrix<float,1,48> & v5 = v3.v2;
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
    Eigen::Matrix<float,1,115> v30;
    v30.setZero();
    float * v31;
    v31 = &v30(0,0);
    Union4 v32;
    v32 = Union4{Union4_1{}};
    bool v33;
    v33 = false;
    int v34;
    v34 = 8;
    unsigned int v35;
    v35 = 5u;
    method_5(v31, v8, v32, v33, v28, v34, v35);
    Eigen::Matrix<float,1,48> v36;
    Eigen::Matrix<float,1,115> & v37 = v3.v1;
    Eigen::Matrix<float,1,115> & v38 = v37;
    Eigen::Matrix<float,1,48> & v39 = v3.v2;
    Eigen::Matrix<float,1,48> & v40 = v39;
    Eigen::Matrix<float,1,115> & v41 = v30;
    Eigen::Matrix<float,1,48> & v42 = v36;
    method_10(v38, v40, v41, v42);
    static_array<float,12> v45;
    int v47;
    v47 = 0;
    while (while_method_11(v47)){
        float v49;
        v49 = v36(0,v47);
        v45[v47] = v49;
        v47 += 1 ;
    }
    static_array<float,12> v52;
    int v54;
    v54 = 0;
    while (while_method_11(v54)){
        int v56;
        v56 = v54 + 12;
        float v57;
        v57 = v36(0,v56);
        v52[v54] = v57;
        v54 += 1 ;
    }
    static_array<float,12> v60;
    int v62;
    v62 = 0;
    while (while_method_11(v62)){
        int v64;
        v64 = v62 + 24;
        float v65;
        v65 = v36(0,v64);
        v60[v62] = v65;
        v62 += 1 ;
    }
    static_array<float,12> v68;
    int v70;
    v70 = 0;
    while (while_method_11(v70)){
        int v72;
        v72 = v70 + 36;
        float v73;
        v73 = v36(0,v72);
        v68[v70] = v73;
        v70 += 1 ;
    }
    printf("{%s = %s","action_history", "[");
    int v120;
    v120 = v8.length;
    bool v121;
    v121 = 100 < v120;
    int v122;
    if (v121){
        v122 = 100;
    } else {
        v122 = v120;
    }
    int v123;
    v123 = 0;
    while (while_method_7(v122, v123)){
        Union3 v127;
        v127 = v8[v123];
        printf("");
        method_12(v127);
        printf("");
        int v129;
        v129 = v123 + 1;
        int v130;
        v130 = v8.length;
        bool v131;
        v131 = v129 < v130;
        if (v131){
            printf("%s","; ");
        } else {
        }
        v123 += 1 ;
    }
    int v132;
    v132 = v8.length;
    bool v133;
    v133 = v132 > 100;
    if (v133){
        printf("%s","; ...");
    } else {
    }
    printf("%s","]");
    printf("; %s = %s; %s = %s; %s = %s","card", "King", "is_first", "false", "l", "[");
    int v134;
    v134 = 0;
    while (while_method_9(v134)){
        int v138;
        v138 = v28[v134];
        printf("%d",v138);
        int v140;
        v140 = v134 + 1;
        bool v141;
        v141 = v140 < 3;
        if (v141){
            printf("%s","; ");
        } else {
        }
        v134 += 1 ;
    }
    printf("%s","]");
    printf("; %s = %d; %s = %u}\n","pot", 8, "stack", 5u);
    fflush(stdout);
    printf("{%s = %s","average_policy", "[");
    int v250;
    v250 = 0;
    while (while_method_11(v250)){
        float v254;
        v254 = v45[v250];
        printf("%f",v254);
        int v256;
        v256 = v250 + 1;
        bool v257;
        v257 = v256 < 12;
        if (v257){
            printf("%s","; ");
        } else {
        }
        v250 += 1 ;
    }
    printf("%s","]");
    printf("; %s = %s","current_policy", "[");
    int v258;
    v258 = 0;
    while (while_method_11(v258)){
        float v262;
        v262 = v52[v258];
        printf("%f",v262);
        int v264;
        v264 = v258 + 1;
        bool v265;
        v265 = v264 < 12;
        if (v265){
            printf("%s","; ");
        } else {
        }
        v258 += 1 ;
    }
    printf("%s","]");
    printf("; %s = %s","ev_values", "[");
    int v266;
    v266 = 0;
    while (while_method_11(v266)){
        float v270;
        v270 = v60[v266];
        printf("%f",v270);
        int v272;
        v272 = v266 + 1;
        bool v273;
        v273 = v272 < 12;
        if (v273){
            printf("%s","; ");
        } else {
        }
        v266 += 1 ;
    }
    printf("%s","]");
    printf("; %s = %s","ev_weights", "[");
    int v274;
    v274 = 0;
    while (while_method_11(v274)){
        float v278;
        v278 = v68[v274];
        printf("%f",v278);
        int v280;
        v280 = v274 + 1;
        bool v281;
        v281 = v280 < 12;
        if (v281){
            printf("%s","; ");
        } else {
        }
        v274 += 1 ;
    }
    printf("%s","]");
    printf("}\n");
    fflush(stdout);
    return 0;
}
