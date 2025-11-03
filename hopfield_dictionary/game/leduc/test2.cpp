#include "test2.hpp"
inline bool while_method_0(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 5;
    return v1;
}
inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
void method_0(float * v0, static_array_list<Union0,5> v1, Union1 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6){
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
    while (while_method_0(v46, v47)){
        Union0 v51;
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
    while (while_method_1(v76)){
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
    while (while_method_2(v83)){
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
void method_1(Eigen::Matrix<float,1,115> v0, Eigen::Matrix<float,1,48> v1, Eigen::Matrix<float,1,115> v2, Eigen::Matrix<float,1,48> v3){
    v3 = ((v0 * v2.transpose() - v2.rowwise().sum().transpose().replicate(v0.rows(),1)).array() / 0.001f).exp().matrix().transpose() * v1;
    return ;
}
inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 12;
    return v1;
}
int main() {
    Eigen::Matrix<float,1,115> v0;
    Eigen::Matrix<float,1,48> v1;
    int v2;
    v2 = 0;
    StackRefs0 v3{v2, v0, v1};
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
    Eigen::Matrix<float,1,115> v28;
    v28.setZero();
    float * v29;
    v29 = &v28(0,0);
    Union1 v30;
    v30 = Union1{Union1_1{}};
    bool v31;
    v31 = false;
    int v32;
    v32 = 8;
    unsigned int v33;
    v33 = 5u;
    method_0(v29, v6, v30, v31, v26, v32, v33);
    Eigen::Matrix<float,1,48> v34;
    Eigen::Matrix<float,1,115> & v35 = v3.v1;
    Eigen::Matrix<float,1,48> & v36 = v3.v2;
    method_1(v35, v36, v28, v34);
    static_array<float,12> v39;
    int v41;
    v41 = 0;
    while (while_method_3(v41)){
        float v43;
        v43 = v34(0,v41);
        v39[v41] = v43;
        v41 += 1 ;
    }
    static_array<float,12> v46;
    int v48;
    v48 = 0;
    while (while_method_3(v48)){
        int v50;
        v50 = v48 + 12;
        float v51;
        v51 = v34(0,v50);
        v46[v48] = v51;
        v48 += 1 ;
    }
    static_array<float,12> v54;
    int v56;
    v56 = 0;
    while (while_method_3(v56)){
        int v58;
        v58 = v56 + 24;
        float v59;
        v59 = v34(0,v58);
        v54[v56] = v59;
        v56 += 1 ;
    }
    static_array<float,12> v62;
    int v64;
    v64 = 0;
    while (while_method_3(v64)){
        int v66;
        v66 = v64 + 36;
        float v67;
        v67 = v34(0,v66);
        v62[v64] = v67;
        v64 += 1 ;
    }
    printf("{%s = %s","average_policy", "[");
    int v134;
    v134 = 0;
    while (while_method_3(v134)){
        float v138;
        v138 = v39[v134];
        printf("%f",v138);
        int v140;
        v140 = v134 + 1;
        bool v141;
        v141 = v140 < 12;
        if (v141){
            printf("%s","; ");
        } else {
        }
        v134 += 1 ;
    }
    printf("%s","]");
    printf("; %s = %s","current_policy", "[");
    int v142;
    v142 = 0;
    while (while_method_3(v142)){
        float v146;
        v146 = v46[v142];
        printf("%f",v146);
        int v148;
        v148 = v142 + 1;
        bool v149;
        v149 = v148 < 12;
        if (v149){
            printf("%s","; ");
        } else {
        }
        v142 += 1 ;
    }
    printf("%s","]");
    printf("; %s = %s","ev_values", "[");
    int v150;
    v150 = 0;
    while (while_method_3(v150)){
        float v154;
        v154 = v54[v150];
        printf("%f",v154);
        int v156;
        v156 = v150 + 1;
        bool v157;
        v157 = v156 < 12;
        if (v157){
            printf("%s","; ");
        } else {
        }
        v150 += 1 ;
    }
    printf("%s","]");
    printf("; %s = %s","ev_weights", "[");
    int v158;
    v158 = 0;
    while (while_method_3(v158)){
        float v162;
        v162 = v62[v158];
        printf("%f",v162);
        int v164;
        v164 = v158 + 1;
        bool v165;
        v165 = v164 < 12;
        if (v165){
            printf("%s","; ");
        } else {
        }
        v158 += 1 ;
    }
    printf("%s","]");
    printf("}\n");
    fflush(stdout);
    return 0;
}
