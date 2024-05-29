kernel = r"""
template <typename el, int dim> struct static_array { el v[dim]; };
template <typename el, int dim, typename default_int> struct static_array_list { el v[dim]; default_int length; };
#include <mma.h>
using namespace nvcuda;
__device__ inline bool while_method_0(long v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ inline bool while_method_1(long v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
__device__ inline bool while_method_2(long v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_3(long v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_4(long v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1, float * v2) {
    extern __shared__ unsigned char v3[];
    float * v4;
    v4 = reinterpret_cast<float *>(&v3[0ull]);
    float * v5;
    v5 = reinterpret_cast<float *>(&v3[34816ull]);
    float * v6;
    v6 = reinterpret_cast<float *>(&v3[0ull]);
    long v7;
    v7 = threadIdx.x;
    long v8;
    v8 = v7 / 32l;
    bool v9;
    v9 = 0l <= v8;
    bool v10;
    v10 = v9 == false;
    if (v10){
        assert("The index needs to be zero or positive." && v9);
    } else {
    }
    long v11;
    v11 = v8 % 8l;
    long v12;
    v12 = v8 / 8l;
    bool v13;
    v13 = v12 < 2l;
    bool v14;
    v14 = v13 == false;
    if (v14){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v13);
    } else {
    }
    assert("Tensor range check" && 0 <= v12 && v12 < 2l);
    assert("Tensor range check" && 0 <= v11 && v11 < 8l);
    long v15;
    v15 = 16l * v11;
    long v16;
    v16 = 8704l * v12;
    long v17;
    v17 = v16 + v15;
    float * v18;
    v18 = v6+v17;
    assert("Tensor range check" && 0 <= v12 && v12 < 2l);
    long v19;
    v19 = 4352l * v12;
    long v20;
    v20 = threadIdx.x;
    long v21;
    v21 = v20 % 32l;
    bool v22;
    v22 = 0l <= v21;
    bool v23;
    v23 = v22 == false;
    if (v23){
        assert("The index needs to be zero or positive." && v22);
    } else {
    }
    long v24;
    v24 = v21 % 4l;
    long v25;
    v25 = v21 / 4l;
    bool v26;
    v26 = v25 < 8l;
    bool v27;
    v27 = v26 == false;
    if (v27){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v26);
    } else {
    }
    assert("Tensor range check" && 0 <= v25 && v25 < 8l);
    assert("Tensor range check" && 0 <= v24 && v24 < 4l);
    long v28;
    v28 = v24 + v19;
    long v29;
    v29 = 68l * v25;
    long v30;
    v30 = v29 + v28;
    float * v31;
    v31 = v4+v30;
    assert("Tensor range check" && 0 <= v11 && v11 < 8l);
    long v32;
    v32 = 1088l * v11;
    long v33;
    v33 = threadIdx.x;
    long v34;
    v34 = v33 % 32l;
    bool v35;
    v35 = 0l <= v34;
    bool v36;
    v36 = v35 == false;
    if (v36){
        assert("The index needs to be zero or positive." && v35);
    } else {
    }
    long v37;
    v37 = v34 % 4l;
    long v38;
    v38 = v34 / 4l;
    bool v39;
    v39 = v38 < 8l;
    bool v40;
    v40 = v39 == false;
    if (v40){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v39);
    } else {
    }
    assert("Tensor range check" && 0 <= v38 && v38 < 8l);
    assert("Tensor range check" && 0 <= v37 && v37 < 4l);
    long v41;
    v41 = v37 + v32;
    long v42;
    v42 = 68l * v38;
    long v43;
    v43 = v42 + v41;
    float * v44;
    v44 = v5+v43;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v45[4l];
    long v46;
    v46 = 0l;
    while (while_method_0(v46)){
        long v48;
        v48 = 0l;
        while (while_method_0(v48)){
            assert("Tensor range check" && 0 <= v46 && v46 < 64l);
            assert("Tensor range check" && 0 <= v48 && v48 < 64l);
            long v50;
            v50 = 128l * v48;
            long v51;
            v51 = 1048576l * v46;
            long v52;
            v52 = v51 + v50;
            float * v53;
            v53 = v2+v52;
            // Pushing the loop unrolling to: 0
            long v54;
            v54 = threadIdx.x;
            bool v55;
            v55 = 0l <= v54;
            bool v56;
            v56 = v55 == false;
            if (v56){
                assert("The index needs to be zero or positive." && v55);
            } else {
            }
            long v57;
            v57 = v54 % 32l;
            long v58;
            v58 = v54 / 32l;
            bool v59;
            v59 = v58 < 16l;
            bool v60;
            v60 = v59 == false;
            if (v60){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v59);
            } else {
            }
            assert("Tensor range check" && 0 <= v58 && v58 < 16l);
            assert("Tensor range check" && 0 <= v57 && v57 < 32l);
            long v61;
            v61 = 4l * v57;
            long v62;
            v62 = 136l * v58;
            long v63;
            v63 = v62 + v61;
            long v64;
            v64 = 8192l * v58;
            long v65;
            v65 = v64 + v61;
            float * v66;
            v66 = v6+v63;
            float * v67;
            v67 = v53+v65;
            long v68;
            v68 = 0l;
            #pragma unroll
            while (while_method_1(v68)){
                long v70;
                v70 = 0l;
                #pragma unroll
                while (while_method_2(v70)){
                    assert("Tensor range check" && 0 <= v68 && v68 < 8l);
                    assert("Tensor range check" && 0 <= v70 && v70 < 1l);
                    long v72;
                    v72 = 128l * v70;
                    long v73;
                    v73 = 2176l * v68;
                    long v74;
                    v74 = v73 + v72;
                    long v75;
                    v75 = 131072l * v68;
                    long v76;
                    v76 = v75 + v72;
                    int4* v77;
                    v77 = reinterpret_cast<int4*>(v67 + v76);
                    int4* v78;
                    v78 = reinterpret_cast<int4*>(v66 + v74);
                    assert("Pointer alignment check" && (unsigned long long)(v77) % 4l == 0 && (unsigned long long)(v78) % 4l == 0);
                    *v78 = *v77;
                    v70 += 1l ;
                }
                v68 += 1l ;
            }
            __syncthreads();
            long v79;
            v79 = 0l;
            #pragma unroll
            while (while_method_3(v79)){
                long v81;
                v81 = 0l;
                #pragma unroll
                while (while_method_2(v81)){
                    assert("Tensor range check" && 0 <= v79 && v79 < 4l);
                    assert("Tensor range check" && 0 <= v81 && v81 < 1l);
                    long v83;
                    v83 = v79 + v81;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v84 = v45[v83];
                    assert("Tensor range check" && 0 <= v79 && v79 < 4l);
                    assert("Tensor range check" && 0 <= v81 && v81 < 1l);
                    long v85;
                    v85 = 16l * v81;
                    long v86;
                    v86 = 2176l * v79;
                    long v87;
                    v87 = v86 + v85;
                    float * v88;
                    v88 = v18+v87;
                    wmma::load_matrix_sync(v84, v88, 136l, wmma::mem_row_major);
                    v81 += 1l ;
                }
                v79 += 1l ;
            }
            __syncthreads();
            long v89;
            v89 = 0l;
            #pragma unroll
            while (while_method_0(v89)){
                assert("Tensor range check" && 0 <= v46 && v46 < 64l);
                long v91;
                v91 = 524288l * v46;
                assert("Tensor range check" && 0 <= v89 && v89 < 64l);
                long v92;
                v92 = 64l * v89;
                long v93;
                v93 = v92 + v91;
                float * v94;
                v94 = v0+v93;
                assert("Tensor range check" && 0 <= v48 && v48 < 64l);
                long v95;
                v95 = 524288l * v48;
                assert("Tensor range check" && 0 <= v89 && v89 < 64l);
                long v96;
                v96 = v92 + v95;
                float * v97;
                v97 = v1+v96;
                long v98;
                v98 = threadIdx.x;
                bool v99;
                v99 = 0l <= v98;
                bool v100;
                v100 = v99 == false;
                if (v100){
                    assert("The index needs to be zero or positive." && v99);
                } else {
                }
                long v101;
                v101 = v98 % 16l;
                long v102;
                v102 = v98 / 16l;
                bool v103;
                v103 = v102 < 32l;
                bool v104;
                v104 = v103 == false;
                if (v104){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v103);
                } else {
                }
                assert("Tensor range check" && 0 <= v102 && v102 < 32l);
                assert("Tensor range check" && 0 <= v101 && v101 < 16l);
                long v105;
                v105 = 4l * v101;
                long v106;
                v106 = 68l * v102;
                long v107;
                v107 = v106 + v105;
                long v108;
                v108 = 4096l * v102;
                long v109;
                v109 = v108 + v105;
                float * v110;
                v110 = v5+v107;
                float * v111;
                v111 = v97+v109;
                long v112;
                v112 = 0l;
                #pragma unroll
                while (while_method_3(v112)){
                    long v114;
                    v114 = 0l;
                    #pragma unroll
                    while (while_method_2(v114)){
                        assert("Tensor range check" && 0 <= v112 && v112 < 4l);
                        assert("Tensor range check" && 0 <= v114 && v114 < 1l);
                        long v116;
                        v116 = 64l * v114;
                        long v117;
                        v117 = 2176l * v112;
                        long v118;
                        v118 = v117 + v116;
                        long v119;
                        v119 = 131072l * v112;
                        long v120;
                        v120 = v119 + v116;
                        float v121[4l];
                        long v122;
                        v122 = 0l;
                        #pragma unroll
                        while (while_method_3(v122)){
                            assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                            long v124;
                            v124 = v122 + v120;
                            float v125;
                            v125 = v111[v124];
                            float v126;
                            v126 = wmma::__float_to_tf32(v125);
                            assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                            v121[v122] = v126;
                            v122 += 1l ;
                        }
                        int4* v127;
                        v127 = reinterpret_cast<int4*>(v121 + 0l);
                        int4* v128;
                        v128 = reinterpret_cast<int4*>(v110 + v118);
                        assert("Pointer alignment check" && (unsigned long long)(v127) % 4l == 0 && (unsigned long long)(v128) % 4l == 0);
                        *v128 = *v127;
                        v114 += 1l ;
                    }
                    v112 += 1l ;
                }
                long v129;
                v129 = threadIdx.x;
                bool v130;
                v130 = 0l <= v129;
                bool v131;
                v131 = v130 == false;
                if (v131){
                    assert("The index needs to be zero or positive." && v130);
                } else {
                }
                long v132;
                v132 = v129 % 16l;
                long v133;
                v133 = v129 / 16l;
                bool v134;
                v134 = v133 < 32l;
                bool v135;
                v135 = v134 == false;
                if (v135){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v134);
                } else {
                }
                assert("Tensor range check" && 0 <= v133 && v133 < 32l);
                assert("Tensor range check" && 0 <= v132 && v132 < 16l);
                long v136;
                v136 = 4l * v132;
                long v137;
                v137 = 68l * v133;
                long v138;
                v138 = v137 + v136;
                long v139;
                v139 = 4096l * v133;
                long v140;
                v140 = v139 + v136;
                float * v141;
                v141 = v4+v138;
                float * v142;
                v142 = v94+v140;
                long v143;
                v143 = 0l;
                #pragma unroll
                while (while_method_3(v143)){
                    long v145;
                    v145 = 0l;
                    #pragma unroll
                    while (while_method_2(v145)){
                        assert("Tensor range check" && 0 <= v143 && v143 < 4l);
                        assert("Tensor range check" && 0 <= v145 && v145 < 1l);
                        long v147;
                        v147 = 64l * v145;
                        long v148;
                        v148 = 2176l * v143;
                        long v149;
                        v149 = v148 + v147;
                        long v150;
                        v150 = 131072l * v143;
                        long v151;
                        v151 = v150 + v147;
                        float v152[4l];
                        long v153;
                        v153 = 0l;
                        #pragma unroll
                        while (while_method_3(v153)){
                            assert("Tensor range check" && 0 <= v153 && v153 < 4l);
                            long v155;
                            v155 = v153 + v151;
                            float v156;
                            v156 = v142[v155];
                            float v157;
                            v157 = wmma::__float_to_tf32(v156);
                            assert("Tensor range check" && 0 <= v153 && v153 < 4l);
                            v152[v153] = v157;
                            v153 += 1l ;
                        }
                        int4* v158;
                        v158 = reinterpret_cast<int4*>(v152 + 0l);
                        int4* v159;
                        v159 = reinterpret_cast<int4*>(v141 + v149);
                        assert("Pointer alignment check" && (unsigned long long)(v158) % 4l == 0 && (unsigned long long)(v159) % 4l == 0);
                        *v159 = *v158;
                        v145 += 1l ;
                    }
                    v143 += 1l ;
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v160[32l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v161[8l];
                long v162;
                v162 = 0l;
                #pragma unroll
                while (while_method_3(v162)){
                    long v164;
                    v164 = 0l;
                    #pragma unroll
                    while (while_method_1(v164)){
                        assert("Tensor range check" && 0 <= v162 && v162 < 4l);
                        assert("Tensor range check" && 0 <= v164 && v164 < 8l);
                        long v166;
                        v166 = 8l * v162;
                        long v167;
                        v167 = v166 + v164;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v168 = v160[v167];
                        assert("Tensor range check" && 0 <= v162 && v162 < 4l);
                        long v169;
                        v169 = 1088l * v162;
                        assert("Tensor range check" && 0 <= v164 && v164 < 8l);
                        long v170;
                        v170 = 8l * v164;
                        long v171;
                        v171 = v170 + v169;
                        long v172;
                        v172 = 0l;
                        #pragma unroll
                        while (while_method_4(v172)){
                            long v174;
                            v174 = 0l;
                            #pragma unroll
                            while (while_method_4(v174)){
                                assert("Tensor range check" && 0 <= v172 && v172 < 2l);
                                assert("Tensor range check" && 0 <= v174 && v174 < 2l);
                                long v176;
                                v176 = 544l * v174;
                                long v177;
                                v177 = v176 + v171;
                                long v178;
                                v178 = 4l * v172;
                                long v179;
                                v179 = v178 + v177;
                                float v180;
                                v180 = v31[v179];
                                bool v181;
                                v181 = 0l <= v174;
                                bool v183;
                                if (v181){
                                    bool v182;
                                    v182 = v174 < 2l;
                                    v183 = v182;
                                } else {
                                    v183 = false;
                                }
                                bool v184;
                                v184 = v183 == false;
                                if (v184){
                                    assert("The indices should be inside the range of the dimension." && v183);
                                } else {
                                }
                                bool v185;
                                v185 = 0l <= v172;
                                bool v187;
                                if (v185){
                                    bool v186;
                                    v186 = v172 < 2l;
                                    v187 = v186;
                                } else {
                                    v187 = false;
                                }
                                bool v188;
                                v188 = v187 == false;
                                if (v188){
                                    assert("The indices should be inside the range of the dimension." && v187);
                                } else {
                                }
                                long v189;
                                v189 = v172 * 2l;
                                long v190;
                                v190 = v174 + v189;
                                v168.x[v190] = v180;
                                v174 += 1l ;
                            }
                            v172 += 1l ;
                        }
                        v164 += 1l ;
                    }
                    v162 += 1l ;
                }
                long v191;
                v191 = 0l;
                #pragma unroll
                while (while_method_2(v191)){
                    long v193;
                    v193 = 0l;
                    #pragma unroll
                    while (while_method_1(v193)){
                        assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                        assert("Tensor range check" && 0 <= v193 && v193 < 8l);
                        long v195;
                        v195 = 8l * v191;
                        long v196;
                        v196 = v195 + v193;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v197 = v161[v196];
                        assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                        long v198;
                        v198 = 1088l * v191;
                        assert("Tensor range check" && 0 <= v193 && v193 < 8l);
                        long v199;
                        v199 = 8l * v193;
                        long v200;
                        v200 = v199 + v198;
                        long v201;
                        v201 = 0l;
                        #pragma unroll
                        while (while_method_4(v201)){
                            long v203;
                            v203 = 0l;
                            #pragma unroll
                            while (while_method_4(v203)){
                                assert("Tensor range check" && 0 <= v201 && v201 < 2l);
                                assert("Tensor range check" && 0 <= v203 && v203 < 2l);
                                long v205;
                                v205 = 4l * v203;
                                long v206;
                                v206 = v205 + v200;
                                long v207;
                                v207 = 544l * v201;
                                long v208;
                                v208 = v207 + v206;
                                float v209;
                                v209 = v44[v208];
                                bool v210;
                                v210 = 0l <= v203;
                                bool v212;
                                if (v210){
                                    bool v211;
                                    v211 = v203 < 2l;
                                    v212 = v211;
                                } else {
                                    v212 = false;
                                }
                                bool v213;
                                v213 = v212 == false;
                                if (v213){
                                    assert("The indices should be inside the range of the dimension." && v212);
                                } else {
                                }
                                bool v214;
                                v214 = 0l <= v201;
                                bool v216;
                                if (v214){
                                    bool v215;
                                    v215 = v201 < 2l;
                                    v216 = v215;
                                } else {
                                    v216 = false;
                                }
                                bool v217;
                                v217 = v216 == false;
                                if (v217){
                                    assert("The indices should be inside the range of the dimension." && v216);
                                } else {
                                }
                                long v218;
                                v218 = v201 * 2l;
                                long v219;
                                v219 = v203 + v218;
                                v197.x[v219] = v209;
                                v203 += 1l ;
                            }
                            v201 += 1l ;
                        }
                        v193 += 1l ;
                    }
                    v191 += 1l ;
                }
                __syncthreads();
                long v220;
                v220 = 0l;
                #pragma unroll
                while (while_method_3(v220)){
                    long v222;
                    v222 = 0l;
                    #pragma unroll
                    while (while_method_2(v222)){
                        long v224;
                        v224 = 0l;
                        #pragma unroll
                        while (while_method_1(v224)){
                            assert("Tensor range check" && 0 <= v220 && v220 < 4l);
                            assert("Tensor range check" && 0 <= v222 && v222 < 1l);
                            long v226;
                            v226 = v220 + v222;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v227 = v45[v226];
                            assert("Tensor range check" && 0 <= v220 && v220 < 4l);
                            assert("Tensor range check" && 0 <= v224 && v224 < 8l);
                            long v228;
                            v228 = 8l * v220;
                            long v229;
                            v229 = v228 + v224;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v230 = v160[v229];
                            assert("Tensor range check" && 0 <= v222 && v222 < 1l);
                            assert("Tensor range check" && 0 <= v224 && v224 < 8l);
                            long v231;
                            v231 = 8l * v222;
                            long v232;
                            v232 = v231 + v224;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v233 = v161[v232];
                            wmma::mma_sync(v227, v230, v233, v227);
                            v224 += 1l ;
                        }
                        v222 += 1l ;
                    }
                    v220 += 1l ;
                }
                v89 += 1l ;
            }
            long v234;
            v234 = 0l;
            #pragma unroll
            while (while_method_3(v234)){
                long v236;
                v236 = 0l;
                #pragma unroll
                while (while_method_2(v236)){
                    assert("Tensor range check" && 0 <= v234 && v234 < 4l);
                    assert("Tensor range check" && 0 <= v236 && v236 < 1l);
                    long v238;
                    v238 = v234 + v236;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v239 = v45[v238];
                    assert("Tensor range check" && 0 <= v234 && v234 < 4l);
                    assert("Tensor range check" && 0 <= v236 && v236 < 1l);
                    long v240;
                    v240 = 16l * v236;
                    long v241;
                    v241 = 2176l * v234;
                    long v242;
                    v242 = v241 + v240;
                    float * v243;
                    v243 = v18+v242;
                    wmma::store_matrix_sync(v243, v239, 136l, wmma::mem_row_major);
                    v236 += 1l ;
                }
                v234 += 1l ;
            }
            __syncthreads();
            long v244;
            v244 = threadIdx.x;
            bool v245;
            v245 = 0l <= v244;
            bool v246;
            v246 = v245 == false;
            if (v246){
                assert("The index needs to be zero or positive." && v245);
            } else {
            }
            long v247;
            v247 = v244 % 32l;
            long v248;
            v248 = v244 / 32l;
            bool v249;
            v249 = v248 < 16l;
            bool v250;
            v250 = v249 == false;
            if (v250){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v249);
            } else {
            }
            assert("Tensor range check" && 0 <= v248 && v248 < 16l);
            assert("Tensor range check" && 0 <= v247 && v247 < 32l);
            long v251;
            v251 = 4l * v247;
            long v252;
            v252 = 8192l * v248;
            long v253;
            v253 = v252 + v251;
            long v254;
            v254 = 136l * v248;
            long v255;
            v255 = v254 + v251;
            float * v256;
            v256 = v53+v253;
            float * v257;
            v257 = v6+v255;
            long v258;
            v258 = 0l;
            #pragma unroll
            while (while_method_1(v258)){
                long v260;
                v260 = 0l;
                #pragma unroll
                while (while_method_2(v260)){
                    assert("Tensor range check" && 0 <= v258 && v258 < 8l);
                    assert("Tensor range check" && 0 <= v260 && v260 < 1l);
                    long v262;
                    v262 = 128l * v260;
                    long v263;
                    v263 = 131072l * v258;
                    long v264;
                    v264 = v263 + v262;
                    long v265;
                    v265 = 2176l * v258;
                    long v266;
                    v266 = v265 + v262;
                    int4* v267;
                    v267 = reinterpret_cast<int4*>(v257 + v266);
                    int4* v268;
                    v268 = reinterpret_cast<int4*>(v256 + v264);
                    assert("Pointer alignment check" && (unsigned long long)(v267) % 4l == 0 && (unsigned long long)(v268) % 4l == 0);
                    *v268 = *v267;
                    v260 += 1l ;
                }
                v258 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v48 += 1l ;
        }
        v46 += 1l ;
    }
    return ;
}
"""
class static_array(list):
    def __init__(self, length):
        for _ in range(length):
            self.append(None)

class static_array_list(static_array):
    def __init__(self, length):
        super().__init__(length)
        self.length = 0
import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

from max_blocks_per_sm import max_blocks_per_sm
options = []
options.append('--diag-suppress=550,20012')
options.append('--dopt=on')
options.append('--restrict')
options.append('--maxrregcount=128')
raw_module = cp.RawModule(code=kernel, backend='nvrtc', enable_cooperative_groups=True, options=tuple(options))
def main():
    v0 = cp.random.normal(0.0,1.0,67108864,dtype=cp.float32)
    v1 = v0.size
    v2 = 67108864 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,33554432,dtype=cp.float32)
    v6 = v5.size
    v7 = 33554432 == v6
    del v6
    v8 = v7 == False
    if v8:
        v9 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v7, v9
        del v9
    else:
        pass
    del v7, v8
    v10 = cp.random.normal(0.0,1.0,33554432,dtype=cp.float32)
    v11 = v10.size
    v12 = 33554432 == v11
    del v11
    v13 = v12 == False
    if v13:
        v14 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v12, v14
        del v14
    else:
        pass
    del v12, v13
    v15 = v10.reshape((8192, 4096))
    v16 = v5.reshape((8192, 4096))
    v17 = cp.transpose(v16)
    del v16
    v18 = v0.reshape((8192, 8192))
    v19 = (cp.matmul(v15,v17) + v18).flatten()
    del v15, v17, v18
    v20 = v19.size
    v21 = 67108864 == v20
    del v20
    v22 = v21 == False
    if v22:
        v23 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v21, v23
        del v23
    else:
        pass
    del v21, v22
    max_blocks_per_sm(cp.cuda.Device(),raw_module.get_function('entry0'),512,is_print=True)
    v24 = 0
    v25 = raw_module.get_function(f"entry{v24}")
    del v24
    v25.max_dynamic_shared_size_bytes = 69632 
    v25((1,),(512,),(v10, v5, v0),shared_mem=69632)
    del v5, v10, v25
    v26 = cp.max(cp.abs(v0-v19))
    del v0, v19
    return v26

if __name__ == '__main__': print(main())
