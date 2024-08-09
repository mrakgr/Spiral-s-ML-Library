kernel = r"""
template <typename el, int dim> struct static_array { el v[dim]; };
template <typename el, int dim, typename default_int> struct static_array_list { el v[dim]; default_int length; };
#include <mma.h>
using namespace nvcuda;
__device__ inline bool while_method_0(long v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ inline bool while_method_1(long v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ inline bool while_method_2(long v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_3(long v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ inline bool while_method_4(long v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1, float * v2) {
    extern __shared__ unsigned char v3[];
    float * v4;
    v4 = reinterpret_cast<float *>(&v3[0ull]);
    float * v5;
    v5 = reinterpret_cast<float *>(&v3[768ull]);
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
    v11 = v8 % 1l;
    bool v12;
    v12 = v8 < 1l;
    bool v13;
    v13 = v12 == false;
    if (v13){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v12);
    } else {
    }
    assert("Tensor range check" && 0 <= v8 && v8 < 1l);
    assert("Tensor range check" && 0 <= v11 && v11 < 1l);
    long v14;
    v14 = 16l * v11;
    long v15;
    v15 = 384l * v8;
    long v16;
    v16 = v15 + v14;
    float * v17;
    v17 = v6+v16;
    assert("Tensor range check" && 0 <= v8 && v8 < 1l);
    long v18;
    v18 = 192l * v8;
    long v19;
    v19 = threadIdx.x;
    long v20;
    v20 = v19 % 32l;
    bool v21;
    v21 = 0l <= v20;
    bool v22;
    v22 = v21 == false;
    if (v22){
        assert("The index needs to be zero or positive." && v21);
    } else {
    }
    long v23;
    v23 = v20 % 4l;
    long v24;
    v24 = v20 / 4l;
    bool v25;
    v25 = v24 < 8l;
    bool v26;
    v26 = v25 == false;
    if (v26){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v25);
    } else {
    }
    assert("Tensor range check" && 0 <= v24 && v24 < 8l);
    assert("Tensor range check" && 0 <= v23 && v23 < 4l);
    long v27;
    v27 = v23 + v18;
    long v28;
    v28 = 12l * v24;
    long v29;
    v29 = v28 + v27;
    float * v30;
    v30 = v4+v29;
    assert("Tensor range check" && 0 <= v11 && v11 < 1l);
    long v31;
    v31 = 192l * v11;
    long v32;
    v32 = threadIdx.x;
    long v33;
    v33 = v32 % 32l;
    bool v34;
    v34 = 0l <= v33;
    bool v35;
    v35 = v34 == false;
    if (v35){
        assert("The index needs to be zero or positive." && v34);
    } else {
    }
    long v36;
    v36 = v33 % 4l;
    long v37;
    v37 = v33 / 4l;
    bool v38;
    v38 = v37 < 8l;
    bool v39;
    v39 = v38 == false;
    if (v39){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v38);
    } else {
    }
    assert("Tensor range check" && 0 <= v37 && v37 < 8l);
    assert("Tensor range check" && 0 <= v36 && v36 < 4l);
    long v40;
    v40 = v36 + v31;
    long v41;
    v41 = 12l * v37;
    long v42;
    v42 = v41 + v40;
    float * v43;
    v43 = v5+v42;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v44[1l];
    long v45;
    v45 = 0l;
    while (while_method_0(v45)){
        long v47;
        v47 = 0l;
        while (while_method_0(v47)){
            assert("Tensor range check" && 0 <= v45 && v45 < 32l);
            assert("Tensor range check" && 0 <= v47 && v47 < 32l);
            long v49;
            v49 = 16l * v47;
            long v50;
            v50 = 8192l * v45;
            long v51;
            v51 = v50 + v49;
            float * v52;
            v52 = v2+v51;
            // Pushing the loop unrolling to: 0
            long v53;
            v53 = threadIdx.x;
            bool v54;
            v54 = 0l <= v53;
            bool v55;
            v55 = v54 == false;
            if (v55){
                assert("The index needs to be zero or positive." && v54);
            } else {
            }
            long v56;
            v56 = v53 % 4l;
            long v57;
            v57 = v53 / 4l;
            bool v58;
            v58 = v57 < 8l;
            bool v59;
            v59 = v58 == false;
            if (v59){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v58);
            } else {
            }
            assert("Tensor range check" && 0 <= v57 && v57 < 8l);
            assert("Tensor range check" && 0 <= v56 && v56 < 4l);
            long v60;
            v60 = 4l * v56;
            long v61;
            v61 = 24l * v57;
            long v62;
            v62 = v61 + v60;
            long v63;
            v63 = 512l * v57;
            long v64;
            v64 = v63 + v60;
            float * v65;
            v65 = v6+v62;
            float * v66;
            v66 = v52+v64;
            long v67;
            v67 = 0l;
            #pragma unroll
            while (while_method_1(v67)){
                long v69;
                v69 = 0l;
                #pragma unroll
                while (while_method_2(v69)){
                    assert("Tensor range check" && 0 <= v67 && v67 < 2l);
                    assert("Tensor range check" && 0 <= v69 && v69 < 1l);
                    long v71;
                    v71 = 16l * v69;
                    long v72;
                    v72 = 192l * v67;
                    long v73;
                    v73 = v72 + v71;
                    long v74;
                    v74 = 4096l * v67;
                    long v75;
                    v75 = v74 + v71;
                    int4* v76;
                    v76 = reinterpret_cast<int4*>(v66 + v75);
                    int4* v77;
                    v77 = reinterpret_cast<int4*>(v65 + v73);
                    assert("Pointer alignment check" && (unsigned long long)(v76) % 4l == 0 && (unsigned long long)(v77) % 4l == 0);
                    *v77 = *v76;
                    v69 += 1l ;
                }
                v67 += 1l ;
            }
            barrier_cta_sync 0;
            long v78;
            v78 = 0l;
            #pragma unroll
            while (while_method_2(v78)){
                long v80;
                v80 = 0l;
                #pragma unroll
                while (while_method_2(v80)){
                    assert("Tensor range check" && 0 <= v78 && v78 < 1l);
                    assert("Tensor range check" && 0 <= v80 && v80 < 1l);
                    long v82;
                    v82 = v78 + v80;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v83 = v44[v82];
                    assert("Tensor range check" && 0 <= v78 && v78 < 1l);
                    assert("Tensor range check" && 0 <= v80 && v80 < 1l);
                    long v84;
                    v84 = 16l * v80;
                    long v85;
                    v85 = 384l * v78;
                    long v86;
                    v86 = v85 + v84;
                    float * v87;
                    v87 = v17+v86;
                    wmma::load_matrix_sync(v83, v87, 24l, wmma::mem_row_major);
                    v80 += 1l ;
                }
                v78 += 1l ;
            }
            barrier_cta_sync 0;
            long v88;
            v88 = 0l;
            #pragma unroll
            while (while_method_3(v88)){
                assert("Tensor range check" && 0 <= v45 && v45 < 32l);
                assert("Tensor range check" && 0 <= v88 && v88 < 64l);
                long v90;
                v90 = 8l * v88;
                long v91;
                v91 = v90 + v50;
                float * v92;
                v92 = v0+v91;
                assert("Tensor range check" && 0 <= v47 && v47 < 32l);
                long v93;
                v93 = 8192l * v47;
                assert("Tensor range check" && 0 <= v88 && v88 < 64l);
                long v94;
                v94 = v90 + v93;
                float * v95;
                v95 = v1+v94;
                long v96;
                v96 = threadIdx.x;
                bool v97;
                v97 = 0l <= v96;
                bool v98;
                v98 = v97 == false;
                if (v98){
                    assert("The index needs to be zero or positive." && v97);
                } else {
                }
                long v99;
                v99 = v96 % 2l;
                long v100;
                v100 = v96 / 2l;
                bool v101;
                v101 = v100 < 16l;
                bool v102;
                v102 = v101 == false;
                if (v102){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v101);
                } else {
                }
                assert("Tensor range check" && 0 <= v100 && v100 < 16l);
                assert("Tensor range check" && 0 <= v99 && v99 < 2l);
                long v103;
                v103 = 4l * v99;
                long v104;
                v104 = 12l * v100;
                long v105;
                v105 = v104 + v103;
                long v106;
                v106 = 512l * v100;
                long v107;
                v107 = v106 + v103;
                float * v108;
                v108 = v5+v105;
                float * v109;
                v109 = v95+v107;
                long v110;
                v110 = 0l;
                #pragma unroll
                while (while_method_2(v110)){
                    long v112;
                    v112 = 0l;
                    #pragma unroll
                    while (while_method_2(v112)){
                        assert("Tensor range check" && 0 <= v110 && v110 < 1l);
                        assert("Tensor range check" && 0 <= v112 && v112 < 1l);
                        long v114;
                        v114 = 8l * v112;
                        long v115;
                        v115 = 192l * v110;
                        long v116;
                        v116 = v115 + v114;
                        long v117;
                        v117 = 8192l * v110;
                        long v118;
                        v118 = v117 + v114;
                        float v119[4l];
                        long v120;
                        v120 = 0l;
                        #pragma unroll
                        while (while_method_4(v120)){
                            assert("Tensor range check" && 0 <= v120 && v120 < 4l);
                            long v122;
                            v122 = v120 + v118;
                            float v123;
                            v123 = v109[v122];
                            float v124;
                            v124 = wmma::__float_to_tf32(v123);
                            assert("Tensor range check" && 0 <= v120 && v120 < 4l);
                            v119[v120] = v124;
                            v120 += 1l ;
                        }
                        int4* v125;
                        v125 = reinterpret_cast<int4*>(v119 + 0l);
                        int4* v126;
                        v126 = reinterpret_cast<int4*>(v108 + v116);
                        assert("Pointer alignment check" && (unsigned long long)(v125) % 4l == 0 && (unsigned long long)(v126) % 4l == 0);
                        *v126 = *v125;
                        v112 += 1l ;
                    }
                    v110 += 1l ;
                }
                long v127;
                v127 = threadIdx.x;
                bool v128;
                v128 = 0l <= v127;
                bool v129;
                v129 = v128 == false;
                if (v129){
                    assert("The index needs to be zero or positive." && v128);
                } else {
                }
                long v130;
                v130 = v127 % 2l;
                long v131;
                v131 = v127 / 2l;
                bool v132;
                v132 = v131 < 16l;
                bool v133;
                v133 = v132 == false;
                if (v133){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v132);
                } else {
                }
                assert("Tensor range check" && 0 <= v131 && v131 < 16l);
                assert("Tensor range check" && 0 <= v130 && v130 < 2l);
                long v134;
                v134 = 4l * v130;
                long v135;
                v135 = 12l * v131;
                long v136;
                v136 = v135 + v134;
                long v137;
                v137 = 512l * v131;
                long v138;
                v138 = v137 + v134;
                float * v139;
                v139 = v4+v136;
                float * v140;
                v140 = v92+v138;
                long v141;
                v141 = 0l;
                #pragma unroll
                while (while_method_2(v141)){
                    long v143;
                    v143 = 0l;
                    #pragma unroll
                    while (while_method_2(v143)){
                        assert("Tensor range check" && 0 <= v141 && v141 < 1l);
                        assert("Tensor range check" && 0 <= v143 && v143 < 1l);
                        long v145;
                        v145 = 8l * v143;
                        long v146;
                        v146 = 192l * v141;
                        long v147;
                        v147 = v146 + v145;
                        long v148;
                        v148 = 8192l * v141;
                        long v149;
                        v149 = v148 + v145;
                        float v150[4l];
                        long v151;
                        v151 = 0l;
                        #pragma unroll
                        while (while_method_4(v151)){
                            assert("Tensor range check" && 0 <= v151 && v151 < 4l);
                            long v153;
                            v153 = v151 + v149;
                            float v154;
                            v154 = v140[v153];
                            float v155;
                            v155 = wmma::__float_to_tf32(v154);
                            assert("Tensor range check" && 0 <= v151 && v151 < 4l);
                            v150[v151] = v155;
                            v151 += 1l ;
                        }
                        int4* v156;
                        v156 = reinterpret_cast<int4*>(v150 + 0l);
                        int4* v157;
                        v157 = reinterpret_cast<int4*>(v139 + v147);
                        assert("Pointer alignment check" && (unsigned long long)(v156) % 4l == 0 && (unsigned long long)(v157) % 4l == 0);
                        *v157 = *v156;
                        v143 += 1l ;
                    }
                    v141 += 1l ;
                }
                barrier_cta_sync 0;
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v158[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v159[1l];
                long v160;
                v160 = 0l;
                #pragma unroll
                while (while_method_2(v160)){
                    long v162;
                    v162 = 0l;
                    #pragma unroll
                    while (while_method_2(v162)){
                        assert("Tensor range check" && 0 <= v160 && v160 < 1l);
                        assert("Tensor range check" && 0 <= v162 && v162 < 1l);
                        long v164;
                        v164 = v160 + v162;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v165 = v158[v164];
                        assert("Tensor range check" && 0 <= v160 && v160 < 1l);
                        long v166;
                        v166 = 192l * v160;
                        assert("Tensor range check" && 0 <= v162 && v162 < 1l);
                        long v167;
                        v167 = 8l * v162;
                        long v168;
                        v168 = v167 + v166;
                        long v169;
                        v169 = 0l;
                        #pragma unroll
                        while (while_method_1(v169)){
                            long v171;
                            v171 = 0l;
                            #pragma unroll
                            while (while_method_1(v171)){
                                assert("Tensor range check" && 0 <= v169 && v169 < 2l);
                                assert("Tensor range check" && 0 <= v171 && v171 < 2l);
                                long v173;
                                v173 = 96l * v171;
                                long v174;
                                v174 = v173 + v168;
                                long v175;
                                v175 = 4l * v169;
                                long v176;
                                v176 = v175 + v174;
                                float v177;
                                v177 = v30[v176];
                                bool v178;
                                v178 = 0l <= v171;
                                bool v180;
                                if (v178){
                                    bool v179;
                                    v179 = v171 < 2l;
                                    v180 = v179;
                                } else {
                                    v180 = false;
                                }
                                bool v181;
                                v181 = v180 == false;
                                if (v181){
                                    assert("The indices should be inside the range of the dimension." && v180);
                                } else {
                                }
                                bool v182;
                                v182 = 0l <= v169;
                                bool v184;
                                if (v182){
                                    bool v183;
                                    v183 = v169 < 2l;
                                    v184 = v183;
                                } else {
                                    v184 = false;
                                }
                                bool v185;
                                v185 = v184 == false;
                                if (v185){
                                    assert("The indices should be inside the range of the dimension." && v184);
                                } else {
                                }
                                long v186;
                                v186 = v169 * 2l;
                                long v187;
                                v187 = v171 + v186;
                                v165.x[v187] = v177;
                                v171 += 1l ;
                            }
                            v169 += 1l ;
                        }
                        v162 += 1l ;
                    }
                    v160 += 1l ;
                }
                long v188;
                v188 = 0l;
                #pragma unroll
                while (while_method_2(v188)){
                    long v190;
                    v190 = 0l;
                    #pragma unroll
                    while (while_method_2(v190)){
                        assert("Tensor range check" && 0 <= v188 && v188 < 1l);
                        assert("Tensor range check" && 0 <= v190 && v190 < 1l);
                        long v192;
                        v192 = v188 + v190;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v193 = v159[v192];
                        assert("Tensor range check" && 0 <= v188 && v188 < 1l);
                        long v194;
                        v194 = 192l * v188;
                        assert("Tensor range check" && 0 <= v190 && v190 < 1l);
                        long v195;
                        v195 = 8l * v190;
                        long v196;
                        v196 = v195 + v194;
                        long v197;
                        v197 = 0l;
                        #pragma unroll
                        while (while_method_1(v197)){
                            long v199;
                            v199 = 0l;
                            #pragma unroll
                            while (while_method_1(v199)){
                                assert("Tensor range check" && 0 <= v197 && v197 < 2l);
                                assert("Tensor range check" && 0 <= v199 && v199 < 2l);
                                long v201;
                                v201 = 4l * v199;
                                long v202;
                                v202 = v201 + v196;
                                long v203;
                                v203 = 96l * v197;
                                long v204;
                                v204 = v203 + v202;
                                float v205;
                                v205 = v43[v204];
                                bool v206;
                                v206 = 0l <= v199;
                                bool v208;
                                if (v206){
                                    bool v207;
                                    v207 = v199 < 2l;
                                    v208 = v207;
                                } else {
                                    v208 = false;
                                }
                                bool v209;
                                v209 = v208 == false;
                                if (v209){
                                    assert("The indices should be inside the range of the dimension." && v208);
                                } else {
                                }
                                bool v210;
                                v210 = 0l <= v197;
                                bool v212;
                                if (v210){
                                    bool v211;
                                    v211 = v197 < 2l;
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
                                long v214;
                                v214 = v197 * 2l;
                                long v215;
                                v215 = v199 + v214;
                                v193.x[v215] = v205;
                                v199 += 1l ;
                            }
                            v197 += 1l ;
                        }
                        v190 += 1l ;
                    }
                    v188 += 1l ;
                }
                barrier_cta_sync 0;
                long v216;
                v216 = 0l;
                #pragma unroll
                while (while_method_2(v216)){
                    long v218;
                    v218 = 0l;
                    #pragma unroll
                    while (while_method_2(v218)){
                        long v220;
                        v220 = 0l;
                        #pragma unroll
                        while (while_method_2(v220)){
                            assert("Tensor range check" && 0 <= v216 && v216 < 1l);
                            assert("Tensor range check" && 0 <= v218 && v218 < 1l);
                            long v222;
                            v222 = v216 + v218;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v223 = v44[v222];
                            assert("Tensor range check" && 0 <= v216 && v216 < 1l);
                            assert("Tensor range check" && 0 <= v220 && v220 < 1l);
                            long v224;
                            v224 = v216 + v220;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v225 = v158[v224];
                            assert("Tensor range check" && 0 <= v218 && v218 < 1l);
                            assert("Tensor range check" && 0 <= v220 && v220 < 1l);
                            long v226;
                            v226 = v218 + v220;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v227 = v159[v226];
                            wmma::mma_sync(v223, v225, v227, v223);
                            v220 += 1l ;
                        }
                        v218 += 1l ;
                    }
                    v216 += 1l ;
                }
                v88 += 1l ;
            }
            long v228;
            v228 = 0l;
            #pragma unroll
            while (while_method_2(v228)){
                long v230;
                v230 = 0l;
                #pragma unroll
                while (while_method_2(v230)){
                    assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                    assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                    long v232;
                    v232 = v228 + v230;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v233 = v44[v232];
                    assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                    assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                    long v234;
                    v234 = 16l * v230;
                    long v235;
                    v235 = 384l * v228;
                    long v236;
                    v236 = v235 + v234;
                    float * v237;
                    v237 = v17+v236;
                    wmma::store_matrix_sync(v237, v233, 24l, wmma::mem_row_major);
                    v230 += 1l ;
                }
                v228 += 1l ;
            }
            barrier_cta_sync 0;
            long v238;
            v238 = threadIdx.x;
            bool v239;
            v239 = 0l <= v238;
            bool v240;
            v240 = v239 == false;
            if (v240){
                assert("The index needs to be zero or positive." && v239);
            } else {
            }
            long v241;
            v241 = v238 % 4l;
            long v242;
            v242 = v238 / 4l;
            bool v243;
            v243 = v242 < 8l;
            bool v244;
            v244 = v243 == false;
            if (v244){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v243);
            } else {
            }
            assert("Tensor range check" && 0 <= v242 && v242 < 8l);
            assert("Tensor range check" && 0 <= v241 && v241 < 4l);
            long v245;
            v245 = 4l * v241;
            long v246;
            v246 = 512l * v242;
            long v247;
            v247 = v246 + v245;
            long v248;
            v248 = 24l * v242;
            long v249;
            v249 = v248 + v245;
            float * v250;
            v250 = v52+v247;
            float * v251;
            v251 = v6+v249;
            long v252;
            v252 = 0l;
            #pragma unroll
            while (while_method_1(v252)){
                long v254;
                v254 = 0l;
                #pragma unroll
                while (while_method_2(v254)){
                    assert("Tensor range check" && 0 <= v252 && v252 < 2l);
                    assert("Tensor range check" && 0 <= v254 && v254 < 1l);
                    long v256;
                    v256 = 16l * v254;
                    long v257;
                    v257 = 4096l * v252;
                    long v258;
                    v258 = v257 + v256;
                    long v259;
                    v259 = 192l * v252;
                    long v260;
                    v260 = v259 + v256;
                    int4* v261;
                    v261 = reinterpret_cast<int4*>(v251 + v260);
                    int4* v262;
                    v262 = reinterpret_cast<int4*>(v250 + v258);
                    assert("Pointer alignment check" && (unsigned long long)(v261) % 4l == 0 && (unsigned long long)(v262) % 4l == 0);
                    *v262 = *v261;
                    v254 += 1l ;
                }
                v252 += 1l ;
            }
            barrier_cta_sync 0;
            // Poping the loop unrolling to: 0
            v47 += 1l ;
        }
        v45 += 1l ;
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
raw_module = cp.RawModule(code=kernel, backend='nvrtc', enable_cooperative_groups=True, options=tuple(options))
def main():
    v0 = cp.random.normal(0.0,1.0,262144,dtype=cp.float32) # type: ignore
    v1 = cp.random.normal(0.0,1.0,262144,dtype=cp.float32) # type: ignore
    v2 = cp.random.normal(0.0,1.0,262144,dtype=cp.float32) # type: ignore
    v3 = v2.reshape((512, 512))
    v4 = v1.reshape((512, 512))
    v5 = cp.transpose(v4)
    del v4
    v6 = v0.reshape((512, 512))
    v7 = (cp.matmul(v3,v5) + v6).flatten()
    del v3, v5, v6
    v8 = v7.size
    v9 = 262144 == v8
    del v8
    v10 = v9 == False
    if v10:
        v11 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v9, v11
        del v11
    else:
        pass
    del v9, v10
    max_blocks_per_sm(cp.cuda.Device(),raw_module.get_function('entry0'),32,is_print=True)
    v12 = 0
    v13 = raw_module.get_function(f"entry{v12}")
    del v12
    v13.max_dynamic_shared_size_bytes = 1536 
    v13((1,),(32,),(v2, v1, v0),shared_mem=1536)
    del v1, v2, v13
    v14 = cp.max(cp.abs(v0-v7))
    del v0, v7
    return v14

if __name__ == '__main__': print(main())
