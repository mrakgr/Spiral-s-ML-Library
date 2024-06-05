kernel = r"""
template <typename el, int dim> struct static_array { el v[dim]; };
template <typename el, int dim, typename default_int> struct static_array_list { el v[dim]; default_int length; };
#include <mma.h>
using namespace nvcuda;
__device__ void method_1(float * v0, float * v1, float * v2);
__device__ void method_0(unsigned char * v0, unsigned char * v1);
__device__ void method_3(float * v0, float * v1);
__device__ void method_2(unsigned char * v0, unsigned char * v1);
__device__ void method_5(float * v0, float * v1, float * v2);
__device__ void method_4(unsigned char * v0, unsigned char * v1);
__device__ void method_6(unsigned char * v0, unsigned char * v1);
__device__ void method_7(unsigned char * v0, unsigned char * v1);
__device__ void method_8(unsigned char * v0, unsigned char * v1);
__device__ inline bool while_method_0(long v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_1(long v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_2(long v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ void method_1(float * v0, float * v1, float * v2){
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
            assert("Tensor range check" && 0 <= v45 && v45 < 1l);
            assert("Tensor range check" && 0 <= v47 && v47 < 1l);
            long v49;
            v49 = 16l * v47;
            long v50;
            v50 = 256l * v45;
            long v51;
            v51 = v50 + v49;
            float * v52;
            v52 = v0+v51;
            // Pushing the loop unrolling to: 0
            long v53;
            v53 = 0l;
            #pragma unroll
            while (while_method_0(v53)){
                long v55;
                v55 = 0l;
                #pragma unroll
                while (while_method_0(v55)){
                    assert("Tensor range check" && 0 <= v53 && v53 < 1l);
                    assert("Tensor range check" && 0 <= v55 && v55 < 1l);
                    long v57;
                    v57 = v53 + v55;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v58 = v44[v57];
                    wmma::fill_fragment(v58, 0.0f);
                    v55 += 1l ;
                }
                v53 += 1l ;
            }
            long v59;
            v59 = 0l;
            #pragma unroll
            while (while_method_0(v59)){
                assert("Tensor range check" && 0 <= v45 && v45 < 1l);
                long v61;
                v61 = 128l * v45;
                assert("Tensor range check" && 0 <= v59 && v59 < 1l);
                long v62;
                v62 = 8l * v59;
                long v63;
                v63 = v62 + v61;
                float * v64;
                v64 = v2+v63;
                assert("Tensor range check" && 0 <= v47 && v47 < 1l);
                long v65;
                v65 = 128l * v47;
                assert("Tensor range check" && 0 <= v59 && v59 < 1l);
                long v66;
                v66 = v62 + v65;
                float * v67;
                v67 = v1+v66;
                long v68;
                v68 = threadIdx.x;
                bool v69;
                v69 = 0l <= v68;
                bool v70;
                v70 = v69 == false;
                if (v70){
                    assert("The index needs to be zero or positive." && v69);
                } else {
                }
                long v71;
                v71 = v68 % 2l;
                long v72;
                v72 = v68 / 2l;
                bool v73;
                v73 = v72 < 16l;
                bool v74;
                v74 = v73 == false;
                if (v74){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v73);
                } else {
                }
                assert("Tensor range check" && 0 <= v72 && v72 < 16l);
                assert("Tensor range check" && 0 <= v71 && v71 < 2l);
                long v75;
                v75 = 4l * v71;
                long v76;
                v76 = 12l * v72;
                long v77;
                v77 = v76 + v75;
                long v78;
                v78 = 8l * v72;
                long v79;
                v79 = v78 + v75;
                float * v80;
                v80 = v5+v77;
                float * v81;
                v81 = v67+v79;
                long v82;
                v82 = 0l;
                #pragma unroll
                while (while_method_0(v82)){
                    long v84;
                    v84 = 0l;
                    #pragma unroll
                    while (while_method_0(v84)){
                        assert("Tensor range check" && 0 <= v82 && v82 < 1l);
                        assert("Tensor range check" && 0 <= v84 && v84 < 1l);
                        long v86;
                        v86 = 8l * v84;
                        long v87;
                        v87 = 192l * v82;
                        long v88;
                        v88 = v87 + v86;
                        long v89;
                        v89 = 128l * v82;
                        long v90;
                        v90 = v89 + v86;
                        float v91[4l];
                        long v92;
                        v92 = 0l;
                        #pragma unroll
                        while (while_method_1(v92)){
                            assert("Tensor range check" && 0 <= v92 && v92 < 4l);
                            long v94;
                            v94 = v92 + v90;
                            float v95;
                            v95 = v81[v94];
                            float v96;
                            v96 = wmma::__float_to_tf32(v95);
                            assert("Tensor range check" && 0 <= v92 && v92 < 4l);
                            v91[v92] = v96;
                            v92 += 1l ;
                        }
                        int4* v97;
                        v97 = reinterpret_cast<int4*>(v91 + 0l);
                        int4* v98;
                        v98 = reinterpret_cast<int4*>(v80 + v88);
                        assert("Pointer alignment check" && (unsigned long long)(v97) % 4l == 0 && (unsigned long long)(v98) % 4l == 0);
                        *v98 = *v97;
                        v84 += 1l ;
                    }
                    v82 += 1l ;
                }
                long v99;
                v99 = threadIdx.x;
                bool v100;
                v100 = 0l <= v99;
                bool v101;
                v101 = v100 == false;
                if (v101){
                    assert("The index needs to be zero or positive." && v100);
                } else {
                }
                long v102;
                v102 = v99 % 2l;
                long v103;
                v103 = v99 / 2l;
                bool v104;
                v104 = v103 < 16l;
                bool v105;
                v105 = v104 == false;
                if (v105){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v104);
                } else {
                }
                assert("Tensor range check" && 0 <= v103 && v103 < 16l);
                assert("Tensor range check" && 0 <= v102 && v102 < 2l);
                long v106;
                v106 = 4l * v102;
                long v107;
                v107 = 12l * v103;
                long v108;
                v108 = v107 + v106;
                long v109;
                v109 = 8l * v103;
                long v110;
                v110 = v109 + v106;
                float * v111;
                v111 = v4+v108;
                float * v112;
                v112 = v64+v110;
                long v113;
                v113 = 0l;
                #pragma unroll
                while (while_method_0(v113)){
                    long v115;
                    v115 = 0l;
                    #pragma unroll
                    while (while_method_0(v115)){
                        assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                        assert("Tensor range check" && 0 <= v115 && v115 < 1l);
                        long v117;
                        v117 = 8l * v115;
                        long v118;
                        v118 = 192l * v113;
                        long v119;
                        v119 = v118 + v117;
                        long v120;
                        v120 = 128l * v113;
                        long v121;
                        v121 = v120 + v117;
                        float v122[4l];
                        long v123;
                        v123 = 0l;
                        #pragma unroll
                        while (while_method_1(v123)){
                            assert("Tensor range check" && 0 <= v123 && v123 < 4l);
                            long v125;
                            v125 = v123 + v121;
                            float v126;
                            v126 = v112[v125];
                            float v127;
                            v127 = wmma::__float_to_tf32(v126);
                            assert("Tensor range check" && 0 <= v123 && v123 < 4l);
                            v122[v123] = v127;
                            v123 += 1l ;
                        }
                        int4* v128;
                        v128 = reinterpret_cast<int4*>(v122 + 0l);
                        int4* v129;
                        v129 = reinterpret_cast<int4*>(v111 + v119);
                        assert("Pointer alignment check" && (unsigned long long)(v128) % 4l == 0 && (unsigned long long)(v129) % 4l == 0);
                        *v129 = *v128;
                        v115 += 1l ;
                    }
                    v113 += 1l ;
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v130[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v131[1l];
                long v132;
                v132 = 0l;
                #pragma unroll
                while (while_method_0(v132)){
                    long v134;
                    v134 = 0l;
                    #pragma unroll
                    while (while_method_0(v134)){
                        assert("Tensor range check" && 0 <= v132 && v132 < 1l);
                        assert("Tensor range check" && 0 <= v134 && v134 < 1l);
                        long v136;
                        v136 = v132 + v134;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v137 = v130[v136];
                        assert("Tensor range check" && 0 <= v132 && v132 < 1l);
                        long v138;
                        v138 = 192l * v132;
                        assert("Tensor range check" && 0 <= v134 && v134 < 1l);
                        long v139;
                        v139 = 8l * v134;
                        long v140;
                        v140 = v139 + v138;
                        long v141;
                        v141 = 0l;
                        #pragma unroll
                        while (while_method_2(v141)){
                            long v143;
                            v143 = 0l;
                            #pragma unroll
                            while (while_method_2(v143)){
                                assert("Tensor range check" && 0 <= v141 && v141 < 2l);
                                assert("Tensor range check" && 0 <= v143 && v143 < 2l);
                                long v145;
                                v145 = 96l * v143;
                                long v146;
                                v146 = v145 + v140;
                                long v147;
                                v147 = 4l * v141;
                                long v148;
                                v148 = v147 + v146;
                                float v149;
                                v149 = v30[v148];
                                bool v150;
                                v150 = 0l <= v143;
                                bool v152;
                                if (v150){
                                    bool v151;
                                    v151 = v143 < 2l;
                                    v152 = v151;
                                } else {
                                    v152 = false;
                                }
                                bool v153;
                                v153 = v152 == false;
                                if (v153){
                                    assert("The indices should be inside the range of the dimension." && v152);
                                } else {
                                }
                                bool v154;
                                v154 = 0l <= v141;
                                bool v156;
                                if (v154){
                                    bool v155;
                                    v155 = v141 < 2l;
                                    v156 = v155;
                                } else {
                                    v156 = false;
                                }
                                bool v157;
                                v157 = v156 == false;
                                if (v157){
                                    assert("The indices should be inside the range of the dimension." && v156);
                                } else {
                                }
                                long v158;
                                v158 = v141 * 2l;
                                long v159;
                                v159 = v143 + v158;
                                v137.x[v159] = v149;
                                v143 += 1l ;
                            }
                            v141 += 1l ;
                        }
                        v134 += 1l ;
                    }
                    v132 += 1l ;
                }
                long v160;
                v160 = 0l;
                #pragma unroll
                while (while_method_0(v160)){
                    long v162;
                    v162 = 0l;
                    #pragma unroll
                    while (while_method_0(v162)){
                        assert("Tensor range check" && 0 <= v160 && v160 < 1l);
                        assert("Tensor range check" && 0 <= v162 && v162 < 1l);
                        long v164;
                        v164 = v160 + v162;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v165 = v131[v164];
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
                        while (while_method_2(v169)){
                            long v171;
                            v171 = 0l;
                            #pragma unroll
                            while (while_method_2(v171)){
                                assert("Tensor range check" && 0 <= v169 && v169 < 2l);
                                assert("Tensor range check" && 0 <= v171 && v171 < 2l);
                                long v173;
                                v173 = 4l * v171;
                                long v174;
                                v174 = v173 + v168;
                                long v175;
                                v175 = 96l * v169;
                                long v176;
                                v176 = v175 + v174;
                                float v177;
                                v177 = v43[v176];
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
                __syncthreads();
                long v188;
                v188 = 0l;
                #pragma unroll
                while (while_method_0(v188)){
                    long v190;
                    v190 = 0l;
                    #pragma unroll
                    while (while_method_0(v190)){
                        long v192;
                        v192 = 0l;
                        #pragma unroll
                        while (while_method_0(v192)){
                            assert("Tensor range check" && 0 <= v188 && v188 < 1l);
                            assert("Tensor range check" && 0 <= v190 && v190 < 1l);
                            long v194;
                            v194 = v188 + v190;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v195 = v44[v194];
                            assert("Tensor range check" && 0 <= v188 && v188 < 1l);
                            assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                            long v196;
                            v196 = v188 + v192;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v197 = v130[v196];
                            assert("Tensor range check" && 0 <= v190 && v190 < 1l);
                            assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                            long v198;
                            v198 = v190 + v192;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v199 = v131[v198];
                            wmma::mma_sync(v195, v197, v199, v195);
                            v192 += 1l ;
                        }
                        v190 += 1l ;
                    }
                    v188 += 1l ;
                }
                v59 += 1l ;
            }
            long v200;
            v200 = 0l;
            #pragma unroll
            while (while_method_0(v200)){
                long v202;
                v202 = 0l;
                #pragma unroll
                while (while_method_0(v202)){
                    assert("Tensor range check" && 0 <= v200 && v200 < 1l);
                    assert("Tensor range check" && 0 <= v202 && v202 < 1l);
                    long v204;
                    v204 = v200 + v202;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v205 = v44[v204];
                    assert("Tensor range check" && 0 <= v200 && v200 < 1l);
                    assert("Tensor range check" && 0 <= v202 && v202 < 1l);
                    long v206;
                    v206 = 16l * v202;
                    long v207;
                    v207 = 384l * v200;
                    long v208;
                    v208 = v207 + v206;
                    float * v209;
                    v209 = v17+v208;
                    wmma::store_matrix_sync(v209, v205, 24l, wmma::mem_row_major);
                    v202 += 1l ;
                }
                v200 += 1l ;
            }
            __syncthreads();
            long v210;
            v210 = threadIdx.x;
            bool v211;
            v211 = 0l <= v210;
            bool v212;
            v212 = v211 == false;
            if (v212){
                assert("The index needs to be zero or positive." && v211);
            } else {
            }
            long v213;
            v213 = v210 % 4l;
            long v214;
            v214 = v210 / 4l;
            bool v215;
            v215 = v214 < 8l;
            bool v216;
            v216 = v215 == false;
            if (v216){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v215);
            } else {
            }
            assert("Tensor range check" && 0 <= v214 && v214 < 8l);
            assert("Tensor range check" && 0 <= v213 && v213 < 4l);
            long v217;
            v217 = 4l * v213;
            long v218;
            v218 = 16l * v214;
            long v219;
            v219 = v218 + v217;
            long v220;
            v220 = 24l * v214;
            long v221;
            v221 = v220 + v217;
            float * v222;
            v222 = v52+v219;
            float * v223;
            v223 = v6+v221;
            long v224;
            v224 = 0l;
            #pragma unroll
            while (while_method_2(v224)){
                long v226;
                v226 = 0l;
                #pragma unroll
                while (while_method_0(v226)){
                    assert("Tensor range check" && 0 <= v224 && v224 < 2l);
                    assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                    long v228;
                    v228 = 16l * v226;
                    long v229;
                    v229 = 128l * v224;
                    long v230;
                    v230 = v229 + v228;
                    long v231;
                    v231 = 192l * v224;
                    long v232;
                    v232 = v231 + v228;
                    int4* v233;
                    v233 = reinterpret_cast<int4*>(v223 + v232);
                    int4* v234;
                    v234 = reinterpret_cast<int4*>(v222 + v230);
                    assert("Pointer alignment check" && (unsigned long long)(v233) % 4l == 0 && (unsigned long long)(v234) % 4l == 0);
                    *v234 = *v233;
                    v226 += 1l ;
                }
                v224 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v47 += 1l ;
        }
        v45 += 1l ;
    }
    return ;
}
__device__ void method_0(unsigned char * v0, unsigned char * v1){
    float * v2;
    v2 = reinterpret_cast<float *>(&v1[0ull]);
    float * v3;
    v3 = reinterpret_cast<float *>(&v0[0ull]);
    float * v4;
    v4 = reinterpret_cast<float *>(&v1[512ull]);
    return method_1(v4, v3, v2);
}
__device__ inline bool while_method_3(long v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ void method_3(float * v0, float * v1){
    long v2;
    v2 = threadIdx.x;
    long v3;
    v3 = v2;
    while (while_method_3(v3)){
        bool v5;
        v5 = 0l <= v3;
        bool v6;
        v6 = v5 == false;
        if (v6){
            assert("The index needs to be zero or positive." && v5);
        } else {
        }
        long v7;
        v7 = v3 % 4l;
        long v8;
        v8 = v3 / 4l;
        bool v9;
        v9 = v8 < 16l;
        bool v10;
        v10 = v9 == false;
        if (v10){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v9);
        } else {
        }
        assert("Tensor range check" && 0 <= v8 && v8 < 16l);
        assert("Tensor range check" && 0 <= v7 && v7 < 4l);
        long v11;
        v11 = 4l * v7;
        long v12;
        v12 = 16l * v8;
        long v13;
        v13 = v12 + v11;
        assert("Tensor range check" && 0 <= v8 && v8 < 16l);
        assert("Tensor range check" && 0 <= v7 && v7 < 4l);
        float v14[4l];
        float v15[4l];
        int4* v16;
        v16 = reinterpret_cast<int4*>(v1 + v13);
        int4* v17;
        v17 = reinterpret_cast<int4*>(v14 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v16) % 4l == 0 && (unsigned long long)(v17) % 4l == 0);
        *v17 = *v16;
        // Pushing the loop unrolling to: 0
        long v18;
        v18 = 0l;
        #pragma unroll
        while (while_method_1(v18)){
            assert("Tensor range check" && 0 <= v18 && v18 < 4l);
            float v20;
            v20 = v14[v18];
            float v21;
            v21 = tanh(v20);
            assert("Tensor range check" && 0 <= v18 && v18 < 4l);
            v15[v18] = v21;
            v18 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v22;
        v22 = reinterpret_cast<int4*>(v15 + 0l);
        int4* v23;
        v23 = reinterpret_cast<int4*>(v0 + v13);
        assert("Pointer alignment check" && (unsigned long long)(v22) % 4l == 0 && (unsigned long long)(v23) % 4l == 0);
        *v23 = *v22;
        v3 += 32l ;
    }
    __syncthreads();
    return ;
}
__device__ void method_2(unsigned char * v0, unsigned char * v1){
    float * v2;
    v2 = reinterpret_cast<float *>(&v1[512ull]);
    float * v3;
    v3 = reinterpret_cast<float *>(&v1[1536ull]);
    return method_3(v3, v2);
}
__device__ void method_5(float * v0, float * v1, float * v2){
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
            assert("Tensor range check" && 0 <= v45 && v45 < 1l);
            assert("Tensor range check" && 0 <= v47 && v47 < 1l);
            long v49;
            v49 = 16l * v47;
            long v50;
            v50 = 256l * v45;
            long v51;
            v51 = v50 + v49;
            float * v52;
            v52 = v0+v51;
            // Pushing the loop unrolling to: 0
            long v53;
            v53 = 0l;
            #pragma unroll
            while (while_method_0(v53)){
                long v55;
                v55 = 0l;
                #pragma unroll
                while (while_method_0(v55)){
                    assert("Tensor range check" && 0 <= v53 && v53 < 1l);
                    assert("Tensor range check" && 0 <= v55 && v55 < 1l);
                    long v57;
                    v57 = v53 + v55;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v58 = v44[v57];
                    wmma::fill_fragment(v58, 0.0f);
                    v55 += 1l ;
                }
                v53 += 1l ;
            }
            long v59;
            v59 = 0l;
            #pragma unroll
            while (while_method_2(v59)){
                assert("Tensor range check" && 0 <= v45 && v45 < 1l);
                assert("Tensor range check" && 0 <= v59 && v59 < 2l);
                long v61;
                v61 = 8l * v59;
                long v62;
                v62 = v61 + v50;
                float * v63;
                v63 = v2+v62;
                assert("Tensor range check" && 0 <= v47 && v47 < 1l);
                long v64;
                v64 = 256l * v47;
                assert("Tensor range check" && 0 <= v59 && v59 < 2l);
                long v65;
                v65 = v61 + v64;
                float * v66;
                v66 = v1+v65;
                long v67;
                v67 = threadIdx.x;
                bool v68;
                v68 = 0l <= v67;
                bool v69;
                v69 = v68 == false;
                if (v69){
                    assert("The index needs to be zero or positive." && v68);
                } else {
                }
                long v70;
                v70 = v67 % 2l;
                long v71;
                v71 = v67 / 2l;
                bool v72;
                v72 = v71 < 16l;
                bool v73;
                v73 = v72 == false;
                if (v73){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v72);
                } else {
                }
                assert("Tensor range check" && 0 <= v71 && v71 < 16l);
                assert("Tensor range check" && 0 <= v70 && v70 < 2l);
                long v74;
                v74 = 4l * v70;
                long v75;
                v75 = 12l * v71;
                long v76;
                v76 = v75 + v74;
                long v77;
                v77 = 16l * v71;
                long v78;
                v78 = v77 + v74;
                float * v79;
                v79 = v5+v76;
                float * v80;
                v80 = v66+v78;
                long v81;
                v81 = 0l;
                #pragma unroll
                while (while_method_0(v81)){
                    long v83;
                    v83 = 0l;
                    #pragma unroll
                    while (while_method_0(v83)){
                        assert("Tensor range check" && 0 <= v81 && v81 < 1l);
                        assert("Tensor range check" && 0 <= v83 && v83 < 1l);
                        long v85;
                        v85 = 8l * v83;
                        long v86;
                        v86 = 192l * v81;
                        long v87;
                        v87 = v86 + v85;
                        long v88;
                        v88 = 256l * v81;
                        long v89;
                        v89 = v88 + v85;
                        float v90[4l];
                        long v91;
                        v91 = 0l;
                        #pragma unroll
                        while (while_method_1(v91)){
                            assert("Tensor range check" && 0 <= v91 && v91 < 4l);
                            long v93;
                            v93 = v91 + v89;
                            float v94;
                            v94 = v80[v93];
                            float v95;
                            v95 = wmma::__float_to_tf32(v94);
                            assert("Tensor range check" && 0 <= v91 && v91 < 4l);
                            v90[v91] = v95;
                            v91 += 1l ;
                        }
                        int4* v96;
                        v96 = reinterpret_cast<int4*>(v90 + 0l);
                        int4* v97;
                        v97 = reinterpret_cast<int4*>(v79 + v87);
                        assert("Pointer alignment check" && (unsigned long long)(v96) % 4l == 0 && (unsigned long long)(v97) % 4l == 0);
                        *v97 = *v96;
                        v83 += 1l ;
                    }
                    v81 += 1l ;
                }
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
                v101 = v98 % 2l;
                long v102;
                v102 = v98 / 2l;
                bool v103;
                v103 = v102 < 16l;
                bool v104;
                v104 = v103 == false;
                if (v104){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v103);
                } else {
                }
                assert("Tensor range check" && 0 <= v102 && v102 < 16l);
                assert("Tensor range check" && 0 <= v101 && v101 < 2l);
                long v105;
                v105 = 4l * v101;
                long v106;
                v106 = 12l * v102;
                long v107;
                v107 = v106 + v105;
                long v108;
                v108 = 16l * v102;
                long v109;
                v109 = v108 + v105;
                float * v110;
                v110 = v4+v107;
                float * v111;
                v111 = v63+v109;
                long v112;
                v112 = 0l;
                #pragma unroll
                while (while_method_0(v112)){
                    long v114;
                    v114 = 0l;
                    #pragma unroll
                    while (while_method_0(v114)){
                        assert("Tensor range check" && 0 <= v112 && v112 < 1l);
                        assert("Tensor range check" && 0 <= v114 && v114 < 1l);
                        long v116;
                        v116 = 8l * v114;
                        long v117;
                        v117 = 192l * v112;
                        long v118;
                        v118 = v117 + v116;
                        long v119;
                        v119 = 256l * v112;
                        long v120;
                        v120 = v119 + v116;
                        float v121[4l];
                        long v122;
                        v122 = 0l;
                        #pragma unroll
                        while (while_method_1(v122)){
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
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v129[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v130[1l];
                long v131;
                v131 = 0l;
                #pragma unroll
                while (while_method_0(v131)){
                    long v133;
                    v133 = 0l;
                    #pragma unroll
                    while (while_method_0(v133)){
                        assert("Tensor range check" && 0 <= v131 && v131 < 1l);
                        assert("Tensor range check" && 0 <= v133 && v133 < 1l);
                        long v135;
                        v135 = v131 + v133;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v136 = v129[v135];
                        assert("Tensor range check" && 0 <= v131 && v131 < 1l);
                        long v137;
                        v137 = 192l * v131;
                        assert("Tensor range check" && 0 <= v133 && v133 < 1l);
                        long v138;
                        v138 = 8l * v133;
                        long v139;
                        v139 = v138 + v137;
                        long v140;
                        v140 = 0l;
                        #pragma unroll
                        while (while_method_2(v140)){
                            long v142;
                            v142 = 0l;
                            #pragma unroll
                            while (while_method_2(v142)){
                                assert("Tensor range check" && 0 <= v140 && v140 < 2l);
                                assert("Tensor range check" && 0 <= v142 && v142 < 2l);
                                long v144;
                                v144 = 96l * v142;
                                long v145;
                                v145 = v144 + v139;
                                long v146;
                                v146 = 4l * v140;
                                long v147;
                                v147 = v146 + v145;
                                float v148;
                                v148 = v30[v147];
                                bool v149;
                                v149 = 0l <= v142;
                                bool v151;
                                if (v149){
                                    bool v150;
                                    v150 = v142 < 2l;
                                    v151 = v150;
                                } else {
                                    v151 = false;
                                }
                                bool v152;
                                v152 = v151 == false;
                                if (v152){
                                    assert("The indices should be inside the range of the dimension." && v151);
                                } else {
                                }
                                bool v153;
                                v153 = 0l <= v140;
                                bool v155;
                                if (v153){
                                    bool v154;
                                    v154 = v140 < 2l;
                                    v155 = v154;
                                } else {
                                    v155 = false;
                                }
                                bool v156;
                                v156 = v155 == false;
                                if (v156){
                                    assert("The indices should be inside the range of the dimension." && v155);
                                } else {
                                }
                                long v157;
                                v157 = v140 * 2l;
                                long v158;
                                v158 = v142 + v157;
                                v136.x[v158] = v148;
                                v142 += 1l ;
                            }
                            v140 += 1l ;
                        }
                        v133 += 1l ;
                    }
                    v131 += 1l ;
                }
                long v159;
                v159 = 0l;
                #pragma unroll
                while (while_method_0(v159)){
                    long v161;
                    v161 = 0l;
                    #pragma unroll
                    while (while_method_0(v161)){
                        assert("Tensor range check" && 0 <= v159 && v159 < 1l);
                        assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                        long v163;
                        v163 = v159 + v161;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v164 = v130[v163];
                        assert("Tensor range check" && 0 <= v159 && v159 < 1l);
                        long v165;
                        v165 = 192l * v159;
                        assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                        long v166;
                        v166 = 8l * v161;
                        long v167;
                        v167 = v166 + v165;
                        long v168;
                        v168 = 0l;
                        #pragma unroll
                        while (while_method_2(v168)){
                            long v170;
                            v170 = 0l;
                            #pragma unroll
                            while (while_method_2(v170)){
                                assert("Tensor range check" && 0 <= v168 && v168 < 2l);
                                assert("Tensor range check" && 0 <= v170 && v170 < 2l);
                                long v172;
                                v172 = 4l * v170;
                                long v173;
                                v173 = v172 + v167;
                                long v174;
                                v174 = 96l * v168;
                                long v175;
                                v175 = v174 + v173;
                                float v176;
                                v176 = v43[v175];
                                bool v177;
                                v177 = 0l <= v170;
                                bool v179;
                                if (v177){
                                    bool v178;
                                    v178 = v170 < 2l;
                                    v179 = v178;
                                } else {
                                    v179 = false;
                                }
                                bool v180;
                                v180 = v179 == false;
                                if (v180){
                                    assert("The indices should be inside the range of the dimension." && v179);
                                } else {
                                }
                                bool v181;
                                v181 = 0l <= v168;
                                bool v183;
                                if (v181){
                                    bool v182;
                                    v182 = v168 < 2l;
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
                                long v185;
                                v185 = v168 * 2l;
                                long v186;
                                v186 = v170 + v185;
                                v164.x[v186] = v176;
                                v170 += 1l ;
                            }
                            v168 += 1l ;
                        }
                        v161 += 1l ;
                    }
                    v159 += 1l ;
                }
                __syncthreads();
                long v187;
                v187 = 0l;
                #pragma unroll
                while (while_method_0(v187)){
                    long v189;
                    v189 = 0l;
                    #pragma unroll
                    while (while_method_0(v189)){
                        long v191;
                        v191 = 0l;
                        #pragma unroll
                        while (while_method_0(v191)){
                            assert("Tensor range check" && 0 <= v187 && v187 < 1l);
                            assert("Tensor range check" && 0 <= v189 && v189 < 1l);
                            long v193;
                            v193 = v187 + v189;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v194 = v44[v193];
                            assert("Tensor range check" && 0 <= v187 && v187 < 1l);
                            assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                            long v195;
                            v195 = v187 + v191;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v196 = v129[v195];
                            assert("Tensor range check" && 0 <= v189 && v189 < 1l);
                            assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                            long v197;
                            v197 = v189 + v191;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v198 = v130[v197];
                            wmma::mma_sync(v194, v196, v198, v194);
                            v191 += 1l ;
                        }
                        v189 += 1l ;
                    }
                    v187 += 1l ;
                }
                v59 += 1l ;
            }
            long v199;
            v199 = 0l;
            #pragma unroll
            while (while_method_0(v199)){
                long v201;
                v201 = 0l;
                #pragma unroll
                while (while_method_0(v201)){
                    assert("Tensor range check" && 0 <= v199 && v199 < 1l);
                    assert("Tensor range check" && 0 <= v201 && v201 < 1l);
                    long v203;
                    v203 = v199 + v201;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v204 = v44[v203];
                    assert("Tensor range check" && 0 <= v199 && v199 < 1l);
                    assert("Tensor range check" && 0 <= v201 && v201 < 1l);
                    long v205;
                    v205 = 16l * v201;
                    long v206;
                    v206 = 384l * v199;
                    long v207;
                    v207 = v206 + v205;
                    float * v208;
                    v208 = v17+v207;
                    wmma::store_matrix_sync(v208, v204, 24l, wmma::mem_row_major);
                    v201 += 1l ;
                }
                v199 += 1l ;
            }
            __syncthreads();
            long v209;
            v209 = threadIdx.x;
            bool v210;
            v210 = 0l <= v209;
            bool v211;
            v211 = v210 == false;
            if (v211){
                assert("The index needs to be zero or positive." && v210);
            } else {
            }
            long v212;
            v212 = v209 % 4l;
            long v213;
            v213 = v209 / 4l;
            bool v214;
            v214 = v213 < 8l;
            bool v215;
            v215 = v214 == false;
            if (v215){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v214);
            } else {
            }
            assert("Tensor range check" && 0 <= v213 && v213 < 8l);
            assert("Tensor range check" && 0 <= v212 && v212 < 4l);
            long v216;
            v216 = 4l * v212;
            long v217;
            v217 = 16l * v213;
            long v218;
            v218 = v217 + v216;
            long v219;
            v219 = 24l * v213;
            long v220;
            v220 = v219 + v216;
            float * v221;
            v221 = v52+v218;
            float * v222;
            v222 = v6+v220;
            long v223;
            v223 = 0l;
            #pragma unroll
            while (while_method_2(v223)){
                long v225;
                v225 = 0l;
                #pragma unroll
                while (while_method_0(v225)){
                    assert("Tensor range check" && 0 <= v223 && v223 < 2l);
                    assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                    long v227;
                    v227 = 16l * v225;
                    long v228;
                    v228 = 128l * v223;
                    long v229;
                    v229 = v228 + v227;
                    long v230;
                    v230 = 192l * v223;
                    long v231;
                    v231 = v230 + v227;
                    int4* v232;
                    v232 = reinterpret_cast<int4*>(v222 + v231);
                    int4* v233;
                    v233 = reinterpret_cast<int4*>(v221 + v229);
                    assert("Pointer alignment check" && (unsigned long long)(v232) % 4l == 0 && (unsigned long long)(v233) % 4l == 0);
                    *v233 = *v232;
                    v225 += 1l ;
                }
                v223 += 1l ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            v47 += 1l ;
        }
        v45 += 1l ;
    }
    return ;
}
__device__ void method_4(unsigned char * v0, unsigned char * v1){
    float * v2;
    v2 = reinterpret_cast<float *>(&v1[1536ull]);
    float * v3;
    v3 = reinterpret_cast<float *>(&v0[512ull]);
    float * v4;
    v4 = reinterpret_cast<float *>(&v1[2560ull]);
    return method_5(v4, v3, v2);
}
__device__ void method_6(unsigned char * v0, unsigned char * v1){
    float * v2;
    v2 = reinterpret_cast<float *>(&v1[2560ull]);
    float * v3;
    v3 = reinterpret_cast<float *>(&v1[3584ull]);
    return method_3(v3, v2);
}
__device__ void method_7(unsigned char * v0, unsigned char * v1){
    float * v2;
    v2 = reinterpret_cast<float *>(&v1[3584ull]);
    float * v3;
    v3 = reinterpret_cast<float *>(&v0[1536ull]);
    float * v4;
    v4 = reinterpret_cast<float *>(&v1[4608ull]);
    return method_5(v4, v3, v2);
}
__device__ void method_8(unsigned char * v0, unsigned char * v1){
    float * v2;
    v2 = reinterpret_cast<float *>(&v1[4608ull]);
    float * v3;
    v3 = reinterpret_cast<float *>(&v1[5632ull]);
    return method_3(v3, v2);
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    method_0(v0, v1);
    method_2(v0, v1);
    method_4(v0, v1);
    method_6(v0, v1);
    method_7(v0, v1);
    return method_8(v0, v1);
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

options = []
options.append('--diag-suppress=550,20012')
options.append('--dopt=on')
options.append('--restrict')
raw_module = cp.RawModule(code=kernel, backend='nvrtc', enable_cooperative_groups=True, options=tuple(options))
def method0(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method2(v0 : char) -> None:
    print(v0, end="")
    del v0
    return 
def method3(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method4(v0 : f32) -> None:
    print("{:.6f}".format(v0), end="")
    del v0
    return 
def method1(v0 : cp.ndarray, v1 : i32, v2 : i32, v3 : i32, v4 : i32, v5 : i32) -> None:
    v6 = 0
    v7 = '['
    method2(v7)
    del v7
    v8 = 0
    while method3(v4, v8):
        v10 = v6
        v11 = v10 >= 100
        del v10
        if v11:
            v12 = " ..."
            method0(v12)
            del v12
            break
        else:
            pass
        del v11
        v13 = v8 == 0
        v14 = v13 != True
        del v13
        if v14:
            v15 = "; "
            method0(v15)
        else:
            pass
        del v14
        v16 = '['
        method2(v16)
        del v16
        v17 = 0
        while method3(v5, v17):
            v19 = v6
            v20 = v19 >= 100
            del v19
            if v20:
                v21 = " ..."
                method0(v21)
                del v21
                break
            else:
                pass
            del v20
            v22 = v17 == 0
            v23 = v22 != True
            del v22
            if v23:
                v24 = "; "
                method0(v24)
            else:
                pass
            del v23
            v25 = v6 + 1
            v6 = v25
            del v25
            v26 = v8 * v2
            v27 = v1 + v26
            del v26
            v28 = v17 * v3
            v29 = v27 + v28
            del v27, v28
            v30 = v0[v29].item()
            del v29
            method4(v30)
            del v30
            v17 += 1 
        del v17
        v31 = ']'
        method2(v31)
        del v31
        v8 += 1 
    del v0, v1, v2, v3, v4, v5, v6, v8
    v32 = ']'
    return method2(v32)
def main():
    v0 = cp.empty(2560,dtype=cp.uint8)
    v1 = cp.empty(6656,dtype=cp.uint8)
    v2 = v0[0:0+4*128].view(cp.float32)
    v3 = cp.random.normal(0.0,1.0,128,dtype=cp.float32) # type: ignore
    cp.copyto(v2[0:0+128],v3[0:0+128])
    del v2, v3
    v4 = v0[512:512+4*256].view(cp.float32)
    v5 = cp.random.normal(0.0,1.0,256,dtype=cp.float32) # type: ignore
    cp.copyto(v4[0:0+256],v5[0:0+256])
    del v4, v5
    v6 = v0[1536:1536+4*256].view(cp.float32)
    v7 = cp.random.normal(0.0,1.0,256,dtype=cp.float32) # type: ignore
    cp.copyto(v6[0:0+256],v7[0:0+256])
    del v6, v7
    v8 = "Here are the weight matrices."
    method0(v8)
    del v8
    print()
    v9 = v0[0:0+4*128].view(cp.float32)
    v10 = 0
    v11 = 8
    v12 = 1
    v13 = 16
    v14 = 8
    method1(v9, v10, v11, v12, v13, v14)
    del v9, v10, v11, v12, v13, v14
    print()
    v15 = v0[512:512+4*256].view(cp.float32)
    v16 = 0
    v17 = 16
    v18 = 1
    v19 = 16
    v20 = 16
    method1(v15, v16, v17, v18, v19, v20)
    del v15, v16, v17, v18, v19, v20
    print()
    v21 = v0[1536:1536+4*256].view(cp.float32)
    v22 = 0
    v23 = 16
    v24 = 1
    v25 = 16
    v26 = 16
    method1(v21, v22, v23, v24, v25, v26)
    del v21, v22, v23, v24, v25, v26
    print()
    v27 = v1[0:0+4*128].view(cp.float32)
    v28 = cp.random.normal(0.0,1.0,128,dtype=cp.float32) # type: ignore
    cp.copyto(v27[0:0+128],v28[0:0+128])
    del v28
    v29 = 0
    v30 = 8
    v31 = 1
    v32 = 16
    v33 = 8
    method1(v27, v29, v30, v31, v32, v33)
    del v27, v29, v30, v31, v32, v33
    print()
    v34 = "Here is the output tensor."
    method0(v34)
    del v34
    print()
    v35 = 0
    v36 = raw_module.get_function(f"entry{v35}")
    del v35
    v36.max_dynamic_shared_size_bytes = 1536 
    v36((1,),(32,),(v0, v1),shared_mem=1536)
    del v0, v36
    v37 = v1[5632:5632+4*256].view(cp.float32)
    del v1
    v38 = 0
    v39 = 16
    v40 = 1
    v41 = 16
    v42 = 16
    method1(v37, v38, v39, v40, v41, v42)
    del v37, v38, v39, v40, v41, v42
    print()
    return 

if __name__ == '__main__': print(main())
