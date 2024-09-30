kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <mma.h>
using namespace nvcuda;
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>
using default_int = int;
using default_uint = unsigned int;
template <typename el>
struct sptr // Shared pointer for the Spiral datatypes. They have to have the refc field inside them to work.
{
    el* base;

    __device__ sptr() : base(nullptr) {}
    __device__ sptr(el* ptr) : base(ptr) { this->base->refc++; }

    __device__ ~sptr()
    {
        if (this->base != nullptr && --this->base->refc == 0)
        {
            delete this->base;
            this->base = nullptr;
        }
    }

    __device__ sptr(sptr& x)
    {
        this->base = x.base;
        this->base->refc++;
    }

    __device__ sptr(sptr&& x)
    {
        this->base = x.base;
        x.base = nullptr;
    }

    __device__ sptr& operator=(sptr& x)
    {
        if (this->base != x.base)
        {
            delete this->base;
            this->base = x.base;
            this->base->refc++;
        }
        return *this;
    }

    __device__ sptr& operator=(sptr&& x)
    {
        if (this->base != x.base)
        {
            delete this->base;
            this->base = x.base;
            x.base = nullptr;
        }
        return *this;
    }
};

template <typename el>
struct csptr : public sptr<el>
{ // Shared pointer for closures specifically.
    using sptr<el>::sptr;
    template <typename... Args>
    __device__ auto operator()(Args... args) -> decltype(this->base->operator()(args...))
    {
        return this->base->operator()(args...);
    }
};

template <typename el, default_int max_length>
struct static_array
{
    el ptr[max_length];
    __device__ el& operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < max_length);
        return this->ptr[i];
    }
};

template <typename el, default_int max_length>
struct static_array_list
{
    default_int length{ 0 };
    el ptr[max_length];

    __device__ el& operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < this->length);
        return this->ptr[i];
    }
    __device__ void push(el& x) {
        ptr[this->length++] = x;
        assert("The array after pushing should not be greater than max length." && this->length <= max_length);
    }
    __device__ void push(el&& x) {
        ptr[this->length++] = std::move(x);
        assert("The array after pushing should not be greater than max length." && this->length <= max_length);
    }
    __device__ el pop() {
        assert("The array before popping should be greater than 0." && 0 < this->length);
        auto x = ptr[--this->length];
        ptr[this->length].~el();
        new (&ptr[this->length]) el();
        return x;
    }
    // Should be used only during initialization.
    __device__ void unsafe_set_length(default_int i) {
        assert("The new length should be in range." && 0 <= i && i <= max_length);
        this->length = i;
    }
};

template <typename el, default_int max_length>
struct dynamic_array_base
{
    int refc{ 0 };
    el* ptr;

    __device__ dynamic_array_base() : ptr(new el[max_length]) {}
    __device__ ~dynamic_array_base() { delete[] this->ptr; }

    __device__ el& operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < this->length);
        return this->ptr[i];
    }
};

template <typename el, default_int max_length>
struct dynamic_array
{
    sptr<dynamic_array_base<el, max_length>> ptr;

    __device__ dynamic_array() = default;
    __device__ dynamic_array(bool t) : ptr(new dynamic_array_base<el, max_length>()) {}
    __device__ el& operator[](default_int i) {
        return this->ptr.base->operator[](i);
    }
};

template <typename el, default_int max_length>
struct dynamic_array_list_base
{
    int refc{ 0 };
    default_int length{ 0 };
    el* ptr;

    __device__ dynamic_array_list_base() : ptr(new el[max_length]) {}
    __device__ dynamic_array_list_base(default_int l) : ptr(new el[max_length]) { this->unsafe_set_length(l); }
    __device__ ~dynamic_array_list_base() { delete[] this->ptr; }

    __device__ el& operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < this->length);
        return this->ptr[i];
    }
    __device__ void push(el& x) {
        ptr[this->length++] = x;
        assert("The array after pushing should not be greater than max length." && this->length <= max_length);
    }
    __device__ void push(el&& x) {
        ptr[this->length++] = std::move(x);
        assert("The array after pushing should not be greater than max length." && this->length <= max_length);
    }
    __device__ el pop() {
        assert("The array before popping should be greater than 0." && 0 < this->length);
        auto x = ptr[--this->length];
        ptr[this->length].~el();
        new (&ptr[this->length]) el();
        return x;
    }
    // Should be used only during initialization.
    __device__ void unsafe_set_length(default_int i) {
        assert("The new length should be in range." && 0 <= i && i <= max_length);
        this->length = i;
    }
};

template <typename el, default_int max_length>
struct dynamic_array_list
{
    sptr<dynamic_array_list_base<el, max_length>> ptr;

    __device__ dynamic_array_list() = default;
    __device__ dynamic_array_list(default_int l) : ptr(new dynamic_array_list_base<el, max_length>(l)) {}

    __device__ el& operator[](default_int i) {
        return this->ptr.base->operator[](i);
    }
    __device__ void push(el& x) {
        this->ptr.base->push(x);
    }
    __device__ void push(el&& x) {
        this->ptr.base->push(std::move(x));
    }
    __device__ el pop() {
        return this->ptr.base->pop();
    }
    // Should be used only during initialization.
    __device__ void unsafe_set_length(default_int i) {
        this->ptr.base->unsafe_set_length(i);
    }
    __device__ default_int length_() {
        return this->ptr.base->length;
    }
};

struct Union0;
struct Union0_0 { // None
};
struct Union0_1 { // Some
    int v0;
    __device__ Union0_1(int t0) : v0(t0) {}
    __device__ Union0_1() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // None
        Union0_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // None
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Some
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // None
            case 1: new (&this->case1) Union0_1(x.case1); break; // Some
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{x};
        }
        return *this;
    }
    __device__ Union0 & operator=(Union0 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // None
            case 1: this->case1.~Union0_1(); break; // Some
        }
        this->tag = 255;
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 64;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 16;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 64;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1, float * v2) {
    cuda::pipeline<cuda::thread_scope_thread> v3 = cuda::make_pipeline();
    extern __shared__ unsigned char v4[];
    float * v5;
    v5 = reinterpret_cast<float *>(&v4[0ull]);
    float * v7;
    v7 = reinterpret_cast<float *>(&v4[34816ull]);
    float * v9;
    v9 = reinterpret_cast<float *>(&v4[0ull]);
    int v11;
    v11 = threadIdx.x;
    int v12;
    v12 = v11 / 32;
    bool v13;
    v13 = 0 <= v12;
    bool v14;
    v14 = v13 == false;
    if (v14){
        assert("The index needs to be zero or positive." && v13);
    } else {
    }
    int v16;
    v16 = v12 % 8;
    int v17;
    v17 = v12 / 8;
    bool v18;
    v18 = v17 < 1;
    bool v19;
    v19 = v18 == false;
    if (v19){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v18);
    } else {
    }
    assert("Tensor range check" && 0 <= v17 && v17 < 1);
    assert("Tensor range check" && 0 <= v16 && v16 < 8);
    int v21;
    v21 = 16 * v16;
    int v22;
    v22 = 17408 * v17;
    int v23;
    v23 = v22 + v21;
    float * v24;
    v24 = v9+v23;
    assert("Tensor range check" && 0 <= v17 && v17 < 1);
    int v26;
    v26 = 8704 * v17;
    int v27;
    v27 = threadIdx.x;
    int v28;
    v28 = v27 % 32;
    bool v29;
    v29 = 0 <= v28;
    bool v30;
    v30 = v29 == false;
    if (v30){
        assert("The index needs to be zero or positive." && v29);
    } else {
    }
    int v32;
    v32 = v28 % 4;
    int v33;
    v33 = v28 / 4;
    bool v34;
    v34 = v33 < 8;
    bool v35;
    v35 = v34 == false;
    if (v35){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v34);
    } else {
    }
    assert("Tensor range check" && 0 <= v33 && v33 < 8);
    assert("Tensor range check" && 0 <= v32 && v32 < 4);
    int v37;
    v37 = v32 + v26;
    int v38;
    v38 = 68 * v33;
    int v39;
    v39 = v38 + v37;
    float * v40;
    v40 = v5+v39;
    assert("Tensor range check" && 0 <= v16 && v16 < 8);
    int v42;
    v42 = 1088 * v16;
    int v43;
    v43 = threadIdx.x;
    int v44;
    v44 = v43 % 32;
    bool v45;
    v45 = 0 <= v44;
    bool v46;
    v46 = v45 == false;
    if (v46){
        assert("The index needs to be zero or positive." && v45);
    } else {
    }
    int v48;
    v48 = v44 % 4;
    int v49;
    v49 = v44 / 4;
    bool v50;
    v50 = v49 < 8;
    bool v51;
    v51 = v50 == false;
    if (v51){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v50);
    } else {
    }
    assert("Tensor range check" && 0 <= v49 && v49 < 8);
    assert("Tensor range check" && 0 <= v48 && v48 < 4);
    int v53;
    v53 = v48 + v42;
    int v54;
    v54 = 68 * v49;
    int v55;
    v55 = v54 + v53;
    float * v56;
    v56 = v7+v55;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v58[8];
    int v59;
    v59 = 0;
    while (while_method_0(v59)){
        int v61;
        v61 = 0;
        while (while_method_0(v61)){
            assert("Tensor range check" && 0 <= v59 && v59 < 64);
            assert("Tensor range check" && 0 <= v61 && v61 < 64);
            int v63;
            v63 = 128 * v61;
            int v64;
            v64 = 1048576 * v59;
            int v65;
            v65 = v64 + v63;
            float * v66;
            v66 = v2+v65;
            // Pushing the loop unrolling to: 0
            int v68;
            v68 = threadIdx.x;
            bool v69;
            v69 = 0 <= v68;
            bool v70;
            v70 = v69 == false;
            if (v70){
                assert("The index needs to be zero or positive." && v69);
            } else {
            }
            int v72;
            v72 = v68 % 32;
            int v73;
            v73 = v68 / 32;
            bool v74;
            v74 = v73 < 8;
            bool v75;
            v75 = v74 == false;
            if (v75){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v74);
            } else {
            }
            assert("Tensor range check" && 0 <= v73 && v73 < 8);
            assert("Tensor range check" && 0 <= v72 && v72 < 32);
            int v77;
            v77 = 4 * v72;
            int v78;
            v78 = 136 * v73;
            int v79;
            v79 = v78 + v77;
            int v80;
            v80 = 8192 * v73;
            int v81;
            v81 = v80 + v77;
            float * v82;
            v82 = v9+v79;
            float * v84;
            v84 = v66+v81;
            int v86;
            v86 = 0;
            #pragma unroll
            while (while_method_1(v86)){
                int v88;
                v88 = 0;
                #pragma unroll
                while (while_method_2(v88)){
                    assert("Tensor range check" && 0 <= v86 && v86 < 16);
                    assert("Tensor range check" && 0 <= v88 && v88 < 1);
                    int v90;
                    v90 = 128 * v88;
                    int v91;
                    v91 = 1088 * v86;
                    int v92;
                    v92 = v91 + v90;
                    int v93;
                    v93 = 65536 * v86;
                    int v94;
                    v94 = v93 + v90;
                    int4* v95;
                    v95 = reinterpret_cast<int4*>(v84 + v94);
                    int4* v96;
                    v96 = reinterpret_cast<int4*>(v82 + v92);
                    assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v95) % 16 == 0 && reinterpret_cast<unsigned long long>(v96) % 16 == 0);
                    *v96 = *v95;
                    v88 += 1 ;
                }
                v86 += 1 ;
            }
            __syncthreads();
            int v97;
            v97 = 0;
            #pragma unroll
            while (while_method_3(v97)){
                int v99;
                v99 = 0;
                #pragma unroll
                while (while_method_2(v99)){
                    assert("Tensor range check" && 0 <= v97 && v97 < 8);
                    assert("Tensor range check" && 0 <= v99 && v99 < 1);
                    int v101;
                    v101 = v97 + v99;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v102 = v58[v101];
                    assert("Tensor range check" && 0 <= v97 && v97 < 8);
                    assert("Tensor range check" && 0 <= v99 && v99 < 1);
                    int v103;
                    v103 = 16 * v99;
                    int v104;
                    v104 = 2176 * v97;
                    int v105;
                    v105 = v104 + v103;
                    float * v106;
                    v106 = v24+v105;
                    wmma::load_matrix_sync(v102, v106, 136, wmma::mem_row_major);
                    v99 += 1 ;
                }
                v97 += 1 ;
            }
            __syncthreads();
            // Poping the loop unrolling to: 0
            int v108;
            v108 = 0;
            while (while_method_4(v108)){
                int v110;
                v110 = v108 + 1;
                bool v111;
                v111 = v108 == 0;
                int v112;
                v112 = v108 % 2;
                bool v113;
                v113 = 0 <= v108;
                bool v114;
                v114 = v113 == false;
                if (v114){
                    assert("The index needs to be zero or positive." && v113);
                } else {
                }
                bool v116;
                v116 = v108 < 64;
                bool v117;
                v117 = v116 == false;
                if (v117){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v116);
                } else {
                }
                bool v119;
                v119 = v110 < 64;
                Union0 v125;
                if (v119){
                    bool v120;
                    v120 = 0 <= v110;
                    bool v121;
                    v121 = v120 == false;
                    if (v121){
                        assert("The index needs to be zero or positive." && v120);
                    } else {
                    }
                    v125 = Union0{Union0_1{v110}};
                } else {
                    v125 = Union0{Union0_0{}};
                }
                assert("Tensor range check" && 0 <= v59 && v59 < 64);
                int v126;
                v126 = 524288 * v59;
                assert("Tensor range check" && 0 <= v108 && v108 < 64);
                int v127;
                v127 = 64 * v108;
                int v128;
                v128 = v127 + v126;
                float * v129;
                v129 = v0+v128;
                assert("Tensor range check" && 0 <= v61 && v61 < 64);
                int v131;
                v131 = 524288 * v61;
                if (v111){
                    assert("Tensor range check" && 0 <= v108 && v108 < 64);
                    int v132;
                    v132 = v127 + v131;
                    float * v133;
                    v133 = v1+v132;
                    // Pushing the loop unrolling to: 0
                    v3.producer_acquire();
                    int v135;
                    v135 = threadIdx.x;
                    bool v136;
                    v136 = 0 <= v135;
                    bool v137;
                    v137 = v136 == false;
                    if (v137){
                        assert("The index needs to be zero or positive." && v136);
                    } else {
                    }
                    int v139;
                    v139 = v135 % 16;
                    int v140;
                    v140 = v135 / 16;
                    bool v141;
                    v141 = v140 < 16;
                    bool v142;
                    v142 = v141 == false;
                    if (v142){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v141);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v140 && v140 < 16);
                    assert("Tensor range check" && 0 <= v139 && v139 < 16);
                    int v144;
                    v144 = 4 * v139;
                    int v145;
                    v145 = 68 * v140;
                    int v146;
                    v146 = v145 + v144;
                    int v147;
                    v147 = 4096 * v140;
                    int v148;
                    v148 = v147 + v144;
                    float * v149;
                    v149 = v7+v146;
                    float * v151;
                    v151 = v133+v148;
                    int v153;
                    v153 = 0;
                    #pragma unroll
                    while (while_method_3(v153)){
                        int v155;
                        v155 = 0;
                        #pragma unroll
                        while (while_method_2(v155)){
                            assert("Tensor range check" && 0 <= v153 && v153 < 8);
                            assert("Tensor range check" && 0 <= v155 && v155 < 1);
                            int v157;
                            v157 = 64 * v155;
                            int v158;
                            v158 = 1088 * v153;
                            int v159;
                            v159 = v158 + v157;
                            int v160;
                            v160 = 65536 * v153;
                            int v161;
                            v161 = v160 + v157;
                            constexpr int v162 = sizeof(float) * 4;
                            assert("Pointer alignment check" && (unsigned long long)(v151 + v161) % v162 == 0 && (unsigned long long)(v149 + v159) % v162 == 0);
                            cuda::memcpy_async(v149 + v159, v151 + v161, cuda::aligned_size_t<v162>(v162), v3);
                            v155 += 1 ;
                        }
                        v153 += 1 ;
                    }
                    v3.producer_commit();
                    // Poping the loop unrolling to: 0
                } else {
                }
                // Pushing the loop unrolling to: 0
                int v163;
                v163 = threadIdx.x;
                bool v164;
                v164 = 0 <= v163;
                bool v165;
                v165 = v164 == false;
                if (v165){
                    assert("The index needs to be zero or positive." && v164);
                } else {
                }
                int v167;
                v167 = v163 % 16;
                int v168;
                v168 = v163 / 16;
                bool v169;
                v169 = v168 < 16;
                bool v170;
                v170 = v169 == false;
                if (v170){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v169);
                } else {
                }
                assert("Tensor range check" && 0 <= v168 && v168 < 16);
                assert("Tensor range check" && 0 <= v167 && v167 < 16);
                int v172;
                v172 = 4 * v167;
                int v173;
                v173 = 68 * v168;
                int v174;
                v174 = v173 + v172;
                int v175;
                v175 = 4096 * v168;
                int v176;
                v176 = v175 + v172;
                float * v177;
                v177 = v5+v174;
                float * v179;
                v179 = v129+v176;
                int v181;
                v181 = 0;
                #pragma unroll
                while (while_method_3(v181)){
                    int v183;
                    v183 = 0;
                    #pragma unroll
                    while (while_method_2(v183)){
                        assert("Tensor range check" && 0 <= v181 && v181 < 8);
                        assert("Tensor range check" && 0 <= v183 && v183 < 1);
                        int v185;
                        v185 = 64 * v183;
                        int v186;
                        v186 = 1088 * v181;
                        int v187;
                        v187 = v186 + v185;
                        int v188;
                        v188 = 65536 * v181;
                        int v189;
                        v189 = v188 + v185;
                        int4* v190;
                        v190 = reinterpret_cast<int4*>(v179 + v189);
                        int4* v191;
                        v191 = reinterpret_cast<int4*>(v177 + v187);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v190) % 16 == 0 && reinterpret_cast<unsigned long long>(v191) % 16 == 0);
                        *v191 = *v190;
                        v183 += 1 ;
                    }
                    v181 += 1 ;
                }
                // Poping the loop unrolling to: 0
                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v192[1];
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v193[8];
                cuda::pipeline_consumer_wait_prior<0>(v3);;
                __syncthreads();
                // Pushing the loop unrolling to: 0
                int v194;
                v194 = 0;
                #pragma unroll
                while (while_method_2(v194)){
                    int v196;
                    v196 = 0;
                    #pragma unroll
                    while (while_method_3(v196)){
                        assert("Tensor range check" && 0 <= v194 && v194 < 1);
                        assert("Tensor range check" && 0 <= v196 && v196 < 8);
                        int v198;
                        v198 = 8 * v194;
                        int v199;
                        v199 = v198 + v196;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v200 = v193[v199];
                        assert("Tensor range check" && 0 <= v194 && v194 < 1);
                        int v201;
                        v201 = 1088 * v194;
                        assert("Tensor range check" && 0 <= v196 && v196 < 8);
                        int v202;
                        v202 = 8 * v196;
                        int v203;
                        v203 = v202 + v201;
                        int v204;
                        v204 = 0;
                        #pragma unroll
                        while (while_method_5(v204)){
                            int v206;
                            v206 = 0;
                            #pragma unroll
                            while (while_method_5(v206)){
                                assert("Tensor range check" && 0 <= v204 && v204 < 2);
                                assert("Tensor range check" && 0 <= v206 && v206 < 2);
                                int v208;
                                v208 = 4 * v206;
                                int v209;
                                v209 = v208 + v203;
                                int v210;
                                v210 = 544 * v204;
                                int v211;
                                v211 = v210 + v209;
                                float v212;
                                v212 = v56[v211];
                                bool v213;
                                v213 = 0 <= v206;
                                bool v215;
                                if (v213){
                                    bool v214;
                                    v214 = v206 < 2;
                                    v215 = v214;
                                } else {
                                    v215 = false;
                                }
                                bool v216;
                                v216 = v215 == false;
                                if (v216){
                                    assert("The indices should be inside the range of the dimension." && v215);
                                } else {
                                }
                                bool v218;
                                v218 = 0 <= v204;
                                bool v220;
                                if (v218){
                                    bool v219;
                                    v219 = v204 < 2;
                                    v220 = v219;
                                } else {
                                    v220 = false;
                                }
                                bool v221;
                                v221 = v220 == false;
                                if (v221){
                                    assert("The indices should be inside the range of the dimension." && v220);
                                } else {
                                }
                                int v223;
                                v223 = v204 * 2;
                                int v224;
                                v224 = v206 + v223;
                                v200.x[v224] = wmma::__float_to_tf32(v212);
                                v206 += 1 ;
                            }
                            v204 += 1 ;
                        }
                        v196 += 1 ;
                    }
                    v194 += 1 ;
                }
                // Poping the loop unrolling to: 0
                v3.consumer_release();
                switch (v125.tag) {
                    case 0: { // None
                        break;
                    }
                    case 1: { // Some
                        int v225 = v125.case1.v0;
                        assert("Tensor range check" && 0 <= v225 && v225 < 64);
                        int v226;
                        v226 = 64 * v225;
                        int v227;
                        v227 = v226 + v131;
                        float * v228;
                        v228 = v1+v227;
                        __syncthreads();
                        // Pushing the loop unrolling to: 0
                        v3.producer_acquire();
                        int v230;
                        v230 = threadIdx.x;
                        bool v231;
                        v231 = 0 <= v230;
                        bool v232;
                        v232 = v231 == false;
                        if (v232){
                            assert("The index needs to be zero or positive." && v231);
                        } else {
                        }
                        int v234;
                        v234 = v230 % 16;
                        int v235;
                        v235 = v230 / 16;
                        bool v236;
                        v236 = v235 < 16;
                        bool v237;
                        v237 = v236 == false;
                        if (v237){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v236);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v235 && v235 < 16);
                        assert("Tensor range check" && 0 <= v234 && v234 < 16);
                        int v239;
                        v239 = 4 * v234;
                        int v240;
                        v240 = 68 * v235;
                        int v241;
                        v241 = v240 + v239;
                        int v242;
                        v242 = 4096 * v235;
                        int v243;
                        v243 = v242 + v239;
                        float * v244;
                        v244 = v7+v241;
                        float * v246;
                        v246 = v228+v243;
                        int v248;
                        v248 = 0;
                        #pragma unroll
                        while (while_method_3(v248)){
                            int v250;
                            v250 = 0;
                            #pragma unroll
                            while (while_method_2(v250)){
                                assert("Tensor range check" && 0 <= v248 && v248 < 8);
                                assert("Tensor range check" && 0 <= v250 && v250 < 1);
                                int v252;
                                v252 = 64 * v250;
                                int v253;
                                v253 = 1088 * v248;
                                int v254;
                                v254 = v253 + v252;
                                int v255;
                                v255 = 65536 * v248;
                                int v256;
                                v256 = v255 + v252;
                                constexpr int v257 = sizeof(float) * 4;
                                assert("Pointer alignment check" && (unsigned long long)(v246 + v256) % v257 == 0 && (unsigned long long)(v244 + v254) % v257 == 0);
                                cuda::memcpy_async(v244 + v254, v246 + v256, cuda::aligned_size_t<v257>(v257), v3);
                                v250 += 1 ;
                            }
                            v248 += 1 ;
                        }
                        v3.producer_commit();
                        // Poping the loop unrolling to: 0
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                // Pushing the loop unrolling to: 0
                int v258;
                v258 = 0;
                #pragma unroll
                while (while_method_3(v258)){
                    int v260;
                    v260 = 0;
                    #pragma unroll
                    while (while_method_3(v260)){
                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v262 = v192[0];
                        assert("Tensor range check" && 0 <= v258 && v258 < 8);
                        int v263;
                        v263 = 1088 * v258;
                        assert("Tensor range check" && 0 <= v260 && v260 < 8);
                        int v264;
                        v264 = 8 * v260;
                        int v265;
                        v265 = v264 + v263;
                        int v266;
                        v266 = 0;
                        #pragma unroll
                        while (while_method_5(v266)){
                            int v268;
                            v268 = 0;
                            #pragma unroll
                            while (while_method_5(v268)){
                                assert("Tensor range check" && 0 <= v266 && v266 < 2);
                                assert("Tensor range check" && 0 <= v268 && v268 < 2);
                                int v270;
                                v270 = 544 * v268;
                                int v271;
                                v271 = v270 + v265;
                                int v272;
                                v272 = 4 * v266;
                                int v273;
                                v273 = v272 + v271;
                                float v274;
                                v274 = v40[v273];
                                bool v275;
                                v275 = 0 <= v268;
                                bool v277;
                                if (v275){
                                    bool v276;
                                    v276 = v268 < 2;
                                    v277 = v276;
                                } else {
                                    v277 = false;
                                }
                                bool v278;
                                v278 = v277 == false;
                                if (v278){
                                    assert("The indices should be inside the range of the dimension." && v277);
                                } else {
                                }
                                bool v280;
                                v280 = 0 <= v266;
                                bool v282;
                                if (v280){
                                    bool v281;
                                    v281 = v266 < 2;
                                    v282 = v281;
                                } else {
                                    v282 = false;
                                }
                                bool v283;
                                v283 = v282 == false;
                                if (v283){
                                    assert("The indices should be inside the range of the dimension." && v282);
                                } else {
                                }
                                int v285;
                                v285 = v266 * 2;
                                int v286;
                                v286 = v268 + v285;
                                v262.x[v286] = wmma::__float_to_tf32(v274);
                                v268 += 1 ;
                            }
                            v266 += 1 ;
                        }
                        int v287;
                        v287 = 0;
                        #pragma unroll
                        while (while_method_2(v287)){
                            assert("Tensor range check" && 0 <= v258 && v258 < 8);
                            assert("Tensor range check" && 0 <= v287 && v287 < 1);
                            int v289;
                            v289 = v258 + v287;
                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v290 = v58[v289];
                            assert("Tensor range check" && 0 <= v287 && v287 < 1);
                            assert("Tensor range check" && 0 <= v260 && v260 < 8);
                            int v291;
                            v291 = 8 * v287;
                            int v292;
                            v292 = v291 + v260;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v293 = v193[v292];
                            wmma::mma_sync(v290, v262, v293, v290);
                            v287 += 1 ;
                        }
                        v260 += 1 ;
                    }
                    v258 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                v108 = v110;
            }
            // Pushing the loop unrolling to: 0
            int v294;
            v294 = 0;
            #pragma unroll
            while (while_method_3(v294)){
                int v296;
                v296 = 0;
                #pragma unroll
                while (while_method_2(v296)){
                    assert("Tensor range check" && 0 <= v294 && v294 < 8);
                    assert("Tensor range check" && 0 <= v296 && v296 < 1);
                    int v298;
                    v298 = v294 + v296;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v299 = v58[v298];
                    assert("Tensor range check" && 0 <= v294 && v294 < 8);
                    assert("Tensor range check" && 0 <= v296 && v296 < 1);
                    int v300;
                    v300 = 16 * v296;
                    int v301;
                    v301 = 2176 * v294;
                    int v302;
                    v302 = v301 + v300;
                    float * v303;
                    v303 = v24+v302;
                    wmma::store_matrix_sync(v303, v299, 136, wmma::mem_row_major);
                    v296 += 1 ;
                }
                v294 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            // Pushing the loop unrolling to: 0
            int v305;
            v305 = threadIdx.x;
            bool v306;
            v306 = 0 <= v305;
            bool v307;
            v307 = v306 == false;
            if (v307){
                assert("The index needs to be zero or positive." && v306);
            } else {
            }
            int v309;
            v309 = v305 % 32;
            int v310;
            v310 = v305 / 32;
            bool v311;
            v311 = v310 < 8;
            bool v312;
            v312 = v311 == false;
            if (v312){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v311);
            } else {
            }
            assert("Tensor range check" && 0 <= v310 && v310 < 8);
            assert("Tensor range check" && 0 <= v309 && v309 < 32);
            int v314;
            v314 = 4 * v309;
            int v315;
            v315 = 8192 * v310;
            int v316;
            v316 = v315 + v314;
            int v317;
            v317 = 136 * v310;
            int v318;
            v318 = v317 + v314;
            float * v319;
            v319 = v66+v316;
            float * v321;
            v321 = v9+v318;
            int v323;
            v323 = 0;
            #pragma unroll
            while (while_method_1(v323)){
                int v325;
                v325 = 0;
                #pragma unroll
                while (while_method_2(v325)){
                    assert("Tensor range check" && 0 <= v323 && v323 < 16);
                    assert("Tensor range check" && 0 <= v325 && v325 < 1);
                    int v327;
                    v327 = 128 * v325;
                    int v328;
                    v328 = 65536 * v323;
                    int v329;
                    v329 = v328 + v327;
                    int v330;
                    v330 = 1088 * v323;
                    int v331;
                    v331 = v330 + v327;
                    int4* v332;
                    v332 = reinterpret_cast<int4*>(v321 + v331);
                    int4* v333;
                    v333 = reinterpret_cast<int4*>(v319 + v329);
                    assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v332) % 16 == 0 && reinterpret_cast<unsigned long long>(v333) % 16 == 0);
                    *v333 = *v332;
                    v325 += 1 ;
                }
                v323 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            v61 += 1 ;
        }
        v59 += 1 ;
    }
    return ;
}
"""
class static_array():
    def __init__(self, length):
        self.ptr = []
        for _ in range(length):
            self.ptr.append(None)

    def __getitem__(self, index):
        assert 0 <= index < len(self.ptr), "The get index needs to be in range."
        return self.ptr[index]
    
    def __setitem__(self, index, value):
        assert 0 <= index < len(self.ptr), "The set index needs to be in range."
        self.ptr[index] = value

class static_array_list(static_array):
    def __init__(self, length):
        super().__init__(length)
        self.length = 0

    def __getitem__(self, index):
        assert 0 <= index < self.length, "The get index needs to be in range."
        return self.ptr[index]
    
    def __setitem__(self, index, value):
        assert 0 <= index < self.length, "The set index needs to be in range."
        self.ptr[index] = value

    def push(self,value):
        assert (self.length < len(self.ptr)), "The length before pushing has to be less than the maximum length of the array."
        self.ptr[self.length] = value
        self.length += 1

    def pop(self):
        assert (0 < self.length), "The length before popping has to be greater than 0."
        self.length -= 1
        return self.ptr[self.length]

    def unsafe_set_length(self,i):
        assert 0 <= i <= len(self.ptr), "The new length has to be in range."
        self.length = i

class dynamic_array(static_array): 
    pass

class dynamic_array_list(static_array_list):
    def length_(self): return self.length

import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = int; i16 = int; i32 = int; i64 = int; u8 = int; u16 = int; u32 = int; u64 = int; f32 = float; f64 = float; char = str; string = str

options = []
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=255')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def main_body():
    v0 = cp.random.normal(0.0,1.0,67108864,dtype=cp.float32) # type: ignore
    v1 = cp.random.normal(0.0,1.0,33554432,dtype=cp.float32) # type: ignore
    v2 = cp.random.normal(0.0,1.0,33554432,dtype=cp.float32) # type: ignore
    v3 = v2.reshape((8192, 4096))
    v4 = v1.reshape((8192, 4096))
    v5 = cp.transpose(v4)
    del v4
    v6 = v0.reshape((8192, 8192))
    v7 = (cp.matmul(v3,v5) + v6).flatten()
    del v3, v5, v6
    v8 = v7.size
    v9 = 67108864 == v8
    del v8
    v10 = v9 == False
    if v10:
        v11 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v9, v11
        del v11
    else:
        pass
    del v9, v10
    v12 = cp.cuda.Device().attributes['MultiProcessorCount']
    v13 = v12 == 24
    del v12
    v14 = v13 == False
    if v14:
        v15 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v13, v15
        del v15
    else:
        pass
    del v13, v14
    v16 = 0
    v17 = raw_module.get_function(f"entry{v16}")
    del v16
    v17.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v17((24,),(256,),(v2, v1, v0),shared_mem=98304)
    del v1, v2, v17
    v18 = cp.max(cp.abs(v0-v7))
    del v0, v7
    return v18

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
