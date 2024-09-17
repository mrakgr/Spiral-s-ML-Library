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
    v1 = v0 < 8192;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 64;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 4;
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
    v18 = v17 < 2;
    bool v19;
    v19 = v18 == false;
    if (v19){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v18);
    } else {
    }
    assert("Tensor range check" && 0 <= v17 && v17 < 2);
    assert("Tensor range check" && 0 <= v16 && v16 < 8);
    int v21;
    v21 = 16 * v16;
    int v22;
    v22 = 4352 * v17;
    int v23;
    v23 = v22 + v21;
    float * v24;
    v24 = v9+v23;
    assert("Tensor range check" && 0 <= v16 && v16 < 8);
    int v26;
    v26 = 1088 * v16;
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
    v40 = v7+v39;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v42[2];
    int v43;
    v43 = blockIdx.x;
    int v44;
    v44 = v43;
    while (while_method_0(v44)){
        bool v46;
        v46 = 0 <= v44;
        bool v47;
        v47 = v46 == false;
        if (v47){
            assert("The index needs to be zero or positive." && v46);
        } else {
        }
        int v49;
        v49 = v44 % 64;
        int v50;
        v50 = v44 / 64;
        bool v51;
        v51 = v50 < 128;
        bool v52;
        v52 = v51 == false;
        if (v52){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v51);
        } else {
        }
        assert("Tensor range check" && 0 <= v50 && v50 < 128);
        assert("Tensor range check" && 0 <= v49 && v49 < 64);
        int v54;
        v54 = 128 * v49;
        int v55;
        v55 = 524288 * v50;
        int v56;
        v56 = v55 + v54;
        float * v57;
        v57 = v2+v56;
        // Pushing the loop unrolling to: 0
        int v59;
        v59 = threadIdx.x;
        bool v60;
        v60 = 0 <= v59;
        bool v61;
        v61 = v60 == false;
        if (v61){
            assert("The index needs to be zero or positive." && v60);
        } else {
        }
        int v63;
        v63 = v59 % 64;
        int v64;
        v64 = v59 / 64;
        bool v65;
        v65 = v64 < 8;
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v65);
        } else {
        }
        assert("Tensor range check" && 0 <= v64 && v64 < 8);
        assert("Tensor range check" && 0 <= v63 && v63 < 64);
        int v68;
        v68 = 2 * v63;
        int v69;
        v69 = 136 * v64;
        int v70;
        v70 = v69 + v68;
        int v71;
        v71 = 8192 * v64;
        int v72;
        v72 = v71 + v68;
        float * v73;
        v73 = v9+v70;
        float * v75;
        v75 = v57+v72;
        int v77;
        v77 = 0;
        #pragma unroll
        while (while_method_1(v77)){
            int v79;
            v79 = 0;
            #pragma unroll
            while (while_method_2(v79)){
                assert("Tensor range check" && 0 <= v77 && v77 < 8);
                assert("Tensor range check" && 0 <= v79 && v79 < 1);
                int v81;
                v81 = 128 * v79;
                int v82;
                v82 = 1088 * v77;
                int v83;
                v83 = v82 + v81;
                int v84;
                v84 = 65536 * v77;
                int v85;
                v85 = v84 + v81;
                int2* v86;
                v86 = reinterpret_cast<int2*>(v75 + v85);
                int2* v87;
                v87 = reinterpret_cast<int2*>(v73 + v83);
                assert("Pointer alignment check" && (unsigned long long)(v86) % 2 == 0 && (unsigned long long)(v87) % 2 == 0);
                *v87 = *v86;
                v79 += 1 ;
            }
            v77 += 1 ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0));
        int v88;
        v88 = 0;
        #pragma unroll
        while (while_method_3(v88)){
            int v90;
            v90 = 0;
            #pragma unroll
            while (while_method_2(v90)){
                assert("Tensor range check" && 0 <= v88 && v88 < 2);
                assert("Tensor range check" && 0 <= v90 && v90 < 1);
                int v92;
                v92 = v88 + v90;
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v93 = v42[v92];
                assert("Tensor range check" && 0 <= v88 && v88 < 2);
                assert("Tensor range check" && 0 <= v90 && v90 < 1);
                int v94;
                v94 = 16 * v90;
                int v95;
                v95 = 2176 * v88;
                int v96;
                v96 = v95 + v94;
                float * v97;
                v97 = v24+v96;
                wmma::load_matrix_sync(v93, v97, 136, wmma::mem_row_major);
                v90 += 1 ;
            }
            v88 += 1 ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0));
        // Poping the loop unrolling to: 0
        int v99;
        v99 = 0;
        while (while_method_4(v99)){
            int v101;
            v101 = v99 + 1;
            bool v102;
            v102 = v99 == 0;
            int v103;
            v103 = v99 % 2;
            bool v104;
            v104 = 0 <= v99;
            bool v105;
            v105 = v104 == false;
            if (v105){
                assert("The index needs to be zero or positive." && v104);
            } else {
            }
            bool v107;
            v107 = v99 < 64;
            bool v108;
            v108 = v107 == false;
            if (v108){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v107);
            } else {
            }
            bool v110;
            v110 = v101 < 64;
            Union0 v116;
            if (v110){
                bool v111;
                v111 = 0 <= v101;
                bool v112;
                v112 = v111 == false;
                if (v112){
                    assert("The index needs to be zero or positive." && v111);
                } else {
                }
                v116 = Union0{Union0_1{v101}};
            } else {
                v116 = Union0{Union0_0{}};
            }
            assert("Tensor range check" && 0 <= v50 && v50 < 128);
            int v117;
            v117 = 262144 * v50;
            assert("Tensor range check" && 0 <= v49 && v49 < 64);
            int v118;
            v118 = 524288 * v49;
            if (v102){
                assert("Tensor range check" && 0 <= v99 && v99 < 64);
                int v119;
                v119 = 64 * v99;
                int v120;
                v120 = v119 + v117;
                float * v121;
                v121 = v0+v120;
                assert("Tensor range check" && 0 <= v99 && v99 < 64);
                int v123;
                v123 = v119 + v118;
                float * v124;
                v124 = v1+v123;
                // Pushing the loop unrolling to: 0
                v3.producer_acquire();
                assert("Tensor range check" && 0 <= v103 && v103 < 2);
                int v126;
                v126 = 4352 * v103;
                int v127;
                v127 = threadIdx.x;
                bool v128;
                v128 = 0 <= v127;
                bool v129;
                v129 = v128 == false;
                if (v129){
                    assert("The index needs to be zero or positive." && v128);
                } else {
                }
                int v131;
                v131 = v127 % 32;
                int v132;
                v132 = v127 / 32;
                bool v133;
                v133 = v132 < 16;
                bool v134;
                v134 = v133 == false;
                if (v134){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v133);
                } else {
                }
                assert("Tensor range check" && 0 <= v132 && v132 < 16);
                assert("Tensor range check" && 0 <= v131 && v131 < 32);
                int v136;
                v136 = 2 * v131;
                int v137;
                v137 = v136 + v126;
                int v138;
                v138 = 68 * v132;
                int v139;
                v139 = v138 + v137;
                int v140;
                v140 = 4096 * v132;
                int v141;
                v141 = v140 + v136;
                float * v142;
                v142 = v5+v139;
                float * v144;
                v144 = v121+v141;
                int v146;
                v146 = 0;
                #pragma unroll
                while (while_method_5(v146)){
                    int v148;
                    v148 = 0;
                    #pragma unroll
                    while (while_method_2(v148)){
                        assert("Tensor range check" && 0 <= v146 && v146 < 4);
                        assert("Tensor range check" && 0 <= v148 && v148 < 1);
                        int v150;
                        v150 = 64 * v148;
                        int v151;
                        v151 = 1088 * v146;
                        int v152;
                        v152 = v151 + v150;
                        int v153;
                        v153 = 65536 * v146;
                        int v154;
                        v154 = v153 + v150;
                        constexpr int v155 = sizeof(float) * 2;
                        assert("Pointer alignment check" && (unsigned long long)(v144 + v154) % v155 == 0 && (unsigned long long)(v142 + v152) % v155 == 0);
                        cuda::memcpy_async(v142 + v152, v144 + v154, cuda::aligned_size_t<v155>(v155), v3);
                        v148 += 1 ;
                    }
                    v146 += 1 ;
                }
                int v156;
                v156 = threadIdx.x;
                bool v157;
                v157 = 0 <= v156;
                bool v158;
                v158 = v157 == false;
                if (v158){
                    assert("The index needs to be zero or positive." && v157);
                } else {
                }
                int v160;
                v160 = v156 % 32;
                int v161;
                v161 = v156 / 32;
                bool v162;
                v162 = v161 < 16;
                bool v163;
                v163 = v162 == false;
                if (v163){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v162);
                } else {
                }
                assert("Tensor range check" && 0 <= v161 && v161 < 16);
                assert("Tensor range check" && 0 <= v160 && v160 < 32);
                int v165;
                v165 = 2 * v160;
                int v166;
                v166 = 68 * v161;
                int v167;
                v167 = v166 + v165;
                int v168;
                v168 = 4096 * v161;
                int v169;
                v169 = v168 + v165;
                float * v170;
                v170 = v7+v167;
                float * v172;
                v172 = v124+v169;
                int v174;
                v174 = 0;
                #pragma unroll
                while (while_method_1(v174)){
                    int v176;
                    v176 = 0;
                    #pragma unroll
                    while (while_method_2(v176)){
                        assert("Tensor range check" && 0 <= v174 && v174 < 8);
                        assert("Tensor range check" && 0 <= v176 && v176 < 1);
                        int v178;
                        v178 = 64 * v176;
                        int v179;
                        v179 = 1088 * v174;
                        int v180;
                        v180 = v179 + v178;
                        int v181;
                        v181 = 65536 * v174;
                        int v182;
                        v182 = v181 + v178;
                        constexpr int v183 = sizeof(float) * 2;
                        assert("Pointer alignment check" && (unsigned long long)(v172 + v182) % v183 == 0 && (unsigned long long)(v170 + v180) % v183 == 0);
                        cuda::memcpy_async(v170 + v180, v172 + v182, cuda::aligned_size_t<v183>(v183), v3);
                        v176 += 1 ;
                    }
                    v174 += 1 ;
                }
                v3.producer_commit();
                // Poping the loop unrolling to: 0
            } else {
            }
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v184[1];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v185[8];
            cuda::pipeline_consumer_wait_prior<0>(v3);;
            asm("barrier.cta.sync %0;" :: "r"(0));
            // Pushing the loop unrolling to: 0
            int v186;
            v186 = 0;
            #pragma unroll
            while (while_method_2(v186)){
                int v188;
                v188 = 0;
                #pragma unroll
                while (while_method_1(v188)){
                    assert("Tensor range check" && 0 <= v186 && v186 < 1);
                    assert("Tensor range check" && 0 <= v188 && v188 < 8);
                    int v190;
                    v190 = 8 * v186;
                    int v191;
                    v191 = v190 + v188;
                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v192 = v185[v191];
                    assert("Tensor range check" && 0 <= v186 && v186 < 1);
                    int v193;
                    v193 = 1088 * v186;
                    assert("Tensor range check" && 0 <= v188 && v188 < 8);
                    int v194;
                    v194 = 8 * v188;
                    int v195;
                    v195 = v194 + v193;
                    int v196;
                    v196 = 0;
                    #pragma unroll
                    while (while_method_3(v196)){
                        int v198;
                        v198 = 0;
                        #pragma unroll
                        while (while_method_3(v198)){
                            assert("Tensor range check" && 0 <= v196 && v196 < 2);
                            assert("Tensor range check" && 0 <= v198 && v198 < 2);
                            int v200;
                            v200 = 4 * v198;
                            int v201;
                            v201 = v200 + v195;
                            int v202;
                            v202 = 544 * v196;
                            int v203;
                            v203 = v202 + v201;
                            float v204;
                            v204 = v40[v203];
                            bool v205;
                            v205 = 0 <= v198;
                            bool v207;
                            if (v205){
                                bool v206;
                                v206 = v198 < 2;
                                v207 = v206;
                            } else {
                                v207 = false;
                            }
                            bool v208;
                            v208 = v207 == false;
                            if (v208){
                                assert("The indices should be inside the range of the dimension." && v207);
                            } else {
                            }
                            bool v210;
                            v210 = 0 <= v196;
                            bool v212;
                            if (v210){
                                bool v211;
                                v211 = v196 < 2;
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
                            int v215;
                            v215 = v196 * 2;
                            int v216;
                            v216 = v198 + v215;
                            v192.x[v216] = wmma::__float_to_tf32(v204);
                            v198 += 1 ;
                        }
                        v196 += 1 ;
                    }
                    v188 += 1 ;
                }
                v186 += 1 ;
            }
            // Poping the loop unrolling to: 0
            v3.consumer_release();
            switch (v116.tag) {
                case 0: { // None
                    break;
                }
                case 1: { // Some
                    int v217 = v116.case1.v0;
                    assert("Tensor range check" && 0 <= v217 && v217 < 64);
                    int v218;
                    v218 = 64 * v217;
                    int v219;
                    v219 = v218 + v117;
                    float * v220;
                    v220 = v0+v219;
                    assert("Tensor range check" && 0 <= v217 && v217 < 64);
                    int v222;
                    v222 = v218 + v118;
                    float * v223;
                    v223 = v1+v222;
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    // Pushing the loop unrolling to: 0
                    v3.producer_acquire();
                    int v225;
                    v225 = v103 ^ 1;
                    assert("Tensor range check" && 0 <= v225 && v225 < 2);
                    int v226;
                    v226 = 4352 * v225;
                    int v227;
                    v227 = threadIdx.x;
                    bool v228;
                    v228 = 0 <= v227;
                    bool v229;
                    v229 = v228 == false;
                    if (v229){
                        assert("The index needs to be zero or positive." && v228);
                    } else {
                    }
                    int v231;
                    v231 = v227 % 32;
                    int v232;
                    v232 = v227 / 32;
                    bool v233;
                    v233 = v232 < 16;
                    bool v234;
                    v234 = v233 == false;
                    if (v234){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v233);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v232 && v232 < 16);
                    assert("Tensor range check" && 0 <= v231 && v231 < 32);
                    int v236;
                    v236 = 2 * v231;
                    int v237;
                    v237 = v236 + v226;
                    int v238;
                    v238 = 68 * v232;
                    int v239;
                    v239 = v238 + v237;
                    int v240;
                    v240 = 4096 * v232;
                    int v241;
                    v241 = v240 + v236;
                    float * v242;
                    v242 = v5+v239;
                    float * v244;
                    v244 = v220+v241;
                    int v246;
                    v246 = 0;
                    #pragma unroll
                    while (while_method_5(v246)){
                        int v248;
                        v248 = 0;
                        #pragma unroll
                        while (while_method_2(v248)){
                            assert("Tensor range check" && 0 <= v246 && v246 < 4);
                            assert("Tensor range check" && 0 <= v248 && v248 < 1);
                            int v250;
                            v250 = 64 * v248;
                            int v251;
                            v251 = 1088 * v246;
                            int v252;
                            v252 = v251 + v250;
                            int v253;
                            v253 = 65536 * v246;
                            int v254;
                            v254 = v253 + v250;
                            constexpr int v255 = sizeof(float) * 2;
                            assert("Pointer alignment check" && (unsigned long long)(v244 + v254) % v255 == 0 && (unsigned long long)(v242 + v252) % v255 == 0);
                            cuda::memcpy_async(v242 + v252, v244 + v254, cuda::aligned_size_t<v255>(v255), v3);
                            v248 += 1 ;
                        }
                        v246 += 1 ;
                    }
                    int v256;
                    v256 = threadIdx.x;
                    bool v257;
                    v257 = 0 <= v256;
                    bool v258;
                    v258 = v257 == false;
                    if (v258){
                        assert("The index needs to be zero or positive." && v257);
                    } else {
                    }
                    int v260;
                    v260 = v256 % 32;
                    int v261;
                    v261 = v256 / 32;
                    bool v262;
                    v262 = v261 < 16;
                    bool v263;
                    v263 = v262 == false;
                    if (v263){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v262);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v261 && v261 < 16);
                    assert("Tensor range check" && 0 <= v260 && v260 < 32);
                    int v265;
                    v265 = 2 * v260;
                    int v266;
                    v266 = 68 * v261;
                    int v267;
                    v267 = v266 + v265;
                    int v268;
                    v268 = 4096 * v261;
                    int v269;
                    v269 = v268 + v265;
                    float * v270;
                    v270 = v7+v267;
                    float * v272;
                    v272 = v223+v269;
                    int v274;
                    v274 = 0;
                    #pragma unroll
                    while (while_method_1(v274)){
                        int v276;
                        v276 = 0;
                        #pragma unroll
                        while (while_method_2(v276)){
                            assert("Tensor range check" && 0 <= v274 && v274 < 8);
                            assert("Tensor range check" && 0 <= v276 && v276 < 1);
                            int v278;
                            v278 = 64 * v276;
                            int v279;
                            v279 = 1088 * v274;
                            int v280;
                            v280 = v279 + v278;
                            int v281;
                            v281 = 65536 * v274;
                            int v282;
                            v282 = v281 + v278;
                            constexpr int v283 = sizeof(float) * 2;
                            assert("Pointer alignment check" && (unsigned long long)(v272 + v282) % v283 == 0 && (unsigned long long)(v270 + v280) % v283 == 0);
                            cuda::memcpy_async(v270 + v280, v272 + v282, cuda::aligned_size_t<v283>(v283), v3);
                            v276 += 1 ;
                        }
                        v274 += 1 ;
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
            int v284;
            v284 = 0;
            #pragma unroll
            while (while_method_3(v284)){
                int v286;
                v286 = 0;
                #pragma unroll
                while (while_method_1(v286)){
                    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v288 = v184[0];
                    assert("Tensor range check" && 0 <= v103 && v103 < 2);
                    int v289;
                    v289 = 4352 * v103;
                    assert("Tensor range check" && 0 <= v17 && v17 < 2);
                    int v290;
                    v290 = 2176 * v17;
                    int v291;
                    v291 = v290 + v289;
                    int v292;
                    v292 = threadIdx.x;
                    int v293;
                    v293 = v292 % 32;
                    bool v294;
                    v294 = 0 <= v293;
                    bool v295;
                    v295 = v294 == false;
                    if (v295){
                        assert("The index needs to be zero or positive." && v294);
                    } else {
                    }
                    int v297;
                    v297 = v293 % 4;
                    int v298;
                    v298 = v293 / 4;
                    bool v299;
                    v299 = v298 < 8;
                    bool v300;
                    v300 = v299 == false;
                    if (v300){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v299);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v298 && v298 < 8);
                    assert("Tensor range check" && 0 <= v297 && v297 < 4);
                    int v302;
                    v302 = v297 + v291;
                    int v303;
                    v303 = 68 * v298;
                    int v304;
                    v304 = v303 + v302;
                    float * v305;
                    v305 = v5+v304;
                    assert("Tensor range check" && 0 <= v284 && v284 < 2);
                    int v307;
                    v307 = 1088 * v284;
                    assert("Tensor range check" && 0 <= v286 && v286 < 8);
                    int v308;
                    v308 = 8 * v286;
                    int v309;
                    v309 = v308 + v307;
                    int v310;
                    v310 = 0;
                    #pragma unroll
                    while (while_method_3(v310)){
                        int v312;
                        v312 = 0;
                        #pragma unroll
                        while (while_method_3(v312)){
                            assert("Tensor range check" && 0 <= v310 && v310 < 2);
                            assert("Tensor range check" && 0 <= v312 && v312 < 2);
                            int v314;
                            v314 = 544 * v312;
                            int v315;
                            v315 = v314 + v309;
                            int v316;
                            v316 = 4 * v310;
                            int v317;
                            v317 = v316 + v315;
                            float v318;
                            v318 = v305[v317];
                            bool v319;
                            v319 = 0 <= v312;
                            bool v321;
                            if (v319){
                                bool v320;
                                v320 = v312 < 2;
                                v321 = v320;
                            } else {
                                v321 = false;
                            }
                            bool v322;
                            v322 = v321 == false;
                            if (v322){
                                assert("The indices should be inside the range of the dimension." && v321);
                            } else {
                            }
                            bool v324;
                            v324 = 0 <= v310;
                            bool v326;
                            if (v324){
                                bool v325;
                                v325 = v310 < 2;
                                v326 = v325;
                            } else {
                                v326 = false;
                            }
                            bool v327;
                            v327 = v326 == false;
                            if (v327){
                                assert("The indices should be inside the range of the dimension." && v326);
                            } else {
                            }
                            int v329;
                            v329 = v310 * 2;
                            int v330;
                            v330 = v312 + v329;
                            v288.x[v330] = wmma::__float_to_tf32(v318);
                            v312 += 1 ;
                        }
                        v310 += 1 ;
                    }
                    int v331;
                    v331 = 0;
                    #pragma unroll
                    while (while_method_2(v331)){
                        assert("Tensor range check" && 0 <= v284 && v284 < 2);
                        assert("Tensor range check" && 0 <= v331 && v331 < 1);
                        int v333;
                        v333 = v284 + v331;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v334 = v42[v333];
                        assert("Tensor range check" && 0 <= v331 && v331 < 1);
                        assert("Tensor range check" && 0 <= v286 && v286 < 8);
                        int v335;
                        v335 = 8 * v331;
                        int v336;
                        v336 = v335 + v286;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v337 = v185[v336];
                        wmma::mma_sync(v334, v288, v337, v334);
                        v331 += 1 ;
                    }
                    v286 += 1 ;
                }
                v284 += 1 ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0));
            v99 = v101;
        }
        // Pushing the loop unrolling to: 0
        int v338;
        v338 = 0;
        #pragma unroll
        while (while_method_3(v338)){
            int v340;
            v340 = 0;
            #pragma unroll
            while (while_method_2(v340)){
                assert("Tensor range check" && 0 <= v338 && v338 < 2);
                assert("Tensor range check" && 0 <= v340 && v340 < 1);
                int v342;
                v342 = v338 + v340;
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v343 = v42[v342];
                assert("Tensor range check" && 0 <= v338 && v338 < 2);
                assert("Tensor range check" && 0 <= v340 && v340 < 1);
                int v344;
                v344 = 16 * v340;
                int v345;
                v345 = 2176 * v338;
                int v346;
                v346 = v345 + v344;
                float * v347;
                v347 = v24+v346;
                wmma::store_matrix_sync(v347, v343, 136, wmma::mem_row_major);
                v340 += 1 ;
            }
            v338 += 1 ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0));
        // Pushing the loop unrolling to: 0
        int v349;
        v349 = threadIdx.x;
        bool v350;
        v350 = 0 <= v349;
        bool v351;
        v351 = v350 == false;
        if (v351){
            assert("The index needs to be zero or positive." && v350);
        } else {
        }
        int v353;
        v353 = v349 % 64;
        int v354;
        v354 = v349 / 64;
        bool v355;
        v355 = v354 < 8;
        bool v356;
        v356 = v355 == false;
        if (v356){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v355);
        } else {
        }
        assert("Tensor range check" && 0 <= v354 && v354 < 8);
        assert("Tensor range check" && 0 <= v353 && v353 < 64);
        int v358;
        v358 = 2 * v353;
        int v359;
        v359 = 8192 * v354;
        int v360;
        v360 = v359 + v358;
        int v361;
        v361 = 136 * v354;
        int v362;
        v362 = v361 + v358;
        float * v363;
        v363 = v57+v360;
        float * v365;
        v365 = v9+v362;
        int v367;
        v367 = 0;
        #pragma unroll
        while (while_method_1(v367)){
            int v369;
            v369 = 0;
            #pragma unroll
            while (while_method_2(v369)){
                assert("Tensor range check" && 0 <= v367 && v367 < 8);
                assert("Tensor range check" && 0 <= v369 && v369 < 1);
                int v371;
                v371 = 128 * v369;
                int v372;
                v372 = 65536 * v367;
                int v373;
                v373 = v372 + v371;
                int v374;
                v374 = 1088 * v367;
                int v375;
                v375 = v374 + v371;
                int2* v376;
                v376 = reinterpret_cast<int2*>(v365 + v375);
                int2* v377;
                v377 = reinterpret_cast<int2*>(v363 + v373);
                assert("Pointer alignment check" && (unsigned long long)(v376) % 2 == 0 && (unsigned long long)(v377) % 2 == 0);
                *v377 = *v376;
                v369 += 1 ;
            }
            v367 += 1 ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0));
        v44 += 24 ;
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
options.append('--define-macro=NDEBUG')
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=128')
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
    print(f'Threads per block, blocks per grid: {512}, {24}')
    v17((24,),(512,),(v2, v1, v0),shared_mem=98304)
    del v1, v2, v17
    v18 = cp.max(cp.abs(v0-v7))
    del v0, v7
    return v18

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
