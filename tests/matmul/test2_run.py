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
    v1 = v0 < 8192l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1, float * v2) {
    cuda::pipeline<cuda::thread_scope_thread> v3 = cuda::make_pipeline();
    extern __shared__ unsigned char v4[];
    float * v5;
    v5 = reinterpret_cast<float *>(&v4[0ull]);
    float * v7;
    v7 = reinterpret_cast<float *>(&v4[69632ull]);
    float * v9;
    v9 = reinterpret_cast<float *>(&v4[0ull]);
    int v11;
    v11 = threadIdx.x;
    int v12;
    v12 = v11 / 32l;
    bool v13;
    v13 = 0l <= v12;
    bool v14;
    v14 = v13 == false;
    if (v14){
        assert("The index needs to be zero or positive." && v13);
    } else {
    }
    int v16;
    v16 = v12 % 4l;
    int v17;
    v17 = v12 / 4l;
    bool v18;
    v18 = v17 < 4l;
    bool v19;
    v19 = v18 == false;
    if (v19){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v18);
    } else {
    }
    assert("Tensor range check" && 0 <= v17 && v17 < 4l);
    assert("Tensor range check" && 0 <= v16 && v16 < 4l);
    int v21;
    v21 = 16l * v16;
    int v22;
    v22 = 2304l * v17;
    int v23;
    v23 = v22 + v21;
    float * v24;
    v24 = v9+v23;
    assert("Tensor range check" && 0 <= v16 && v16 < 4l);
    int v26;
    v26 = 1088l * v16;
    int v27;
    v27 = threadIdx.x;
    int v28;
    v28 = v27 % 32l;
    bool v29;
    v29 = 0l <= v28;
    bool v30;
    v30 = v29 == false;
    if (v30){
        assert("The index needs to be zero or positive." && v29);
    } else {
    }
    int v32;
    v32 = v28 % 4l;
    int v33;
    v33 = v28 / 4l;
    bool v34;
    v34 = v33 < 8l;
    bool v35;
    v35 = v34 == false;
    if (v35){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v34);
    } else {
    }
    assert("Tensor range check" && 0 <= v33 && v33 < 8l);
    assert("Tensor range check" && 0 <= v32 && v32 < 4l);
    int v37;
    v37 = v32 + v26;
    int v38;
    v38 = 68l * v33;
    int v39;
    v39 = v38 + v37;
    float * v40;
    v40 = v7+v39;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v42[2l];
    int v43;
    v43 = blockIdx.x;
    int v44;
    v44 = v43;
    while (while_method_0(v44)){
        bool v46;
        v46 = 0l <= v44;
        bool v47;
        v47 = v46 == false;
        if (v47){
            assert("The index needs to be zero or positive." && v46);
        } else {
        }
        int v49;
        v49 = v44 % 128l;
        int v50;
        v50 = v44 / 128l;
        bool v51;
        v51 = v50 < 64l;
        bool v52;
        v52 = v51 == false;
        if (v52){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v51);
        } else {
        }
        assert("Tensor range check" && 0 <= v50 && v50 < 64l);
        assert("Tensor range check" && 0 <= v49 && v49 < 128l);
        int v54;
        v54 = 64l * v49;
        int v55;
        v55 = 1048576l * v50;
        int v56;
        v56 = v55 + v54;
        float * v57;
        v57 = v2+v56;
        // Pushing the loop unrolling to: 0
        int v59;
        v59 = threadIdx.x;
        bool v60;
        v60 = 0l <= v59;
        bool v61;
        v61 = v60 == false;
        if (v61){
            assert("The index needs to be zero or positive." && v60);
        } else {
        }
        int v63;
        v63 = v59 % 16l;
        int v64;
        v64 = v59 / 16l;
        bool v65;
        v65 = v64 < 32l;
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v65);
        } else {
        }
        assert("Tensor range check" && 0 <= v64 && v64 < 32l);
        assert("Tensor range check" && 0 <= v63 && v63 < 16l);
        int v68;
        v68 = 4l * v63;
        int v69;
        v69 = 72l * v64;
        int v70;
        v70 = v69 + v68;
        int v71;
        v71 = 8192l * v64;
        int v72;
        v72 = v71 + v68;
        float * v73;
        v73 = v9+v70;
        float * v75;
        v75 = v57+v72;
        int v77;
        v77 = 0l;
        #pragma unroll
        while (while_method_1(v77)){
            int v79;
            v79 = 0l;
            #pragma unroll
            while (while_method_2(v79)){
                assert("Tensor range check" && 0 <= v77 && v77 < 4l);
                assert("Tensor range check" && 0 <= v79 && v79 < 1l);
                int v81;
                v81 = 64l * v79;
                int v82;
                v82 = 2304l * v77;
                int v83;
                v83 = v82 + v81;
                int v84;
                v84 = 262144l * v77;
                int v85;
                v85 = v84 + v81;
                int4* v86;
                v86 = reinterpret_cast<int4*>(v75 + v85);
                int4* v87;
                v87 = reinterpret_cast<int4*>(v73 + v83);
                assert("Pointer alignment check" && (unsigned long long)(v86) % 4l == 0 && (unsigned long long)(v87) % 4l == 0);
                *v87 = *v86;
                v79 += 1l ;
            }
            v77 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v88;
        v88 = 0l;
        #pragma unroll
        while (while_method_3(v88)){
            int v90;
            v90 = 0l;
            #pragma unroll
            while (while_method_2(v90)){
                assert("Tensor range check" && 0 <= v88 && v88 < 2l);
                assert("Tensor range check" && 0 <= v90 && v90 < 1l);
                int v92;
                v92 = v88 + v90;
                wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v93 = v42[v92];
                assert("Tensor range check" && 0 <= v88 && v88 < 2l);
                assert("Tensor range check" && 0 <= v90 && v90 < 1l);
                int v94;
                v94 = 16l * v90;
                int v95;
                v95 = 1152l * v88;
                int v96;
                v96 = v95 + v94;
                float * v97;
                v97 = v24+v96;
                wmma::load_matrix_sync(v93, v97, 72l, wmma::mem_row_major);
                v90 += 1l ;
            }
            v88 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        // Poping the loop unrolling to: 0
        int v99;
        v99 = 0l;
        while (while_method_4(v99)){
            int v101;
            v101 = v99 + 1l;
            bool v102;
            v102 = v99 == 0l;
            int v103;
            v103 = v99 % 2l;
            bool v104;
            v104 = 0l <= v99;
            bool v105;
            v105 = v104 == false;
            if (v105){
                assert("The index needs to be zero or positive." && v104);
            } else {
            }
            bool v107;
            v107 = v99 < 64l;
            bool v108;
            v108 = v107 == false;
            if (v108){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v107);
            } else {
            }
            bool v110;
            v110 = v101 < 64l;
            Union0 v116;
            if (v110){
                bool v111;
                v111 = 0l <= v101;
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
            assert("Tensor range check" && 0 <= v50 && v50 < 64l);
            int v117;
            v117 = 524288l * v50;
            assert("Tensor range check" && 0 <= v49 && v49 < 128l);
            int v118;
            v118 = 262144l * v49;
            assert("Tensor range check" && 0 <= v99 && v99 < 64l);
            int v119;
            v119 = 64l * v99;
            int v120;
            v120 = v119 + v117;
            float * v121;
            v121 = v0+v120;
            assert("Tensor range check" && 0 <= v99 && v99 < 64l);
            int v123;
            v123 = v119 + v118;
            float * v124;
            v124 = v1+v123;
            // Pushing the loop unrolling to: 0
            v3.producer_acquire();
            int v126;
            v126 = threadIdx.x;
            bool v127;
            v127 = 0l <= v126;
            bool v128;
            v128 = v127 == false;
            if (v128){
                assert("The index needs to be zero or positive." && v127);
            } else {
            }
            int v130;
            v130 = v126 % 16l;
            int v131;
            v131 = v126 / 16l;
            bool v132;
            v132 = v131 < 32l;
            bool v133;
            v133 = v132 == false;
            if (v133){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v132);
            } else {
            }
            assert("Tensor range check" && 0 <= v131 && v131 < 32l);
            assert("Tensor range check" && 0 <= v130 && v130 < 16l);
            int v135;
            v135 = 4l * v130;
            int v136;
            v136 = 68l * v131;
            int v137;
            v137 = v136 + v135;
            int v138;
            v138 = 4096l * v131;
            int v139;
            v139 = v138 + v135;
            float * v140;
            v140 = v5+v137;
            float * v142;
            v142 = v121+v139;
            int v144;
            v144 = 0l;
            #pragma unroll
            while (while_method_1(v144)){
                int v146;
                v146 = 0l;
                #pragma unroll
                while (while_method_2(v146)){
                    assert("Tensor range check" && 0 <= v144 && v144 < 4l);
                    assert("Tensor range check" && 0 <= v146 && v146 < 1l);
                    int v148;
                    v148 = 64l * v146;
                    int v149;
                    v149 = 2176l * v144;
                    int v150;
                    v150 = v149 + v148;
                    int v151;
                    v151 = 131072l * v144;
                    int v152;
                    v152 = v151 + v148;
                    constexpr int v153 = sizeof(float) * 4l;
                    assert("Pointer alignment check" && (unsigned long long)(v142 + v152) % v153 == 0 && (unsigned long long)(v140 + v150) % v153 == 0);
                    cuda::memcpy_async(v140 + v150, v142 + v152, cuda::aligned_size_t<v153>(v153), v3);
                    v146 += 1l ;
                }
                v144 += 1l ;
            }
            int v154;
            v154 = threadIdx.x;
            bool v155;
            v155 = 0l <= v154;
            bool v156;
            v156 = v155 == false;
            if (v156){
                assert("The index needs to be zero or positive." && v155);
            } else {
            }
            int v158;
            v158 = v154 % 16l;
            int v159;
            v159 = v154 / 16l;
            bool v160;
            v160 = v159 < 32l;
            bool v161;
            v161 = v160 == false;
            if (v161){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v160);
            } else {
            }
            assert("Tensor range check" && 0 <= v159 && v159 < 32l);
            assert("Tensor range check" && 0 <= v158 && v158 < 16l);
            int v163;
            v163 = 4l * v158;
            int v164;
            v164 = 68l * v159;
            int v165;
            v165 = v164 + v163;
            int v166;
            v166 = 4096l * v159;
            int v167;
            v167 = v166 + v163;
            float * v168;
            v168 = v7+v165;
            float * v170;
            v170 = v124+v167;
            int v172;
            v172 = 0l;
            #pragma unroll
            while (while_method_3(v172)){
                int v174;
                v174 = 0l;
                #pragma unroll
                while (while_method_2(v174)){
                    assert("Tensor range check" && 0 <= v172 && v172 < 2l);
                    assert("Tensor range check" && 0 <= v174 && v174 < 1l);
                    int v176;
                    v176 = 64l * v174;
                    int v177;
                    v177 = 2176l * v172;
                    int v178;
                    v178 = v177 + v176;
                    int v179;
                    v179 = 131072l * v172;
                    int v180;
                    v180 = v179 + v176;
                    constexpr int v181 = sizeof(float) * 4l;
                    assert("Pointer alignment check" && (unsigned long long)(v170 + v180) % v181 == 0 && (unsigned long long)(v168 + v178) % v181 == 0);
                    cuda::memcpy_async(v168 + v178, v170 + v180, cuda::aligned_size_t<v181>(v181), v3);
                    v174 += 1l ;
                }
                v172 += 1l ;
            }
            v3.producer_commit();
            // Poping the loop unrolling to: 0
            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v182[1l];
            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v183[8l];
            cuda::pipeline_consumer_wait_prior<0>(v3);;
            asm("barrier.cta.sync %0;" :: "r"(0l));
            // Pushing the loop unrolling to: 0
            int v184;
            v184 = 0l;
            #pragma unroll
            while (while_method_2(v184)){
                int v186;
                v186 = 0l;
                #pragma unroll
                while (while_method_5(v186)){
                    assert("Tensor range check" && 0 <= v184 && v184 < 1l);
                    assert("Tensor range check" && 0 <= v186 && v186 < 8l);
                    int v188;
                    v188 = 8l * v184;
                    int v189;
                    v189 = v188 + v186;
                    wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v190 = v183[v189];
                    assert("Tensor range check" && 0 <= v184 && v184 < 1l);
                    int v191;
                    v191 = 1088l * v184;
                    assert("Tensor range check" && 0 <= v186 && v186 < 8l);
                    int v192;
                    v192 = 8l * v186;
                    int v193;
                    v193 = v192 + v191;
                    int v194;
                    v194 = 0l;
                    #pragma unroll
                    while (while_method_3(v194)){
                        int v196;
                        v196 = 0l;
                        #pragma unroll
                        while (while_method_3(v196)){
                            assert("Tensor range check" && 0 <= v194 && v194 < 2l);
                            assert("Tensor range check" && 0 <= v196 && v196 < 2l);
                            int v198;
                            v198 = 4l * v196;
                            int v199;
                            v199 = v198 + v193;
                            int v200;
                            v200 = 544l * v194;
                            int v201;
                            v201 = v200 + v199;
                            float v202;
                            v202 = v40[v201];
                            bool v203;
                            v203 = 0l <= v196;
                            bool v205;
                            if (v203){
                                bool v204;
                                v204 = v196 < 2l;
                                v205 = v204;
                            } else {
                                v205 = false;
                            }
                            bool v206;
                            v206 = v205 == false;
                            if (v206){
                                assert("The indices should be inside the range of the dimension." && v205);
                            } else {
                            }
                            bool v208;
                            v208 = 0l <= v194;
                            bool v210;
                            if (v208){
                                bool v209;
                                v209 = v194 < 2l;
                                v210 = v209;
                            } else {
                                v210 = false;
                            }
                            bool v211;
                            v211 = v210 == false;
                            if (v211){
                                assert("The indices should be inside the range of the dimension." && v210);
                            } else {
                            }
                            int v213;
                            v213 = v194 * 2l;
                            int v214;
                            v214 = v196 + v213;
                            v190.x[v214] = v202;
                            v196 += 1l ;
                        }
                        v194 += 1l ;
                    }
                    v186 += 1l ;
                }
                v184 += 1l ;
            }
            // Poping the loop unrolling to: 0
            // Pushing the loop unrolling to: 0
            int v215;
            v215 = 0l;
            #pragma unroll
            while (while_method_3(v215)){
                int v217;
                v217 = 0l;
                #pragma unroll
                while (while_method_5(v217)){
                    wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v219 = v182[0l];
                    assert("Tensor range check" && 0 <= v17 && v17 < 4l);
                    int v220;
                    v220 = 2176l * v17;
                    int v221;
                    v221 = threadIdx.x;
                    int v222;
                    v222 = v221 % 32l;
                    bool v223;
                    v223 = 0l <= v222;
                    bool v224;
                    v224 = v223 == false;
                    if (v224){
                        assert("The index needs to be zero or positive." && v223);
                    } else {
                    }
                    int v226;
                    v226 = v222 % 4l;
                    int v227;
                    v227 = v222 / 4l;
                    bool v228;
                    v228 = v227 < 8l;
                    bool v229;
                    v229 = v228 == false;
                    if (v229){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v228);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v227 && v227 < 8l);
                    assert("Tensor range check" && 0 <= v226 && v226 < 4l);
                    int v231;
                    v231 = v226 + v220;
                    int v232;
                    v232 = 68l * v227;
                    int v233;
                    v233 = v232 + v231;
                    float * v234;
                    v234 = v5+v233;
                    assert("Tensor range check" && 0 <= v215 && v215 < 2l);
                    int v236;
                    v236 = 1088l * v215;
                    assert("Tensor range check" && 0 <= v217 && v217 < 8l);
                    int v237;
                    v237 = 8l * v217;
                    int v238;
                    v238 = v237 + v236;
                    int v239;
                    v239 = 0l;
                    #pragma unroll
                    while (while_method_3(v239)){
                        int v241;
                        v241 = 0l;
                        #pragma unroll
                        while (while_method_3(v241)){
                            assert("Tensor range check" && 0 <= v239 && v239 < 2l);
                            assert("Tensor range check" && 0 <= v241 && v241 < 2l);
                            int v243;
                            v243 = 544l * v241;
                            int v244;
                            v244 = v243 + v238;
                            int v245;
                            v245 = 4l * v239;
                            int v246;
                            v246 = v245 + v244;
                            float v247;
                            v247 = v234[v246];
                            bool v248;
                            v248 = 0l <= v241;
                            bool v250;
                            if (v248){
                                bool v249;
                                v249 = v241 < 2l;
                                v250 = v249;
                            } else {
                                v250 = false;
                            }
                            bool v251;
                            v251 = v250 == false;
                            if (v251){
                                assert("The indices should be inside the range of the dimension." && v250);
                            } else {
                            }
                            bool v253;
                            v253 = 0l <= v239;
                            bool v255;
                            if (v253){
                                bool v254;
                                v254 = v239 < 2l;
                                v255 = v254;
                            } else {
                                v255 = false;
                            }
                            bool v256;
                            v256 = v255 == false;
                            if (v256){
                                assert("The indices should be inside the range of the dimension." && v255);
                            } else {
                            }
                            int v258;
                            v258 = v239 * 2l;
                            int v259;
                            v259 = v241 + v258;
                            v219.x[v259] = v247;
                            v241 += 1l ;
                        }
                        v239 += 1l ;
                    }
                    int v260;
                    v260 = 0l;
                    #pragma unroll
                    while (while_method_2(v260)){
                        assert("Tensor range check" && 0 <= v215 && v215 < 2l);
                        assert("Tensor range check" && 0 <= v260 && v260 < 1l);
                        int v262;
                        v262 = v215 + v260;
                        wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v263 = v42[v262];
                        assert("Tensor range check" && 0 <= v260 && v260 < 1l);
                        assert("Tensor range check" && 0 <= v217 && v217 < 8l);
                        int v264;
                        v264 = 8l * v260;
                        int v265;
                        v265 = v264 + v217;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v266 = v183[v265];
                        wmma::mma_sync(v263, v219, v266, v263);
                        v260 += 1l ;
                    }
                    v217 += 1l ;
                }
                v215 += 1l ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0l));
            v3.consumer_release();
            v99 = v101;
        }
        // Pushing the loop unrolling to: 0
        int v267;
        v267 = 0l;
        #pragma unroll
        while (while_method_3(v267)){
            int v269;
            v269 = 0l;
            #pragma unroll
            while (while_method_2(v269)){
                assert("Tensor range check" && 0 <= v267 && v267 < 2l);
                assert("Tensor range check" && 0 <= v269 && v269 < 1l);
                int v271;
                v271 = v267 + v269;
                wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v272 = v42[v271];
                assert("Tensor range check" && 0 <= v267 && v267 < 2l);
                assert("Tensor range check" && 0 <= v269 && v269 < 1l);
                int v273;
                v273 = 16l * v269;
                int v274;
                v274 = 1152l * v267;
                int v275;
                v275 = v274 + v273;
                float * v276;
                v276 = v24+v275;
                wmma::store_matrix_sync(v276, v272, 72l, wmma::mem_row_major);
                v269 += 1l ;
            }
            v267 += 1l ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0l));
        // Pushing the loop unrolling to: 0
        int v278;
        v278 = threadIdx.x;
        bool v279;
        v279 = 0l <= v278;
        bool v280;
        v280 = v279 == false;
        if (v280){
            assert("The index needs to be zero or positive." && v279);
        } else {
        }
        int v282;
        v282 = v278 % 16l;
        int v283;
        v283 = v278 / 16l;
        bool v284;
        v284 = v283 < 32l;
        bool v285;
        v285 = v284 == false;
        if (v285){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v284);
        } else {
        }
        assert("Tensor range check" && 0 <= v283 && v283 < 32l);
        assert("Tensor range check" && 0 <= v282 && v282 < 16l);
        int v287;
        v287 = 4l * v282;
        int v288;
        v288 = 8192l * v283;
        int v289;
        v289 = v288 + v287;
        int v290;
        v290 = 72l * v283;
        int v291;
        v291 = v290 + v287;
        float * v292;
        v292 = v57+v289;
        float * v294;
        v294 = v9+v291;
        int v296;
        v296 = 0l;
        #pragma unroll
        while (while_method_1(v296)){
            int v298;
            v298 = 0l;
            #pragma unroll
            while (while_method_2(v298)){
                assert("Tensor range check" && 0 <= v296 && v296 < 4l);
                assert("Tensor range check" && 0 <= v298 && v298 < 1l);
                int v300;
                v300 = 64l * v298;
                int v301;
                v301 = 262144l * v296;
                int v302;
                v302 = v301 + v300;
                int v303;
                v303 = 2304l * v296;
                int v304;
                v304 = v303 + v300;
                int4* v305;
                v305 = reinterpret_cast<int4*>(v294 + v304);
                int4* v306;
                v306 = reinterpret_cast<int4*>(v292 + v302);
                assert("Pointer alignment check" && (unsigned long long)(v305) % 4l == 0 && (unsigned long long)(v306) % 4l == 0);
                *v306 = *v305;
                v298 += 1l ;
            }
            v296 += 1l ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0l));
        v44 += 24l ;
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
    v17.max_dynamic_shared_size_bytes = 87040 
    print(f'Threads per block, blocks per grid: {512}, {24}')
    v17((24,),(512,),(v2, v1, v0),shared_mem=87040)
    del v1, v2, v17
    v18 = cp.max(cp.abs(v0-v7))
    del v0, v7
    return v18

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
