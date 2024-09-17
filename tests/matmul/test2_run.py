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
    v1 = v0 < 2048l;
    return v1;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 1024l;
    return v1;
}
__device__ inline bool while_method_7(int v0){
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
            if (v102){
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
                assert("Tensor range check" && 0 <= v103 && v103 < 2l);
                int v126;
                v126 = 8704l * v103;
                int v127;
                v127 = threadIdx.x;
                bool v128;
                v128 = 0l <= v127;
                bool v129;
                v129 = v128 == false;
                if (v129){
                    assert("The index needs to be zero or positive." && v128);
                } else {
                }
                int v131;
                v131 = v127 % 16l;
                int v132;
                v132 = v127 / 16l;
                bool v133;
                v133 = v132 < 32l;
                bool v134;
                v134 = v133 == false;
                if (v134){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v133);
                } else {
                }
                assert("Tensor range check" && 0 <= v132 && v132 < 32l);
                assert("Tensor range check" && 0 <= v131 && v131 < 16l);
                int v136;
                v136 = 4l * v131;
                int v137;
                v137 = v136 + v126;
                int v138;
                v138 = 68l * v132;
                int v139;
                v139 = v138 + v137;
                int v140;
                v140 = 4096l * v132;
                int v141;
                v141 = v140 + v136;
                float * v142;
                v142 = v5+v139;
                float * v144;
                v144 = v121+v141;
                int v146;
                v146 = 0l;
                #pragma unroll
                while (while_method_1(v146)){
                    int v148;
                    v148 = 0l;
                    #pragma unroll
                    while (while_method_2(v148)){
                        assert("Tensor range check" && 0 <= v146 && v146 < 4l);
                        assert("Tensor range check" && 0 <= v148 && v148 < 1l);
                        int v150;
                        v150 = 64l * v148;
                        int v151;
                        v151 = 2176l * v146;
                        int v152;
                        v152 = v151 + v150;
                        int v153;
                        v153 = 131072l * v146;
                        int v154;
                        v154 = v153 + v150;
                        constexpr int v155 = sizeof(float) * 4l;
                        assert("Pointer alignment check" && (unsigned long long)(v144 + v154) % v155 == 0 && (unsigned long long)(v142 + v152) % v155 == 0);
                        cuda::memcpy_async(v142 + v152, v144 + v154, cuda::aligned_size_t<v155>(v155), v3);
                        v148 += 1l ;
                    }
                    v146 += 1l ;
                }
                int v156;
                v156 = threadIdx.x;
                bool v157;
                v157 = 0l <= v156;
                bool v158;
                v158 = v157 == false;
                if (v158){
                    assert("The index needs to be zero or positive." && v157);
                } else {
                }
                int v160;
                v160 = v156 % 16l;
                int v161;
                v161 = v156 / 16l;
                bool v162;
                v162 = v161 < 32l;
                bool v163;
                v163 = v162 == false;
                if (v163){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v162);
                } else {
                }
                assert("Tensor range check" && 0 <= v161 && v161 < 32l);
                assert("Tensor range check" && 0 <= v160 && v160 < 16l);
                int v165;
                v165 = 4l * v160;
                int v166;
                v166 = 68l * v161;
                int v167;
                v167 = v166 + v165;
                int v168;
                v168 = 4096l * v161;
                int v169;
                v169 = v168 + v165;
                float * v170;
                v170 = v7+v167;
                float * v172;
                v172 = v124+v169;
                int v174;
                v174 = 0l;
                #pragma unroll
                while (while_method_3(v174)){
                    int v176;
                    v176 = 0l;
                    #pragma unroll
                    while (while_method_2(v176)){
                        assert("Tensor range check" && 0 <= v174 && v174 < 2l);
                        assert("Tensor range check" && 0 <= v176 && v176 < 1l);
                        int v178;
                        v178 = 64l * v176;
                        int v179;
                        v179 = 2176l * v174;
                        int v180;
                        v180 = v179 + v178;
                        int v181;
                        v181 = 131072l * v174;
                        int v182;
                        v182 = v181 + v178;
                        constexpr int v183 = sizeof(float) * 4l;
                        assert("Pointer alignment check" && (unsigned long long)(v172 + v182) % v183 == 0 && (unsigned long long)(v170 + v180) % v183 == 0);
                        cuda::memcpy_async(v170 + v180, v172 + v182, cuda::aligned_size_t<v183>(v183), v3);
                        v176 += 1l ;
                    }
                    v174 += 1l ;
                }
                v3.producer_commit();
                // Poping the loop unrolling to: 0
            } else {
            }
            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v184[1l];
            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v185[8l];
            cuda::pipeline_consumer_wait_prior<0>(v3);;
            // Pushing the loop unrolling to: 0
            assert("Tensor range check" && 0 <= v103 && v103 < 2l);
            int v186;
            v186 = 8704l * v103;
            int v187;
            v187 = threadIdx.x;
            int v188;
            v188 = v187;
            #pragma unroll
            while (while_method_5(v188)){
                bool v190;
                v190 = 0l <= v188;
                bool v191;
                v191 = v190 == false;
                if (v191){
                    assert("The index needs to be zero or positive." && v190);
                } else {
                }
                int v193;
                v193 = v188 % 16l;
                int v194;
                v194 = v188 / 16l;
                bool v195;
                v195 = v194 < 128l;
                bool v196;
                v196 = v195 == false;
                if (v196){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v195);
                } else {
                }
                assert("Tensor range check" && 0 <= v194 && v194 < 128l);
                assert("Tensor range check" && 0 <= v193 && v193 < 16l);
                int v198;
                v198 = 4l * v193;
                int v199;
                v199 = v198 + v186;
                int v200;
                v200 = 68l * v194;
                int v201;
                v201 = v200 + v199;
                float v202[4l];
                int4* v203;
                v203 = reinterpret_cast<int4*>(v5 + v201);
                int4* v204;
                v204 = reinterpret_cast<int4*>(v202 + 0l);
                assert("Pointer alignment check" && (unsigned long long)(v203) % 4l == 0 && (unsigned long long)(v204) % 4l == 0);
                *v204 = *v203;
                // Pushing the loop unrolling to: 0
                int v205;
                v205 = 0l;
                #pragma unroll
                while (while_method_1(v205)){
                    assert("Tensor range check" && 0 <= v205 && v205 < 4l);
                    float v207;
                    v207 = v202[v205];
                    float v208;
                    v208 = wmma::__float_to_tf32(v207);
                    assert("Tensor range check" && 0 <= v205 && v205 < 4l);
                    v202[v205] = v208;
                    v205 += 1l ;
                }
                // Poping the loop unrolling to: 0
                int4* v209;
                v209 = reinterpret_cast<int4*>(v202 + 0l);
                int4* v210;
                v210 = reinterpret_cast<int4*>(v5 + v201);
                assert("Pointer alignment check" && (unsigned long long)(v209) % 4l == 0 && (unsigned long long)(v210) % 4l == 0);
                *v210 = *v209;
                v188 += 512l ;
            }
            int v211;
            v211 = threadIdx.x;
            int v212;
            v212 = v211;
            #pragma unroll
            while (while_method_6(v212)){
                bool v214;
                v214 = 0l <= v212;
                bool v215;
                v215 = v214 == false;
                if (v215){
                    assert("The index needs to be zero or positive." && v214);
                } else {
                }
                int v217;
                v217 = v212 % 16l;
                int v218;
                v218 = v212 / 16l;
                bool v219;
                v219 = v218 < 64l;
                bool v220;
                v220 = v219 == false;
                if (v220){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v219);
                } else {
                }
                assert("Tensor range check" && 0 <= v218 && v218 < 64l);
                assert("Tensor range check" && 0 <= v217 && v217 < 16l);
                int v222;
                v222 = 4l * v217;
                int v223;
                v223 = 68l * v218;
                int v224;
                v224 = v223 + v222;
                float v225[4l];
                int4* v226;
                v226 = reinterpret_cast<int4*>(v7 + v224);
                int4* v227;
                v227 = reinterpret_cast<int4*>(v225 + 0l);
                assert("Pointer alignment check" && (unsigned long long)(v226) % 4l == 0 && (unsigned long long)(v227) % 4l == 0);
                *v227 = *v226;
                // Pushing the loop unrolling to: 0
                int v228;
                v228 = 0l;
                #pragma unroll
                while (while_method_1(v228)){
                    assert("Tensor range check" && 0 <= v228 && v228 < 4l);
                    float v230;
                    v230 = v225[v228];
                    float v231;
                    v231 = wmma::__float_to_tf32(v230);
                    assert("Tensor range check" && 0 <= v228 && v228 < 4l);
                    v225[v228] = v231;
                    v228 += 1l ;
                }
                // Poping the loop unrolling to: 0
                int4* v232;
                v232 = reinterpret_cast<int4*>(v225 + 0l);
                int4* v233;
                v233 = reinterpret_cast<int4*>(v7 + v224);
                assert("Pointer alignment check" && (unsigned long long)(v232) % 4l == 0 && (unsigned long long)(v233) % 4l == 0);
                *v233 = *v232;
                v212 += 512l ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0l));
            // Pushing the loop unrolling to: 0
            int v234;
            v234 = 0l;
            #pragma unroll
            while (while_method_2(v234)){
                int v236;
                v236 = 0l;
                #pragma unroll
                while (while_method_7(v236)){
                    assert("Tensor range check" && 0 <= v234 && v234 < 1l);
                    assert("Tensor range check" && 0 <= v236 && v236 < 8l);
                    int v238;
                    v238 = 8l * v234;
                    int v239;
                    v239 = v238 + v236;
                    wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v240 = v185[v239];
                    assert("Tensor range check" && 0 <= v234 && v234 < 1l);
                    int v241;
                    v241 = 1088l * v234;
                    assert("Tensor range check" && 0 <= v236 && v236 < 8l);
                    int v242;
                    v242 = 8l * v236;
                    int v243;
                    v243 = v242 + v241;
                    int v244;
                    v244 = 0l;
                    #pragma unroll
                    while (while_method_3(v244)){
                        int v246;
                        v246 = 0l;
                        #pragma unroll
                        while (while_method_3(v246)){
                            assert("Tensor range check" && 0 <= v244 && v244 < 2l);
                            assert("Tensor range check" && 0 <= v246 && v246 < 2l);
                            int v248;
                            v248 = 4l * v246;
                            int v249;
                            v249 = v248 + v243;
                            int v250;
                            v250 = 544l * v244;
                            int v251;
                            v251 = v250 + v249;
                            float v252;
                            v252 = v40[v251];
                            bool v253;
                            v253 = 0l <= v246;
                            bool v255;
                            if (v253){
                                bool v254;
                                v254 = v246 < 2l;
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
                            bool v258;
                            v258 = 0l <= v244;
                            bool v260;
                            if (v258){
                                bool v259;
                                v259 = v244 < 2l;
                                v260 = v259;
                            } else {
                                v260 = false;
                            }
                            bool v261;
                            v261 = v260 == false;
                            if (v261){
                                assert("The indices should be inside the range of the dimension." && v260);
                            } else {
                            }
                            int v263;
                            v263 = v244 * 2l;
                            int v264;
                            v264 = v246 + v263;
                            v240.x[v264] = v252;
                            v246 += 1l ;
                        }
                        v244 += 1l ;
                    }
                    v236 += 1l ;
                }
                v234 += 1l ;
            }
            // Poping the loop unrolling to: 0
            v3.consumer_release();
            switch (v116.tag) {
                case 0: { // None
                    break;
                }
                case 1: { // Some
                    int v265 = v116.case1.v0;
                    assert("Tensor range check" && 0 <= v265 && v265 < 64l);
                    int v266;
                    v266 = 64l * v265;
                    int v267;
                    v267 = v266 + v117;
                    float * v268;
                    v268 = v0+v267;
                    assert("Tensor range check" && 0 <= v265 && v265 < 64l);
                    int v270;
                    v270 = v266 + v118;
                    float * v271;
                    v271 = v1+v270;
                    asm("barrier.cta.sync %0;" :: "r"(0l));
                    // Pushing the loop unrolling to: 0
                    v3.producer_acquire();
                    int v273;
                    v273 = v103 ^ 1l;
                    assert("Tensor range check" && 0 <= v273 && v273 < 2l);
                    int v274;
                    v274 = 8704l * v273;
                    int v275;
                    v275 = threadIdx.x;
                    bool v276;
                    v276 = 0l <= v275;
                    bool v277;
                    v277 = v276 == false;
                    if (v277){
                        assert("The index needs to be zero or positive." && v276);
                    } else {
                    }
                    int v279;
                    v279 = v275 % 16l;
                    int v280;
                    v280 = v275 / 16l;
                    bool v281;
                    v281 = v280 < 32l;
                    bool v282;
                    v282 = v281 == false;
                    if (v282){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v281);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v280 && v280 < 32l);
                    assert("Tensor range check" && 0 <= v279 && v279 < 16l);
                    int v284;
                    v284 = 4l * v279;
                    int v285;
                    v285 = v284 + v274;
                    int v286;
                    v286 = 68l * v280;
                    int v287;
                    v287 = v286 + v285;
                    int v288;
                    v288 = 4096l * v280;
                    int v289;
                    v289 = v288 + v284;
                    float * v290;
                    v290 = v5+v287;
                    float * v292;
                    v292 = v268+v289;
                    int v294;
                    v294 = 0l;
                    #pragma unroll
                    while (while_method_1(v294)){
                        int v296;
                        v296 = 0l;
                        #pragma unroll
                        while (while_method_2(v296)){
                            assert("Tensor range check" && 0 <= v294 && v294 < 4l);
                            assert("Tensor range check" && 0 <= v296 && v296 < 1l);
                            int v298;
                            v298 = 64l * v296;
                            int v299;
                            v299 = 2176l * v294;
                            int v300;
                            v300 = v299 + v298;
                            int v301;
                            v301 = 131072l * v294;
                            int v302;
                            v302 = v301 + v298;
                            constexpr int v303 = sizeof(float) * 4l;
                            assert("Pointer alignment check" && (unsigned long long)(v292 + v302) % v303 == 0 && (unsigned long long)(v290 + v300) % v303 == 0);
                            cuda::memcpy_async(v290 + v300, v292 + v302, cuda::aligned_size_t<v303>(v303), v3);
                            v296 += 1l ;
                        }
                        v294 += 1l ;
                    }
                    int v304;
                    v304 = threadIdx.x;
                    bool v305;
                    v305 = 0l <= v304;
                    bool v306;
                    v306 = v305 == false;
                    if (v306){
                        assert("The index needs to be zero or positive." && v305);
                    } else {
                    }
                    int v308;
                    v308 = v304 % 16l;
                    int v309;
                    v309 = v304 / 16l;
                    bool v310;
                    v310 = v309 < 32l;
                    bool v311;
                    v311 = v310 == false;
                    if (v311){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v310);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v309 && v309 < 32l);
                    assert("Tensor range check" && 0 <= v308 && v308 < 16l);
                    int v313;
                    v313 = 4l * v308;
                    int v314;
                    v314 = 68l * v309;
                    int v315;
                    v315 = v314 + v313;
                    int v316;
                    v316 = 4096l * v309;
                    int v317;
                    v317 = v316 + v313;
                    float * v318;
                    v318 = v7+v315;
                    float * v320;
                    v320 = v271+v317;
                    int v322;
                    v322 = 0l;
                    #pragma unroll
                    while (while_method_3(v322)){
                        int v324;
                        v324 = 0l;
                        #pragma unroll
                        while (while_method_2(v324)){
                            assert("Tensor range check" && 0 <= v322 && v322 < 2l);
                            assert("Tensor range check" && 0 <= v324 && v324 < 1l);
                            int v326;
                            v326 = 64l * v324;
                            int v327;
                            v327 = 2176l * v322;
                            int v328;
                            v328 = v327 + v326;
                            int v329;
                            v329 = 131072l * v322;
                            int v330;
                            v330 = v329 + v326;
                            constexpr int v331 = sizeof(float) * 4l;
                            assert("Pointer alignment check" && (unsigned long long)(v320 + v330) % v331 == 0 && (unsigned long long)(v318 + v328) % v331 == 0);
                            cuda::memcpy_async(v318 + v328, v320 + v330, cuda::aligned_size_t<v331>(v331), v3);
                            v324 += 1l ;
                        }
                        v322 += 1l ;
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
            int v332;
            v332 = 0l;
            #pragma unroll
            while (while_method_3(v332)){
                int v334;
                v334 = 0l;
                #pragma unroll
                while (while_method_7(v334)){
                    wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v336 = v184[0l];
                    assert("Tensor range check" && 0 <= v103 && v103 < 2l);
                    assert("Tensor range check" && 0 <= v17 && v17 < 4l);
                    int v337;
                    v337 = 2176l * v17;
                    int v338;
                    v338 = v337 + v186;
                    int v339;
                    v339 = threadIdx.x;
                    int v340;
                    v340 = v339 % 32l;
                    bool v341;
                    v341 = 0l <= v340;
                    bool v342;
                    v342 = v341 == false;
                    if (v342){
                        assert("The index needs to be zero or positive." && v341);
                    } else {
                    }
                    int v344;
                    v344 = v340 % 4l;
                    int v345;
                    v345 = v340 / 4l;
                    bool v346;
                    v346 = v345 < 8l;
                    bool v347;
                    v347 = v346 == false;
                    if (v347){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v346);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v345 && v345 < 8l);
                    assert("Tensor range check" && 0 <= v344 && v344 < 4l);
                    int v349;
                    v349 = v344 + v338;
                    int v350;
                    v350 = 68l * v345;
                    int v351;
                    v351 = v350 + v349;
                    float * v352;
                    v352 = v5+v351;
                    assert("Tensor range check" && 0 <= v332 && v332 < 2l);
                    int v354;
                    v354 = 1088l * v332;
                    assert("Tensor range check" && 0 <= v334 && v334 < 8l);
                    int v355;
                    v355 = 8l * v334;
                    int v356;
                    v356 = v355 + v354;
                    int v357;
                    v357 = 0l;
                    #pragma unroll
                    while (while_method_3(v357)){
                        int v359;
                        v359 = 0l;
                        #pragma unroll
                        while (while_method_3(v359)){
                            assert("Tensor range check" && 0 <= v357 && v357 < 2l);
                            assert("Tensor range check" && 0 <= v359 && v359 < 2l);
                            int v361;
                            v361 = 544l * v359;
                            int v362;
                            v362 = v361 + v356;
                            int v363;
                            v363 = 4l * v357;
                            int v364;
                            v364 = v363 + v362;
                            float v365;
                            v365 = v352[v364];
                            bool v366;
                            v366 = 0l <= v359;
                            bool v368;
                            if (v366){
                                bool v367;
                                v367 = v359 < 2l;
                                v368 = v367;
                            } else {
                                v368 = false;
                            }
                            bool v369;
                            v369 = v368 == false;
                            if (v369){
                                assert("The indices should be inside the range of the dimension." && v368);
                            } else {
                            }
                            bool v371;
                            v371 = 0l <= v357;
                            bool v373;
                            if (v371){
                                bool v372;
                                v372 = v357 < 2l;
                                v373 = v372;
                            } else {
                                v373 = false;
                            }
                            bool v374;
                            v374 = v373 == false;
                            if (v374){
                                assert("The indices should be inside the range of the dimension." && v373);
                            } else {
                            }
                            int v376;
                            v376 = v357 * 2l;
                            int v377;
                            v377 = v359 + v376;
                            v336.x[v377] = v365;
                            v359 += 1l ;
                        }
                        v357 += 1l ;
                    }
                    int v378;
                    v378 = 0l;
                    #pragma unroll
                    while (while_method_2(v378)){
                        assert("Tensor range check" && 0 <= v332 && v332 < 2l);
                        assert("Tensor range check" && 0 <= v378 && v378 < 1l);
                        int v380;
                        v380 = v332 + v378;
                        wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v381 = v42[v380];
                        assert("Tensor range check" && 0 <= v378 && v378 < 1l);
                        assert("Tensor range check" && 0 <= v334 && v334 < 8l);
                        int v382;
                        v382 = 8l * v378;
                        int v383;
                        v383 = v382 + v334;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v384 = v185[v383];
                        wmma::mma_sync(v381, v336, v384, v381);
                        v378 += 1l ;
                    }
                    v334 += 1l ;
                }
                v332 += 1l ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0l));
            v99 = v101;
        }
        // Pushing the loop unrolling to: 0
        int v385;
        v385 = 0l;
        #pragma unroll
        while (while_method_3(v385)){
            int v387;
            v387 = 0l;
            #pragma unroll
            while (while_method_2(v387)){
                assert("Tensor range check" && 0 <= v385 && v385 < 2l);
                assert("Tensor range check" && 0 <= v387 && v387 < 1l);
                int v389;
                v389 = v385 + v387;
                wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v390 = v42[v389];
                assert("Tensor range check" && 0 <= v385 && v385 < 2l);
                assert("Tensor range check" && 0 <= v387 && v387 < 1l);
                int v391;
                v391 = 16l * v387;
                int v392;
                v392 = 1152l * v385;
                int v393;
                v393 = v392 + v391;
                float * v394;
                v394 = v24+v393;
                wmma::store_matrix_sync(v394, v390, 72l, wmma::mem_row_major);
                v387 += 1l ;
            }
            v385 += 1l ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0l));
        // Pushing the loop unrolling to: 0
        int v396;
        v396 = threadIdx.x;
        bool v397;
        v397 = 0l <= v396;
        bool v398;
        v398 = v397 == false;
        if (v398){
            assert("The index needs to be zero or positive." && v397);
        } else {
        }
        int v400;
        v400 = v396 % 16l;
        int v401;
        v401 = v396 / 16l;
        bool v402;
        v402 = v401 < 32l;
        bool v403;
        v403 = v402 == false;
        if (v403){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v402);
        } else {
        }
        assert("Tensor range check" && 0 <= v401 && v401 < 32l);
        assert("Tensor range check" && 0 <= v400 && v400 < 16l);
        int v405;
        v405 = 4l * v400;
        int v406;
        v406 = 8192l * v401;
        int v407;
        v407 = v406 + v405;
        int v408;
        v408 = 72l * v401;
        int v409;
        v409 = v408 + v405;
        float * v410;
        v410 = v57+v407;
        float * v412;
        v412 = v9+v409;
        int v414;
        v414 = 0l;
        #pragma unroll
        while (while_method_1(v414)){
            int v416;
            v416 = 0l;
            #pragma unroll
            while (while_method_2(v416)){
                assert("Tensor range check" && 0 <= v414 && v414 < 4l);
                assert("Tensor range check" && 0 <= v416 && v416 < 1l);
                int v418;
                v418 = 64l * v416;
                int v419;
                v419 = 262144l * v414;
                int v420;
                v420 = v419 + v418;
                int v421;
                v421 = 2304l * v414;
                int v422;
                v422 = v421 + v418;
                int4* v423;
                v423 = reinterpret_cast<int4*>(v412 + v422);
                int4* v424;
                v424 = reinterpret_cast<int4*>(v410 + v420);
                assert("Pointer alignment check" && (unsigned long long)(v423) % 4l == 0 && (unsigned long long)(v424) % 4l == 0);
                *v424 = *v423;
                v416 += 1l ;
            }
            v414 += 1l ;
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
