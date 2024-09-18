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
    v1 = v0 < 4096;
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
    v1 = v0 < 4;
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
    v22 = 8704 * v17;
    int v23;
    v23 = v22 + v21;
    float * v24;
    v24 = v9+v23;
    assert("Tensor range check" && 0 <= v17 && v17 < 2);
    int v26;
    v26 = 4352 * v17;
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
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v58[4];
    int v59;
    v59 = blockIdx.x;
    int v60;
    v60 = v59;
    while (while_method_0(v60)){
        bool v62;
        v62 = 0 <= v60;
        bool v63;
        v63 = v62 == false;
        if (v63){
            assert("The index needs to be zero or positive." && v62);
        } else {
        }
        int v65;
        v65 = v60 % 64;
        int v66;
        v66 = v60 / 64;
        bool v67;
        v67 = v66 < 64;
        bool v68;
        v68 = v67 == false;
        if (v68){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v67);
        } else {
        }
        assert("Tensor range check" && 0 <= v66 && v66 < 64);
        assert("Tensor range check" && 0 <= v65 && v65 < 64);
        int v70;
        v70 = 128 * v65;
        int v71;
        v71 = 1048576 * v66;
        int v72;
        v72 = v71 + v70;
        float * v73;
        v73 = v2+v72;
        // Pushing the loop unrolling to: 0
        int v75;
        v75 = threadIdx.x;
        bool v76;
        v76 = 0 <= v75;
        bool v77;
        v77 = v76 == false;
        if (v77){
            assert("The index needs to be zero or positive." && v76);
        } else {
        }
        int v79;
        v79 = v75 % 32;
        int v80;
        v80 = v75 / 32;
        bool v81;
        v81 = v80 < 16;
        bool v82;
        v82 = v81 == false;
        if (v82){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v81);
        } else {
        }
        assert("Tensor range check" && 0 <= v80 && v80 < 16);
        assert("Tensor range check" && 0 <= v79 && v79 < 32);
        int v84;
        v84 = 4 * v79;
        int v85;
        v85 = 136 * v80;
        int v86;
        v86 = v85 + v84;
        int v87;
        v87 = 8192 * v80;
        int v88;
        v88 = v87 + v84;
        float * v89;
        v89 = v9+v86;
        float * v91;
        v91 = v73+v88;
        int v93;
        v93 = 0;
        #pragma unroll
        while (while_method_1(v93)){
            int v95;
            v95 = 0;
            #pragma unroll
            while (while_method_2(v95)){
                assert("Tensor range check" && 0 <= v93 && v93 < 8);
                assert("Tensor range check" && 0 <= v95 && v95 < 1);
                int v97;
                v97 = 128 * v95;
                int v98;
                v98 = 2176 * v93;
                int v99;
                v99 = v98 + v97;
                int v100;
                v100 = 131072 * v93;
                int v101;
                v101 = v100 + v97;
                int4* v102;
                v102 = reinterpret_cast<int4*>(v91 + v101);
                int4* v103;
                v103 = reinterpret_cast<int4*>(v89 + v99);
                assert("Pointer alignment check" && (unsigned long long)(v102) % 4 == 0 && (unsigned long long)(v103) % 4 == 0);
                *v103 = *v102;
                v95 += 1 ;
            }
            v93 += 1 ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0));
        int v104;
        v104 = 0;
        #pragma unroll
        while (while_method_3(v104)){
            int v106;
            v106 = 0;
            #pragma unroll
            while (while_method_2(v106)){
                assert("Tensor range check" && 0 <= v104 && v104 < 4);
                assert("Tensor range check" && 0 <= v106 && v106 < 1);
                int v108;
                v108 = v104 + v106;
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v109 = v58[v108];
                assert("Tensor range check" && 0 <= v104 && v104 < 4);
                assert("Tensor range check" && 0 <= v106 && v106 < 1);
                int v110;
                v110 = 16 * v106;
                int v111;
                v111 = 2176 * v104;
                int v112;
                v112 = v111 + v110;
                float * v113;
                v113 = v24+v112;
                wmma::load_matrix_sync(v109, v113, 136, wmma::mem_row_major);
                v106 += 1 ;
            }
            v104 += 1 ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0));
        // Poping the loop unrolling to: 0
        int v115;
        v115 = 0;
        while (while_method_4(v115)){
            int v117;
            v117 = v115 + 1;
            bool v118;
            v118 = v115 == 0;
            int v119;
            v119 = v115 % 2;
            bool v120;
            v120 = 0 <= v115;
            bool v121;
            v121 = v120 == false;
            if (v121){
                assert("The index needs to be zero or positive." && v120);
            } else {
            }
            bool v123;
            v123 = v115 < 64;
            bool v124;
            v124 = v123 == false;
            if (v124){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v123);
            } else {
            }
            bool v126;
            v126 = v117 < 64;
            Union0 v132;
            if (v126){
                bool v127;
                v127 = 0 <= v117;
                bool v128;
                v128 = v127 == false;
                if (v128){
                    assert("The index needs to be zero or positive." && v127);
                } else {
                }
                v132 = Union0{Union0_1{v117}};
            } else {
                v132 = Union0{Union0_0{}};
            }
            assert("Tensor range check" && 0 <= v66 && v66 < 64);
            int v133;
            v133 = 524288 * v66;
            assert("Tensor range check" && 0 <= v115 && v115 < 64);
            int v134;
            v134 = 64 * v115;
            int v135;
            v135 = v134 + v133;
            float * v136;
            v136 = v0+v135;
            assert("Tensor range check" && 0 <= v65 && v65 < 64);
            int v138;
            v138 = 524288 * v65;
            if (v118){
                assert("Tensor range check" && 0 <= v115 && v115 < 64);
                int v139;
                v139 = v134 + v138;
                float * v140;
                v140 = v1+v139;
                // Pushing the loop unrolling to: 0
                v3.producer_acquire();
                int v142;
                v142 = threadIdx.x;
                bool v143;
                v143 = 0 <= v142;
                bool v144;
                v144 = v143 == false;
                if (v144){
                    assert("The index needs to be zero or positive." && v143);
                } else {
                }
                int v146;
                v146 = v142 % 16;
                int v147;
                v147 = v142 / 16;
                bool v148;
                v148 = v147 < 32;
                bool v149;
                v149 = v148 == false;
                if (v149){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v148);
                } else {
                }
                assert("Tensor range check" && 0 <= v147 && v147 < 32);
                assert("Tensor range check" && 0 <= v146 && v146 < 16);
                int v151;
                v151 = 4 * v146;
                int v152;
                v152 = 68 * v147;
                int v153;
                v153 = v152 + v151;
                int v154;
                v154 = 4096 * v147;
                int v155;
                v155 = v154 + v151;
                float * v156;
                v156 = v7+v153;
                float * v158;
                v158 = v140+v155;
                int v160;
                v160 = 0;
                #pragma unroll
                while (while_method_3(v160)){
                    int v162;
                    v162 = 0;
                    #pragma unroll
                    while (while_method_2(v162)){
                        assert("Tensor range check" && 0 <= v160 && v160 < 4);
                        assert("Tensor range check" && 0 <= v162 && v162 < 1);
                        int v164;
                        v164 = 64 * v162;
                        int v165;
                        v165 = 2176 * v160;
                        int v166;
                        v166 = v165 + v164;
                        int v167;
                        v167 = 131072 * v160;
                        int v168;
                        v168 = v167 + v164;
                        constexpr int v169 = sizeof(float) * 4;
                        assert("Pointer alignment check" && (unsigned long long)(v158 + v168) % v169 == 0 && (unsigned long long)(v156 + v166) % v169 == 0);
                        cuda::memcpy_async(v156 + v166, v158 + v168, cuda::aligned_size_t<v169>(v169), v3);
                        v162 += 1 ;
                    }
                    v160 += 1 ;
                }
                v3.producer_commit();
                // Poping the loop unrolling to: 0
            } else {
            }
            int v170;
            v170 = threadIdx.x;
            bool v171;
            v171 = 0 <= v170;
            bool v172;
            v172 = v171 == false;
            if (v172){
                assert("The index needs to be zero or positive." && v171);
            } else {
            }
            int v174;
            v174 = v170 % 16;
            int v175;
            v175 = v170 / 16;
            bool v176;
            v176 = v175 < 32;
            bool v177;
            v177 = v176 == false;
            if (v177){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v176);
            } else {
            }
            assert("Tensor range check" && 0 <= v175 && v175 < 32);
            assert("Tensor range check" && 0 <= v174 && v174 < 16);
            int v179;
            v179 = 4 * v174;
            int v180;
            v180 = 68 * v175;
            int v181;
            v181 = v180 + v179;
            int v182;
            v182 = 4096 * v175;
            int v183;
            v183 = v182 + v179;
            float * v184;
            v184 = v5+v181;
            float * v186;
            v186 = v136+v183;
            int v188;
            v188 = 0;
            while (while_method_3(v188)){
                int v190;
                v190 = 0;
                while (while_method_2(v190)){
                    assert("Tensor range check" && 0 <= v188 && v188 < 4);
                    assert("Tensor range check" && 0 <= v190 && v190 < 1);
                    int v192;
                    v192 = 64 * v190;
                    int v193;
                    v193 = 2176 * v188;
                    int v194;
                    v194 = v193 + v192;
                    int v195;
                    v195 = 131072 * v188;
                    int v196;
                    v196 = v195 + v192;
                    int4* v197;
                    v197 = reinterpret_cast<int4*>(v186 + v196);
                    int4* v198;
                    v198 = reinterpret_cast<int4*>(v184 + v194);
                    assert("Pointer alignment check" && (unsigned long long)(v197) % 4 == 0 && (unsigned long long)(v198) % 4 == 0);
                    *v198 = *v197;
                    v190 += 1 ;
                }
                v188 += 1 ;
            }
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v199[1];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v200[8];
            cuda::pipeline_consumer_wait_prior<0>(v3);;
            asm("barrier.cta.sync %0;" :: "r"(0));
            // Pushing the loop unrolling to: 0
            int v201;
            v201 = 0;
            #pragma unroll
            while (while_method_2(v201)){
                int v203;
                v203 = 0;
                #pragma unroll
                while (while_method_1(v203)){
                    assert("Tensor range check" && 0 <= v201 && v201 < 1);
                    assert("Tensor range check" && 0 <= v203 && v203 < 8);
                    int v205;
                    v205 = 8 * v201;
                    int v206;
                    v206 = v205 + v203;
                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v207 = v200[v206];
                    assert("Tensor range check" && 0 <= v201 && v201 < 1);
                    int v208;
                    v208 = 1088 * v201;
                    assert("Tensor range check" && 0 <= v203 && v203 < 8);
                    int v209;
                    v209 = 8 * v203;
                    int v210;
                    v210 = v209 + v208;
                    int v211;
                    v211 = 0;
                    #pragma unroll
                    while (while_method_5(v211)){
                        int v213;
                        v213 = 0;
                        #pragma unroll
                        while (while_method_5(v213)){
                            assert("Tensor range check" && 0 <= v211 && v211 < 2);
                            assert("Tensor range check" && 0 <= v213 && v213 < 2);
                            int v215;
                            v215 = 4 * v213;
                            int v216;
                            v216 = v215 + v210;
                            int v217;
                            v217 = 544 * v211;
                            int v218;
                            v218 = v217 + v216;
                            float v219;
                            v219 = v56[v218];
                            bool v220;
                            v220 = 0 <= v213;
                            bool v222;
                            if (v220){
                                bool v221;
                                v221 = v213 < 2;
                                v222 = v221;
                            } else {
                                v222 = false;
                            }
                            bool v223;
                            v223 = v222 == false;
                            if (v223){
                                assert("The indices should be inside the range of the dimension." && v222);
                            } else {
                            }
                            bool v225;
                            v225 = 0 <= v211;
                            bool v227;
                            if (v225){
                                bool v226;
                                v226 = v211 < 2;
                                v227 = v226;
                            } else {
                                v227 = false;
                            }
                            bool v228;
                            v228 = v227 == false;
                            if (v228){
                                assert("The indices should be inside the range of the dimension." && v227);
                            } else {
                            }
                            int v230;
                            v230 = v211 * 2;
                            int v231;
                            v231 = v213 + v230;
                            v207.x[v231] = wmma::__float_to_tf32(v219);
                            v213 += 1 ;
                        }
                        v211 += 1 ;
                    }
                    v203 += 1 ;
                }
                v201 += 1 ;
            }
            // Poping the loop unrolling to: 0
            v3.consumer_release();
            switch (v132.tag) {
                case 0: { // None
                    break;
                }
                case 1: { // Some
                    int v232 = v132.case1.v0;
                    assert("Tensor range check" && 0 <= v232 && v232 < 64);
                    int v233;
                    v233 = 64 * v232;
                    int v234;
                    v234 = v233 + v138;
                    float * v235;
                    v235 = v1+v234;
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    // Pushing the loop unrolling to: 0
                    v3.producer_acquire();
                    int v237;
                    v237 = threadIdx.x;
                    bool v238;
                    v238 = 0 <= v237;
                    bool v239;
                    v239 = v238 == false;
                    if (v239){
                        assert("The index needs to be zero or positive." && v238);
                    } else {
                    }
                    int v241;
                    v241 = v237 % 16;
                    int v242;
                    v242 = v237 / 16;
                    bool v243;
                    v243 = v242 < 32;
                    bool v244;
                    v244 = v243 == false;
                    if (v244){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v243);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v242 && v242 < 32);
                    assert("Tensor range check" && 0 <= v241 && v241 < 16);
                    int v246;
                    v246 = 4 * v241;
                    int v247;
                    v247 = 68 * v242;
                    int v248;
                    v248 = v247 + v246;
                    int v249;
                    v249 = 4096 * v242;
                    int v250;
                    v250 = v249 + v246;
                    float * v251;
                    v251 = v7+v248;
                    float * v253;
                    v253 = v235+v250;
                    int v255;
                    v255 = 0;
                    #pragma unroll
                    while (while_method_3(v255)){
                        int v257;
                        v257 = 0;
                        #pragma unroll
                        while (while_method_2(v257)){
                            assert("Tensor range check" && 0 <= v255 && v255 < 4);
                            assert("Tensor range check" && 0 <= v257 && v257 < 1);
                            int v259;
                            v259 = 64 * v257;
                            int v260;
                            v260 = 2176 * v255;
                            int v261;
                            v261 = v260 + v259;
                            int v262;
                            v262 = 131072 * v255;
                            int v263;
                            v263 = v262 + v259;
                            constexpr int v264 = sizeof(float) * 4;
                            assert("Pointer alignment check" && (unsigned long long)(v253 + v263) % v264 == 0 && (unsigned long long)(v251 + v261) % v264 == 0);
                            cuda::memcpy_async(v251 + v261, v253 + v263, cuda::aligned_size_t<v264>(v264), v3);
                            v257 += 1 ;
                        }
                        v255 += 1 ;
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
            int v265;
            v265 = 0;
            #pragma unroll
            while (while_method_3(v265)){
                int v267;
                v267 = 0;
                #pragma unroll
                while (while_method_1(v267)){
                    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v269 = v199[0];
                    assert("Tensor range check" && 0 <= v265 && v265 < 4);
                    int v270;
                    v270 = 1088 * v265;
                    assert("Tensor range check" && 0 <= v267 && v267 < 8);
                    int v271;
                    v271 = 8 * v267;
                    int v272;
                    v272 = v271 + v270;
                    int v273;
                    v273 = 0;
                    #pragma unroll
                    while (while_method_5(v273)){
                        int v275;
                        v275 = 0;
                        #pragma unroll
                        while (while_method_5(v275)){
                            assert("Tensor range check" && 0 <= v273 && v273 < 2);
                            assert("Tensor range check" && 0 <= v275 && v275 < 2);
                            int v277;
                            v277 = 544 * v275;
                            int v278;
                            v278 = v277 + v272;
                            int v279;
                            v279 = 4 * v273;
                            int v280;
                            v280 = v279 + v278;
                            float v281;
                            v281 = v40[v280];
                            bool v282;
                            v282 = 0 <= v275;
                            bool v284;
                            if (v282){
                                bool v283;
                                v283 = v275 < 2;
                                v284 = v283;
                            } else {
                                v284 = false;
                            }
                            bool v285;
                            v285 = v284 == false;
                            if (v285){
                                assert("The indices should be inside the range of the dimension." && v284);
                            } else {
                            }
                            bool v287;
                            v287 = 0 <= v273;
                            bool v289;
                            if (v287){
                                bool v288;
                                v288 = v273 < 2;
                                v289 = v288;
                            } else {
                                v289 = false;
                            }
                            bool v290;
                            v290 = v289 == false;
                            if (v290){
                                assert("The indices should be inside the range of the dimension." && v289);
                            } else {
                            }
                            int v292;
                            v292 = v273 * 2;
                            int v293;
                            v293 = v275 + v292;
                            v269.x[v293] = wmma::__float_to_tf32(v281);
                            v275 += 1 ;
                        }
                        v273 += 1 ;
                    }
                    int v294;
                    v294 = 0;
                    #pragma unroll
                    while (while_method_2(v294)){
                        assert("Tensor range check" && 0 <= v265 && v265 < 4);
                        assert("Tensor range check" && 0 <= v294 && v294 < 1);
                        int v296;
                        v296 = v265 + v294;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v297 = v58[v296];
                        assert("Tensor range check" && 0 <= v294 && v294 < 1);
                        assert("Tensor range check" && 0 <= v267 && v267 < 8);
                        int v298;
                        v298 = 8 * v294;
                        int v299;
                        v299 = v298 + v267;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v300 = v200[v299];
                        wmma::mma_sync(v297, v269, v300, v297);
                        v294 += 1 ;
                    }
                    v267 += 1 ;
                }
                v265 += 1 ;
            }
            // Poping the loop unrolling to: 0
            asm("barrier.cta.sync %0;" :: "r"(0));
            v115 = v117;
        }
        // Pushing the loop unrolling to: 0
        int v301;
        v301 = 0;
        #pragma unroll
        while (while_method_3(v301)){
            int v303;
            v303 = 0;
            #pragma unroll
            while (while_method_2(v303)){
                assert("Tensor range check" && 0 <= v301 && v301 < 4);
                assert("Tensor range check" && 0 <= v303 && v303 < 1);
                int v305;
                v305 = v301 + v303;
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v306 = v58[v305];
                assert("Tensor range check" && 0 <= v301 && v301 < 4);
                assert("Tensor range check" && 0 <= v303 && v303 < 1);
                int v307;
                v307 = 16 * v303;
                int v308;
                v308 = 2176 * v301;
                int v309;
                v309 = v308 + v307;
                float * v310;
                v310 = v24+v309;
                wmma::store_matrix_sync(v310, v306, 136, wmma::mem_row_major);
                v303 += 1 ;
            }
            v301 += 1 ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0));
        // Pushing the loop unrolling to: 0
        int v312;
        v312 = threadIdx.x;
        bool v313;
        v313 = 0 <= v312;
        bool v314;
        v314 = v313 == false;
        if (v314){
            assert("The index needs to be zero or positive." && v313);
        } else {
        }
        int v316;
        v316 = v312 % 32;
        int v317;
        v317 = v312 / 32;
        bool v318;
        v318 = v317 < 16;
        bool v319;
        v319 = v318 == false;
        if (v319){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v318);
        } else {
        }
        assert("Tensor range check" && 0 <= v317 && v317 < 16);
        assert("Tensor range check" && 0 <= v316 && v316 < 32);
        int v321;
        v321 = 4 * v316;
        int v322;
        v322 = 8192 * v317;
        int v323;
        v323 = v322 + v321;
        int v324;
        v324 = 136 * v317;
        int v325;
        v325 = v324 + v321;
        float * v326;
        v326 = v73+v323;
        float * v328;
        v328 = v9+v325;
        int v330;
        v330 = 0;
        #pragma unroll
        while (while_method_1(v330)){
            int v332;
            v332 = 0;
            #pragma unroll
            while (while_method_2(v332)){
                assert("Tensor range check" && 0 <= v330 && v330 < 8);
                assert("Tensor range check" && 0 <= v332 && v332 < 1);
                int v334;
                v334 = 128 * v332;
                int v335;
                v335 = 131072 * v330;
                int v336;
                v336 = v335 + v334;
                int v337;
                v337 = 2176 * v330;
                int v338;
                v338 = v337 + v334;
                int4* v339;
                v339 = reinterpret_cast<int4*>(v328 + v338);
                int4* v340;
                v340 = reinterpret_cast<int4*>(v326 + v336);
                assert("Pointer alignment check" && (unsigned long long)(v339) % 4 == 0 && (unsigned long long)(v340) % 4 == 0);
                *v340 = *v339;
                v332 += 1 ;
            }
            v330 += 1 ;
        }
        // Poping the loop unrolling to: 0
        asm("barrier.cta.sync %0;" :: "r"(0));
        v60 += 24 ;
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
