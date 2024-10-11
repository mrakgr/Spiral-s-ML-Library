kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups/reduce.h>
#include <curand_kernel.h>
#include <cooperative_groups/scan.h>
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

struct Tuple0;
struct Tuple1;
struct Tuple2;
struct Closure0 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure1 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple0 {
    float v0;
    bool v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple0 operator()(Tuple0 tup0, Tuple0 tup1){
        float v0 = tup0.v0; bool v1 = tup0.v1; float v2 = tup1.v0; bool v3 = tup1.v1;
        if (v1){
            if (v3){
                bool v4;
                v4 = v0 >= v2;
                float v5;
                if (v4){
                    v5 = v0;
                } else {
                    v5 = v2;
                }
                return Tuple0{v5, true};
            } else {
                return Tuple0{v0, v1};
            }
        } else {
            if (v3){
                return Tuple0{v2, v3};
            } else {
                return Tuple0{v0, v1};
            }
        }
    }
};
struct Tuple1 {
    float v0;
    int v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple1{v0, v1};
        } else {
            return Tuple1{v2, v3};
        }
    }
};
struct Tuple2 {
    int v0;
    bool v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure5 {
    __device__ Tuple2 operator()(Tuple2 tup0, Tuple2 tup1){
        int v0 = tup0.v0; bool v1 = tup0.v1; int v2 = tup1.v0; bool v3 = tup1.v1;
        if (v1){
            if (v3){
                bool v4;
                v4 = v0 < v2;
                int v5;
                if (v4){
                    v5 = v0;
                } else {
                    v5 = v2;
                }
                return Tuple2{v5, true};
            } else {
                return Tuple2{v0, v1};
            }
        } else {
            if (v3){
                return Tuple2{v2, v3};
            } else {
                return Tuple2{v0, v1};
            }
        }
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 64;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    int v8;
    v8 = threadIdx.x;
    int v9;
    v9 = blockIdx.x;
    int v10;
    v10 = v9 * 256;
    int v11;
    v11 = v8 + v10;
    assert("Tensor range check" && 0 <= v11 && v11 < 6144);
    int v12;
    v12 = 16 * v11;
    int v13;
    v13 = threadIdx.x;
    int v14;
    v14 = blockIdx.x;
    int v15;
    v15 = v14 * 256;
    int v16;
    v16 = v13 + v15;
    assert("Tensor range check" && 0 <= v16 && v16 < 6144);
    int v17;
    v17 = 16 * v16;
    int v18;
    v18 = threadIdx.x;
    int v19;
    v19 = blockIdx.x;
    int v20;
    v20 = v19 * 256;
    int v21;
    v21 = v18 + v20;
    assert("Tensor range check" && 0 <= v21 && v21 < 6144);
    int v22;
    v22 = 16 * v21;
    int v23;
    v23 = threadIdx.x;
    int v24;
    v24 = blockIdx.x;
    int v25;
    v25 = v24 * 256;
    int v26;
    v26 = v23 + v25;
    assert("Tensor range check" && 0 <= v26 && v26 < 6144);
    int v27;
    v27 = 16 * v26;
    int v28;
    v28 = threadIdx.x;
    int v29;
    v29 = blockIdx.x;
    int v30;
    v30 = v29 * 256;
    int v31;
    v31 = v28 + v30;
    assert("Tensor range check" && 0 <= v31 && v31 < 6144);
    int v32;
    v32 = 16 * v31;
    float * v33;
    v33 = v1+v12;
    int * v35;
    v35 = v2+v27;
    int * v37;
    v37 = v3+v27;
    int v39;
    v39 = sizeof(float *);
    unsigned long long v40;
    v40 = (unsigned long long)v39;
    unsigned long long v41;
    v41 = 256ull * v40;
    unsigned long long v42;
    v42 = v41 + 16ull;
    unsigned long long v43;
    v43 = v42 - 1ull;
    unsigned long long v44;
    v44 = v43 % 16ull;
    unsigned long long v45;
    v45 = v43 - v44;
    int v46;
    v46 = sizeof(int *);
    unsigned long long v47;
    v47 = (unsigned long long)v46;
    unsigned long long v48;
    v48 = 256ull * v47;
    unsigned long long v49;
    v49 = v45 + v48;
    unsigned long long v50;
    v50 = v49 + 16ull;
    unsigned long long v51;
    v51 = v50 - 1ull;
    unsigned long long v52;
    v52 = v51 % 16ull;
    unsigned long long v53;
    v53 = v51 - v52;
    unsigned long long v54;
    v54 = v53 + v48;
    bool v55;
    v55 = v54 <= 98304ull;
    bool v56;
    v56 = v55 == false;
    if (v56){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v55);
    } else {
    }
    extern __shared__ unsigned char v58[];
    bool v59;
    v59 = v54 <= v54;
    bool v60;
    v60 = v59 == false;
    if (v60){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v59);
    } else {
    }
    float * * v62;
    v62 = reinterpret_cast<float * *>(&v58[0ull]);
    int * * v64;
    v64 = reinterpret_cast<int * *>(&v58[v45]);
    int * * v66;
    v66 = reinterpret_cast<int * *>(&v58[v53]);
    int v68;
    v68 = threadIdx.x;
    assert("Tensor range check" && 0 <= v68 && v68 < 256);
    v62[v68] = v33;
    v64[v68] = v35;
    v66[v68] = v37;
    __syncthreads();
    bool v69;
    v69 = 0 <= v68;
    bool v70;
    v70 = v69 == false;
    if (v70){
        assert("The index needs to be zero or positive." && v69);
    } else {
    }
    int v72;
    v72 = v68 % 4;
    int v73;
    v73 = v68 / 4;
    bool v74;
    v74 = v73 < 64;
    bool v75;
    v75 = v74 == false;
    if (v75){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v74);
    } else {
    }
    assert("Tensor range check" && 0 <= v73 && v73 < 64);
    int v77;
    v77 = 0;
    while (while_method_0(v77)){
        bool v79;
        v79 = 0 <= v73;
        bool v80;
        v80 = v79 && v74;
        bool v81;
        v81 = v80 == false;
        if (v81){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v80);
        } else {
        }
        bool v83;
        v83 = 0 <= v77;
        bool v85;
        if (v83){
            bool v84;
            v84 = v77 < 4;
            v85 = v84;
        } else {
            v85 = false;
        }
        bool v86;
        v86 = v85 == false;
        if (v86){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v85);
        } else {
        }
        int v88;
        v88 = v77 * 64;
        int v89;
        v89 = v88 + v73;
        assert("Tensor range check" && 0 <= v77 && v77 < 4);
        int v90;
        v90 = 64 * v77;
        int v91;
        v91 = v90 + v73;
        float * v92;
        v92 = v62[v91];
        int * v93;
        v93 = v64[v91];
        int * v94;
        v94 = v66[v91];
        int v95;
        v95 = blockIdx.x;
        int v96;
        v96 = v95 * 256;
        int v97;
        v97 = v96 + v89;
        assert("Tensor range check" && 0 <= v72 && v72 < 4);
        int v98;
        v98 = 4 * v72;
        float v99[4];
        int v100[4];
        int v101;
        v101 = 0;
        while (while_method_1(v101)){
            assert("Tensor range check" && 0 <= v101 && v101 < 1);
            int v103;
            v103 = 4 * v101;
            assert("Tensor range check" && 0 <= v101 && v101 < 1);
            int v104;
            v104 = 16 * v101;
            int v105;
            v105 = v104 + v98;
            int4* v106;
            v106 = reinterpret_cast<int4*>(v92 + v105);
            int4* v107;
            v107 = reinterpret_cast<int4*>(v99 + v103);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v106) % 16 == 0 && reinterpret_cast<unsigned long long>(v107) % 16 == 0);
            *v107 = *v106;
            v101 += 1 ;
        }
        int v108;
        v108 = 0;
        while (while_method_1(v108)){
            int v110;
            v110 = 0;
            while (while_method_0(v110)){
                bool v112;
                v112 = 0 <= v110;
                bool v114;
                if (v112){
                    bool v113;
                    v113 = v110 < 4;
                    v114 = v113;
                } else {
                    v114 = false;
                }
                bool v115;
                v115 = v114 == false;
                if (v115){
                    assert("The indices should be inside the range of the dimension." && v114);
                } else {
                }
                bool v117;
                v117 = 0 <= v72;
                bool v119;
                if (v117){
                    bool v118;
                    v118 = v72 < 4;
                    v119 = v118;
                } else {
                    v119 = false;
                }
                bool v120;
                v120 = v119 == false;
                if (v120){
                    assert("The indices should be inside the range of the dimension." && v119);
                } else {
                }
                int v122;
                v122 = v72 * 4;
                int v123;
                v123 = v110 + v122;
                bool v124;
                v124 = 0 <= v108;
                bool v126;
                if (v124){
                    bool v125;
                    v125 = v108 < 1;
                    v126 = v125;
                } else {
                    v126 = false;
                }
                bool v127;
                v127 = v126 == false;
                if (v127){
                    assert("The indices should be inside the range of the dimension." && v126);
                } else {
                }
                int v129;
                v129 = v108 * 16;
                int v130;
                v130 = v123 + v129;
                assert("Tensor range check" && 0 <= v108 && v108 < 1);
                assert("Tensor range check" && 0 <= v110 && v110 < 4);
                int v131;
                v131 = 4 * v108;
                int v132;
                v132 = v131 + v110;
                v100[v132] = v130;
                v110 += 1 ;
            }
            v108 += 1 ;
        }
        int v133[4];
        int v134[4];
        int v135;
        v135 = 0;
        while (while_method_1(v135)){
            int v137;
            v137 = 0;
            while (while_method_0(v137)){
                assert("Tensor range check" && 0 <= v135 && v135 < 1);
                assert("Tensor range check" && 0 <= v137 && v137 < 4);
                int v139;
                v139 = 4 * v135;
                int v140;
                v140 = v139 + v137;
                int v141;
                v141 = v100[v140];
                assert("Tensor range check" && 0 <= v135 && v135 < 1);
                assert("Tensor range check" && 0 <= v137 && v137 < 4);
                v133[v140] = v97;
                v134[v140] = v141;
                v137 += 1 ;
            }
            v135 += 1 ;
        }
        int v142;
        v142 = 0;
        while (while_method_1(v142)){
            assert("Tensor range check" && 0 <= v142 && v142 < 1);
            int v144;
            v144 = 16 * v142;
            int v145;
            v145 = v144 + v98;
            assert("Tensor range check" && 0 <= v142 && v142 < 1);
            int v146;
            v146 = 4 * v142;
            int4* v147;
            v147 = reinterpret_cast<int4*>(v133 + v146);
            int4* v148;
            v148 = reinterpret_cast<int4*>(v93 + v145);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v147) % 16 == 0 && reinterpret_cast<unsigned long long>(v148) % 16 == 0);
            *v148 = *v147;
            int4* v149;
            v149 = reinterpret_cast<int4*>(v134 + v146);
            int4* v150;
            v150 = reinterpret_cast<int4*>(v94 + v145);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v149) % 16 == 0 && reinterpret_cast<unsigned long long>(v150) % 16 == 0);
            *v150 = *v149;
            v142 += 1 ;
        }
        assert("Tensor range check" && 0 <= v89 && v89 < 256);
        v77 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v68 && v68 < 256);
    __syncthreads();
    float * v151;
    v151 = v1+v12;
    unsigned long long v153;
    v153 = v45 + 1024ull;
    bool v154;
    v154 = v153 <= 98304ull;
    bool v155;
    v155 = v154 == false;
    if (v155){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v154);
    } else {
    }
    extern __shared__ unsigned char v157[];
    bool v158;
    v158 = v153 <= v153;
    bool v159;
    v159 = v158 == false;
    if (v159){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v158);
    } else {
    }
    float * * v161;
    v161 = reinterpret_cast<float * *>(&v157[0ull]);
    int * v163;
    v163 = reinterpret_cast<int *>(&v157[v45]);
    int v165;
    v165 = threadIdx.x;
    assert("Tensor range check" && 0 <= v165 && v165 < 256);
    v161[v165] = v151;
    __syncthreads();
    bool v166;
    v166 = 0 <= v165;
    bool v167;
    v167 = v166 == false;
    if (v167){
        assert("The index needs to be zero or positive." && v166);
    } else {
    }
    int v169;
    v169 = v165 % 4;
    int v170;
    v170 = v165 / 4;
    bool v171;
    v171 = v170 < 64;
    bool v172;
    v172 = v171 == false;
    if (v172){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v171);
    } else {
    }
    assert("Tensor range check" && 0 <= v170 && v170 < 64);
    int v174;
    v174 = 0;
    while (while_method_0(v174)){
        bool v176;
        v176 = 0 <= v170;
        bool v177;
        v177 = v176 && v171;
        bool v178;
        v178 = v177 == false;
        if (v178){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v177);
        } else {
        }
        bool v180;
        v180 = 0 <= v174;
        bool v182;
        if (v180){
            bool v181;
            v181 = v174 < 4;
            v182 = v181;
        } else {
            v182 = false;
        }
        bool v183;
        v183 = v182 == false;
        if (v183){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v182);
        } else {
        }
        int v185;
        v185 = v174 * 64;
        int v186;
        v186 = v185 + v170;
        assert("Tensor range check" && 0 <= v174 && v174 < 4);
        int v187;
        v187 = 64 * v174;
        int v188;
        v188 = v187 + v170;
        float * v189;
        v189 = v161[v188];
        int v190;
        v190 = blockIdx.x;
        int v191;
        v191 = v190 * 256;
        int v192;
        v192 = v191 + v186;
        assert("Tensor range check" && 0 <= v169 && v169 < 4);
        int v193;
        v193 = 4 * v169;
        float v194[4];
        int v195[4];
        int v196;
        v196 = 0;
        while (while_method_1(v196)){
            assert("Tensor range check" && 0 <= v196 && v196 < 1);
            int v198;
            v198 = 4 * v196;
            assert("Tensor range check" && 0 <= v196 && v196 < 1);
            int v199;
            v199 = 16 * v196;
            int v200;
            v200 = v199 + v193;
            int4* v201;
            v201 = reinterpret_cast<int4*>(v189 + v200);
            int4* v202;
            v202 = reinterpret_cast<int4*>(v194 + v198);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v201) % 16 == 0 && reinterpret_cast<unsigned long long>(v202) % 16 == 0);
            *v202 = *v201;
            v196 += 1 ;
        }
        int v203;
        v203 = 0;
        while (while_method_1(v203)){
            int v205;
            v205 = 0;
            while (while_method_0(v205)){
                bool v207;
                v207 = 0 <= v205;
                bool v209;
                if (v207){
                    bool v208;
                    v208 = v205 < 4;
                    v209 = v208;
                } else {
                    v209 = false;
                }
                bool v210;
                v210 = v209 == false;
                if (v210){
                    assert("The indices should be inside the range of the dimension." && v209);
                } else {
                }
                bool v212;
                v212 = 0 <= v169;
                bool v214;
                if (v212){
                    bool v213;
                    v213 = v169 < 4;
                    v214 = v213;
                } else {
                    v214 = false;
                }
                bool v215;
                v215 = v214 == false;
                if (v215){
                    assert("The indices should be inside the range of the dimension." && v214);
                } else {
                }
                int v217;
                v217 = v169 * 4;
                int v218;
                v218 = v205 + v217;
                bool v219;
                v219 = 0 <= v203;
                bool v221;
                if (v219){
                    bool v220;
                    v220 = v203 < 1;
                    v221 = v220;
                } else {
                    v221 = false;
                }
                bool v222;
                v222 = v221 == false;
                if (v222){
                    assert("The indices should be inside the range of the dimension." && v221);
                } else {
                }
                int v224;
                v224 = v203 * 16;
                int v225;
                v225 = v218 + v224;
                assert("Tensor range check" && 0 <= v203 && v203 < 1);
                assert("Tensor range check" && 0 <= v205 && v205 < 4);
                int v226;
                v226 = 4 * v203;
                int v227;
                v227 = v226 + v205;
                v195[v227] = v225;
                v205 += 1 ;
            }
            v203 += 1 ;
        }
        int v228;
        v228 = 0;
        while (while_method_1(v228)){
            assert("Tensor range check" && 0 <= v228 && v228 < 1);
            assert("Tensor range check" && 0 <= v228 && v228 < 1);
            v228 += 1 ;
        }
        assert("Tensor range check" && 0 <= v186 && v186 < 256);
        v163[v186] = v192;
        v174 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v165 && v165 < 256);
    int v230;
    v230 = v163[v165];
    __syncthreads();
    int v231;
    v231 = threadIdx.x;
    int v232;
    v232 = blockIdx.x;
    int v233;
    v233 = v232 * 256;
    int v234;
    v234 = v231 + v233;
    assert("Tensor range check" && 0 <= v234 && v234 < 6144);
    v4[v234] = v230;
    float * v235;
    v235 = v1+v12;
    float * v237;
    v237 = v6+v32;
    unsigned long long v239;
    v239 = v45 + v41;
    bool v240;
    v240 = v239 <= 98304ull;
    bool v241;
    v241 = v240 == false;
    if (v241){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v240);
    } else {
    }
    extern __shared__ unsigned char v243[];
    bool v244;
    v244 = v239 <= v239;
    bool v245;
    v245 = v244 == false;
    if (v245){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v244);
    } else {
    }
    float * * v247;
    v247 = reinterpret_cast<float * *>(&v243[0ull]);
    float * * v249;
    v249 = reinterpret_cast<float * *>(&v243[v45]);
    int v251;
    v251 = threadIdx.x;
    assert("Tensor range check" && 0 <= v251 && v251 < 256);
    v247[v251] = v235;
    v249[v251] = v237;
    __syncthreads();
    bool v252;
    v252 = 0 <= v251;
    bool v253;
    v253 = v252 == false;
    if (v253){
        assert("The index needs to be zero or positive." && v252);
    } else {
    }
    int v255;
    v255 = v251 % 4;
    int v256;
    v256 = v251 / 4;
    bool v257;
    v257 = v256 < 64;
    bool v258;
    v258 = v257 == false;
    if (v258){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v257);
    } else {
    }
    assert("Tensor range check" && 0 <= v256 && v256 < 64);
    int v260;
    v260 = 0;
    while (while_method_0(v260)){
        bool v262;
        v262 = 0 <= v256;
        bool v263;
        v263 = v262 && v257;
        bool v264;
        v264 = v263 == false;
        if (v264){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v263);
        } else {
        }
        bool v266;
        v266 = 0 <= v260;
        bool v268;
        if (v266){
            bool v267;
            v267 = v260 < 4;
            v268 = v267;
        } else {
            v268 = false;
        }
        bool v269;
        v269 = v268 == false;
        if (v269){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v268);
        } else {
        }
        int v271;
        v271 = v260 * 64;
        int v272;
        v272 = v271 + v256;
        assert("Tensor range check" && 0 <= v260 && v260 < 4);
        int v273;
        v273 = 64 * v260;
        int v274;
        v274 = v273 + v256;
        float * v275;
        v275 = v247[v274];
        float * v276;
        v276 = v249[v274];
        int v277;
        v277 = blockIdx.x;
        int v278;
        v278 = v277 * 256;
        int v279;
        v279 = v278 + v272;
        assert("Tensor range check" && 0 <= v255 && v255 < 4);
        int v280;
        v280 = 4 * v255;
        float v281[4];
        int v282[4];
        int v283;
        v283 = 0;
        while (while_method_1(v283)){
            assert("Tensor range check" && 0 <= v283 && v283 < 1);
            int v285;
            v285 = 4 * v283;
            assert("Tensor range check" && 0 <= v283 && v283 < 1);
            int v286;
            v286 = 16 * v283;
            int v287;
            v287 = v286 + v280;
            int4* v288;
            v288 = reinterpret_cast<int4*>(v275 + v287);
            int4* v289;
            v289 = reinterpret_cast<int4*>(v281 + v285);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v288) % 16 == 0 && reinterpret_cast<unsigned long long>(v289) % 16 == 0);
            *v289 = *v288;
            v283 += 1 ;
        }
        int v290;
        v290 = 0;
        while (while_method_1(v290)){
            int v292;
            v292 = 0;
            while (while_method_0(v292)){
                bool v294;
                v294 = 0 <= v292;
                bool v296;
                if (v294){
                    bool v295;
                    v295 = v292 < 4;
                    v296 = v295;
                } else {
                    v296 = false;
                }
                bool v297;
                v297 = v296 == false;
                if (v297){
                    assert("The indices should be inside the range of the dimension." && v296);
                } else {
                }
                bool v299;
                v299 = 0 <= v255;
                bool v301;
                if (v299){
                    bool v300;
                    v300 = v255 < 4;
                    v301 = v300;
                } else {
                    v301 = false;
                }
                bool v302;
                v302 = v301 == false;
                if (v302){
                    assert("The indices should be inside the range of the dimension." && v301);
                } else {
                }
                int v304;
                v304 = v255 * 4;
                int v305;
                v305 = v292 + v304;
                bool v306;
                v306 = 0 <= v290;
                bool v308;
                if (v306){
                    bool v307;
                    v307 = v290 < 1;
                    v308 = v307;
                } else {
                    v308 = false;
                }
                bool v309;
                v309 = v308 == false;
                if (v309){
                    assert("The indices should be inside the range of the dimension." && v308);
                } else {
                }
                int v311;
                v311 = v290 * 16;
                int v312;
                v312 = v305 + v311;
                assert("Tensor range check" && 0 <= v290 && v290 < 1);
                assert("Tensor range check" && 0 <= v292 && v292 < 4);
                int v313;
                v313 = 4 * v290;
                int v314;
                v314 = v313 + v292;
                v282[v314] = v312;
                v292 += 1 ;
            }
            v290 += 1 ;
        }
        int v315;
        v315 = 0;
        while (while_method_1(v315)){
            assert("Tensor range check" && 0 <= v315 && v315 < 1);
            int v317;
            v317 = 16 * v315;
            int v318;
            v318 = v317 + v280;
            assert("Tensor range check" && 0 <= v315 && v315 < 1);
            int v319;
            v319 = 4 * v315;
            int4* v320;
            v320 = reinterpret_cast<int4*>(v281 + v319);
            int4* v321;
            v321 = reinterpret_cast<int4*>(v276 + v318);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v320) % 16 == 0 && reinterpret_cast<unsigned long long>(v321) % 16 == 0);
            *v321 = *v320;
            v315 += 1 ;
        }
        assert("Tensor range check" && 0 <= v272 && v272 < 256);
        v260 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v251 && v251 < 256);
    __syncthreads();
    float * v322;
    v322 = v1+v12;
    float * v324;
    v324 = v7+v22;
    if (v241){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v240);
    } else {
    }
    extern __shared__ unsigned char v327[];
    if (v245){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v244);
    } else {
    }
    float * * v329;
    v329 = reinterpret_cast<float * *>(&v327[0ull]);
    float * * v331;
    v331 = reinterpret_cast<float * *>(&v327[v45]);
    int v333;
    v333 = threadIdx.x;
    assert("Tensor range check" && 0 <= v333 && v333 < 256);
    v329[v333] = v322;
    v331[v333] = v324;
    __syncthreads();
    bool v334;
    v334 = 0 <= v333;
    bool v335;
    v335 = v334 == false;
    if (v335){
        assert("The index needs to be zero or positive." && v334);
    } else {
    }
    int v337;
    v337 = v333 % 4;
    int v338;
    v338 = v333 / 4;
    bool v339;
    v339 = v338 < 64;
    bool v340;
    v340 = v339 == false;
    if (v340){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v339);
    } else {
    }
    assert("Tensor range check" && 0 <= v338 && v338 < 64);
    int v342;
    v342 = 0;
    while (while_method_0(v342)){
        bool v344;
        v344 = 0 <= v338;
        bool v345;
        v345 = v344 && v339;
        bool v346;
        v346 = v345 == false;
        if (v346){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v345);
        } else {
        }
        bool v348;
        v348 = 0 <= v342;
        bool v350;
        if (v348){
            bool v349;
            v349 = v342 < 4;
            v350 = v349;
        } else {
            v350 = false;
        }
        bool v351;
        v351 = v350 == false;
        if (v351){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v350);
        } else {
        }
        int v353;
        v353 = v342 * 64;
        int v354;
        v354 = v353 + v338;
        assert("Tensor range check" && 0 <= v342 && v342 < 4);
        int v355;
        v355 = 64 * v342;
        int v356;
        v356 = v355 + v338;
        float * v357;
        v357 = v329[v356];
        float * v358;
        v358 = v331[v356];
        int v359;
        v359 = blockIdx.x;
        int v360;
        v360 = v359 * 256;
        int v361;
        v361 = v360 + v354;
        assert("Tensor range check" && 0 <= v337 && v337 < 4);
        int v362;
        v362 = 4 * v337;
        float v363[4];
        int v364[4];
        int v365;
        v365 = 0;
        while (while_method_1(v365)){
            assert("Tensor range check" && 0 <= v365 && v365 < 1);
            int v367;
            v367 = 4 * v365;
            assert("Tensor range check" && 0 <= v365 && v365 < 1);
            int v368;
            v368 = 16 * v365;
            int v369;
            v369 = v368 + v362;
            int4* v370;
            v370 = reinterpret_cast<int4*>(v357 + v369);
            int4* v371;
            v371 = reinterpret_cast<int4*>(v363 + v367);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v370) % 16 == 0 && reinterpret_cast<unsigned long long>(v371) % 16 == 0);
            *v371 = *v370;
            v365 += 1 ;
        }
        int v372;
        v372 = 0;
        while (while_method_1(v372)){
            int v374;
            v374 = 0;
            while (while_method_0(v374)){
                bool v376;
                v376 = 0 <= v374;
                bool v378;
                if (v376){
                    bool v377;
                    v377 = v374 < 4;
                    v378 = v377;
                } else {
                    v378 = false;
                }
                bool v379;
                v379 = v378 == false;
                if (v379){
                    assert("The indices should be inside the range of the dimension." && v378);
                } else {
                }
                bool v381;
                v381 = 0 <= v337;
                bool v383;
                if (v381){
                    bool v382;
                    v382 = v337 < 4;
                    v383 = v382;
                } else {
                    v383 = false;
                }
                bool v384;
                v384 = v383 == false;
                if (v384){
                    assert("The indices should be inside the range of the dimension." && v383);
                } else {
                }
                int v386;
                v386 = v337 * 4;
                int v387;
                v387 = v374 + v386;
                bool v388;
                v388 = 0 <= v372;
                bool v390;
                if (v388){
                    bool v389;
                    v389 = v372 < 1;
                    v390 = v389;
                } else {
                    v390 = false;
                }
                bool v391;
                v391 = v390 == false;
                if (v391){
                    assert("The indices should be inside the range of the dimension." && v390);
                } else {
                }
                int v393;
                v393 = v372 * 16;
                int v394;
                v394 = v387 + v393;
                assert("Tensor range check" && 0 <= v372 && v372 < 1);
                assert("Tensor range check" && 0 <= v374 && v374 < 4);
                int v395;
                v395 = 4 * v372;
                int v396;
                v396 = v395 + v374;
                v364[v396] = v394;
                v374 += 1 ;
            }
            v372 += 1 ;
        }
        bool v397[4];
        int v398;
        v398 = 0;
        while (while_method_1(v398)){
            int v400;
            v400 = 0;
            while (while_method_0(v400)){
                assert("Tensor range check" && 0 <= v398 && v398 < 1);
                assert("Tensor range check" && 0 <= v400 && v400 < 4);
                int v402;
                v402 = 4 * v398;
                int v403;
                v403 = v402 + v400;
                float v404;
                v404 = v363[v403];
                int v405;
                v405 = v364[v403];
                bool v406;
                v406 = v405 < 3;
                assert("Tensor range check" && 0 <= v398 && v398 < 1);
                assert("Tensor range check" && 0 <= v400 && v400 < 4);
                v397[v403] = v406;
                v400 += 1 ;
            }
            v398 += 1 ;
        }
        float v407[4];
        int v408;
        v408 = 0;
        while (while_method_1(v408)){
            int v410;
            v410 = 0;
            while (while_method_0(v410)){
                assert("Tensor range check" && 0 <= v408 && v408 < 1);
                assert("Tensor range check" && 0 <= v410 && v410 < 4);
                int v412;
                v412 = 4 * v408;
                int v413;
                v413 = v412 + v410;
                float v414;
                v414 = v363[v413];
                bool v415;
                v415 = v397[v413];
                float v418;
                if (v415){
                    bool v416;
                    v416 = 0.0f >= v414;
                    if (v416){
                        v418 = 0.0f;
                    } else {
                        v418 = v414;
                    }
                } else {
                    v418 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v408 && v408 < 1);
                assert("Tensor range check" && 0 <= v410 && v410 < 4);
                v407[v413] = v418;
                v410 += 1 ;
            }
            v408 += 1 ;
        }
        float v419;
        v419 = 0.0f;
        int v420;
        v420 = 0;
        while (while_method_1(v420)){
            int v422;
            v422 = 0;
            while (while_method_0(v422)){
                assert("Tensor range check" && 0 <= v420 && v420 < 1);
                assert("Tensor range check" && 0 <= v422 && v422 < 4);
                int v424;
                v424 = 4 * v420;
                int v425;
                v425 = v424 + v422;
                float v426;
                v426 = v407[v425];
                float v427;
                v427 = v419 + v426;
                v419 = v427;
                v422 += 1 ;
            }
            v420 += 1 ;
        }
        auto v428 = cooperative_groups::coalesced_threads();
        int v429;
        v429 = threadIdx.x;
        int v430;
        v430 = v429 / 4;
        auto v431 = cooperative_groups::labeled_partition(v428,v430);
        Closure0 v432{};
        float v433;
        v433 = cooperative_groups::reduce(v431, v419, v432);
        int v434[4];
        int v435;
        v435 = 0;
        while (while_method_1(v435)){
            int v437;
            v437 = 0;
            while (while_method_0(v437)){
                assert("Tensor range check" && 0 <= v435 && v435 < 1);
                assert("Tensor range check" && 0 <= v437 && v437 < 4);
                int v439;
                v439 = 4 * v435;
                int v440;
                v440 = v439 + v437;
                bool v441;
                v441 = v397[v440];
                int v442;
                if (v441){
                    v442 = 1;
                } else {
                    v442 = 0;
                }
                assert("Tensor range check" && 0 <= v435 && v435 < 1);
                assert("Tensor range check" && 0 <= v437 && v437 < 4);
                v434[v440] = v442;
                v437 += 1 ;
            }
            v435 += 1 ;
        }
        int v443;
        v443 = 0;
        int v444;
        v444 = 0;
        while (while_method_1(v444)){
            int v446;
            v446 = 0;
            while (while_method_0(v446)){
                assert("Tensor range check" && 0 <= v444 && v444 < 1);
                assert("Tensor range check" && 0 <= v446 && v446 < 4);
                int v448;
                v448 = 4 * v444;
                int v449;
                v449 = v448 + v446;
                int v450;
                v450 = v434[v449];
                int v451;
                v451 = v443 + v450;
                v443 = v451;
                v446 += 1 ;
            }
            v444 += 1 ;
        }
        auto v452 = cooperative_groups::coalesced_threads();
        int v453;
        v453 = threadIdx.x;
        int v454;
        v454 = v453 / 4;
        auto v455 = cooperative_groups::labeled_partition(v452,v454);
        Closure1 v456{};
        int v457;
        v457 = cooperative_groups::reduce(v455, v443, v456);
        float v458;
        v458 = (float)v457;
        float v459;
        v459 = 1.0f / v458;
        float v460[4];
        int v461;
        v461 = 0;
        while (while_method_1(v461)){
            int v463;
            v463 = 0;
            while (while_method_0(v463)){
                assert("Tensor range check" && 0 <= v461 && v461 < 1);
                assert("Tensor range check" && 0 <= v463 && v463 < 4);
                int v465;
                v465 = 4 * v461;
                int v466;
                v466 = v465 + v463;
                float v467;
                v467 = v407[v466];
                bool v468;
                v468 = v397[v466];
                bool v469;
                v469 = v468 == false;
                float v474;
                if (v469){
                    v474 = 0.0f;
                } else {
                    bool v470;
                    v470 = v433 == 0.0f;
                    bool v471;
                    v471 = v470 != true;
                    if (v471){
                        float v472;
                        v472 = v467 / v433;
                        v474 = v472;
                    } else {
                        v474 = v459;
                    }
                }
                assert("Tensor range check" && 0 <= v461 && v461 < 1);
                assert("Tensor range check" && 0 <= v463 && v463 < 4);
                v460[v466] = v474;
                v463 += 1 ;
            }
            v461 += 1 ;
        }
        int v475;
        v475 = 0;
        while (while_method_1(v475)){
            assert("Tensor range check" && 0 <= v475 && v475 < 1);
            int v477;
            v477 = 16 * v475;
            int v478;
            v478 = v477 + v362;
            assert("Tensor range check" && 0 <= v475 && v475 < 1);
            int v479;
            v479 = 4 * v475;
            int4* v480;
            v480 = reinterpret_cast<int4*>(v460 + v479);
            int4* v481;
            v481 = reinterpret_cast<int4*>(v358 + v478);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v480) % 16 == 0 && reinterpret_cast<unsigned long long>(v481) % 16 == 0);
            *v481 = *v480;
            v475 += 1 ;
        }
        assert("Tensor range check" && 0 <= v354 && v354 < 256);
        v342 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v333 && v333 < 256);
    __syncthreads();
    int v482;
    v482 = threadIdx.x;
    int v483;
    v483 = blockIdx.x;
    int v484;
    v484 = v483 * 256;
    int v485;
    v485 = v482 + v484;
    unsigned long long v486;
    v486 = (unsigned long long)v485;
    curandStatePhilox4_32_10_t v487;
    curand_init(12344321ull,v486,0ull,&v487);
    float * v488;
    v488 = v1+v12;
    if (v155){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v154);
    } else {
    }
    extern __shared__ unsigned char v491[];
    if (v159){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v158);
    } else {
    }
    float * * v493;
    v493 = reinterpret_cast<float * *>(&v491[0ull]);
    int * v495;
    v495 = reinterpret_cast<int *>(&v491[v45]);
    int v497;
    v497 = threadIdx.x;
    assert("Tensor range check" && 0 <= v497 && v497 < 256);
    v493[v497] = v488;
    __syncthreads();
    bool v498;
    v498 = 0 <= v497;
    bool v499;
    v499 = v498 == false;
    if (v499){
        assert("The index needs to be zero or positive." && v498);
    } else {
    }
    int v501;
    v501 = v497 % 4;
    int v502;
    v502 = v497 / 4;
    bool v503;
    v503 = v502 < 64;
    bool v504;
    v504 = v503 == false;
    if (v504){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v503);
    } else {
    }
    assert("Tensor range check" && 0 <= v502 && v502 < 64);
    int v506;
    v506 = 0;
    while (while_method_0(v506)){
        bool v508;
        v508 = 0 <= v502;
        bool v509;
        v509 = v508 && v503;
        bool v510;
        v510 = v509 == false;
        if (v510){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v509);
        } else {
        }
        bool v512;
        v512 = 0 <= v506;
        bool v514;
        if (v512){
            bool v513;
            v513 = v506 < 4;
            v514 = v513;
        } else {
            v514 = false;
        }
        bool v515;
        v515 = v514 == false;
        if (v515){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v514);
        } else {
        }
        int v517;
        v517 = v506 * 64;
        int v518;
        v518 = v517 + v502;
        assert("Tensor range check" && 0 <= v506 && v506 < 4);
        int v519;
        v519 = 64 * v506;
        int v520;
        v520 = v519 + v502;
        float * v521;
        v521 = v493[v520];
        int v522;
        v522 = blockIdx.x;
        int v523;
        v523 = v522 * 256;
        int v524;
        v524 = v523 + v518;
        assert("Tensor range check" && 0 <= v501 && v501 < 4);
        int v525;
        v525 = 4 * v501;
        float v526[4];
        int v527[4];
        int v528;
        v528 = 0;
        while (while_method_1(v528)){
            assert("Tensor range check" && 0 <= v528 && v528 < 1);
            int v530;
            v530 = 4 * v528;
            assert("Tensor range check" && 0 <= v528 && v528 < 1);
            int v531;
            v531 = 16 * v528;
            int v532;
            v532 = v531 + v525;
            int4* v533;
            v533 = reinterpret_cast<int4*>(v521 + v532);
            int4* v534;
            v534 = reinterpret_cast<int4*>(v526 + v530);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v533) % 16 == 0 && reinterpret_cast<unsigned long long>(v534) % 16 == 0);
            *v534 = *v533;
            v528 += 1 ;
        }
        int v535;
        v535 = 0;
        while (while_method_1(v535)){
            int v537;
            v537 = 0;
            while (while_method_0(v537)){
                bool v539;
                v539 = 0 <= v537;
                bool v541;
                if (v539){
                    bool v540;
                    v540 = v537 < 4;
                    v541 = v540;
                } else {
                    v541 = false;
                }
                bool v542;
                v542 = v541 == false;
                if (v542){
                    assert("The indices should be inside the range of the dimension." && v541);
                } else {
                }
                bool v544;
                v544 = 0 <= v501;
                bool v546;
                if (v544){
                    bool v545;
                    v545 = v501 < 4;
                    v546 = v545;
                } else {
                    v546 = false;
                }
                bool v547;
                v547 = v546 == false;
                if (v547){
                    assert("The indices should be inside the range of the dimension." && v546);
                } else {
                }
                int v549;
                v549 = v501 * 4;
                int v550;
                v550 = v537 + v549;
                bool v551;
                v551 = 0 <= v535;
                bool v553;
                if (v551){
                    bool v552;
                    v552 = v535 < 1;
                    v553 = v552;
                } else {
                    v553 = false;
                }
                bool v554;
                v554 = v553 == false;
                if (v554){
                    assert("The indices should be inside the range of the dimension." && v553);
                } else {
                }
                int v556;
                v556 = v535 * 16;
                int v557;
                v557 = v550 + v556;
                assert("Tensor range check" && 0 <= v535 && v535 < 1);
                assert("Tensor range check" && 0 <= v537 && v537 < 4);
                int v558;
                v558 = 4 * v535;
                int v559;
                v559 = v558 + v537;
                v527[v559] = v557;
                v537 += 1 ;
            }
            v535 += 1 ;
        }
        bool v560[4];
        int v561;
        v561 = 0;
        while (while_method_1(v561)){
            int v563;
            v563 = 0;
            while (while_method_0(v563)){
                assert("Tensor range check" && 0 <= v561 && v561 < 1);
                assert("Tensor range check" && 0 <= v563 && v563 < 4);
                int v565;
                v565 = 4 * v561;
                int v566;
                v566 = v565 + v563;
                float v567;
                v567 = v526[v566];
                int v568;
                v568 = v527[v566];
                bool v569;
                v569 = v568 < 3;
                assert("Tensor range check" && 0 <= v561 && v561 < 1);
                assert("Tensor range check" && 0 <= v563 && v563 < 4);
                v560[v566] = v569;
                v563 += 1 ;
            }
            v561 += 1 ;
        }
        float v570[4];
        int v571;
        v571 = 0;
        while (while_method_1(v571)){
            int v573;
            v573 = 0;
            while (while_method_0(v573)){
                assert("Tensor range check" && 0 <= v571 && v571 < 1);
                assert("Tensor range check" && 0 <= v573 && v573 < 4);
                int v575;
                v575 = 4 * v571;
                int v576;
                v576 = v575 + v573;
                float v577;
                v577 = v526[v576];
                bool v578;
                v578 = v560[v576];
                float v579;
                if (v578){
                    v579 = v577;
                } else {
                    v579 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v571 && v571 < 1);
                assert("Tensor range check" && 0 <= v573 && v573 < 4);
                v570[v576] = v579;
                v573 += 1 ;
            }
            v571 += 1 ;
        }
        float v580;
        v580 = 0.0f;
        int v581;
        v581 = 0;
        while (while_method_1(v581)){
            int v583;
            v583 = 0;
            while (while_method_0(v583)){
                assert("Tensor range check" && 0 <= v581 && v581 < 1);
                assert("Tensor range check" && 0 <= v583 && v583 < 4);
                int v585;
                v585 = 4 * v581;
                int v586;
                v586 = v585 + v583;
                float v587;
                v587 = v570[v586];
                float v588;
                v588 = v580 + v587;
                v580 = v588;
                v583 += 1 ;
            }
            v581 += 1 ;
        }
        auto v589 = cooperative_groups::coalesced_threads();
        int v590;
        v590 = threadIdx.x;
        int v591;
        v591 = v590 / 4;
        auto v592 = cooperative_groups::labeled_partition(v589,v591);
        Closure0 v593{};
        float v594;
        v594 = cooperative_groups::reduce(v592, v580, v593);
        int v595[4];
        int v596;
        v596 = 0;
        while (while_method_1(v596)){
            int v598;
            v598 = 0;
            while (while_method_0(v598)){
                assert("Tensor range check" && 0 <= v596 && v596 < 1);
                assert("Tensor range check" && 0 <= v598 && v598 < 4);
                int v600;
                v600 = 4 * v596;
                int v601;
                v601 = v600 + v598;
                bool v602;
                v602 = v560[v601];
                int v603;
                if (v602){
                    v603 = 1;
                } else {
                    v603 = 0;
                }
                assert("Tensor range check" && 0 <= v596 && v596 < 1);
                assert("Tensor range check" && 0 <= v598 && v598 < 4);
                v595[v601] = v603;
                v598 += 1 ;
            }
            v596 += 1 ;
        }
        int v604;
        v604 = 0;
        int v605;
        v605 = 0;
        while (while_method_1(v605)){
            int v607;
            v607 = 0;
            while (while_method_0(v607)){
                assert("Tensor range check" && 0 <= v605 && v605 < 1);
                assert("Tensor range check" && 0 <= v607 && v607 < 4);
                int v609;
                v609 = 4 * v605;
                int v610;
                v610 = v609 + v607;
                int v611;
                v611 = v595[v610];
                int v612;
                v612 = v604 + v611;
                v604 = v612;
                v607 += 1 ;
            }
            v605 += 1 ;
        }
        auto v613 = cooperative_groups::coalesced_threads();
        int v614;
        v614 = threadIdx.x;
        int v615;
        v615 = v614 / 4;
        auto v616 = cooperative_groups::labeled_partition(v613,v615);
        Closure1 v617{};
        int v618;
        v618 = cooperative_groups::reduce(v616, v604, v617);
        float v619;
        v619 = (float)v618;
        float v620;
        v620 = v594 / v619;
        float v621[4];
        int v622;
        v622 = 0;
        while (while_method_1(v622)){
            int v624;
            v624 = 0;
            while (while_method_0(v624)){
                assert("Tensor range check" && 0 <= v622 && v622 < 1);
                assert("Tensor range check" && 0 <= v624 && v624 < 4);
                int v626;
                v626 = 4 * v622;
                int v627;
                v627 = v626 + v624;
                float v628;
                v628 = v526[v627];
                bool v629;
                v629 = v560[v627];
                float v630;
                if (v629){
                    v630 = v628;
                } else {
                    v630 = -1.0f / 0.0f;
                }
                float v631;
                v631 = v630 - v620;
                float v632;
                v632 = exp(v631);
                bool v633;
                v633 = v632 < 1.0f / 0.0f;
                bool v634;
                v634 = v633 == false;
                if (v634){
                    assert("The softmax values must not grow too large." && v633);
                } else {
                }
                bool v636;
                v636 = isnan(v632);
                bool v637;
                v637 = v636 == false;
                bool v638;
                v638 = v637 == false;
                if (v638){
                    assert("The softmax values must not be nans." && v637);
                } else {
                }
                assert("Tensor range check" && 0 <= v622 && v622 < 1);
                assert("Tensor range check" && 0 <= v624 && v624 < 4);
                v621[v627] = v632;
                v624 += 1 ;
            }
            v622 += 1 ;
        }
        float v640;
        v640 = 0.0f;
        int v641;
        v641 = 0;
        while (while_method_1(v641)){
            int v643;
            v643 = 0;
            while (while_method_0(v643)){
                assert("Tensor range check" && 0 <= v641 && v641 < 1);
                assert("Tensor range check" && 0 <= v643 && v643 < 4);
                int v645;
                v645 = 4 * v641;
                int v646;
                v646 = v645 + v643;
                float v647;
                v647 = v621[v646];
                float v648;
                v648 = v640 + v647;
                v640 = v648;
                v643 += 1 ;
            }
            v641 += 1 ;
        }
        auto v649 = cooperative_groups::coalesced_threads();
        int v650;
        v650 = threadIdx.x;
        int v651;
        v651 = v650 / 4;
        auto v652 = cooperative_groups::labeled_partition(v649,v651);
        float v653;
        v653 = cooperative_groups::reduce(v652, v640, v593);
        float v654[4];
        int v655;
        v655 = 0;
        while (while_method_1(v655)){
            int v657;
            v657 = 0;
            while (while_method_0(v657)){
                assert("Tensor range check" && 0 <= v655 && v655 < 1);
                assert("Tensor range check" && 0 <= v657 && v657 < 4);
                int v659;
                v659 = 4 * v655;
                int v660;
                v660 = v659 + v657;
                float v661;
                v661 = v621[v660];
                float v662;
                v662 = v661 / v653;
                assert("Tensor range check" && 0 <= v655 && v655 < 1);
                assert("Tensor range check" && 0 <= v657 && v657 < 4);
                v654[v660] = v662;
                v657 += 1 ;
            }
            v655 += 1 ;
        }
        float v663[4];
        float v664;
        v664 = 0.0f;
        int v665;
        v665 = 0;
        while (while_method_1(v665)){
            assert("Tensor range check" && 0 <= v665 && v665 < 1);
            int v667;
            v667 = 4 * v665;
            assert("Tensor range check" && 0 <= v665 && v665 < 1);
            float v668;
            v668 = 0.0f;
            int v669;
            v669 = 0;
            while (while_method_0(v669)){
                assert("Tensor range check" && 0 <= v669 && v669 < 4);
                int v671;
                v671 = v669 + v667;
                float v672;
                v672 = v654[v671];
                float v673;
                v673 = v668 + v672;
                v668 = v673;
                v669 += 1 ;
            }
            auto v674 = cooperative_groups::coalesced_threads();
            int v675;
            v675 = threadIdx.x;
            int v676;
            v676 = v675 / 4;
            auto v677 = cooperative_groups::labeled_partition(v674,v676);
            Closure2 v678{};
            float v679;
            v679 = cooperative_groups::inclusive_scan(v677, v668, v678);
            float v680;
            v680 = v677.shfl_up(v679,1);
            bool v681;
            v681 = v677.thread_rank() == 0;
            float v682;
            if (v681){
                v682 = 0.0f;
            } else {
                v682 = v680;
            }
            float v683;
            v683 = v677.shfl(v679,v677.num_threads()-1);
            float v684;
            v684 = v664 + v682;
            float v685;
            v685 = v684;
            int v686;
            v686 = 0;
            while (while_method_0(v686)){
                assert("Tensor range check" && 0 <= v686 && v686 < 4);
                int v688;
                v688 = v686 + v667;
                float v689;
                v689 = v654[v688];
                float v690;
                v690 = v685 + v689;
                assert("Tensor range check" && 0 <= v686 && v686 < 4);
                v663[v688] = v690;
                v685 = v690;
                v686 += 1 ;
            }
            float v691;
            v691 = v664 + v683;
            v664 = v691;
            v665 += 1 ;
        }
        float v692[4];
        bool v693[4];
        int v694;
        v694 = 0;
        while (while_method_1(v694)){
            int v696;
            v696 = 0;
            while (while_method_0(v696)){
                assert("Tensor range check" && 0 <= v694 && v694 < 1);
                assert("Tensor range check" && 0 <= v696 && v696 < 4);
                int v698;
                v698 = 4 * v694;
                int v699;
                v699 = v698 + v696;
                float v700;
                v700 = v663[v699];
                float v701;
                v701 = v654[v699];
                bool v702;
                v702 = v701 > 0.0f;
                assert("Tensor range check" && 0 <= v694 && v694 < 1);
                assert("Tensor range check" && 0 <= v696 && v696 < 4);
                v692[v699] = v700;
                v693[v699] = v702;
                v696 += 1 ;
            }
            v694 += 1 ;
        }
        float v703; bool v704;
        Tuple0 tmp0 = Tuple0{-1.0f / 0.0f, false};
        v703 = tmp0.v0; v704 = tmp0.v1;
        int v705;
        v705 = 0;
        while (while_method_1(v705)){
            int v707;
            v707 = 0;
            while (while_method_0(v707)){
                assert("Tensor range check" && 0 <= v705 && v705 < 1);
                assert("Tensor range check" && 0 <= v707 && v707 < 4);
                int v709;
                v709 = 4 * v705;
                int v710;
                v710 = v709 + v707;
                float v711;
                v711 = v692[v710];
                bool v712;
                v712 = v693[v710];
                float v719; bool v720;
                if (v704){
                    if (v712){
                        bool v713;
                        v713 = v703 >= v711;
                        float v714;
                        if (v713){
                            v714 = v703;
                        } else {
                            v714 = v711;
                        }
                        v719 = v714; v720 = true;
                    } else {
                        v719 = v703; v720 = v704;
                    }
                } else {
                    if (v712){
                        v719 = v711; v720 = v712;
                    } else {
                        v719 = v703; v720 = v704;
                    }
                }
                v703 = v719;
                v704 = v720;
                v707 += 1 ;
            }
            v705 += 1 ;
        }
        auto v721 = cooperative_groups::coalesced_threads();
        int v722;
        v722 = threadIdx.x;
        int v723;
        v723 = v722 / 4;
        auto v724 = cooperative_groups::labeled_partition(v721,v723);
        Closure3 v725{};
        float v726; bool v727;
        Tuple0 tmp1 = cooperative_groups::reduce(v724, Tuple0{v703, v704}, v725);
        v726 = tmp1.v0; v727 = tmp1.v1;
        bool v728;
        v728 = v727 == false;
        if (v728){
            assert("The local reduce must be true." && v727);
        } else {
        }
        float v730[4];
        int v731[4];
        int v732;
        v732 = 0;
        while (while_method_1(v732)){
            int v734;
            v734 = 0;
            while (while_method_0(v734)){
                assert("Tensor range check" && 0 <= v732 && v732 < 1);
                assert("Tensor range check" && 0 <= v734 && v734 < 4);
                int v736;
                v736 = 4 * v732;
                int v737;
                v737 = v736 + v734;
                int v738;
                v738 = v527[v737];
                float v739;
                v739 = curand_uniform(&v487);
                assert("Tensor range check" && 0 <= v732 && v732 < 1);
                assert("Tensor range check" && 0 <= v734 && v734 < 4);
                v730[v737] = v739;
                v731[v737] = v738;
                v734 += 1 ;
            }
            v732 += 1 ;
        }
        float v740; int v741;
        Tuple1 tmp2 = Tuple1{0.0f, 2147483647};
        v740 = tmp2.v0; v741 = tmp2.v1;
        int v742;
        v742 = 0;
        while (while_method_1(v742)){
            int v744;
            v744 = 0;
            while (while_method_0(v744)){
                assert("Tensor range check" && 0 <= v742 && v742 < 1);
                assert("Tensor range check" && 0 <= v744 && v744 < 4);
                int v746;
                v746 = 4 * v742;
                int v747;
                v747 = v746 + v744;
                float v748;
                v748 = v730[v747];
                int v749;
                v749 = v731[v747];
                bool v750;
                v750 = v741 < v749;
                float v751; int v752;
                if (v750){
                    v751 = v740; v752 = v741;
                } else {
                    v751 = v748; v752 = v749;
                }
                v740 = v751;
                v741 = v752;
                v744 += 1 ;
            }
            v742 += 1 ;
        }
        auto v753 = cooperative_groups::coalesced_threads();
        int v754;
        v754 = threadIdx.x;
        int v755;
        v755 = v754 / 4;
        auto v756 = cooperative_groups::labeled_partition(v753,v755);
        Closure4 v757{};
        float v758; int v759;
        Tuple1 tmp3 = cooperative_groups::reduce(v756, Tuple1{v740, v741}, v757);
        v758 = tmp3.v0; v759 = tmp3.v1;
        float v760;
        v760 = v726 * v758;
        int v761[4];
        bool v762[4];
        int v763;
        v763 = 0;
        while (while_method_1(v763)){
            int v765;
            v765 = 0;
            while (while_method_0(v765)){
                assert("Tensor range check" && 0 <= v763 && v763 < 1);
                assert("Tensor range check" && 0 <= v765 && v765 < 4);
                int v767;
                v767 = 4 * v763;
                int v768;
                v768 = v767 + v765;
                float v769;
                v769 = v692[v768];
                bool v770;
                v770 = v693[v768];
                int v771;
                v771 = v527[v768];
                int v774; bool v775;
                if (v770){
                    float v772;
                    v772 = v769 - v760;
                    bool v773;
                    v773 = v772 >= 0.0f;
                    v774 = v771; v775 = v773;
                } else {
                    v774 = 2147483647; v775 = false;
                }
                assert("Tensor range check" && 0 <= v763 && v763 < 1);
                assert("Tensor range check" && 0 <= v765 && v765 < 4);
                v761[v768] = v774;
                v762[v768] = v775;
                v765 += 1 ;
            }
            v763 += 1 ;
        }
        int v776; bool v777;
        Tuple2 tmp4 = Tuple2{2147483647, false};
        v776 = tmp4.v0; v777 = tmp4.v1;
        int v778;
        v778 = 0;
        while (while_method_1(v778)){
            int v780;
            v780 = 0;
            while (while_method_0(v780)){
                assert("Tensor range check" && 0 <= v778 && v778 < 1);
                assert("Tensor range check" && 0 <= v780 && v780 < 4);
                int v782;
                v782 = 4 * v778;
                int v783;
                v783 = v782 + v780;
                int v784;
                v784 = v761[v783];
                bool v785;
                v785 = v762[v783];
                int v792; bool v793;
                if (v777){
                    if (v785){
                        bool v786;
                        v786 = v776 < v784;
                        int v787;
                        if (v786){
                            v787 = v776;
                        } else {
                            v787 = v784;
                        }
                        v792 = v787; v793 = true;
                    } else {
                        v792 = v776; v793 = v777;
                    }
                } else {
                    if (v785){
                        v792 = v784; v793 = v785;
                    } else {
                        v792 = v776; v793 = v777;
                    }
                }
                v776 = v792;
                v777 = v793;
                v780 += 1 ;
            }
            v778 += 1 ;
        }
        auto v794 = cooperative_groups::coalesced_threads();
        int v795;
        v795 = threadIdx.x;
        int v796;
        v796 = v795 / 4;
        auto v797 = cooperative_groups::labeled_partition(v794,v796);
        Closure5 v798{};
        int v799; bool v800;
        Tuple2 tmp5 = cooperative_groups::reduce(v797, Tuple2{v776, v777}, v798);
        v799 = tmp5.v0; v800 = tmp5.v1;
        bool v801;
        v801 = v800 == false;
        if (v801){
            assert("The local reduce must be true." && v800);
        } else {
        }
        int v803;
        v803 = 0;
        while (while_method_1(v803)){
            assert("Tensor range check" && 0 <= v803 && v803 < 1);
            assert("Tensor range check" && 0 <= v803 && v803 < 1);
            v803 += 1 ;
        }
        assert("Tensor range check" && 0 <= v518 && v518 < 256);
        v495[v518] = v799;
        v506 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v497 && v497 < 256);
    int v805;
    v805 = v495[v497];
    __syncthreads();
    int v806;
    v806 = threadIdx.x;
    int v807;
    v807 = blockIdx.x;
    int v808;
    v808 = v807 * 256;
    int v809;
    v809 = v806 + v808;
    assert("Tensor range check" && 0 <= v809 && v809 < 6144);
    v5[v809] = v805;
    return ;
}
extern "C" __global__ void entry1(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    int v8;
    v8 = threadIdx.x;
    int v9;
    v9 = blockIdx.x;
    int v10;
    v10 = v9 * 256;
    int v11;
    v11 = v8 + v10;
    assert("Tensor range check" && 0 <= v11 && v11 < 6144);
    int v12;
    v12 = 256 * v11;
    int v13;
    v13 = threadIdx.x;
    int v14;
    v14 = blockIdx.x;
    int v15;
    v15 = v14 * 256;
    int v16;
    v16 = v13 + v15;
    assert("Tensor range check" && 0 <= v16 && v16 < 6144);
    int v17;
    v17 = 256 * v16;
    int v18;
    v18 = threadIdx.x;
    int v19;
    v19 = blockIdx.x;
    int v20;
    v20 = v19 * 256;
    int v21;
    v21 = v18 + v20;
    assert("Tensor range check" && 0 <= v21 && v21 < 6144);
    int v22;
    v22 = 256 * v21;
    int v23;
    v23 = threadIdx.x;
    int v24;
    v24 = blockIdx.x;
    int v25;
    v25 = v24 * 256;
    int v26;
    v26 = v23 + v25;
    assert("Tensor range check" && 0 <= v26 && v26 < 6144);
    int v27;
    v27 = 256 * v26;
    int v28;
    v28 = threadIdx.x;
    int v29;
    v29 = blockIdx.x;
    int v30;
    v30 = v29 * 256;
    int v31;
    v31 = v28 + v30;
    assert("Tensor range check" && 0 <= v31 && v31 < 6144);
    int v32;
    v32 = 256 * v31;
    float * v33;
    v33 = v1+v12;
    int * v35;
    v35 = v2+v27;
    int * v37;
    v37 = v3+v27;
    int v39;
    v39 = sizeof(float *);
    unsigned long long v40;
    v40 = (unsigned long long)v39;
    unsigned long long v41;
    v41 = 256ull * v40;
    unsigned long long v42;
    v42 = v41 + 16ull;
    unsigned long long v43;
    v43 = v42 - 1ull;
    unsigned long long v44;
    v44 = v43 % 16ull;
    unsigned long long v45;
    v45 = v43 - v44;
    int v46;
    v46 = sizeof(int *);
    unsigned long long v47;
    v47 = (unsigned long long)v46;
    unsigned long long v48;
    v48 = 256ull * v47;
    unsigned long long v49;
    v49 = v45 + v48;
    unsigned long long v50;
    v50 = v49 + 16ull;
    unsigned long long v51;
    v51 = v50 - 1ull;
    unsigned long long v52;
    v52 = v51 % 16ull;
    unsigned long long v53;
    v53 = v51 - v52;
    unsigned long long v54;
    v54 = v53 + v48;
    bool v55;
    v55 = v54 <= 98304ull;
    bool v56;
    v56 = v55 == false;
    if (v56){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v55);
    } else {
    }
    extern __shared__ unsigned char v58[];
    bool v59;
    v59 = v54 <= v54;
    bool v60;
    v60 = v59 == false;
    if (v60){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v59);
    } else {
    }
    float * * v62;
    v62 = reinterpret_cast<float * *>(&v58[0ull]);
    int * * v64;
    v64 = reinterpret_cast<int * *>(&v58[v45]);
    int * * v66;
    v66 = reinterpret_cast<int * *>(&v58[v53]);
    int v68;
    v68 = threadIdx.x;
    assert("Tensor range check" && 0 <= v68 && v68 < 256);
    v62[v68] = v33;
    v64[v68] = v35;
    v66[v68] = v37;
    __syncthreads();
    bool v69;
    v69 = 0 <= v68;
    bool v70;
    v70 = v69 == false;
    if (v70){
        assert("The index needs to be zero or positive." && v69);
    } else {
    }
    int v72;
    v72 = v68 % 64;
    int v73;
    v73 = v68 / 64;
    bool v74;
    v74 = v73 < 4;
    bool v75;
    v75 = v74 == false;
    if (v75){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v74);
    } else {
    }
    assert("Tensor range check" && 0 <= v73 && v73 < 4);
    int v77;
    v77 = 0;
    while (while_method_2(v77)){
        bool v79;
        v79 = 0 <= v73;
        bool v80;
        v80 = v79 && v74;
        bool v81;
        v81 = v80 == false;
        if (v81){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v80);
        } else {
        }
        bool v83;
        v83 = 0 <= v77;
        bool v85;
        if (v83){
            bool v84;
            v84 = v77 < 64;
            v85 = v84;
        } else {
            v85 = false;
        }
        bool v86;
        v86 = v85 == false;
        if (v86){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v85);
        } else {
        }
        int v88;
        v88 = v77 * 4;
        int v89;
        v89 = v88 + v73;
        assert("Tensor range check" && 0 <= v77 && v77 < 64);
        int v90;
        v90 = 4 * v77;
        int v91;
        v91 = v90 + v73;
        float * v92;
        v92 = v62[v91];
        int * v93;
        v93 = v64[v91];
        int * v94;
        v94 = v66[v91];
        int v95;
        v95 = blockIdx.x;
        int v96;
        v96 = v95 * 256;
        int v97;
        v97 = v96 + v89;
        assert("Tensor range check" && 0 <= v72 && v72 < 64);
        int v98;
        v98 = 4 * v72;
        float v99[4];
        int v100[4];
        int v101;
        v101 = 0;
        while (while_method_1(v101)){
            assert("Tensor range check" && 0 <= v101 && v101 < 1);
            int v103;
            v103 = 4 * v101;
            assert("Tensor range check" && 0 <= v101 && v101 < 1);
            int v104;
            v104 = 256 * v101;
            int v105;
            v105 = v104 + v98;
            int4* v106;
            v106 = reinterpret_cast<int4*>(v92 + v105);
            int4* v107;
            v107 = reinterpret_cast<int4*>(v99 + v103);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v106) % 16 == 0 && reinterpret_cast<unsigned long long>(v107) % 16 == 0);
            *v107 = *v106;
            v101 += 1 ;
        }
        int v108;
        v108 = 0;
        while (while_method_1(v108)){
            int v110;
            v110 = 0;
            while (while_method_0(v110)){
                bool v112;
                v112 = 0 <= v110;
                bool v114;
                if (v112){
                    bool v113;
                    v113 = v110 < 4;
                    v114 = v113;
                } else {
                    v114 = false;
                }
                bool v115;
                v115 = v114 == false;
                if (v115){
                    assert("The indices should be inside the range of the dimension." && v114);
                } else {
                }
                bool v117;
                v117 = 0 <= v72;
                bool v119;
                if (v117){
                    bool v118;
                    v118 = v72 < 64;
                    v119 = v118;
                } else {
                    v119 = false;
                }
                bool v120;
                v120 = v119 == false;
                if (v120){
                    assert("The indices should be inside the range of the dimension." && v119);
                } else {
                }
                int v122;
                v122 = v72 * 4;
                int v123;
                v123 = v110 + v122;
                bool v124;
                v124 = 0 <= v108;
                bool v126;
                if (v124){
                    bool v125;
                    v125 = v108 < 1;
                    v126 = v125;
                } else {
                    v126 = false;
                }
                bool v127;
                v127 = v126 == false;
                if (v127){
                    assert("The indices should be inside the range of the dimension." && v126);
                } else {
                }
                int v129;
                v129 = v108 * 256;
                int v130;
                v130 = v123 + v129;
                assert("Tensor range check" && 0 <= v108 && v108 < 1);
                assert("Tensor range check" && 0 <= v110 && v110 < 4);
                int v131;
                v131 = 4 * v108;
                int v132;
                v132 = v131 + v110;
                v100[v132] = v130;
                v110 += 1 ;
            }
            v108 += 1 ;
        }
        int v133[4];
        int v134[4];
        int v135;
        v135 = 0;
        while (while_method_1(v135)){
            int v137;
            v137 = 0;
            while (while_method_0(v137)){
                assert("Tensor range check" && 0 <= v135 && v135 < 1);
                assert("Tensor range check" && 0 <= v137 && v137 < 4);
                int v139;
                v139 = 4 * v135;
                int v140;
                v140 = v139 + v137;
                int v141;
                v141 = v100[v140];
                assert("Tensor range check" && 0 <= v135 && v135 < 1);
                assert("Tensor range check" && 0 <= v137 && v137 < 4);
                v133[v140] = v97;
                v134[v140] = v141;
                v137 += 1 ;
            }
            v135 += 1 ;
        }
        int v142;
        v142 = 0;
        while (while_method_1(v142)){
            assert("Tensor range check" && 0 <= v142 && v142 < 1);
            int v144;
            v144 = 256 * v142;
            int v145;
            v145 = v144 + v98;
            assert("Tensor range check" && 0 <= v142 && v142 < 1);
            int v146;
            v146 = 4 * v142;
            int4* v147;
            v147 = reinterpret_cast<int4*>(v133 + v146);
            int4* v148;
            v148 = reinterpret_cast<int4*>(v93 + v145);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v147) % 16 == 0 && reinterpret_cast<unsigned long long>(v148) % 16 == 0);
            *v148 = *v147;
            int4* v149;
            v149 = reinterpret_cast<int4*>(v134 + v146);
            int4* v150;
            v150 = reinterpret_cast<int4*>(v94 + v145);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v149) % 16 == 0 && reinterpret_cast<unsigned long long>(v150) % 16 == 0);
            *v150 = *v149;
            v142 += 1 ;
        }
        assert("Tensor range check" && 0 <= v89 && v89 < 256);
        v77 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v68 && v68 < 256);
    __syncthreads();
    float * v151;
    v151 = v1+v12;
    unsigned long long v153;
    v153 = v45 + 1024ull;
    bool v154;
    v154 = v153 <= 98304ull;
    bool v155;
    v155 = v154 == false;
    if (v155){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v154);
    } else {
    }
    extern __shared__ unsigned char v157[];
    bool v158;
    v158 = v153 <= v153;
    bool v159;
    v159 = v158 == false;
    if (v159){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v158);
    } else {
    }
    float * * v161;
    v161 = reinterpret_cast<float * *>(&v157[0ull]);
    int * v163;
    v163 = reinterpret_cast<int *>(&v157[v45]);
    int v165;
    v165 = threadIdx.x;
    assert("Tensor range check" && 0 <= v165 && v165 < 256);
    v161[v165] = v151;
    __syncthreads();
    bool v166;
    v166 = 0 <= v165;
    bool v167;
    v167 = v166 == false;
    if (v167){
        assert("The index needs to be zero or positive." && v166);
    } else {
    }
    int v169;
    v169 = v165 % 64;
    int v170;
    v170 = v165 / 64;
    bool v171;
    v171 = v170 < 4;
    bool v172;
    v172 = v171 == false;
    if (v172){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v171);
    } else {
    }
    assert("Tensor range check" && 0 <= v170 && v170 < 4);
    int v174;
    v174 = 0;
    while (while_method_2(v174)){
        bool v176;
        v176 = 0 <= v170;
        bool v177;
        v177 = v176 && v171;
        bool v178;
        v178 = v177 == false;
        if (v178){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v177);
        } else {
        }
        bool v180;
        v180 = 0 <= v174;
        bool v182;
        if (v180){
            bool v181;
            v181 = v174 < 64;
            v182 = v181;
        } else {
            v182 = false;
        }
        bool v183;
        v183 = v182 == false;
        if (v183){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v182);
        } else {
        }
        int v185;
        v185 = v174 * 4;
        int v186;
        v186 = v185 + v170;
        assert("Tensor range check" && 0 <= v174 && v174 < 64);
        int v187;
        v187 = 4 * v174;
        int v188;
        v188 = v187 + v170;
        float * v189;
        v189 = v161[v188];
        int v190;
        v190 = blockIdx.x;
        int v191;
        v191 = v190 * 256;
        int v192;
        v192 = v191 + v186;
        assert("Tensor range check" && 0 <= v169 && v169 < 64);
        int v193;
        v193 = 4 * v169;
        float v194[4];
        int v195[4];
        int v196;
        v196 = 0;
        while (while_method_1(v196)){
            assert("Tensor range check" && 0 <= v196 && v196 < 1);
            int v198;
            v198 = 4 * v196;
            assert("Tensor range check" && 0 <= v196 && v196 < 1);
            int v199;
            v199 = 256 * v196;
            int v200;
            v200 = v199 + v193;
            int4* v201;
            v201 = reinterpret_cast<int4*>(v189 + v200);
            int4* v202;
            v202 = reinterpret_cast<int4*>(v194 + v198);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v201) % 16 == 0 && reinterpret_cast<unsigned long long>(v202) % 16 == 0);
            *v202 = *v201;
            v196 += 1 ;
        }
        int v203;
        v203 = 0;
        while (while_method_1(v203)){
            int v205;
            v205 = 0;
            while (while_method_0(v205)){
                bool v207;
                v207 = 0 <= v205;
                bool v209;
                if (v207){
                    bool v208;
                    v208 = v205 < 4;
                    v209 = v208;
                } else {
                    v209 = false;
                }
                bool v210;
                v210 = v209 == false;
                if (v210){
                    assert("The indices should be inside the range of the dimension." && v209);
                } else {
                }
                bool v212;
                v212 = 0 <= v169;
                bool v214;
                if (v212){
                    bool v213;
                    v213 = v169 < 64;
                    v214 = v213;
                } else {
                    v214 = false;
                }
                bool v215;
                v215 = v214 == false;
                if (v215){
                    assert("The indices should be inside the range of the dimension." && v214);
                } else {
                }
                int v217;
                v217 = v169 * 4;
                int v218;
                v218 = v205 + v217;
                bool v219;
                v219 = 0 <= v203;
                bool v221;
                if (v219){
                    bool v220;
                    v220 = v203 < 1;
                    v221 = v220;
                } else {
                    v221 = false;
                }
                bool v222;
                v222 = v221 == false;
                if (v222){
                    assert("The indices should be inside the range of the dimension." && v221);
                } else {
                }
                int v224;
                v224 = v203 * 256;
                int v225;
                v225 = v218 + v224;
                assert("Tensor range check" && 0 <= v203 && v203 < 1);
                assert("Tensor range check" && 0 <= v205 && v205 < 4);
                int v226;
                v226 = 4 * v203;
                int v227;
                v227 = v226 + v205;
                v195[v227] = v225;
                v205 += 1 ;
            }
            v203 += 1 ;
        }
        int v228;
        v228 = 0;
        while (while_method_1(v228)){
            assert("Tensor range check" && 0 <= v228 && v228 < 1);
            assert("Tensor range check" && 0 <= v228 && v228 < 1);
            v228 += 1 ;
        }
        assert("Tensor range check" && 0 <= v186 && v186 < 256);
        v163[v186] = v192;
        v174 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v165 && v165 < 256);
    int v230;
    v230 = v163[v165];
    __syncthreads();
    int v231;
    v231 = threadIdx.x;
    int v232;
    v232 = blockIdx.x;
    int v233;
    v233 = v232 * 256;
    int v234;
    v234 = v231 + v233;
    assert("Tensor range check" && 0 <= v234 && v234 < 6144);
    v4[v234] = v230;
    float * v235;
    v235 = v1+v12;
    float * v237;
    v237 = v6+v32;
    unsigned long long v239;
    v239 = v45 + v41;
    bool v240;
    v240 = v239 <= 98304ull;
    bool v241;
    v241 = v240 == false;
    if (v241){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v240);
    } else {
    }
    extern __shared__ unsigned char v243[];
    bool v244;
    v244 = v239 <= v239;
    bool v245;
    v245 = v244 == false;
    if (v245){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v244);
    } else {
    }
    float * * v247;
    v247 = reinterpret_cast<float * *>(&v243[0ull]);
    float * * v249;
    v249 = reinterpret_cast<float * *>(&v243[v45]);
    int v251;
    v251 = threadIdx.x;
    assert("Tensor range check" && 0 <= v251 && v251 < 256);
    v247[v251] = v235;
    v249[v251] = v237;
    __syncthreads();
    bool v252;
    v252 = 0 <= v251;
    bool v253;
    v253 = v252 == false;
    if (v253){
        assert("The index needs to be zero or positive." && v252);
    } else {
    }
    int v255;
    v255 = v251 % 64;
    int v256;
    v256 = v251 / 64;
    bool v257;
    v257 = v256 < 4;
    bool v258;
    v258 = v257 == false;
    if (v258){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v257);
    } else {
    }
    assert("Tensor range check" && 0 <= v256 && v256 < 4);
    int v260;
    v260 = 0;
    while (while_method_2(v260)){
        bool v262;
        v262 = 0 <= v256;
        bool v263;
        v263 = v262 && v257;
        bool v264;
        v264 = v263 == false;
        if (v264){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v263);
        } else {
        }
        bool v266;
        v266 = 0 <= v260;
        bool v268;
        if (v266){
            bool v267;
            v267 = v260 < 64;
            v268 = v267;
        } else {
            v268 = false;
        }
        bool v269;
        v269 = v268 == false;
        if (v269){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v268);
        } else {
        }
        int v271;
        v271 = v260 * 4;
        int v272;
        v272 = v271 + v256;
        assert("Tensor range check" && 0 <= v260 && v260 < 64);
        int v273;
        v273 = 4 * v260;
        int v274;
        v274 = v273 + v256;
        float * v275;
        v275 = v247[v274];
        float * v276;
        v276 = v249[v274];
        int v277;
        v277 = blockIdx.x;
        int v278;
        v278 = v277 * 256;
        int v279;
        v279 = v278 + v272;
        assert("Tensor range check" && 0 <= v255 && v255 < 64);
        int v280;
        v280 = 4 * v255;
        float v281[4];
        int v282[4];
        int v283;
        v283 = 0;
        while (while_method_1(v283)){
            assert("Tensor range check" && 0 <= v283 && v283 < 1);
            int v285;
            v285 = 4 * v283;
            assert("Tensor range check" && 0 <= v283 && v283 < 1);
            int v286;
            v286 = 256 * v283;
            int v287;
            v287 = v286 + v280;
            int4* v288;
            v288 = reinterpret_cast<int4*>(v275 + v287);
            int4* v289;
            v289 = reinterpret_cast<int4*>(v281 + v285);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v288) % 16 == 0 && reinterpret_cast<unsigned long long>(v289) % 16 == 0);
            *v289 = *v288;
            v283 += 1 ;
        }
        int v290;
        v290 = 0;
        while (while_method_1(v290)){
            int v292;
            v292 = 0;
            while (while_method_0(v292)){
                bool v294;
                v294 = 0 <= v292;
                bool v296;
                if (v294){
                    bool v295;
                    v295 = v292 < 4;
                    v296 = v295;
                } else {
                    v296 = false;
                }
                bool v297;
                v297 = v296 == false;
                if (v297){
                    assert("The indices should be inside the range of the dimension." && v296);
                } else {
                }
                bool v299;
                v299 = 0 <= v255;
                bool v301;
                if (v299){
                    bool v300;
                    v300 = v255 < 64;
                    v301 = v300;
                } else {
                    v301 = false;
                }
                bool v302;
                v302 = v301 == false;
                if (v302){
                    assert("The indices should be inside the range of the dimension." && v301);
                } else {
                }
                int v304;
                v304 = v255 * 4;
                int v305;
                v305 = v292 + v304;
                bool v306;
                v306 = 0 <= v290;
                bool v308;
                if (v306){
                    bool v307;
                    v307 = v290 < 1;
                    v308 = v307;
                } else {
                    v308 = false;
                }
                bool v309;
                v309 = v308 == false;
                if (v309){
                    assert("The indices should be inside the range of the dimension." && v308);
                } else {
                }
                int v311;
                v311 = v290 * 256;
                int v312;
                v312 = v305 + v311;
                assert("Tensor range check" && 0 <= v290 && v290 < 1);
                assert("Tensor range check" && 0 <= v292 && v292 < 4);
                int v313;
                v313 = 4 * v290;
                int v314;
                v314 = v313 + v292;
                v282[v314] = v312;
                v292 += 1 ;
            }
            v290 += 1 ;
        }
        int v315;
        v315 = 0;
        while (while_method_1(v315)){
            assert("Tensor range check" && 0 <= v315 && v315 < 1);
            int v317;
            v317 = 256 * v315;
            int v318;
            v318 = v317 + v280;
            assert("Tensor range check" && 0 <= v315 && v315 < 1);
            int v319;
            v319 = 4 * v315;
            int4* v320;
            v320 = reinterpret_cast<int4*>(v281 + v319);
            int4* v321;
            v321 = reinterpret_cast<int4*>(v276 + v318);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v320) % 16 == 0 && reinterpret_cast<unsigned long long>(v321) % 16 == 0);
            *v321 = *v320;
            v315 += 1 ;
        }
        assert("Tensor range check" && 0 <= v272 && v272 < 256);
        v260 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v251 && v251 < 256);
    __syncthreads();
    float * v322;
    v322 = v1+v12;
    float * v324;
    v324 = v7+v22;
    if (v241){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v240);
    } else {
    }
    extern __shared__ unsigned char v327[];
    if (v245){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v244);
    } else {
    }
    float * * v329;
    v329 = reinterpret_cast<float * *>(&v327[0ull]);
    float * * v331;
    v331 = reinterpret_cast<float * *>(&v327[v45]);
    int v333;
    v333 = threadIdx.x;
    assert("Tensor range check" && 0 <= v333 && v333 < 256);
    v329[v333] = v322;
    v331[v333] = v324;
    __syncthreads();
    bool v334;
    v334 = 0 <= v333;
    bool v335;
    v335 = v334 == false;
    if (v335){
        assert("The index needs to be zero or positive." && v334);
    } else {
    }
    int v337;
    v337 = v333 % 64;
    int v338;
    v338 = v333 / 64;
    bool v339;
    v339 = v338 < 4;
    bool v340;
    v340 = v339 == false;
    if (v340){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v339);
    } else {
    }
    assert("Tensor range check" && 0 <= v338 && v338 < 4);
    int v342;
    v342 = 0;
    while (while_method_2(v342)){
        bool v344;
        v344 = 0 <= v338;
        bool v345;
        v345 = v344 && v339;
        bool v346;
        v346 = v345 == false;
        if (v346){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v345);
        } else {
        }
        bool v348;
        v348 = 0 <= v342;
        bool v350;
        if (v348){
            bool v349;
            v349 = v342 < 64;
            v350 = v349;
        } else {
            v350 = false;
        }
        bool v351;
        v351 = v350 == false;
        if (v351){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v350);
        } else {
        }
        int v353;
        v353 = v342 * 4;
        int v354;
        v354 = v353 + v338;
        assert("Tensor range check" && 0 <= v342 && v342 < 64);
        int v355;
        v355 = 4 * v342;
        int v356;
        v356 = v355 + v338;
        float * v357;
        v357 = v329[v356];
        float * v358;
        v358 = v331[v356];
        int v359;
        v359 = blockIdx.x;
        int v360;
        v360 = v359 * 256;
        int v361;
        v361 = v360 + v354;
        assert("Tensor range check" && 0 <= v337 && v337 < 64);
        int v362;
        v362 = 4 * v337;
        float v363[4];
        int v364[4];
        int v365;
        v365 = 0;
        while (while_method_1(v365)){
            assert("Tensor range check" && 0 <= v365 && v365 < 1);
            int v367;
            v367 = 4 * v365;
            assert("Tensor range check" && 0 <= v365 && v365 < 1);
            int v368;
            v368 = 256 * v365;
            int v369;
            v369 = v368 + v362;
            int4* v370;
            v370 = reinterpret_cast<int4*>(v357 + v369);
            int4* v371;
            v371 = reinterpret_cast<int4*>(v363 + v367);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v370) % 16 == 0 && reinterpret_cast<unsigned long long>(v371) % 16 == 0);
            *v371 = *v370;
            v365 += 1 ;
        }
        int v372;
        v372 = 0;
        while (while_method_1(v372)){
            int v374;
            v374 = 0;
            while (while_method_0(v374)){
                bool v376;
                v376 = 0 <= v374;
                bool v378;
                if (v376){
                    bool v377;
                    v377 = v374 < 4;
                    v378 = v377;
                } else {
                    v378 = false;
                }
                bool v379;
                v379 = v378 == false;
                if (v379){
                    assert("The indices should be inside the range of the dimension." && v378);
                } else {
                }
                bool v381;
                v381 = 0 <= v337;
                bool v383;
                if (v381){
                    bool v382;
                    v382 = v337 < 64;
                    v383 = v382;
                } else {
                    v383 = false;
                }
                bool v384;
                v384 = v383 == false;
                if (v384){
                    assert("The indices should be inside the range of the dimension." && v383);
                } else {
                }
                int v386;
                v386 = v337 * 4;
                int v387;
                v387 = v374 + v386;
                bool v388;
                v388 = 0 <= v372;
                bool v390;
                if (v388){
                    bool v389;
                    v389 = v372 < 1;
                    v390 = v389;
                } else {
                    v390 = false;
                }
                bool v391;
                v391 = v390 == false;
                if (v391){
                    assert("The indices should be inside the range of the dimension." && v390);
                } else {
                }
                int v393;
                v393 = v372 * 256;
                int v394;
                v394 = v387 + v393;
                assert("Tensor range check" && 0 <= v372 && v372 < 1);
                assert("Tensor range check" && 0 <= v374 && v374 < 4);
                int v395;
                v395 = 4 * v372;
                int v396;
                v396 = v395 + v374;
                v364[v396] = v394;
                v374 += 1 ;
            }
            v372 += 1 ;
        }
        bool v397[4];
        int v398;
        v398 = 0;
        while (while_method_1(v398)){
            int v400;
            v400 = 0;
            while (while_method_0(v400)){
                assert("Tensor range check" && 0 <= v398 && v398 < 1);
                assert("Tensor range check" && 0 <= v400 && v400 < 4);
                int v402;
                v402 = 4 * v398;
                int v403;
                v403 = v402 + v400;
                float v404;
                v404 = v363[v403];
                int v405;
                v405 = v364[v403];
                bool v406;
                v406 = v405 < 3;
                assert("Tensor range check" && 0 <= v398 && v398 < 1);
                assert("Tensor range check" && 0 <= v400 && v400 < 4);
                v397[v403] = v406;
                v400 += 1 ;
            }
            v398 += 1 ;
        }
        float v407[4];
        int v408;
        v408 = 0;
        while (while_method_1(v408)){
            int v410;
            v410 = 0;
            while (while_method_0(v410)){
                assert("Tensor range check" && 0 <= v408 && v408 < 1);
                assert("Tensor range check" && 0 <= v410 && v410 < 4);
                int v412;
                v412 = 4 * v408;
                int v413;
                v413 = v412 + v410;
                float v414;
                v414 = v363[v413];
                bool v415;
                v415 = v397[v413];
                float v418;
                if (v415){
                    bool v416;
                    v416 = 0.0f >= v414;
                    if (v416){
                        v418 = 0.0f;
                    } else {
                        v418 = v414;
                    }
                } else {
                    v418 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v408 && v408 < 1);
                assert("Tensor range check" && 0 <= v410 && v410 < 4);
                v407[v413] = v418;
                v410 += 1 ;
            }
            v408 += 1 ;
        }
        float v419;
        v419 = 0.0f;
        int v420;
        v420 = 0;
        while (while_method_1(v420)){
            int v422;
            v422 = 0;
            while (while_method_0(v422)){
                assert("Tensor range check" && 0 <= v420 && v420 < 1);
                assert("Tensor range check" && 0 <= v422 && v422 < 4);
                int v424;
                v424 = 4 * v420;
                int v425;
                v425 = v424 + v422;
                float v426;
                v426 = v407[v425];
                float v427;
                v427 = v419 + v426;
                v419 = v427;
                v422 += 1 ;
            }
            v420 += 1 ;
        }
        auto v428 = cooperative_groups::coalesced_threads();
        Closure0 v429{};
        float v430;
        v430 = cooperative_groups::reduce(v428, v419, v429);
        int v431;
        v431 = threadIdx.x;
        int v432;
        v432 = v431 / 32;
        unsigned long long v433;
        v433 = v239 + 16ull;
        unsigned long long v434;
        v434 = v433 - 1ull;
        unsigned long long v435;
        v435 = v434 % 16ull;
        unsigned long long v436;
        v436 = v434 - v435;
        unsigned long long v437;
        v437 = v436 + 32ull;
        bool v438;
        v438 = v437 <= 98304ull;
        bool v439;
        v439 = v438 == false;
        if (v439){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v438);
        } else {
        }
        extern __shared__ unsigned char v441[];
        bool v442;
        v442 = v437 <= v437;
        bool v443;
        v443 = v442 == false;
        if (v443){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v442);
        } else {
        }
        float * v445;
        v445 = reinterpret_cast<float *>(&v441[v436]);
        bool v447;
        v447 = 0 <= v432;
        bool v448;
        v448 = v447 == false;
        if (v448){
            assert("The index needs to be zero or positive." && v447);
        } else {
        }
        int v450;
        v450 = v432 % 2;
        int v451;
        v451 = v432 / 2;
        bool v452;
        v452 = v451 < 4;
        bool v453;
        v453 = v452 == false;
        if (v453){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v452);
        } else {
        }
        assert("Tensor range check" && 0 <= v451 && v451 < 4);
        assert("Tensor range check" && 0 <= v450 && v450 < 2);
        int v455;
        v455 = 2 * v451;
        int v456;
        v456 = v455 + v450;
        v445[v456] = v430;
        int v457;
        v457 = v451 + 1;
        bool v458;
        v458 = v457 < 16;
        bool v459;
        v459 = v458 == false;
        if (v459){
            assert("The barrier_id has to be less than 16." && v458);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v457), "r"(64));
        int v461;
        v461 = threadIdx.x;
        int v462;
        v462 = v461 % 32;
        bool v463;
        v463 = v462 < 2;
        float v466;
        if (v463){
            assert("Tensor range check" && 0 <= v451 && v451 < 4);
            assert("Tensor range check" && 0 <= v462 && v462 < 2);
            int v464;
            v464 = v455 + v462;
            float v465;
            v465 = v445[v464];
            v466 = v465;
        } else {
            v466 = 0.0f;
        }
        __syncthreads();
        float v467;
        v467 = cooperative_groups::reduce(v428, v466, v429);
        int v468[4];
        int v469;
        v469 = 0;
        while (while_method_1(v469)){
            int v471;
            v471 = 0;
            while (while_method_0(v471)){
                assert("Tensor range check" && 0 <= v469 && v469 < 1);
                assert("Tensor range check" && 0 <= v471 && v471 < 4);
                int v473;
                v473 = 4 * v469;
                int v474;
                v474 = v473 + v471;
                bool v475;
                v475 = v397[v474];
                int v476;
                if (v475){
                    v476 = 1;
                } else {
                    v476 = 0;
                }
                assert("Tensor range check" && 0 <= v469 && v469 < 1);
                assert("Tensor range check" && 0 <= v471 && v471 < 4);
                v468[v474] = v476;
                v471 += 1 ;
            }
            v469 += 1 ;
        }
        int v477;
        v477 = 0;
        int v478;
        v478 = 0;
        while (while_method_1(v478)){
            int v480;
            v480 = 0;
            while (while_method_0(v480)){
                assert("Tensor range check" && 0 <= v478 && v478 < 1);
                assert("Tensor range check" && 0 <= v480 && v480 < 4);
                int v482;
                v482 = 4 * v478;
                int v483;
                v483 = v482 + v480;
                int v484;
                v484 = v468[v483];
                int v485;
                v485 = v477 + v484;
                v477 = v485;
                v480 += 1 ;
            }
            v478 += 1 ;
        }
        auto v486 = cooperative_groups::coalesced_threads();
        Closure1 v487{};
        int v488;
        v488 = cooperative_groups::reduce(v486, v477, v487);
        int v489;
        v489 = threadIdx.x;
        int v490;
        v490 = v489 / 32;
        if (v439){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v438);
        } else {
        }
        extern __shared__ unsigned char v492[];
        if (v443){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v442);
        } else {
        }
        int * v494;
        v494 = reinterpret_cast<int *>(&v492[v436]);
        bool v496;
        v496 = 0 <= v490;
        bool v497;
        v497 = v496 == false;
        if (v497){
            assert("The index needs to be zero or positive." && v496);
        } else {
        }
        int v499;
        v499 = v490 % 2;
        int v500;
        v500 = v490 / 2;
        bool v501;
        v501 = v500 < 4;
        bool v502;
        v502 = v501 == false;
        if (v502){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v501);
        } else {
        }
        assert("Tensor range check" && 0 <= v500 && v500 < 4);
        assert("Tensor range check" && 0 <= v499 && v499 < 2);
        int v504;
        v504 = 2 * v500;
        int v505;
        v505 = v504 + v499;
        v494[v505] = v488;
        int v506;
        v506 = v500 + 1;
        bool v507;
        v507 = v506 < 16;
        bool v508;
        v508 = v507 == false;
        if (v508){
            assert("The barrier_id has to be less than 16." && v507);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v506), "r"(64));
        int v510;
        v510 = threadIdx.x;
        int v511;
        v511 = v510 % 32;
        bool v512;
        v512 = v511 < 2;
        int v515;
        if (v512){
            assert("Tensor range check" && 0 <= v500 && v500 < 4);
            assert("Tensor range check" && 0 <= v511 && v511 < 2);
            int v513;
            v513 = v504 + v511;
            int v514;
            v514 = v494[v513];
            v515 = v514;
        } else {
            v515 = 0;
        }
        __syncthreads();
        int v516;
        v516 = cooperative_groups::reduce(v486, v515, v487);
        float v517;
        v517 = (float)v516;
        float v518;
        v518 = 1.0f / v517;
        float v519[4];
        int v520;
        v520 = 0;
        while (while_method_1(v520)){
            int v522;
            v522 = 0;
            while (while_method_0(v522)){
                assert("Tensor range check" && 0 <= v520 && v520 < 1);
                assert("Tensor range check" && 0 <= v522 && v522 < 4);
                int v524;
                v524 = 4 * v520;
                int v525;
                v525 = v524 + v522;
                float v526;
                v526 = v407[v525];
                bool v527;
                v527 = v397[v525];
                bool v528;
                v528 = v527 == false;
                float v533;
                if (v528){
                    v533 = 0.0f;
                } else {
                    bool v529;
                    v529 = v467 == 0.0f;
                    bool v530;
                    v530 = v529 != true;
                    if (v530){
                        float v531;
                        v531 = v526 / v467;
                        v533 = v531;
                    } else {
                        v533 = v518;
                    }
                }
                assert("Tensor range check" && 0 <= v520 && v520 < 1);
                assert("Tensor range check" && 0 <= v522 && v522 < 4);
                v519[v525] = v533;
                v522 += 1 ;
            }
            v520 += 1 ;
        }
        int v534;
        v534 = 0;
        while (while_method_1(v534)){
            assert("Tensor range check" && 0 <= v534 && v534 < 1);
            int v536;
            v536 = 256 * v534;
            int v537;
            v537 = v536 + v362;
            assert("Tensor range check" && 0 <= v534 && v534 < 1);
            int v538;
            v538 = 4 * v534;
            int4* v539;
            v539 = reinterpret_cast<int4*>(v519 + v538);
            int4* v540;
            v540 = reinterpret_cast<int4*>(v358 + v537);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v539) % 16 == 0 && reinterpret_cast<unsigned long long>(v540) % 16 == 0);
            *v540 = *v539;
            v534 += 1 ;
        }
        assert("Tensor range check" && 0 <= v354 && v354 < 256);
        v342 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v333 && v333 < 256);
    __syncthreads();
    int v541;
    v541 = threadIdx.x;
    int v542;
    v542 = blockIdx.x;
    int v543;
    v543 = v542 * 256;
    int v544;
    v544 = v541 + v543;
    unsigned long long v545;
    v545 = (unsigned long long)v544;
    curandStatePhilox4_32_10_t v546;
    curand_init(12344321ull,v545,0ull,&v546);
    float * v547;
    v547 = v1+v12;
    if (v155){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v154);
    } else {
    }
    extern __shared__ unsigned char v550[];
    if (v159){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v158);
    } else {
    }
    float * * v552;
    v552 = reinterpret_cast<float * *>(&v550[0ull]);
    int * v554;
    v554 = reinterpret_cast<int *>(&v550[v45]);
    int v556;
    v556 = threadIdx.x;
    assert("Tensor range check" && 0 <= v556 && v556 < 256);
    v552[v556] = v547;
    __syncthreads();
    bool v557;
    v557 = 0 <= v556;
    bool v558;
    v558 = v557 == false;
    if (v558){
        assert("The index needs to be zero or positive." && v557);
    } else {
    }
    int v560;
    v560 = v556 % 64;
    int v561;
    v561 = v556 / 64;
    bool v562;
    v562 = v561 < 4;
    bool v563;
    v563 = v562 == false;
    if (v563){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v562);
    } else {
    }
    assert("Tensor range check" && 0 <= v561 && v561 < 4);
    int v565;
    v565 = 0;
    while (while_method_2(v565)){
        bool v567;
        v567 = 0 <= v561;
        bool v568;
        v568 = v567 && v562;
        bool v569;
        v569 = v568 == false;
        if (v569){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v568);
        } else {
        }
        bool v571;
        v571 = 0 <= v565;
        bool v573;
        if (v571){
            bool v572;
            v572 = v565 < 64;
            v573 = v572;
        } else {
            v573 = false;
        }
        bool v574;
        v574 = v573 == false;
        if (v574){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v573);
        } else {
        }
        int v576;
        v576 = v565 * 4;
        int v577;
        v577 = v576 + v561;
        assert("Tensor range check" && 0 <= v565 && v565 < 64);
        int v578;
        v578 = 4 * v565;
        int v579;
        v579 = v578 + v561;
        float * v580;
        v580 = v552[v579];
        int v581;
        v581 = blockIdx.x;
        int v582;
        v582 = v581 * 256;
        int v583;
        v583 = v582 + v577;
        assert("Tensor range check" && 0 <= v560 && v560 < 64);
        int v584;
        v584 = 4 * v560;
        float v585[4];
        int v586[4];
        int v587;
        v587 = 0;
        while (while_method_1(v587)){
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v589;
            v589 = 4 * v587;
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v590;
            v590 = 256 * v587;
            int v591;
            v591 = v590 + v584;
            int4* v592;
            v592 = reinterpret_cast<int4*>(v580 + v591);
            int4* v593;
            v593 = reinterpret_cast<int4*>(v585 + v589);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v592) % 16 == 0 && reinterpret_cast<unsigned long long>(v593) % 16 == 0);
            *v593 = *v592;
            v587 += 1 ;
        }
        int v594;
        v594 = 0;
        while (while_method_1(v594)){
            int v596;
            v596 = 0;
            while (while_method_0(v596)){
                bool v598;
                v598 = 0 <= v596;
                bool v600;
                if (v598){
                    bool v599;
                    v599 = v596 < 4;
                    v600 = v599;
                } else {
                    v600 = false;
                }
                bool v601;
                v601 = v600 == false;
                if (v601){
                    assert("The indices should be inside the range of the dimension." && v600);
                } else {
                }
                bool v603;
                v603 = 0 <= v560;
                bool v605;
                if (v603){
                    bool v604;
                    v604 = v560 < 64;
                    v605 = v604;
                } else {
                    v605 = false;
                }
                bool v606;
                v606 = v605 == false;
                if (v606){
                    assert("The indices should be inside the range of the dimension." && v605);
                } else {
                }
                int v608;
                v608 = v560 * 4;
                int v609;
                v609 = v596 + v608;
                bool v610;
                v610 = 0 <= v594;
                bool v612;
                if (v610){
                    bool v611;
                    v611 = v594 < 1;
                    v612 = v611;
                } else {
                    v612 = false;
                }
                bool v613;
                v613 = v612 == false;
                if (v613){
                    assert("The indices should be inside the range of the dimension." && v612);
                } else {
                }
                int v615;
                v615 = v594 * 256;
                int v616;
                v616 = v609 + v615;
                assert("Tensor range check" && 0 <= v594 && v594 < 1);
                assert("Tensor range check" && 0 <= v596 && v596 < 4);
                int v617;
                v617 = 4 * v594;
                int v618;
                v618 = v617 + v596;
                v586[v618] = v616;
                v596 += 1 ;
            }
            v594 += 1 ;
        }
        bool v619[4];
        int v620;
        v620 = 0;
        while (while_method_1(v620)){
            int v622;
            v622 = 0;
            while (while_method_0(v622)){
                assert("Tensor range check" && 0 <= v620 && v620 < 1);
                assert("Tensor range check" && 0 <= v622 && v622 < 4);
                int v624;
                v624 = 4 * v620;
                int v625;
                v625 = v624 + v622;
                float v626;
                v626 = v585[v625];
                int v627;
                v627 = v586[v625];
                bool v628;
                v628 = v627 < 3;
                assert("Tensor range check" && 0 <= v620 && v620 < 1);
                assert("Tensor range check" && 0 <= v622 && v622 < 4);
                v619[v625] = v628;
                v622 += 1 ;
            }
            v620 += 1 ;
        }
        float v629[4];
        int v630;
        v630 = 0;
        while (while_method_1(v630)){
            int v632;
            v632 = 0;
            while (while_method_0(v632)){
                assert("Tensor range check" && 0 <= v630 && v630 < 1);
                assert("Tensor range check" && 0 <= v632 && v632 < 4);
                int v634;
                v634 = 4 * v630;
                int v635;
                v635 = v634 + v632;
                float v636;
                v636 = v585[v635];
                bool v637;
                v637 = v619[v635];
                float v638;
                if (v637){
                    v638 = v636;
                } else {
                    v638 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v630 && v630 < 1);
                assert("Tensor range check" && 0 <= v632 && v632 < 4);
                v629[v635] = v638;
                v632 += 1 ;
            }
            v630 += 1 ;
        }
        float v639;
        v639 = 0.0f;
        int v640;
        v640 = 0;
        while (while_method_1(v640)){
            int v642;
            v642 = 0;
            while (while_method_0(v642)){
                assert("Tensor range check" && 0 <= v640 && v640 < 1);
                assert("Tensor range check" && 0 <= v642 && v642 < 4);
                int v644;
                v644 = 4 * v640;
                int v645;
                v645 = v644 + v642;
                float v646;
                v646 = v629[v645];
                float v647;
                v647 = v639 + v646;
                v639 = v647;
                v642 += 1 ;
            }
            v640 += 1 ;
        }
        auto v648 = cooperative_groups::coalesced_threads();
        Closure0 v649{};
        float v650;
        v650 = cooperative_groups::reduce(v648, v639, v649);
        int v651;
        v651 = threadIdx.x;
        int v652;
        v652 = v651 / 32;
        unsigned long long v653;
        v653 = v153 + 16ull;
        unsigned long long v654;
        v654 = v653 - 1ull;
        unsigned long long v655;
        v655 = v654 % 16ull;
        unsigned long long v656;
        v656 = v654 - v655;
        unsigned long long v657;
        v657 = v656 + 32ull;
        bool v658;
        v658 = v657 <= 98304ull;
        bool v659;
        v659 = v658 == false;
        if (v659){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v658);
        } else {
        }
        extern __shared__ unsigned char v661[];
        bool v662;
        v662 = v657 <= v657;
        bool v663;
        v663 = v662 == false;
        if (v663){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v662);
        } else {
        }
        float * v665;
        v665 = reinterpret_cast<float *>(&v661[v656]);
        bool v667;
        v667 = 0 <= v652;
        bool v668;
        v668 = v667 == false;
        if (v668){
            assert("The index needs to be zero or positive." && v667);
        } else {
        }
        int v670;
        v670 = v652 % 2;
        int v671;
        v671 = v652 / 2;
        bool v672;
        v672 = v671 < 4;
        bool v673;
        v673 = v672 == false;
        if (v673){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v672);
        } else {
        }
        assert("Tensor range check" && 0 <= v671 && v671 < 4);
        assert("Tensor range check" && 0 <= v670 && v670 < 2);
        int v675;
        v675 = 2 * v671;
        int v676;
        v676 = v675 + v670;
        v665[v676] = v650;
        int v677;
        v677 = v671 + 1;
        bool v678;
        v678 = v677 < 16;
        bool v679;
        v679 = v678 == false;
        if (v679){
            assert("The barrier_id has to be less than 16." && v678);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v677), "r"(64));
        int v681;
        v681 = threadIdx.x;
        int v682;
        v682 = v681 % 32;
        bool v683;
        v683 = v682 < 2;
        float v686;
        if (v683){
            assert("Tensor range check" && 0 <= v671 && v671 < 4);
            assert("Tensor range check" && 0 <= v682 && v682 < 2);
            int v684;
            v684 = v675 + v682;
            float v685;
            v685 = v665[v684];
            v686 = v685;
        } else {
            v686 = 0.0f;
        }
        __syncthreads();
        float v687;
        v687 = cooperative_groups::reduce(v648, v686, v649);
        int v688[4];
        int v689;
        v689 = 0;
        while (while_method_1(v689)){
            int v691;
            v691 = 0;
            while (while_method_0(v691)){
                assert("Tensor range check" && 0 <= v689 && v689 < 1);
                assert("Tensor range check" && 0 <= v691 && v691 < 4);
                int v693;
                v693 = 4 * v689;
                int v694;
                v694 = v693 + v691;
                bool v695;
                v695 = v619[v694];
                int v696;
                if (v695){
                    v696 = 1;
                } else {
                    v696 = 0;
                }
                assert("Tensor range check" && 0 <= v689 && v689 < 1);
                assert("Tensor range check" && 0 <= v691 && v691 < 4);
                v688[v694] = v696;
                v691 += 1 ;
            }
            v689 += 1 ;
        }
        int v697;
        v697 = 0;
        int v698;
        v698 = 0;
        while (while_method_1(v698)){
            int v700;
            v700 = 0;
            while (while_method_0(v700)){
                assert("Tensor range check" && 0 <= v698 && v698 < 1);
                assert("Tensor range check" && 0 <= v700 && v700 < 4);
                int v702;
                v702 = 4 * v698;
                int v703;
                v703 = v702 + v700;
                int v704;
                v704 = v688[v703];
                int v705;
                v705 = v697 + v704;
                v697 = v705;
                v700 += 1 ;
            }
            v698 += 1 ;
        }
        auto v706 = cooperative_groups::coalesced_threads();
        Closure1 v707{};
        int v708;
        v708 = cooperative_groups::reduce(v706, v697, v707);
        int v709;
        v709 = threadIdx.x;
        int v710;
        v710 = v709 / 32;
        if (v659){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v658);
        } else {
        }
        extern __shared__ unsigned char v712[];
        if (v663){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v662);
        } else {
        }
        int * v714;
        v714 = reinterpret_cast<int *>(&v712[v656]);
        bool v716;
        v716 = 0 <= v710;
        bool v717;
        v717 = v716 == false;
        if (v717){
            assert("The index needs to be zero or positive." && v716);
        } else {
        }
        int v719;
        v719 = v710 % 2;
        int v720;
        v720 = v710 / 2;
        bool v721;
        v721 = v720 < 4;
        bool v722;
        v722 = v721 == false;
        if (v722){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v721);
        } else {
        }
        assert("Tensor range check" && 0 <= v720 && v720 < 4);
        assert("Tensor range check" && 0 <= v719 && v719 < 2);
        int v724;
        v724 = 2 * v720;
        int v725;
        v725 = v724 + v719;
        v714[v725] = v708;
        int v726;
        v726 = v720 + 1;
        bool v727;
        v727 = v726 < 16;
        bool v728;
        v728 = v727 == false;
        if (v728){
            assert("The barrier_id has to be less than 16." && v727);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v726), "r"(64));
        int v730;
        v730 = threadIdx.x;
        int v731;
        v731 = v730 % 32;
        bool v732;
        v732 = v731 < 2;
        int v735;
        if (v732){
            assert("Tensor range check" && 0 <= v720 && v720 < 4);
            assert("Tensor range check" && 0 <= v731 && v731 < 2);
            int v733;
            v733 = v724 + v731;
            int v734;
            v734 = v714[v733];
            v735 = v734;
        } else {
            v735 = 0;
        }
        __syncthreads();
        int v736;
        v736 = cooperative_groups::reduce(v706, v735, v707);
        float v737;
        v737 = (float)v736;
        float v738;
        v738 = v687 / v737;
        float v739[4];
        int v740;
        v740 = 0;
        while (while_method_1(v740)){
            int v742;
            v742 = 0;
            while (while_method_0(v742)){
                assert("Tensor range check" && 0 <= v740 && v740 < 1);
                assert("Tensor range check" && 0 <= v742 && v742 < 4);
                int v744;
                v744 = 4 * v740;
                int v745;
                v745 = v744 + v742;
                float v746;
                v746 = v585[v745];
                bool v747;
                v747 = v619[v745];
                float v748;
                if (v747){
                    v748 = v746;
                } else {
                    v748 = -1.0f / 0.0f;
                }
                float v749;
                v749 = v748 - v738;
                float v750;
                v750 = exp(v749);
                bool v751;
                v751 = v750 < 1.0f / 0.0f;
                bool v752;
                v752 = v751 == false;
                if (v752){
                    assert("The softmax values must not grow too large." && v751);
                } else {
                }
                bool v754;
                v754 = isnan(v750);
                bool v755;
                v755 = v754 == false;
                bool v756;
                v756 = v755 == false;
                if (v756){
                    assert("The softmax values must not be nans." && v755);
                } else {
                }
                assert("Tensor range check" && 0 <= v740 && v740 < 1);
                assert("Tensor range check" && 0 <= v742 && v742 < 4);
                v739[v745] = v750;
                v742 += 1 ;
            }
            v740 += 1 ;
        }
        float v758;
        v758 = 0.0f;
        int v759;
        v759 = 0;
        while (while_method_1(v759)){
            int v761;
            v761 = 0;
            while (while_method_0(v761)){
                assert("Tensor range check" && 0 <= v759 && v759 < 1);
                assert("Tensor range check" && 0 <= v761 && v761 < 4);
                int v763;
                v763 = 4 * v759;
                int v764;
                v764 = v763 + v761;
                float v765;
                v765 = v739[v764];
                float v766;
                v766 = v758 + v765;
                v758 = v766;
                v761 += 1 ;
            }
            v759 += 1 ;
        }
        auto v767 = cooperative_groups::coalesced_threads();
        float v768;
        v768 = cooperative_groups::reduce(v767, v758, v649);
        int v769;
        v769 = threadIdx.x;
        int v770;
        v770 = v769 / 32;
        if (v659){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v658);
        } else {
        }
        extern __shared__ unsigned char v772[];
        if (v663){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v662);
        } else {
        }
        float * v774;
        v774 = reinterpret_cast<float *>(&v772[v656]);
        bool v776;
        v776 = 0 <= v770;
        bool v777;
        v777 = v776 == false;
        if (v777){
            assert("The index needs to be zero or positive." && v776);
        } else {
        }
        int v779;
        v779 = v770 % 2;
        int v780;
        v780 = v770 / 2;
        bool v781;
        v781 = v780 < 4;
        bool v782;
        v782 = v781 == false;
        if (v782){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v781);
        } else {
        }
        assert("Tensor range check" && 0 <= v780 && v780 < 4);
        assert("Tensor range check" && 0 <= v779 && v779 < 2);
        int v784;
        v784 = 2 * v780;
        int v785;
        v785 = v784 + v779;
        v774[v785] = v768;
        int v786;
        v786 = v780 + 1;
        bool v787;
        v787 = v786 < 16;
        bool v788;
        v788 = v787 == false;
        if (v788){
            assert("The barrier_id has to be less than 16." && v787);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v786), "r"(64));
        int v790;
        v790 = threadIdx.x;
        int v791;
        v791 = v790 % 32;
        bool v792;
        v792 = v791 < 2;
        float v795;
        if (v792){
            assert("Tensor range check" && 0 <= v780 && v780 < 4);
            assert("Tensor range check" && 0 <= v791 && v791 < 2);
            int v793;
            v793 = v784 + v791;
            float v794;
            v794 = v774[v793];
            v795 = v794;
        } else {
            v795 = 0.0f;
        }
        __syncthreads();
        float v796;
        v796 = cooperative_groups::reduce(v767, v795, v649);
        float v797[4];
        int v798;
        v798 = 0;
        while (while_method_1(v798)){
            int v800;
            v800 = 0;
            while (while_method_0(v800)){
                assert("Tensor range check" && 0 <= v798 && v798 < 1);
                assert("Tensor range check" && 0 <= v800 && v800 < 4);
                int v802;
                v802 = 4 * v798;
                int v803;
                v803 = v802 + v800;
                float v804;
                v804 = v739[v803];
                float v805;
                v805 = v804 / v796;
                assert("Tensor range check" && 0 <= v798 && v798 < 1);
                assert("Tensor range check" && 0 <= v800 && v800 < 4);
                v797[v803] = v805;
                v800 += 1 ;
            }
            v798 += 1 ;
        }
        float v806[4];
        float v807;
        v807 = 0.0f;
        int v808;
        v808 = 0;
        while (while_method_1(v808)){
            assert("Tensor range check" && 0 <= v808 && v808 < 1);
            int v810;
            v810 = 4 * v808;
            assert("Tensor range check" && 0 <= v808 && v808 < 1);
            float v811;
            v811 = 0.0f;
            int v812;
            v812 = 0;
            while (while_method_0(v812)){
                assert("Tensor range check" && 0 <= v812 && v812 < 4);
                int v814;
                v814 = v812 + v810;
                float v815;
                v815 = v797[v814];
                float v816;
                v816 = v811 + v815;
                v811 = v816;
                v812 += 1 ;
            }
            auto v817 = cooperative_groups::coalesced_threads();
            int v818;
            v818 = threadIdx.x;
            int v819;
            v819 = v818 / 32;
            if (v659){
                assert("The dynamic shared memory is insufficient to allocate the tensor." && v658);
            } else {
            }
            extern __shared__ unsigned char v821[];
            if (v663){
                assert("The length of the partition has to be less than or equal to the length of the base array." && v662);
            } else {
            }
            float * v823;
            v823 = reinterpret_cast<float *>(&v821[v656]);
            Closure2 v825{};
            float v826;
            v826 = cooperative_groups::inclusive_scan(v817, v811, v825);
            float v827;
            v827 = v817.shfl_up(v826,1);
            bool v828;
            v828 = v817.thread_rank() == 0;
            float v829;
            if (v828){
                v829 = 0.0f;
            } else {
                v829 = v827;
            }
            float v830;
            v830 = v817.shfl(v826,v817.num_threads()-1);
            bool v831;
            v831 = 0 <= v819;
            bool v832;
            v832 = v831 == false;
            if (v832){
                assert("The index needs to be zero or positive." && v831);
            } else {
            }
            int v834;
            v834 = v819 % 2;
            int v835;
            v835 = v819 / 2;
            bool v836;
            v836 = v835 < 4;
            bool v837;
            v837 = v836 == false;
            if (v837){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v836);
            } else {
            }
            assert("Tensor range check" && 0 <= v835 && v835 < 4);
            assert("Tensor range check" && 0 <= v834 && v834 < 2);
            int v839;
            v839 = 2 * v835;
            int v840;
            v840 = v839 + v834;
            v823[v840] = v830;
            int v841;
            v841 = v835 + 1;
            bool v842;
            v842 = v841 < 16;
            bool v843;
            v843 = v842 == false;
            if (v843){
                assert("The barrier_id has to be less than 16." && v842);
            } else {
            }
            asm("barrier.cta.sync %0, %1;" :: "r"(v841), "r"(64));
            int v845;
            v845 = threadIdx.x;
            int v846;
            v846 = v845 % 32;
            bool v847;
            v847 = v846 < 2;
            float v850;
            if (v847){
                assert("Tensor range check" && 0 <= v835 && v835 < 4);
                assert("Tensor range check" && 0 <= v846 && v846 < 2);
                int v848;
                v848 = v839 + v846;
                float v849;
                v849 = v823[v848];
                v850 = v849;
            } else {
                v850 = 0.0f;
            }
            __syncthreads();
            float v851;
            v851 = cooperative_groups::inclusive_scan(v817, v850, v825);
            float v852;
            v852 = v817.shfl_up(v851,1);
            bool v853;
            v853 = v817.thread_rank() == 0;
            float v854;
            if (v853){
                v854 = 0.0f;
            } else {
                v854 = v852;
            }
            float v855;
            v855 = v817.shfl(v851,v817.num_threads()-1);
            float v856;
            v856 = v817.shfl(v854,v834);
            float v857;
            v857 = v856 + v829;
            float v858;
            v858 = v807 + v857;
            float v859;
            v859 = v858;
            int v860;
            v860 = 0;
            while (while_method_0(v860)){
                assert("Tensor range check" && 0 <= v860 && v860 < 4);
                int v862;
                v862 = v860 + v810;
                float v863;
                v863 = v797[v862];
                float v864;
                v864 = v859 + v863;
                assert("Tensor range check" && 0 <= v860 && v860 < 4);
                v806[v862] = v864;
                v859 = v864;
                v860 += 1 ;
            }
            float v865;
            v865 = v807 + v855;
            v807 = v865;
            v808 += 1 ;
        }
        float v866[4];
        bool v867[4];
        int v868;
        v868 = 0;
        while (while_method_1(v868)){
            int v870;
            v870 = 0;
            while (while_method_0(v870)){
                assert("Tensor range check" && 0 <= v868 && v868 < 1);
                assert("Tensor range check" && 0 <= v870 && v870 < 4);
                int v872;
                v872 = 4 * v868;
                int v873;
                v873 = v872 + v870;
                float v874;
                v874 = v806[v873];
                float v875;
                v875 = v797[v873];
                bool v876;
                v876 = v875 > 0.0f;
                assert("Tensor range check" && 0 <= v868 && v868 < 1);
                assert("Tensor range check" && 0 <= v870 && v870 < 4);
                v866[v873] = v874;
                v867[v873] = v876;
                v870 += 1 ;
            }
            v868 += 1 ;
        }
        float v877; bool v878;
        Tuple0 tmp6 = Tuple0{-1.0f / 0.0f, false};
        v877 = tmp6.v0; v878 = tmp6.v1;
        int v879;
        v879 = 0;
        while (while_method_1(v879)){
            int v881;
            v881 = 0;
            while (while_method_0(v881)){
                assert("Tensor range check" && 0 <= v879 && v879 < 1);
                assert("Tensor range check" && 0 <= v881 && v881 < 4);
                int v883;
                v883 = 4 * v879;
                int v884;
                v884 = v883 + v881;
                float v885;
                v885 = v866[v884];
                bool v886;
                v886 = v867[v884];
                float v893; bool v894;
                if (v878){
                    if (v886){
                        bool v887;
                        v887 = v877 >= v885;
                        float v888;
                        if (v887){
                            v888 = v877;
                        } else {
                            v888 = v885;
                        }
                        v893 = v888; v894 = true;
                    } else {
                        v893 = v877; v894 = v878;
                    }
                } else {
                    if (v886){
                        v893 = v885; v894 = v886;
                    } else {
                        v893 = v877; v894 = v878;
                    }
                }
                v877 = v893;
                v878 = v894;
                v881 += 1 ;
            }
            v879 += 1 ;
        }
        auto v895 = cooperative_groups::coalesced_threads();
        Closure3 v896{};
        float v897; bool v898;
        Tuple0 tmp7 = cooperative_groups::reduce(v895, Tuple0{v877, v878}, v896);
        v897 = tmp7.v0; v898 = tmp7.v1;
        int v899;
        v899 = threadIdx.x;
        int v900;
        v900 = v899 / 32;
        unsigned long long v901;
        v901 = v657 + 16ull;
        unsigned long long v902;
        v902 = v901 - 1ull;
        unsigned long long v903;
        v903 = v902 % 16ull;
        unsigned long long v904;
        v904 = v902 - v903;
        unsigned long long v905;
        v905 = v904 + 8ull;
        bool v906;
        v906 = v905 <= 98304ull;
        bool v907;
        v907 = v906 == false;
        if (v907){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v906);
        } else {
        }
        extern __shared__ unsigned char v909[];
        bool v910;
        v910 = v905 <= v905;
        bool v911;
        v911 = v910 == false;
        if (v911){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v910);
        } else {
        }
        float * v913;
        v913 = reinterpret_cast<float *>(&v909[v656]);
        bool * v915;
        v915 = reinterpret_cast<bool *>(&v909[v904]);
        bool v917;
        v917 = 0 <= v900;
        bool v918;
        v918 = v917 == false;
        if (v918){
            assert("The index needs to be zero or positive." && v917);
        } else {
        }
        int v920;
        v920 = v900 % 2;
        int v921;
        v921 = v900 / 2;
        bool v922;
        v922 = v921 < 4;
        bool v923;
        v923 = v922 == false;
        if (v923){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v922);
        } else {
        }
        assert("Tensor range check" && 0 <= v921 && v921 < 4);
        assert("Tensor range check" && 0 <= v920 && v920 < 2);
        int v925;
        v925 = 2 * v921;
        int v926;
        v926 = v925 + v920;
        v913[v926] = v897;
        v915[v926] = v898;
        int v927;
        v927 = v921 + 1;
        bool v928;
        v928 = v927 < 16;
        bool v929;
        v929 = v928 == false;
        if (v929){
            assert("The barrier_id has to be less than 16." && v928);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v927), "r"(64));
        int v931;
        v931 = threadIdx.x;
        int v932;
        v932 = v931 % 32;
        bool v933;
        v933 = v932 < 2;
        float v937; bool v938;
        if (v933){
            assert("Tensor range check" && 0 <= v921 && v921 < 4);
            assert("Tensor range check" && 0 <= v932 && v932 < 2);
            int v934;
            v934 = v925 + v932;
            float v935;
            v935 = v913[v934];
            bool v936;
            v936 = v915[v934];
            v937 = v935; v938 = v936;
        } else {
            v937 = -1.0f / 0.0f; v938 = false;
        }
        __syncthreads();
        float v939; bool v940;
        Tuple0 tmp8 = cooperative_groups::reduce(v895, Tuple0{v937, v938}, v896);
        v939 = tmp8.v0; v940 = tmp8.v1;
        bool v941;
        v941 = v940 == false;
        if (v941){
            assert("The local reduce must be true." && v940);
        } else {
        }
        float v943[4];
        int v944[4];
        int v945;
        v945 = 0;
        while (while_method_1(v945)){
            int v947;
            v947 = 0;
            while (while_method_0(v947)){
                assert("Tensor range check" && 0 <= v945 && v945 < 1);
                assert("Tensor range check" && 0 <= v947 && v947 < 4);
                int v949;
                v949 = 4 * v945;
                int v950;
                v950 = v949 + v947;
                int v951;
                v951 = v586[v950];
                float v952;
                v952 = curand_uniform(&v546);
                assert("Tensor range check" && 0 <= v945 && v945 < 1);
                assert("Tensor range check" && 0 <= v947 && v947 < 4);
                v943[v950] = v952;
                v944[v950] = v951;
                v947 += 1 ;
            }
            v945 += 1 ;
        }
        float v953; int v954;
        Tuple1 tmp9 = Tuple1{0.0f, 2147483647};
        v953 = tmp9.v0; v954 = tmp9.v1;
        int v955;
        v955 = 0;
        while (while_method_1(v955)){
            int v957;
            v957 = 0;
            while (while_method_0(v957)){
                assert("Tensor range check" && 0 <= v955 && v955 < 1);
                assert("Tensor range check" && 0 <= v957 && v957 < 4);
                int v959;
                v959 = 4 * v955;
                int v960;
                v960 = v959 + v957;
                float v961;
                v961 = v943[v960];
                int v962;
                v962 = v944[v960];
                bool v963;
                v963 = v954 < v962;
                float v964; int v965;
                if (v963){
                    v964 = v953; v965 = v954;
                } else {
                    v964 = v961; v965 = v962;
                }
                v953 = v964;
                v954 = v965;
                v957 += 1 ;
            }
            v955 += 1 ;
        }
        auto v966 = cooperative_groups::coalesced_threads();
        Closure4 v967{};
        float v968; int v969;
        Tuple1 tmp10 = cooperative_groups::reduce(v966, Tuple1{v953, v954}, v967);
        v968 = tmp10.v0; v969 = tmp10.v1;
        int v970;
        v970 = threadIdx.x;
        int v971;
        v971 = v970 / 32;
        unsigned long long v972;
        v972 = v904 + 32ull;
        bool v973;
        v973 = v972 <= 98304ull;
        bool v974;
        v974 = v973 == false;
        if (v974){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v973);
        } else {
        }
        extern __shared__ unsigned char v976[];
        bool v977;
        v977 = v972 <= v972;
        bool v978;
        v978 = v977 == false;
        if (v978){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v977);
        } else {
        }
        float * v980;
        v980 = reinterpret_cast<float *>(&v976[v656]);
        int * v982;
        v982 = reinterpret_cast<int *>(&v976[v904]);
        bool v984;
        v984 = 0 <= v971;
        bool v985;
        v985 = v984 == false;
        if (v985){
            assert("The index needs to be zero or positive." && v984);
        } else {
        }
        int v987;
        v987 = v971 % 2;
        int v988;
        v988 = v971 / 2;
        bool v989;
        v989 = v988 < 4;
        bool v990;
        v990 = v989 == false;
        if (v990){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v989);
        } else {
        }
        assert("Tensor range check" && 0 <= v988 && v988 < 4);
        assert("Tensor range check" && 0 <= v987 && v987 < 2);
        int v992;
        v992 = 2 * v988;
        int v993;
        v993 = v992 + v987;
        v980[v993] = v968;
        v982[v993] = v969;
        int v994;
        v994 = v988 + 1;
        bool v995;
        v995 = v994 < 16;
        bool v996;
        v996 = v995 == false;
        if (v996){
            assert("The barrier_id has to be less than 16." && v995);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v994), "r"(64));
        int v998;
        v998 = threadIdx.x;
        int v999;
        v999 = v998 % 32;
        bool v1000;
        v1000 = v999 < 2;
        float v1004; int v1005;
        if (v1000){
            assert("Tensor range check" && 0 <= v988 && v988 < 4);
            assert("Tensor range check" && 0 <= v999 && v999 < 2);
            int v1001;
            v1001 = v992 + v999;
            float v1002;
            v1002 = v980[v1001];
            int v1003;
            v1003 = v982[v1001];
            v1004 = v1002; v1005 = v1003;
        } else {
            v1004 = 0.0f; v1005 = 2147483647;
        }
        __syncthreads();
        float v1006; int v1007;
        Tuple1 tmp11 = cooperative_groups::reduce(v966, Tuple1{v1004, v1005}, v967);
        v1006 = tmp11.v0; v1007 = tmp11.v1;
        float v1008;
        v1008 = v939 * v1006;
        int v1009[4];
        bool v1010[4];
        int v1011;
        v1011 = 0;
        while (while_method_1(v1011)){
            int v1013;
            v1013 = 0;
            while (while_method_0(v1013)){
                assert("Tensor range check" && 0 <= v1011 && v1011 < 1);
                assert("Tensor range check" && 0 <= v1013 && v1013 < 4);
                int v1015;
                v1015 = 4 * v1011;
                int v1016;
                v1016 = v1015 + v1013;
                float v1017;
                v1017 = v866[v1016];
                bool v1018;
                v1018 = v867[v1016];
                int v1019;
                v1019 = v586[v1016];
                int v1022; bool v1023;
                if (v1018){
                    float v1020;
                    v1020 = v1017 - v1008;
                    bool v1021;
                    v1021 = v1020 >= 0.0f;
                    v1022 = v1019; v1023 = v1021;
                } else {
                    v1022 = 2147483647; v1023 = false;
                }
                assert("Tensor range check" && 0 <= v1011 && v1011 < 1);
                assert("Tensor range check" && 0 <= v1013 && v1013 < 4);
                v1009[v1016] = v1022;
                v1010[v1016] = v1023;
                v1013 += 1 ;
            }
            v1011 += 1 ;
        }
        int v1024; bool v1025;
        Tuple2 tmp12 = Tuple2{2147483647, false};
        v1024 = tmp12.v0; v1025 = tmp12.v1;
        int v1026;
        v1026 = 0;
        while (while_method_1(v1026)){
            int v1028;
            v1028 = 0;
            while (while_method_0(v1028)){
                assert("Tensor range check" && 0 <= v1026 && v1026 < 1);
                assert("Tensor range check" && 0 <= v1028 && v1028 < 4);
                int v1030;
                v1030 = 4 * v1026;
                int v1031;
                v1031 = v1030 + v1028;
                int v1032;
                v1032 = v1009[v1031];
                bool v1033;
                v1033 = v1010[v1031];
                int v1040; bool v1041;
                if (v1025){
                    if (v1033){
                        bool v1034;
                        v1034 = v1024 < v1032;
                        int v1035;
                        if (v1034){
                            v1035 = v1024;
                        } else {
                            v1035 = v1032;
                        }
                        v1040 = v1035; v1041 = true;
                    } else {
                        v1040 = v1024; v1041 = v1025;
                    }
                } else {
                    if (v1033){
                        v1040 = v1032; v1041 = v1033;
                    } else {
                        v1040 = v1024; v1041 = v1025;
                    }
                }
                v1024 = v1040;
                v1025 = v1041;
                v1028 += 1 ;
            }
            v1026 += 1 ;
        }
        auto v1042 = cooperative_groups::coalesced_threads();
        Closure5 v1043{};
        int v1044; bool v1045;
        Tuple2 tmp13 = cooperative_groups::reduce(v1042, Tuple2{v1024, v1025}, v1043);
        v1044 = tmp13.v0; v1045 = tmp13.v1;
        int v1046;
        v1046 = threadIdx.x;
        int v1047;
        v1047 = v1046 / 32;
        if (v907){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v906);
        } else {
        }
        extern __shared__ unsigned char v1049[];
        if (v911){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v910);
        } else {
        }
        int * v1051;
        v1051 = reinterpret_cast<int *>(&v1049[v656]);
        bool * v1053;
        v1053 = reinterpret_cast<bool *>(&v1049[v904]);
        bool v1055;
        v1055 = 0 <= v1047;
        bool v1056;
        v1056 = v1055 == false;
        if (v1056){
            assert("The index needs to be zero or positive." && v1055);
        } else {
        }
        int v1058;
        v1058 = v1047 % 2;
        int v1059;
        v1059 = v1047 / 2;
        bool v1060;
        v1060 = v1059 < 4;
        bool v1061;
        v1061 = v1060 == false;
        if (v1061){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1060);
        } else {
        }
        assert("Tensor range check" && 0 <= v1059 && v1059 < 4);
        assert("Tensor range check" && 0 <= v1058 && v1058 < 2);
        int v1063;
        v1063 = 2 * v1059;
        int v1064;
        v1064 = v1063 + v1058;
        v1051[v1064] = v1044;
        v1053[v1064] = v1045;
        int v1065;
        v1065 = v1059 + 1;
        bool v1066;
        v1066 = v1065 < 16;
        bool v1067;
        v1067 = v1066 == false;
        if (v1067){
            assert("The barrier_id has to be less than 16." && v1066);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v1065), "r"(64));
        int v1069;
        v1069 = threadIdx.x;
        int v1070;
        v1070 = v1069 % 32;
        bool v1071;
        v1071 = v1070 < 2;
        int v1075; bool v1076;
        if (v1071){
            assert("Tensor range check" && 0 <= v1059 && v1059 < 4);
            assert("Tensor range check" && 0 <= v1070 && v1070 < 2);
            int v1072;
            v1072 = v1063 + v1070;
            int v1073;
            v1073 = v1051[v1072];
            bool v1074;
            v1074 = v1053[v1072];
            v1075 = v1073; v1076 = v1074;
        } else {
            v1075 = 2147483647; v1076 = false;
        }
        __syncthreads();
        int v1077; bool v1078;
        Tuple2 tmp14 = cooperative_groups::reduce(v1042, Tuple2{v1075, v1076}, v1043);
        v1077 = tmp14.v0; v1078 = tmp14.v1;
        bool v1079;
        v1079 = v1078 == false;
        if (v1079){
            assert("The local reduce must be true." && v1078);
        } else {
        }
        int v1081;
        v1081 = 0;
        while (while_method_1(v1081)){
            assert("Tensor range check" && 0 <= v1081 && v1081 < 1);
            assert("Tensor range check" && 0 <= v1081 && v1081 < 1);
            v1081 += 1 ;
        }
        assert("Tensor range check" && 0 <= v577 && v577 < 256);
        v554[v577] = v1077;
        v565 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v556 && v556 < 256);
    int v1083;
    v1083 = v554[v556];
    __syncthreads();
    int v1084;
    v1084 = threadIdx.x;
    int v1085;
    v1085 = blockIdx.x;
    int v1086;
    v1086 = v1085 * 256;
    int v1087;
    v1087 = v1084 + v1086;
    assert("Tensor range check" && 0 <= v1087 && v1087 < 6144);
    v5[v1087] = v1083;
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

import sys
import pathlib
options = []
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=255')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method0(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> None:
    v8 = "test_text_outputs/primitives/"
    v9 = "test3/a"
    v10 = "kernel_params.txt"
    v11 = pathlib.Path(v8,v9,v10)
    del v8, v9, v10
    v11.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v11),'w')
    del v11
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
    v17((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v17
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method2(v0 : i32) -> bool:
    v1 = v0 < 6144
    del v0
    return v1
def method3(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def method1(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "input_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method3(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 16
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method4(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "output_sample_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method5(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test3/a"
    v4 = "output_indices_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method2(v35):
        v37 = v33
        v38 = v37 >= 1024
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method3(v43):
            v45 = v33
            v46 = v45 >= 1024
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 16
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{}, {}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method6(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "output_indices_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method7(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "output_op_map.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method3(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 16
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method8(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test3/a"
    v4 = "zip_input_output_identity_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method2(v35):
        v37 = v33
        v38 = v37 >= 1024
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method3(v43):
            v45 = v33
            v46 = v45 >= 1024
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 16
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{:.6f}, {:.6f}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method9(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> None:
    v8 = "test_text_outputs/primitives/"
    v9 = "test3/b"
    v10 = "kernel_params.txt"
    v11 = pathlib.Path(v8,v9,v10)
    del v8, v9, v10
    v11.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v11),'w')
    del v11
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
    v16 = 1
    v17 = raw_module.get_function(f"entry{v16}")
    del v16
    v17.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v17((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v17
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method11(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def method10(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "input_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method11(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 256
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method12(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "output_sample_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method13(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test3/b"
    v4 = "output_indices_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method2(v35):
        v37 = v33
        v38 = v37 >= 1024
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method11(v43):
            v45 = v33
            v46 = v45 >= 1024
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 256
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{}, {}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method14(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "output_indices_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method15(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "output_op_map.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method11(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 256
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method16(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test3/b"
    v4 = "zip_input_output_identity_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method2(v35):
        v37 = v33
        v38 = v37 >= 1024
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method11(v43):
            v45 = v33
            v46 = v45 >= 1024
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 256
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{:.6f}, {:.6f}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def main_body():
    v2 = "{}\n"
    v3 = "3"
    print(v2.format(v3),end="")
    del v3
    cp.random.seed(12344321)
    v4 = cp.arange(0,98304,1,dtype=cp.float32) # type: ignore
    v5 = v4.size
    v6 = 98304 == v5
    del v5
    v7 = v6 == False
    if v7:
        v8 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v6, v8
        del v8
    else:
        pass
    del v6, v7
    v9 = cp.random.normal(0.0,1.0,98304,dtype=cp.float32) # type: ignore
    v10 = cp.empty(98304,dtype=cp.int32)
    v11 = cp.empty(98304,dtype=cp.int32)
    v12 = cp.empty(6144,dtype=cp.int32)
    v13 = cp.empty(6144,dtype=cp.int32)
    v14 = cp.empty(98304,dtype=cp.float32)
    v15 = cp.empty(98304,dtype=cp.float32)
    method0(v4, v9, v10, v11, v12, v13, v14, v15)
    method1(v4)
    del v4
    method4(v13)
    del v13
    method5(v10, v11)
    del v10, v11
    method6(v12)
    del v12
    method7(v15)
    del v15
    method8(v9, v14)
    del v9, v14
    cp.random.seed(12344321)
    v16 = cp.arange(0,1572864,1,dtype=cp.float32) # type: ignore
    v17 = v16.size
    v18 = 1572864 == v17
    del v17
    v19 = v18 == False
    if v19:
        v20 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v18, v20
        del v20
    else:
        pass
    del v18, v19
    v21 = cp.random.normal(0.0,1.0,1572864,dtype=cp.float32) # type: ignore
    v22 = cp.empty(1572864,dtype=cp.int32)
    v23 = cp.empty(1572864,dtype=cp.int32)
    v24 = cp.empty(6144,dtype=cp.int32)
    v25 = cp.empty(6144,dtype=cp.int32)
    v26 = cp.empty(1572864,dtype=cp.float32)
    v27 = cp.empty(1572864,dtype=cp.float32)
    method9(v16, v21, v22, v23, v24, v25, v26, v27)
    method10(v16)
    del v16
    method12(v25)
    del v25
    method13(v22, v23)
    del v22, v23
    method14(v24)
    del v24
    method15(v27)
    del v27
    method16(v21, v26)
    del v21, v26
    cp.cuda.get_current_stream().synchronize()
    v30 = "Done."
    print(v2.format(v30),end="")
    del v2, v30
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
