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
    assert("Tensor range check" && 0 <= v231 && v231 < 6144);
    v4[v231] = v230;
    float * v232;
    v232 = v1+v12;
    float * v234;
    v234 = v6+v32;
    unsigned long long v236;
    v236 = v45 + v41;
    bool v237;
    v237 = v236 <= 98304ull;
    bool v238;
    v238 = v237 == false;
    if (v238){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v237);
    } else {
    }
    extern __shared__ unsigned char v240[];
    bool v241;
    v241 = v236 <= v236;
    bool v242;
    v242 = v241 == false;
    if (v242){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v241);
    } else {
    }
    float * * v244;
    v244 = reinterpret_cast<float * *>(&v240[0ull]);
    float * * v246;
    v246 = reinterpret_cast<float * *>(&v240[v45]);
    int v248;
    v248 = threadIdx.x;
    assert("Tensor range check" && 0 <= v248 && v248 < 256);
    v244[v248] = v232;
    v246[v248] = v234;
    __syncthreads();
    bool v249;
    v249 = 0 <= v248;
    bool v250;
    v250 = v249 == false;
    if (v250){
        assert("The index needs to be zero or positive." && v249);
    } else {
    }
    int v252;
    v252 = v248 % 4;
    int v253;
    v253 = v248 / 4;
    bool v254;
    v254 = v253 < 64;
    bool v255;
    v255 = v254 == false;
    if (v255){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v254);
    } else {
    }
    assert("Tensor range check" && 0 <= v253 && v253 < 64);
    int v257;
    v257 = 0;
    while (while_method_0(v257)){
        bool v259;
        v259 = 0 <= v253;
        bool v260;
        v260 = v259 && v254;
        bool v261;
        v261 = v260 == false;
        if (v261){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v260);
        } else {
        }
        bool v263;
        v263 = 0 <= v257;
        bool v265;
        if (v263){
            bool v264;
            v264 = v257 < 4;
            v265 = v264;
        } else {
            v265 = false;
        }
        bool v266;
        v266 = v265 == false;
        if (v266){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v265);
        } else {
        }
        int v268;
        v268 = v257 * 64;
        int v269;
        v269 = v268 + v253;
        assert("Tensor range check" && 0 <= v257 && v257 < 4);
        int v270;
        v270 = 64 * v257;
        int v271;
        v271 = v270 + v253;
        float * v272;
        v272 = v244[v271];
        float * v273;
        v273 = v246[v271];
        int v274;
        v274 = blockIdx.x;
        int v275;
        v275 = v274 * 256;
        int v276;
        v276 = v275 + v269;
        assert("Tensor range check" && 0 <= v252 && v252 < 4);
        int v277;
        v277 = 4 * v252;
        float v278[4];
        int v279[4];
        int v280;
        v280 = 0;
        while (while_method_1(v280)){
            assert("Tensor range check" && 0 <= v280 && v280 < 1);
            int v282;
            v282 = 4 * v280;
            assert("Tensor range check" && 0 <= v280 && v280 < 1);
            int v283;
            v283 = 16 * v280;
            int v284;
            v284 = v283 + v277;
            int4* v285;
            v285 = reinterpret_cast<int4*>(v272 + v284);
            int4* v286;
            v286 = reinterpret_cast<int4*>(v278 + v282);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v285) % 16 == 0 && reinterpret_cast<unsigned long long>(v286) % 16 == 0);
            *v286 = *v285;
            v280 += 1 ;
        }
        int v287;
        v287 = 0;
        while (while_method_1(v287)){
            int v289;
            v289 = 0;
            while (while_method_0(v289)){
                bool v291;
                v291 = 0 <= v289;
                bool v293;
                if (v291){
                    bool v292;
                    v292 = v289 < 4;
                    v293 = v292;
                } else {
                    v293 = false;
                }
                bool v294;
                v294 = v293 == false;
                if (v294){
                    assert("The indices should be inside the range of the dimension." && v293);
                } else {
                }
                bool v296;
                v296 = 0 <= v252;
                bool v298;
                if (v296){
                    bool v297;
                    v297 = v252 < 4;
                    v298 = v297;
                } else {
                    v298 = false;
                }
                bool v299;
                v299 = v298 == false;
                if (v299){
                    assert("The indices should be inside the range of the dimension." && v298);
                } else {
                }
                int v301;
                v301 = v252 * 4;
                int v302;
                v302 = v289 + v301;
                bool v303;
                v303 = 0 <= v287;
                bool v305;
                if (v303){
                    bool v304;
                    v304 = v287 < 1;
                    v305 = v304;
                } else {
                    v305 = false;
                }
                bool v306;
                v306 = v305 == false;
                if (v306){
                    assert("The indices should be inside the range of the dimension." && v305);
                } else {
                }
                int v308;
                v308 = v287 * 16;
                int v309;
                v309 = v302 + v308;
                assert("Tensor range check" && 0 <= v287 && v287 < 1);
                assert("Tensor range check" && 0 <= v289 && v289 < 4);
                int v310;
                v310 = 4 * v287;
                int v311;
                v311 = v310 + v289;
                v279[v311] = v309;
                v289 += 1 ;
            }
            v287 += 1 ;
        }
        int v312;
        v312 = 0;
        while (while_method_1(v312)){
            assert("Tensor range check" && 0 <= v312 && v312 < 1);
            int v314;
            v314 = 16 * v312;
            int v315;
            v315 = v314 + v277;
            assert("Tensor range check" && 0 <= v312 && v312 < 1);
            int v316;
            v316 = 4 * v312;
            int4* v317;
            v317 = reinterpret_cast<int4*>(v278 + v316);
            int4* v318;
            v318 = reinterpret_cast<int4*>(v273 + v315);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v317) % 16 == 0 && reinterpret_cast<unsigned long long>(v318) % 16 == 0);
            *v318 = *v317;
            v312 += 1 ;
        }
        assert("Tensor range check" && 0 <= v269 && v269 < 256);
        v257 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v248 && v248 < 256);
    __syncthreads();
    float * v319;
    v319 = v1+v12;
    float * v321;
    v321 = v7+v22;
    if (v238){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v237);
    } else {
    }
    extern __shared__ unsigned char v324[];
    if (v242){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v241);
    } else {
    }
    float * * v326;
    v326 = reinterpret_cast<float * *>(&v324[0ull]);
    float * * v328;
    v328 = reinterpret_cast<float * *>(&v324[v45]);
    int v330;
    v330 = threadIdx.x;
    assert("Tensor range check" && 0 <= v330 && v330 < 256);
    v326[v330] = v319;
    v328[v330] = v321;
    __syncthreads();
    bool v331;
    v331 = 0 <= v330;
    bool v332;
    v332 = v331 == false;
    if (v332){
        assert("The index needs to be zero or positive." && v331);
    } else {
    }
    int v334;
    v334 = v330 % 4;
    int v335;
    v335 = v330 / 4;
    bool v336;
    v336 = v335 < 64;
    bool v337;
    v337 = v336 == false;
    if (v337){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v336);
    } else {
    }
    assert("Tensor range check" && 0 <= v335 && v335 < 64);
    int v339;
    v339 = 0;
    while (while_method_0(v339)){
        bool v341;
        v341 = 0 <= v335;
        bool v342;
        v342 = v341 && v336;
        bool v343;
        v343 = v342 == false;
        if (v343){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v342);
        } else {
        }
        bool v345;
        v345 = 0 <= v339;
        bool v347;
        if (v345){
            bool v346;
            v346 = v339 < 4;
            v347 = v346;
        } else {
            v347 = false;
        }
        bool v348;
        v348 = v347 == false;
        if (v348){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v347);
        } else {
        }
        int v350;
        v350 = v339 * 64;
        int v351;
        v351 = v350 + v335;
        assert("Tensor range check" && 0 <= v339 && v339 < 4);
        int v352;
        v352 = 64 * v339;
        int v353;
        v353 = v352 + v335;
        float * v354;
        v354 = v326[v353];
        float * v355;
        v355 = v328[v353];
        int v356;
        v356 = blockIdx.x;
        int v357;
        v357 = v356 * 256;
        int v358;
        v358 = v357 + v351;
        assert("Tensor range check" && 0 <= v334 && v334 < 4);
        int v359;
        v359 = 4 * v334;
        float v360[4];
        int v361[4];
        int v362;
        v362 = 0;
        while (while_method_1(v362)){
            assert("Tensor range check" && 0 <= v362 && v362 < 1);
            int v364;
            v364 = 4 * v362;
            assert("Tensor range check" && 0 <= v362 && v362 < 1);
            int v365;
            v365 = 16 * v362;
            int v366;
            v366 = v365 + v359;
            int4* v367;
            v367 = reinterpret_cast<int4*>(v354 + v366);
            int4* v368;
            v368 = reinterpret_cast<int4*>(v360 + v364);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v367) % 16 == 0 && reinterpret_cast<unsigned long long>(v368) % 16 == 0);
            *v368 = *v367;
            v362 += 1 ;
        }
        int v369;
        v369 = 0;
        while (while_method_1(v369)){
            int v371;
            v371 = 0;
            while (while_method_0(v371)){
                bool v373;
                v373 = 0 <= v371;
                bool v375;
                if (v373){
                    bool v374;
                    v374 = v371 < 4;
                    v375 = v374;
                } else {
                    v375 = false;
                }
                bool v376;
                v376 = v375 == false;
                if (v376){
                    assert("The indices should be inside the range of the dimension." && v375);
                } else {
                }
                bool v378;
                v378 = 0 <= v334;
                bool v380;
                if (v378){
                    bool v379;
                    v379 = v334 < 4;
                    v380 = v379;
                } else {
                    v380 = false;
                }
                bool v381;
                v381 = v380 == false;
                if (v381){
                    assert("The indices should be inside the range of the dimension." && v380);
                } else {
                }
                int v383;
                v383 = v334 * 4;
                int v384;
                v384 = v371 + v383;
                bool v385;
                v385 = 0 <= v369;
                bool v387;
                if (v385){
                    bool v386;
                    v386 = v369 < 1;
                    v387 = v386;
                } else {
                    v387 = false;
                }
                bool v388;
                v388 = v387 == false;
                if (v388){
                    assert("The indices should be inside the range of the dimension." && v387);
                } else {
                }
                int v390;
                v390 = v369 * 16;
                int v391;
                v391 = v384 + v390;
                assert("Tensor range check" && 0 <= v369 && v369 < 1);
                assert("Tensor range check" && 0 <= v371 && v371 < 4);
                int v392;
                v392 = 4 * v369;
                int v393;
                v393 = v392 + v371;
                v361[v393] = v391;
                v371 += 1 ;
            }
            v369 += 1 ;
        }
        bool v394[4];
        int v395;
        v395 = 0;
        while (while_method_1(v395)){
            int v397;
            v397 = 0;
            while (while_method_0(v397)){
                assert("Tensor range check" && 0 <= v395 && v395 < 1);
                assert("Tensor range check" && 0 <= v397 && v397 < 4);
                int v399;
                v399 = 4 * v395;
                int v400;
                v400 = v399 + v397;
                float v401;
                v401 = v360[v400];
                int v402;
                v402 = v361[v400];
                bool v403;
                v403 = v402 < 3;
                assert("Tensor range check" && 0 <= v395 && v395 < 1);
                assert("Tensor range check" && 0 <= v397 && v397 < 4);
                v394[v400] = v403;
                v397 += 1 ;
            }
            v395 += 1 ;
        }
        float v404[4];
        int v405;
        v405 = 0;
        while (while_method_1(v405)){
            int v407;
            v407 = 0;
            while (while_method_0(v407)){
                assert("Tensor range check" && 0 <= v405 && v405 < 1);
                assert("Tensor range check" && 0 <= v407 && v407 < 4);
                int v409;
                v409 = 4 * v405;
                int v410;
                v410 = v409 + v407;
                float v411;
                v411 = v360[v410];
                bool v412;
                v412 = v394[v410];
                float v415;
                if (v412){
                    bool v413;
                    v413 = 0.0f >= v411;
                    if (v413){
                        v415 = 0.0f;
                    } else {
                        v415 = v411;
                    }
                } else {
                    v415 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v405 && v405 < 1);
                assert("Tensor range check" && 0 <= v407 && v407 < 4);
                v404[v410] = v415;
                v407 += 1 ;
            }
            v405 += 1 ;
        }
        float v416;
        v416 = 0.0f;
        int v417;
        v417 = 0;
        while (while_method_1(v417)){
            int v419;
            v419 = 0;
            while (while_method_0(v419)){
                assert("Tensor range check" && 0 <= v417 && v417 < 1);
                assert("Tensor range check" && 0 <= v419 && v419 < 4);
                int v421;
                v421 = 4 * v417;
                int v422;
                v422 = v421 + v419;
                float v423;
                v423 = v404[v422];
                float v424;
                v424 = v416 + v423;
                v416 = v424;
                v419 += 1 ;
            }
            v417 += 1 ;
        }
        auto v425 = cooperative_groups::coalesced_threads();
        int v426;
        v426 = threadIdx.x;
        int v427;
        v427 = v426 / 4;
        auto v428 = cooperative_groups::labeled_partition(v425,v427);
        Closure0 v429{};
        float v430;
        v430 = cooperative_groups::reduce(v428, v416, v429);
        int v431[4];
        int v432;
        v432 = 0;
        while (while_method_1(v432)){
            int v434;
            v434 = 0;
            while (while_method_0(v434)){
                assert("Tensor range check" && 0 <= v432 && v432 < 1);
                assert("Tensor range check" && 0 <= v434 && v434 < 4);
                int v436;
                v436 = 4 * v432;
                int v437;
                v437 = v436 + v434;
                bool v438;
                v438 = v394[v437];
                int v439;
                if (v438){
                    v439 = 1;
                } else {
                    v439 = 0;
                }
                assert("Tensor range check" && 0 <= v432 && v432 < 1);
                assert("Tensor range check" && 0 <= v434 && v434 < 4);
                v431[v437] = v439;
                v434 += 1 ;
            }
            v432 += 1 ;
        }
        int v440;
        v440 = 0;
        int v441;
        v441 = 0;
        while (while_method_1(v441)){
            int v443;
            v443 = 0;
            while (while_method_0(v443)){
                assert("Tensor range check" && 0 <= v441 && v441 < 1);
                assert("Tensor range check" && 0 <= v443 && v443 < 4);
                int v445;
                v445 = 4 * v441;
                int v446;
                v446 = v445 + v443;
                int v447;
                v447 = v431[v446];
                int v448;
                v448 = v440 + v447;
                v440 = v448;
                v443 += 1 ;
            }
            v441 += 1 ;
        }
        auto v449 = cooperative_groups::coalesced_threads();
        int v450;
        v450 = threadIdx.x;
        int v451;
        v451 = v450 / 4;
        auto v452 = cooperative_groups::labeled_partition(v449,v451);
        Closure1 v453{};
        int v454;
        v454 = cooperative_groups::reduce(v452, v440, v453);
        float v455;
        v455 = (float)v454;
        float v456;
        v456 = 1.0f / v455;
        float v457[4];
        int v458;
        v458 = 0;
        while (while_method_1(v458)){
            int v460;
            v460 = 0;
            while (while_method_0(v460)){
                assert("Tensor range check" && 0 <= v458 && v458 < 1);
                assert("Tensor range check" && 0 <= v460 && v460 < 4);
                int v462;
                v462 = 4 * v458;
                int v463;
                v463 = v462 + v460;
                float v464;
                v464 = v404[v463];
                bool v465;
                v465 = v394[v463];
                bool v466;
                v466 = v465 == false;
                float v471;
                if (v466){
                    v471 = 0.0f;
                } else {
                    bool v467;
                    v467 = v430 == 0.0f;
                    bool v468;
                    v468 = v467 != true;
                    if (v468){
                        float v469;
                        v469 = v464 / v430;
                        v471 = v469;
                    } else {
                        v471 = v456;
                    }
                }
                assert("Tensor range check" && 0 <= v458 && v458 < 1);
                assert("Tensor range check" && 0 <= v460 && v460 < 4);
                v457[v463] = v471;
                v460 += 1 ;
            }
            v458 += 1 ;
        }
        int v472;
        v472 = 0;
        while (while_method_1(v472)){
            assert("Tensor range check" && 0 <= v472 && v472 < 1);
            int v474;
            v474 = 16 * v472;
            int v475;
            v475 = v474 + v359;
            assert("Tensor range check" && 0 <= v472 && v472 < 1);
            int v476;
            v476 = 4 * v472;
            int4* v477;
            v477 = reinterpret_cast<int4*>(v457 + v476);
            int4* v478;
            v478 = reinterpret_cast<int4*>(v355 + v475);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v477) % 16 == 0 && reinterpret_cast<unsigned long long>(v478) % 16 == 0);
            *v478 = *v477;
            v472 += 1 ;
        }
        assert("Tensor range check" && 0 <= v351 && v351 < 256);
        v339 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v330 && v330 < 256);
    __syncthreads();
    int v479;
    v479 = threadIdx.x;
    int v480;
    v480 = blockIdx.x;
    int v481;
    v481 = v480 * 256;
    int v482;
    v482 = v479 + v481;
    unsigned long long v483;
    v483 = (unsigned long long)v482;
    curandStatePhilox4_32_10_t v484;
    curand_init(12344321ull,v483,0ull,&v484);
    float * v485;
    v485 = v1+v12;
    if (v155){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v154);
    } else {
    }
    extern __shared__ unsigned char v488[];
    if (v159){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v158);
    } else {
    }
    float * * v490;
    v490 = reinterpret_cast<float * *>(&v488[0ull]);
    int * v492;
    v492 = reinterpret_cast<int *>(&v488[v45]);
    int v494;
    v494 = threadIdx.x;
    assert("Tensor range check" && 0 <= v494 && v494 < 256);
    v490[v494] = v485;
    __syncthreads();
    bool v495;
    v495 = 0 <= v494;
    bool v496;
    v496 = v495 == false;
    if (v496){
        assert("The index needs to be zero or positive." && v495);
    } else {
    }
    int v498;
    v498 = v494 % 4;
    int v499;
    v499 = v494 / 4;
    bool v500;
    v500 = v499 < 64;
    bool v501;
    v501 = v500 == false;
    if (v501){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v500);
    } else {
    }
    assert("Tensor range check" && 0 <= v499 && v499 < 64);
    int v503;
    v503 = 0;
    while (while_method_0(v503)){
        bool v505;
        v505 = 0 <= v499;
        bool v506;
        v506 = v505 && v500;
        bool v507;
        v507 = v506 == false;
        if (v507){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v506);
        } else {
        }
        bool v509;
        v509 = 0 <= v503;
        bool v511;
        if (v509){
            bool v510;
            v510 = v503 < 4;
            v511 = v510;
        } else {
            v511 = false;
        }
        bool v512;
        v512 = v511 == false;
        if (v512){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v511);
        } else {
        }
        int v514;
        v514 = v503 * 64;
        int v515;
        v515 = v514 + v499;
        assert("Tensor range check" && 0 <= v503 && v503 < 4);
        int v516;
        v516 = 64 * v503;
        int v517;
        v517 = v516 + v499;
        float * v518;
        v518 = v490[v517];
        int v519;
        v519 = blockIdx.x;
        int v520;
        v520 = v519 * 256;
        int v521;
        v521 = v520 + v515;
        assert("Tensor range check" && 0 <= v498 && v498 < 4);
        int v522;
        v522 = 4 * v498;
        float v523[4];
        int v524[4];
        int v525;
        v525 = 0;
        while (while_method_1(v525)){
            assert("Tensor range check" && 0 <= v525 && v525 < 1);
            int v527;
            v527 = 4 * v525;
            assert("Tensor range check" && 0 <= v525 && v525 < 1);
            int v528;
            v528 = 16 * v525;
            int v529;
            v529 = v528 + v522;
            int4* v530;
            v530 = reinterpret_cast<int4*>(v518 + v529);
            int4* v531;
            v531 = reinterpret_cast<int4*>(v523 + v527);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v530) % 16 == 0 && reinterpret_cast<unsigned long long>(v531) % 16 == 0);
            *v531 = *v530;
            v525 += 1 ;
        }
        int v532;
        v532 = 0;
        while (while_method_1(v532)){
            int v534;
            v534 = 0;
            while (while_method_0(v534)){
                bool v536;
                v536 = 0 <= v534;
                bool v538;
                if (v536){
                    bool v537;
                    v537 = v534 < 4;
                    v538 = v537;
                } else {
                    v538 = false;
                }
                bool v539;
                v539 = v538 == false;
                if (v539){
                    assert("The indices should be inside the range of the dimension." && v538);
                } else {
                }
                bool v541;
                v541 = 0 <= v498;
                bool v543;
                if (v541){
                    bool v542;
                    v542 = v498 < 4;
                    v543 = v542;
                } else {
                    v543 = false;
                }
                bool v544;
                v544 = v543 == false;
                if (v544){
                    assert("The indices should be inside the range of the dimension." && v543);
                } else {
                }
                int v546;
                v546 = v498 * 4;
                int v547;
                v547 = v534 + v546;
                bool v548;
                v548 = 0 <= v532;
                bool v550;
                if (v548){
                    bool v549;
                    v549 = v532 < 1;
                    v550 = v549;
                } else {
                    v550 = false;
                }
                bool v551;
                v551 = v550 == false;
                if (v551){
                    assert("The indices should be inside the range of the dimension." && v550);
                } else {
                }
                int v553;
                v553 = v532 * 16;
                int v554;
                v554 = v547 + v553;
                assert("Tensor range check" && 0 <= v532 && v532 < 1);
                assert("Tensor range check" && 0 <= v534 && v534 < 4);
                int v555;
                v555 = 4 * v532;
                int v556;
                v556 = v555 + v534;
                v524[v556] = v554;
                v534 += 1 ;
            }
            v532 += 1 ;
        }
        bool v557[4];
        int v558;
        v558 = 0;
        while (while_method_1(v558)){
            int v560;
            v560 = 0;
            while (while_method_0(v560)){
                assert("Tensor range check" && 0 <= v558 && v558 < 1);
                assert("Tensor range check" && 0 <= v560 && v560 < 4);
                int v562;
                v562 = 4 * v558;
                int v563;
                v563 = v562 + v560;
                float v564;
                v564 = v523[v563];
                int v565;
                v565 = v524[v563];
                bool v566;
                v566 = v565 < 3;
                assert("Tensor range check" && 0 <= v558 && v558 < 1);
                assert("Tensor range check" && 0 <= v560 && v560 < 4);
                v557[v563] = v566;
                v560 += 1 ;
            }
            v558 += 1 ;
        }
        float v567[4];
        int v568;
        v568 = 0;
        while (while_method_1(v568)){
            int v570;
            v570 = 0;
            while (while_method_0(v570)){
                assert("Tensor range check" && 0 <= v568 && v568 < 1);
                assert("Tensor range check" && 0 <= v570 && v570 < 4);
                int v572;
                v572 = 4 * v568;
                int v573;
                v573 = v572 + v570;
                float v574;
                v574 = v523[v573];
                bool v575;
                v575 = v557[v573];
                float v576;
                if (v575){
                    v576 = v574;
                } else {
                    v576 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v568 && v568 < 1);
                assert("Tensor range check" && 0 <= v570 && v570 < 4);
                v567[v573] = v576;
                v570 += 1 ;
            }
            v568 += 1 ;
        }
        float v577;
        v577 = 0.0f;
        int v578;
        v578 = 0;
        while (while_method_1(v578)){
            int v580;
            v580 = 0;
            while (while_method_0(v580)){
                assert("Tensor range check" && 0 <= v578 && v578 < 1);
                assert("Tensor range check" && 0 <= v580 && v580 < 4);
                int v582;
                v582 = 4 * v578;
                int v583;
                v583 = v582 + v580;
                float v584;
                v584 = v567[v583];
                float v585;
                v585 = v577 + v584;
                v577 = v585;
                v580 += 1 ;
            }
            v578 += 1 ;
        }
        auto v586 = cooperative_groups::coalesced_threads();
        int v587;
        v587 = threadIdx.x;
        int v588;
        v588 = v587 / 4;
        auto v589 = cooperative_groups::labeled_partition(v586,v588);
        Closure0 v590{};
        float v591;
        v591 = cooperative_groups::reduce(v589, v577, v590);
        int v592[4];
        int v593;
        v593 = 0;
        while (while_method_1(v593)){
            int v595;
            v595 = 0;
            while (while_method_0(v595)){
                assert("Tensor range check" && 0 <= v593 && v593 < 1);
                assert("Tensor range check" && 0 <= v595 && v595 < 4);
                int v597;
                v597 = 4 * v593;
                int v598;
                v598 = v597 + v595;
                bool v599;
                v599 = v557[v598];
                int v600;
                if (v599){
                    v600 = 1;
                } else {
                    v600 = 0;
                }
                assert("Tensor range check" && 0 <= v593 && v593 < 1);
                assert("Tensor range check" && 0 <= v595 && v595 < 4);
                v592[v598] = v600;
                v595 += 1 ;
            }
            v593 += 1 ;
        }
        int v601;
        v601 = 0;
        int v602;
        v602 = 0;
        while (while_method_1(v602)){
            int v604;
            v604 = 0;
            while (while_method_0(v604)){
                assert("Tensor range check" && 0 <= v602 && v602 < 1);
                assert("Tensor range check" && 0 <= v604 && v604 < 4);
                int v606;
                v606 = 4 * v602;
                int v607;
                v607 = v606 + v604;
                int v608;
                v608 = v592[v607];
                int v609;
                v609 = v601 + v608;
                v601 = v609;
                v604 += 1 ;
            }
            v602 += 1 ;
        }
        auto v610 = cooperative_groups::coalesced_threads();
        int v611;
        v611 = threadIdx.x;
        int v612;
        v612 = v611 / 4;
        auto v613 = cooperative_groups::labeled_partition(v610,v612);
        Closure1 v614{};
        int v615;
        v615 = cooperative_groups::reduce(v613, v601, v614);
        float v616;
        v616 = (float)v615;
        float v617;
        v617 = v591 / v616;
        float v618[4];
        int v619;
        v619 = 0;
        while (while_method_1(v619)){
            int v621;
            v621 = 0;
            while (while_method_0(v621)){
                assert("Tensor range check" && 0 <= v619 && v619 < 1);
                assert("Tensor range check" && 0 <= v621 && v621 < 4);
                int v623;
                v623 = 4 * v619;
                int v624;
                v624 = v623 + v621;
                float v625;
                v625 = v523[v624];
                bool v626;
                v626 = v557[v624];
                float v627;
                if (v626){
                    v627 = v625;
                } else {
                    v627 = -1.0f / 0.0f;
                }
                float v628;
                v628 = v627 - v617;
                float v629;
                v629 = exp(v628);
                bool v630;
                v630 = v629 < 1.0f / 0.0f;
                bool v631;
                v631 = v630 == false;
                if (v631){
                    assert("The softmax values must not grow too large." && v630);
                } else {
                }
                bool v633;
                v633 = isnan(v629);
                bool v634;
                v634 = v633 == false;
                bool v635;
                v635 = v634 == false;
                if (v635){
                    assert("The softmax values must not be nans." && v634);
                } else {
                }
                assert("Tensor range check" && 0 <= v619 && v619 < 1);
                assert("Tensor range check" && 0 <= v621 && v621 < 4);
                v618[v624] = v629;
                v621 += 1 ;
            }
            v619 += 1 ;
        }
        float v637;
        v637 = 0.0f;
        int v638;
        v638 = 0;
        while (while_method_1(v638)){
            int v640;
            v640 = 0;
            while (while_method_0(v640)){
                assert("Tensor range check" && 0 <= v638 && v638 < 1);
                assert("Tensor range check" && 0 <= v640 && v640 < 4);
                int v642;
                v642 = 4 * v638;
                int v643;
                v643 = v642 + v640;
                float v644;
                v644 = v618[v643];
                float v645;
                v645 = v637 + v644;
                v637 = v645;
                v640 += 1 ;
            }
            v638 += 1 ;
        }
        auto v646 = cooperative_groups::coalesced_threads();
        int v647;
        v647 = threadIdx.x;
        int v648;
        v648 = v647 / 4;
        auto v649 = cooperative_groups::labeled_partition(v646,v648);
        float v650;
        v650 = cooperative_groups::reduce(v649, v637, v590);
        float v651[4];
        int v652;
        v652 = 0;
        while (while_method_1(v652)){
            int v654;
            v654 = 0;
            while (while_method_0(v654)){
                assert("Tensor range check" && 0 <= v652 && v652 < 1);
                assert("Tensor range check" && 0 <= v654 && v654 < 4);
                int v656;
                v656 = 4 * v652;
                int v657;
                v657 = v656 + v654;
                float v658;
                v658 = v618[v657];
                float v659;
                v659 = v658 / v650;
                assert("Tensor range check" && 0 <= v652 && v652 < 1);
                assert("Tensor range check" && 0 <= v654 && v654 < 4);
                v651[v657] = v659;
                v654 += 1 ;
            }
            v652 += 1 ;
        }
        float v660[4];
        float v661;
        v661 = 0.0f;
        int v662;
        v662 = 0;
        while (while_method_1(v662)){
            assert("Tensor range check" && 0 <= v662 && v662 < 1);
            int v664;
            v664 = 4 * v662;
            assert("Tensor range check" && 0 <= v662 && v662 < 1);
            float v665;
            v665 = 0.0f;
            int v666;
            v666 = 0;
            while (while_method_0(v666)){
                assert("Tensor range check" && 0 <= v666 && v666 < 4);
                int v668;
                v668 = v666 + v664;
                float v669;
                v669 = v651[v668];
                float v670;
                v670 = v665 + v669;
                v665 = v670;
                v666 += 1 ;
            }
            auto v671 = cooperative_groups::coalesced_threads();
            int v672;
            v672 = threadIdx.x;
            int v673;
            v673 = v672 / 4;
            auto v674 = cooperative_groups::labeled_partition(v671,v673);
            Closure2 v675{};
            float v676;
            v676 = cooperative_groups::inclusive_scan(v674, v665, v675);
            float v677;
            v677 = v674.shfl_up(v676,1);
            bool v678;
            v678 = v674.thread_rank() == 0;
            float v679;
            if (v678){
                v679 = 0.0f;
            } else {
                v679 = v677;
            }
            float v680;
            v680 = v674.shfl(v676,v674.num_threads()-1);
            float v681;
            v681 = v661 + v679;
            float v682;
            v682 = v681;
            int v683;
            v683 = 0;
            while (while_method_0(v683)){
                assert("Tensor range check" && 0 <= v683 && v683 < 4);
                int v685;
                v685 = v683 + v664;
                float v686;
                v686 = v651[v685];
                float v687;
                v687 = v682 + v686;
                assert("Tensor range check" && 0 <= v683 && v683 < 4);
                v660[v685] = v687;
                v682 = v687;
                v683 += 1 ;
            }
            float v688;
            v688 = v661 + v680;
            v661 = v688;
            v662 += 1 ;
        }
        float v689[4];
        bool v690[4];
        int v691;
        v691 = 0;
        while (while_method_1(v691)){
            int v693;
            v693 = 0;
            while (while_method_0(v693)){
                assert("Tensor range check" && 0 <= v691 && v691 < 1);
                assert("Tensor range check" && 0 <= v693 && v693 < 4);
                int v695;
                v695 = 4 * v691;
                int v696;
                v696 = v695 + v693;
                float v697;
                v697 = v660[v696];
                float v698;
                v698 = v651[v696];
                bool v699;
                v699 = v698 > 0.0f;
                assert("Tensor range check" && 0 <= v691 && v691 < 1);
                assert("Tensor range check" && 0 <= v693 && v693 < 4);
                v689[v696] = v697;
                v690[v696] = v699;
                v693 += 1 ;
            }
            v691 += 1 ;
        }
        float v700; bool v701;
        Tuple0 tmp0 = Tuple0{-1.0f / 0.0f, false};
        v700 = tmp0.v0; v701 = tmp0.v1;
        int v702;
        v702 = 0;
        while (while_method_1(v702)){
            int v704;
            v704 = 0;
            while (while_method_0(v704)){
                assert("Tensor range check" && 0 <= v702 && v702 < 1);
                assert("Tensor range check" && 0 <= v704 && v704 < 4);
                int v706;
                v706 = 4 * v702;
                int v707;
                v707 = v706 + v704;
                float v708;
                v708 = v689[v707];
                bool v709;
                v709 = v690[v707];
                float v716; bool v717;
                if (v701){
                    if (v709){
                        bool v710;
                        v710 = v700 >= v708;
                        float v711;
                        if (v710){
                            v711 = v700;
                        } else {
                            v711 = v708;
                        }
                        v716 = v711; v717 = true;
                    } else {
                        v716 = v700; v717 = v701;
                    }
                } else {
                    if (v709){
                        v716 = v708; v717 = v709;
                    } else {
                        v716 = v700; v717 = v701;
                    }
                }
                v700 = v716;
                v701 = v717;
                v704 += 1 ;
            }
            v702 += 1 ;
        }
        auto v718 = cooperative_groups::coalesced_threads();
        int v719;
        v719 = threadIdx.x;
        int v720;
        v720 = v719 / 4;
        auto v721 = cooperative_groups::labeled_partition(v718,v720);
        Closure3 v722{};
        float v723; bool v724;
        Tuple0 tmp1 = cooperative_groups::reduce(v721, Tuple0{v700, v701}, v722);
        v723 = tmp1.v0; v724 = tmp1.v1;
        bool v725;
        v725 = v724 == false;
        if (v725){
            assert("The local reduce must be true." && v724);
        } else {
        }
        float v727[4];
        int v728[4];
        int v729;
        v729 = 0;
        while (while_method_1(v729)){
            int v731;
            v731 = 0;
            while (while_method_0(v731)){
                assert("Tensor range check" && 0 <= v729 && v729 < 1);
                assert("Tensor range check" && 0 <= v731 && v731 < 4);
                int v733;
                v733 = 4 * v729;
                int v734;
                v734 = v733 + v731;
                int v735;
                v735 = v524[v734];
                float v736;
                v736 = curand_uniform(&v484);
                assert("Tensor range check" && 0 <= v729 && v729 < 1);
                assert("Tensor range check" && 0 <= v731 && v731 < 4);
                v727[v734] = v736;
                v728[v734] = v735;
                v731 += 1 ;
            }
            v729 += 1 ;
        }
        float v737; int v738;
        Tuple1 tmp2 = Tuple1{0.0f, 2147483647};
        v737 = tmp2.v0; v738 = tmp2.v1;
        int v739;
        v739 = 0;
        while (while_method_1(v739)){
            int v741;
            v741 = 0;
            while (while_method_0(v741)){
                assert("Tensor range check" && 0 <= v739 && v739 < 1);
                assert("Tensor range check" && 0 <= v741 && v741 < 4);
                int v743;
                v743 = 4 * v739;
                int v744;
                v744 = v743 + v741;
                float v745;
                v745 = v727[v744];
                int v746;
                v746 = v728[v744];
                bool v747;
                v747 = v738 < v746;
                float v748; int v749;
                if (v747){
                    v748 = v737; v749 = v738;
                } else {
                    v748 = v745; v749 = v746;
                }
                v737 = v748;
                v738 = v749;
                v741 += 1 ;
            }
            v739 += 1 ;
        }
        auto v750 = cooperative_groups::coalesced_threads();
        int v751;
        v751 = threadIdx.x;
        int v752;
        v752 = v751 / 4;
        auto v753 = cooperative_groups::labeled_partition(v750,v752);
        Closure4 v754{};
        float v755; int v756;
        Tuple1 tmp3 = cooperative_groups::reduce(v753, Tuple1{v737, v738}, v754);
        v755 = tmp3.v0; v756 = tmp3.v1;
        float v757;
        v757 = v723 * v755;
        int v758[4];
        bool v759[4];
        int v760;
        v760 = 0;
        while (while_method_1(v760)){
            int v762;
            v762 = 0;
            while (while_method_0(v762)){
                assert("Tensor range check" && 0 <= v760 && v760 < 1);
                assert("Tensor range check" && 0 <= v762 && v762 < 4);
                int v764;
                v764 = 4 * v760;
                int v765;
                v765 = v764 + v762;
                float v766;
                v766 = v689[v765];
                bool v767;
                v767 = v690[v765];
                int v768;
                v768 = v524[v765];
                int v771; bool v772;
                if (v767){
                    float v769;
                    v769 = v766 - v757;
                    bool v770;
                    v770 = v769 >= 0.0f;
                    v771 = v768; v772 = v770;
                } else {
                    v771 = 2147483647; v772 = false;
                }
                assert("Tensor range check" && 0 <= v760 && v760 < 1);
                assert("Tensor range check" && 0 <= v762 && v762 < 4);
                v758[v765] = v771;
                v759[v765] = v772;
                v762 += 1 ;
            }
            v760 += 1 ;
        }
        int v773; bool v774;
        Tuple2 tmp4 = Tuple2{2147483647, false};
        v773 = tmp4.v0; v774 = tmp4.v1;
        int v775;
        v775 = 0;
        while (while_method_1(v775)){
            int v777;
            v777 = 0;
            while (while_method_0(v777)){
                assert("Tensor range check" && 0 <= v775 && v775 < 1);
                assert("Tensor range check" && 0 <= v777 && v777 < 4);
                int v779;
                v779 = 4 * v775;
                int v780;
                v780 = v779 + v777;
                int v781;
                v781 = v758[v780];
                bool v782;
                v782 = v759[v780];
                int v789; bool v790;
                if (v774){
                    if (v782){
                        bool v783;
                        v783 = v773 < v781;
                        int v784;
                        if (v783){
                            v784 = v773;
                        } else {
                            v784 = v781;
                        }
                        v789 = v784; v790 = true;
                    } else {
                        v789 = v773; v790 = v774;
                    }
                } else {
                    if (v782){
                        v789 = v781; v790 = v782;
                    } else {
                        v789 = v773; v790 = v774;
                    }
                }
                v773 = v789;
                v774 = v790;
                v777 += 1 ;
            }
            v775 += 1 ;
        }
        auto v791 = cooperative_groups::coalesced_threads();
        int v792;
        v792 = threadIdx.x;
        int v793;
        v793 = v792 / 4;
        auto v794 = cooperative_groups::labeled_partition(v791,v793);
        Closure5 v795{};
        int v796; bool v797;
        Tuple2 tmp5 = cooperative_groups::reduce(v794, Tuple2{v773, v774}, v795);
        v796 = tmp5.v0; v797 = tmp5.v1;
        bool v798;
        v798 = v797 == false;
        if (v798){
            assert("The local reduce must be true." && v797);
        } else {
        }
        int v800;
        v800 = 0;
        while (while_method_1(v800)){
            assert("Tensor range check" && 0 <= v800 && v800 < 1);
            assert("Tensor range check" && 0 <= v800 && v800 < 1);
            v800 += 1 ;
        }
        assert("Tensor range check" && 0 <= v515 && v515 < 256);
        v492[v515] = v796;
        v503 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v494 && v494 < 256);
    int v802;
    v802 = v492[v494];
    __syncthreads();
    int v803;
    v803 = threadIdx.x;
    assert("Tensor range check" && 0 <= v803 && v803 < 6144);
    v5[v803] = v802;
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
        v36 = v35 >= 2147483647
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
            v44 = v43 >= 2147483647
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
        v25 = v24 >= 2147483647
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
        v38 = v37 >= 2147483647
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
            v46 = v45 >= 2147483647
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
        v25 = v24 >= 2147483647
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
        v36 = v35 >= 2147483647
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
            v44 = v43 >= 2147483647
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
        v38 = v37 >= 2147483647
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
            v46 = v45 >= 2147483647
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
def main_body():
    cp.random.seed(12344321)
    v0 = cp.arange(0,98304,1,dtype=cp.float32) # type: ignore
    v1 = v0.size
    v2 = 98304 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,98304,dtype=cp.float32) # type: ignore
    v6 = cp.empty(98304,dtype=cp.int32)
    v7 = cp.empty(98304,dtype=cp.int32)
    v8 = cp.empty(6144,dtype=cp.int32)
    v9 = cp.empty(6144,dtype=cp.int32)
    v10 = cp.empty(98304,dtype=cp.float32)
    v11 = cp.empty(98304,dtype=cp.float32)
    method0(v0, v5, v6, v7, v8, v9, v10, v11)
    method1(v0)
    del v0
    method4(v9)
    del v9
    method5(v6, v7)
    del v6, v7
    method6(v8)
    del v8
    method7(v11)
    del v11
    return method8(v5, v10)

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
