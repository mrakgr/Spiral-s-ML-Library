kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
using default_int = long;
using default_uint = unsigned long;
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
struct Tuple0 {
    long v0;
    float v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(long t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure0 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure1 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple1 {
    float v0;
    long v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(float t0, long t1) : v0(t0), v1(t1) {}
};
struct Closure2 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; long v1 = tup0.v1; float v2 = tup1.v0; long v3 = tup1.v1;
        bool v4;
        v4 = v0 > v2;
        if (v4){
            return Tuple1{v0, v1};
        } else {
            return Tuple1{v2, v3};
        }
    }
};
struct Closure3 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure4 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; long v1 = tup0.v1; float v2 = tup1.v0; long v3 = tup1.v1;
        bool v4;
        v4 = v0 >= 0.0f;
        bool v6;
        if (v4){
            bool v5;
            v5 = v2 >= 0.0f;
            v6 = v5;
        } else {
            v6 = false;
        }
        if (v6){
            bool v7;
            v7 = v0 <= v2;
            if (v7){
                return Tuple1{v0, v1};
            } else {
                return Tuple1{v2, v3};
            }
        } else {
            if (v4){
                return Tuple1{v0, v1};
            } else {
                bool v10;
                v10 = v2 >= 0.0f;
                if (v10){
                    return Tuple1{v2, v3};
                } else {
                    return Tuple1{v0, v1};
                }
            }
        }
    }
};
__device__ inline bool while_method_0(long v0){
    bool v1;
    v1 = v0 < 2048l;
    return v1;
}
__device__ inline bool while_method_1(long v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_2(long v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ inline bool while_method_3(long v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
extern "C" __global__ void entry0(long * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, long * v8, long * v9, long * v10, long * v11, long * v12, long * v13) {
    auto v14 = cooperative_groups::this_thread_block();
    float v15;
    v15 = 0.0f;
    long v16;
    v16 = threadIdx.x;
    long v17;
    v17 = v16;
    while (while_method_0(v17)){
        bool v19;
        v19 = 0l <= v17;
        bool v20;
        v20 = v19 == false;
        if (v20){
            assert("The index needs to be zero or positive." && v19);
        } else {
        }
        long v21;
        v21 = v17 % 64l;
        long v22;
        v22 = v17 / 64l;
        bool v23;
        v23 = v22 < 32l;
        bool v24;
        v24 = v23 == false;
        if (v24){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v23);
        } else {
        }
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        assert("Tensor range check" && 0 <= v21 && v21 < 64l);
        long v25;
        v25 = 4l * v21;
        long v26;
        v26 = 256l * v22;
        long v27;
        v27 = v26 + v25;
        float v28[4l];
        int4* v29;
        v29 = reinterpret_cast<int4*>(v1 + v27);
        int4* v30;
        v30 = reinterpret_cast<int4*>(v28 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v29) % 4l == 0 && (unsigned long long)(v30) % 4l == 0);
        *v30 = *v29;
        long v31; float v32;
        Tuple0 tmp0 = Tuple0{0l, v15};
        v31 = tmp0.v0; v32 = tmp0.v1;
        while (while_method_1(v31)){
            assert("Tensor range check" && 0 <= v31 && v31 < 4l);
            float v34;
            v34 = v28[v31];
            float v35;
            v35 = v32 + v34;
            v32 = v35;
            v31 += 1l ;
        }
        v15 = v32;
        v17 += 32l ;
    }
    auto v36 = cooperative_groups::coalesced_threads();
    Closure0 v37{};
    float v38;
    v38 = cooperative_groups::reduce(v36, v15, v37);
    long v39;
    v39 = threadIdx.x;
    long v40;
    v40 = v39 / 32l;
    __shared__ float v41[1l];
    assert("Tensor range check" && 0 <= v40 && v40 < 1l);
    v41[v40] = v38;
    __syncthreads();
    long v42;
    v42 = threadIdx.x;
    long v43;
    v43 = v42 % 32l;
    bool v44;
    v44 = v40 == 0l;
    bool v46;
    if (v44){
        bool v45;
        v45 = v43 < 1l;
        v46 = v45;
    } else {
        v46 = false;
    }
    if (v46){
        auto v47 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v43 && v43 < 1l);
        float v48;
        v48 = v41[v43];
        float v49;
        v49 = cooperative_groups::reduce(v47, v48, v37);
        v3[0l] = v49;
    } else {
    }
    __syncthreads();
    long v50;
    v50 = threadIdx.x;
    bool v51;
    v51 = 0l <= v50;
    bool v52;
    v52 = v51 == false;
    if (v52){
        assert("The index needs to be zero or positive." && v51);
    } else {
    }
    long v53;
    v53 = v50 % 32l;
    long v54;
    v54 = v50 / 32l;
    bool v55;
    v55 = v54 < 1l;
    bool v56;
    v56 = v55 == false;
    if (v56){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v55);
    } else {
    }
    auto & v57 = v14;
    assert("Tensor range check" && 0 <= v54 && v54 < 1l);
    assert("Tensor range check" && 0 <= v53 && v53 < 32l);
    long v58;
    v58 = 4l * v53;
    long v59;
    v59 = 256l * v54;
    long v60;
    v60 = v59 + v58;
    assert("Tensor range check" && 0 <= v54 && v54 < 1l);
    assert("Tensor range check" && 0 <= v53 && v53 < 32l);
    long v61;
    v61 = 0l;
    while (while_method_2(v61)){
        assert("Tensor range check" && 0 <= v61 && v61 < 32l);
        long v63;
        v63 = 256l * v61;
        long v64;
        v64 = v63 + v60;
        assert("Tensor range check" && 0 <= v61 && v61 < 32l);
        long v65[8l];
        long v66[8l];
        long v67;
        v67 = 0l;
        while (while_method_3(v67)){
            assert("Tensor range check" && 0 <= v67 && v67 < 2l);
            long v69;
            v69 = 4l * v67;
            assert("Tensor range check" && 0 <= v67 && v67 < 2l);
            long v70;
            v70 = 128l * v67;
            long v71;
            v71 = v70 + v64;
            int4* v72;
            v72 = reinterpret_cast<int4*>(v0 + v71);
            int4* v73;
            v73 = reinterpret_cast<int4*>(v65 + v69);
            assert("Pointer alignment check" && (unsigned long long)(v72) % 4l == 0 && (unsigned long long)(v73) % 4l == 0);
            *v73 = *v72;
            v67 += 1l ;
        }
        long v74;
        v74 = 0l;
        while (while_method_3(v74)){
            long v76;
            v76 = 0l;
            while (while_method_1(v76)){
                bool v78;
                v78 = 0l <= v76;
                bool v80;
                if (v78){
                    bool v79;
                    v79 = v76 < 4l;
                    v80 = v79;
                } else {
                    v80 = false;
                }
                bool v81;
                v81 = v80 == false;
                if (v81){
                    assert("The indices should be inside the range of the dimension." && v80);
                } else {
                }
                bool v82;
                v82 = 0l <= v53;
                bool v84;
                if (v82){
                    bool v83;
                    v83 = v53 < 32l;
                    v84 = v83;
                } else {
                    v84 = false;
                }
                bool v85;
                v85 = v84 == false;
                if (v85){
                    assert("The indices should be inside the range of the dimension." && v84);
                } else {
                }
                long v86;
                v86 = v53 * 4l;
                long v87;
                v87 = v76 + v86;
                bool v88;
                v88 = 0l <= v74;
                bool v90;
                if (v88){
                    bool v89;
                    v89 = v74 < 2l;
                    v90 = v89;
                } else {
                    v90 = false;
                }
                bool v91;
                v91 = v90 == false;
                if (v91){
                    assert("The indices should be inside the range of the dimension." && v90);
                } else {
                }
                long v92;
                v92 = v74 * 128l;
                long v93;
                v93 = v87 + v92;
                assert("Tensor range check" && 0 <= v74 && v74 < 2l);
                assert("Tensor range check" && 0 <= v76 && v76 < 4l);
                long v94;
                v94 = 4l * v74;
                long v95;
                v95 = v94 + v76;
                v66[v95] = v93;
                v76 += 1l ;
            }
            v74 += 1l ;
        }
        bool v96;
        v96 = 0l <= v54;
        bool v97;
        v97 = v96 && v55;
        bool v98;
        v98 = v97 == false;
        if (v98){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v97);
        } else {
        }
        bool v99;
        v99 = 0l <= v61;
        bool v101;
        if (v99){
            bool v100;
            v100 = v61 < 32l;
            v101 = v100;
        } else {
            v101 = false;
        }
        bool v102;
        v102 = v101 == false;
        if (v102){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v101);
        } else {
        }
        long v103;
        v103 = v61 + v54;
        long v104;
        v104 = 0l;
        while (while_method_3(v104)){
            assert("Tensor range check" && 0 <= v104 && v104 < 2l);
            long v106;
            v106 = 128l * v104;
            long v107;
            v107 = v106 + v64;
            assert("Tensor range check" && 0 <= v104 && v104 < 2l);
            long v108;
            v108 = 4l * v104;
            int4* v109;
            v109 = reinterpret_cast<int4*>(v65 + v108);
            int4* v110;
            v110 = reinterpret_cast<int4*>(v10 + v107);
            assert("Pointer alignment check" && (unsigned long long)(v109) % 4l == 0 && (unsigned long long)(v110) % 4l == 0);
            *v110 = *v109;
            v104 += 1l ;
        }
        v61 += 1l ;
    }
    __syncthreads();
    long v111;
    v111 = threadIdx.x;
    bool v112;
    v112 = 0l <= v111;
    bool v113;
    v113 = v112 == false;
    if (v113){
        assert("The index needs to be zero or positive." && v112);
    } else {
    }
    long v114;
    v114 = v111 % 32l;
    long v115;
    v115 = v111 / 32l;
    bool v116;
    v116 = v115 < 1l;
    bool v117;
    v117 = v116 == false;
    if (v117){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v116);
    } else {
    }
    auto & v118 = v14;
    assert("Tensor range check" && 0 <= v115 && v115 < 1l);
    assert("Tensor range check" && 0 <= v114 && v114 < 32l);
    long v119;
    v119 = 4l * v114;
    long v120;
    v120 = 256l * v115;
    long v121;
    v121 = v120 + v119;
    assert("Tensor range check" && 0 <= v115 && v115 < 1l);
    assert("Tensor range check" && 0 <= v114 && v114 < 32l);
    long v122;
    v122 = 0l;
    while (while_method_2(v122)){
        assert("Tensor range check" && 0 <= v122 && v122 < 32l);
        long v124;
        v124 = 256l * v122;
        long v125;
        v125 = v124 + v121;
        assert("Tensor range check" && 0 <= v122 && v122 < 32l);
        float v126[8l];
        long v127[8l];
        long v128;
        v128 = 0l;
        while (while_method_3(v128)){
            assert("Tensor range check" && 0 <= v128 && v128 < 2l);
            long v130;
            v130 = 4l * v128;
            assert("Tensor range check" && 0 <= v128 && v128 < 2l);
            long v131;
            v131 = 128l * v128;
            long v132;
            v132 = v131 + v125;
            int4* v133;
            v133 = reinterpret_cast<int4*>(v1 + v132);
            int4* v134;
            v134 = reinterpret_cast<int4*>(v126 + v130);
            assert("Pointer alignment check" && (unsigned long long)(v133) % 4l == 0 && (unsigned long long)(v134) % 4l == 0);
            *v134 = *v133;
            v128 += 1l ;
        }
        long v135;
        v135 = 0l;
        while (while_method_3(v135)){
            long v137;
            v137 = 0l;
            while (while_method_1(v137)){
                bool v139;
                v139 = 0l <= v137;
                bool v141;
                if (v139){
                    bool v140;
                    v140 = v137 < 4l;
                    v141 = v140;
                } else {
                    v141 = false;
                }
                bool v142;
                v142 = v141 == false;
                if (v142){
                    assert("The indices should be inside the range of the dimension." && v141);
                } else {
                }
                bool v143;
                v143 = 0l <= v114;
                bool v145;
                if (v143){
                    bool v144;
                    v144 = v114 < 32l;
                    v145 = v144;
                } else {
                    v145 = false;
                }
                bool v146;
                v146 = v145 == false;
                if (v146){
                    assert("The indices should be inside the range of the dimension." && v145);
                } else {
                }
                long v147;
                v147 = v114 * 4l;
                long v148;
                v148 = v137 + v147;
                bool v149;
                v149 = 0l <= v135;
                bool v151;
                if (v149){
                    bool v150;
                    v150 = v135 < 2l;
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
                long v153;
                v153 = v135 * 128l;
                long v154;
                v154 = v148 + v153;
                assert("Tensor range check" && 0 <= v135 && v135 < 2l);
                assert("Tensor range check" && 0 <= v137 && v137 < 4l);
                long v155;
                v155 = 4l * v135;
                long v156;
                v156 = v155 + v137;
                v127[v156] = v154;
                v137 += 1l ;
            }
            v135 += 1l ;
        }
        bool v157;
        v157 = 0l <= v115;
        bool v158;
        v158 = v157 && v116;
        bool v159;
        v159 = v158 == false;
        if (v159){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v158);
        } else {
        }
        bool v160;
        v160 = 0l <= v122;
        bool v162;
        if (v160){
            bool v161;
            v161 = v122 < 32l;
            v162 = v161;
        } else {
            v162 = false;
        }
        bool v163;
        v163 = v162 == false;
        if (v163){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v162);
        } else {
        }
        long v164;
        v164 = v122 + v115;
        long v165[8l];
        long v166[8l];
        long v167;
        v167 = 0l;
        while (while_method_3(v167)){
            long v169;
            v169 = 0l;
            while (while_method_1(v169)){
                assert("Tensor range check" && 0 <= v167 && v167 < 2l);
                assert("Tensor range check" && 0 <= v169 && v169 < 4l);
                long v171;
                v171 = 4l * v167;
                long v172;
                v172 = v171 + v169;
                long v173;
                v173 = v127[v172];
                assert("Tensor range check" && 0 <= v167 && v167 < 2l);
                assert("Tensor range check" && 0 <= v169 && v169 < 4l);
                v165[v172] = v164;
                v166[v172] = v173;
                v169 += 1l ;
            }
            v167 += 1l ;
        }
        long v174;
        v174 = 0l;
        while (while_method_3(v174)){
            assert("Tensor range check" && 0 <= v174 && v174 < 2l);
            long v176;
            v176 = 128l * v174;
            long v177;
            v177 = v176 + v125;
            assert("Tensor range check" && 0 <= v174 && v174 < 2l);
            long v178;
            v178 = 4l * v174;
            int4* v179;
            v179 = reinterpret_cast<int4*>(v165 + v178);
            int4* v180;
            v180 = reinterpret_cast<int4*>(v11 + v177);
            assert("Pointer alignment check" && (unsigned long long)(v179) % 4l == 0 && (unsigned long long)(v180) % 4l == 0);
            *v180 = *v179;
            int4* v181;
            v181 = reinterpret_cast<int4*>(v166 + v178);
            int4* v182;
            v182 = reinterpret_cast<int4*>(v12 + v177);
            assert("Pointer alignment check" && (unsigned long long)(v181) % 4l == 0 && (unsigned long long)(v182) % 4l == 0);
            *v182 = *v181;
            v174 += 1l ;
        }
        v122 += 1l ;
    }
    __syncthreads();
    long v183;
    v183 = threadIdx.x;
    bool v184;
    v184 = 0l <= v183;
    bool v185;
    v185 = v184 == false;
    if (v185){
        assert("The index needs to be zero or positive." && v184);
    } else {
    }
    long v186;
    v186 = v183 % 32l;
    long v187;
    v187 = v183 / 32l;
    bool v188;
    v188 = v187 < 1l;
    bool v189;
    v189 = v188 == false;
    if (v189){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v188);
    } else {
    }
    auto & v190 = v14;
    assert("Tensor range check" && 0 <= v187 && v187 < 1l);
    assert("Tensor range check" && 0 <= v186 && v186 < 32l);
    long v191;
    v191 = 4l * v186;
    long v192;
    v192 = 256l * v187;
    long v193;
    v193 = v192 + v191;
    assert("Tensor range check" && 0 <= v187 && v187 < 1l);
    long v194;
    v194 = 0l;
    while (while_method_2(v194)){
        assert("Tensor range check" && 0 <= v194 && v194 < 32l);
        long v196;
        v196 = 256l * v194;
        long v197;
        v197 = v196 + v193;
        float v198[8l];
        long v199[8l];
        long v200;
        v200 = 0l;
        while (while_method_3(v200)){
            assert("Tensor range check" && 0 <= v200 && v200 < 2l);
            long v202;
            v202 = 4l * v200;
            assert("Tensor range check" && 0 <= v200 && v200 < 2l);
            long v203;
            v203 = 128l * v200;
            long v204;
            v204 = v203 + v197;
            int4* v205;
            v205 = reinterpret_cast<int4*>(v1 + v204);
            int4* v206;
            v206 = reinterpret_cast<int4*>(v198 + v202);
            assert("Pointer alignment check" && (unsigned long long)(v205) % 4l == 0 && (unsigned long long)(v206) % 4l == 0);
            *v206 = *v205;
            v200 += 1l ;
        }
        long v207;
        v207 = 0l;
        while (while_method_3(v207)){
            long v209;
            v209 = 0l;
            while (while_method_1(v209)){
                bool v211;
                v211 = 0l <= v209;
                bool v213;
                if (v211){
                    bool v212;
                    v212 = v209 < 4l;
                    v213 = v212;
                } else {
                    v213 = false;
                }
                bool v214;
                v214 = v213 == false;
                if (v214){
                    assert("The indices should be inside the range of the dimension." && v213);
                } else {
                }
                bool v215;
                v215 = 0l <= v186;
                bool v217;
                if (v215){
                    bool v216;
                    v216 = v186 < 32l;
                    v217 = v216;
                } else {
                    v217 = false;
                }
                bool v218;
                v218 = v217 == false;
                if (v218){
                    assert("The indices should be inside the range of the dimension." && v217);
                } else {
                }
                long v219;
                v219 = v186 * 4l;
                long v220;
                v220 = v209 + v219;
                bool v221;
                v221 = 0l <= v207;
                bool v223;
                if (v221){
                    bool v222;
                    v222 = v207 < 2l;
                    v223 = v222;
                } else {
                    v223 = false;
                }
                bool v224;
                v224 = v223 == false;
                if (v224){
                    assert("The indices should be inside the range of the dimension." && v223);
                } else {
                }
                long v225;
                v225 = v207 * 128l;
                long v226;
                v226 = v220 + v225;
                assert("Tensor range check" && 0 <= v207 && v207 < 2l);
                assert("Tensor range check" && 0 <= v209 && v209 < 4l);
                long v227;
                v227 = 4l * v207;
                long v228;
                v228 = v227 + v209;
                v199[v228] = v226;
                v209 += 1l ;
            }
            v207 += 1l ;
        }
        bool v229;
        v229 = 0l <= v187;
        bool v230;
        v230 = v229 && v188;
        bool v231;
        v231 = v230 == false;
        if (v231){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v230);
        } else {
        }
        bool v232;
        v232 = 0l <= v194;
        bool v234;
        if (v232){
            bool v233;
            v233 = v194 < 32l;
            v234 = v233;
        } else {
            v234 = false;
        }
        bool v235;
        v235 = v234 == false;
        if (v235){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v234);
        } else {
        }
        long v236;
        v236 = v194 + v187;
        assert("Tensor range check" && 0 <= v194 && v194 < 32l);
        v13[v236] = v236;
        v194 += 1l ;
    }
    __syncthreads();
    long v237;
    v237 = threadIdx.x;
    bool v238;
    v238 = 0l <= v237;
    bool v239;
    v239 = v238 == false;
    if (v239){
        assert("The index needs to be zero or positive." && v238);
    } else {
    }
    long v240;
    v240 = v237 % 32l;
    long v241;
    v241 = v237 / 32l;
    bool v242;
    v242 = v241 < 1l;
    bool v243;
    v243 = v242 == false;
    if (v243){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v242);
    } else {
    }
    auto & v244 = v14;
    assert("Tensor range check" && 0 <= v241 && v241 < 1l);
    assert("Tensor range check" && 0 <= v240 && v240 < 32l);
    long v245;
    v245 = 4l * v240;
    long v246;
    v246 = 256l * v241;
    long v247;
    v247 = v246 + v245;
    assert("Tensor range check" && 0 <= v241 && v241 < 1l);
    assert("Tensor range check" && 0 <= v240 && v240 < 32l);
    long v248;
    v248 = 0l;
    while (while_method_2(v248)){
        assert("Tensor range check" && 0 <= v248 && v248 < 32l);
        long v250;
        v250 = 256l * v248;
        long v251;
        v251 = v250 + v247;
        assert("Tensor range check" && 0 <= v248 && v248 < 32l);
        float v252[8l];
        long v253[8l];
        long v254;
        v254 = 0l;
        while (while_method_3(v254)){
            assert("Tensor range check" && 0 <= v254 && v254 < 2l);
            long v256;
            v256 = 4l * v254;
            assert("Tensor range check" && 0 <= v254 && v254 < 2l);
            long v257;
            v257 = 128l * v254;
            long v258;
            v258 = v257 + v251;
            int4* v259;
            v259 = reinterpret_cast<int4*>(v1 + v258);
            int4* v260;
            v260 = reinterpret_cast<int4*>(v252 + v256);
            assert("Pointer alignment check" && (unsigned long long)(v259) % 4l == 0 && (unsigned long long)(v260) % 4l == 0);
            *v260 = *v259;
            v254 += 1l ;
        }
        long v261;
        v261 = 0l;
        while (while_method_3(v261)){
            long v263;
            v263 = 0l;
            while (while_method_1(v263)){
                bool v265;
                v265 = 0l <= v263;
                bool v267;
                if (v265){
                    bool v266;
                    v266 = v263 < 4l;
                    v267 = v266;
                } else {
                    v267 = false;
                }
                bool v268;
                v268 = v267 == false;
                if (v268){
                    assert("The indices should be inside the range of the dimension." && v267);
                } else {
                }
                bool v269;
                v269 = 0l <= v240;
                bool v271;
                if (v269){
                    bool v270;
                    v270 = v240 < 32l;
                    v271 = v270;
                } else {
                    v271 = false;
                }
                bool v272;
                v272 = v271 == false;
                if (v272){
                    assert("The indices should be inside the range of the dimension." && v271);
                } else {
                }
                long v273;
                v273 = v240 * 4l;
                long v274;
                v274 = v263 + v273;
                bool v275;
                v275 = 0l <= v261;
                bool v277;
                if (v275){
                    bool v276;
                    v276 = v261 < 2l;
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
                long v279;
                v279 = v261 * 128l;
                long v280;
                v280 = v274 + v279;
                assert("Tensor range check" && 0 <= v261 && v261 < 2l);
                assert("Tensor range check" && 0 <= v263 && v263 < 4l);
                long v281;
                v281 = 4l * v261;
                long v282;
                v282 = v281 + v263;
                v253[v282] = v280;
                v263 += 1l ;
            }
            v261 += 1l ;
        }
        bool v283;
        v283 = 0l <= v241;
        bool v284;
        v284 = v283 && v242;
        bool v285;
        v285 = v284 == false;
        if (v285){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v284);
        } else {
        }
        bool v286;
        v286 = 0l <= v248;
        bool v288;
        if (v286){
            bool v287;
            v287 = v248 < 32l;
            v288 = v287;
        } else {
            v288 = false;
        }
        bool v289;
        v289 = v288 == false;
        if (v289){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v288);
        } else {
        }
        long v290;
        v290 = v248 + v241;
        float v291;
        v291 = 0.0f;
        long v292;
        v292 = 0l;
        while (while_method_3(v292)){
            long v294;
            v294 = 0l;
            while (while_method_1(v294)){
                assert("Tensor range check" && 0 <= v292 && v292 < 2l);
                assert("Tensor range check" && 0 <= v294 && v294 < 4l);
                long v296;
                v296 = 4l * v292;
                long v297;
                v297 = v296 + v294;
                float v298;
                v298 = v252[v297];
                float v299;
                v299 = v291 + v298;
                v291 = v299;
                v294 += 1l ;
            }
            v292 += 1l ;
        }
        Closure1 v300{};
        float v301;
        v301 = cooperative_groups::reduce(v244, v291, v300);
        float v302;
        v302 = v301 / 256.0f;
        float v303[8l];
        long v304;
        v304 = 0l;
        while (while_method_3(v304)){
            long v306;
            v306 = 0l;
            while (while_method_1(v306)){
                assert("Tensor range check" && 0 <= v304 && v304 < 2l);
                assert("Tensor range check" && 0 <= v306 && v306 < 4l);
                long v308;
                v308 = 4l * v304;
                long v309;
                v309 = v308 + v306;
                float v310;
                v310 = v252[v309];
                float v311;
                v311 = v310 - v302;
                float v312;
                v312 = exp(v311);
                assert("Tensor range check" && 0 <= v304 && v304 < 2l);
                assert("Tensor range check" && 0 <= v306 && v306 < 4l);
                v303[v309] = v312;
                v306 += 1l ;
            }
            v304 += 1l ;
        }
        float v313;
        v313 = 0.0f;
        long v314;
        v314 = 0l;
        while (while_method_3(v314)){
            long v316;
            v316 = 0l;
            while (while_method_1(v316)){
                assert("Tensor range check" && 0 <= v314 && v314 < 2l);
                assert("Tensor range check" && 0 <= v316 && v316 < 4l);
                long v318;
                v318 = 4l * v314;
                long v319;
                v319 = v318 + v316;
                float v320;
                v320 = v303[v319];
                float v321;
                v321 = v313 + v320;
                v313 = v321;
                v316 += 1l ;
            }
            v314 += 1l ;
        }
        float v322;
        v322 = cooperative_groups::reduce(v244, v313, v300);
        float v323[8l];
        long v324;
        v324 = 0l;
        while (while_method_3(v324)){
            long v326;
            v326 = 0l;
            while (while_method_1(v326)){
                assert("Tensor range check" && 0 <= v324 && v324 < 2l);
                assert("Tensor range check" && 0 <= v326 && v326 < 4l);
                long v328;
                v328 = 4l * v324;
                long v329;
                v329 = v328 + v326;
                float v330;
                v330 = v303[v329];
                float v331;
                v331 = v330 / v322;
                assert("Tensor range check" && 0 <= v324 && v324 < 2l);
                assert("Tensor range check" && 0 <= v326 && v326 < 4l);
                v323[v329] = v331;
                v326 += 1l ;
            }
            v324 += 1l ;
        }
        long v332;
        v332 = 0l;
        while (while_method_3(v332)){
            assert("Tensor range check" && 0 <= v332 && v332 < 2l);
            long v334;
            v334 = 128l * v332;
            long v335;
            v335 = v334 + v251;
            assert("Tensor range check" && 0 <= v332 && v332 < 2l);
            long v336;
            v336 = 4l * v332;
            int4* v337;
            v337 = reinterpret_cast<int4*>(v323 + v336);
            int4* v338;
            v338 = reinterpret_cast<int4*>(v4 + v335);
            assert("Pointer alignment check" && (unsigned long long)(v337) % 4l == 0 && (unsigned long long)(v338) % 4l == 0);
            *v338 = *v337;
            v332 += 1l ;
        }
        v248 += 1l ;
    }
    __syncthreads();
    long v339;
    v339 = threadIdx.x;
    bool v340;
    v340 = 0l <= v339;
    bool v341;
    v341 = v340 == false;
    if (v341){
        assert("The index needs to be zero or positive." && v340);
    } else {
    }
    long v342;
    v342 = v339 % 32l;
    long v343;
    v343 = v339 / 32l;
    bool v344;
    v344 = v343 < 1l;
    bool v345;
    v345 = v344 == false;
    if (v345){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v344);
    } else {
    }
    auto & v346 = v14;
    assert("Tensor range check" && 0 <= v343 && v343 < 1l);
    assert("Tensor range check" && 0 <= v342 && v342 < 32l);
    long v347;
    v347 = 4l * v342;
    long v348;
    v348 = 256l * v343;
    long v349;
    v349 = v348 + v347;
    assert("Tensor range check" && 0 <= v343 && v343 < 1l);
    assert("Tensor range check" && 0 <= v342 && v342 < 32l);
    long v350;
    v350 = 0l;
    while (while_method_2(v350)){
        assert("Tensor range check" && 0 <= v350 && v350 < 32l);
        long v352;
        v352 = 256l * v350;
        long v353;
        v353 = v352 + v349;
        assert("Tensor range check" && 0 <= v350 && v350 < 32l);
        float v354[8l];
        long v355[8l];
        long v356;
        v356 = 0l;
        while (while_method_3(v356)){
            assert("Tensor range check" && 0 <= v356 && v356 < 2l);
            long v358;
            v358 = 4l * v356;
            assert("Tensor range check" && 0 <= v356 && v356 < 2l);
            long v359;
            v359 = 128l * v356;
            long v360;
            v360 = v359 + v353;
            int4* v361;
            v361 = reinterpret_cast<int4*>(v1 + v360);
            int4* v362;
            v362 = reinterpret_cast<int4*>(v354 + v358);
            assert("Pointer alignment check" && (unsigned long long)(v361) % 4l == 0 && (unsigned long long)(v362) % 4l == 0);
            *v362 = *v361;
            v356 += 1l ;
        }
        long v363;
        v363 = 0l;
        while (while_method_3(v363)){
            long v365;
            v365 = 0l;
            while (while_method_1(v365)){
                bool v367;
                v367 = 0l <= v365;
                bool v369;
                if (v367){
                    bool v368;
                    v368 = v365 < 4l;
                    v369 = v368;
                } else {
                    v369 = false;
                }
                bool v370;
                v370 = v369 == false;
                if (v370){
                    assert("The indices should be inside the range of the dimension." && v369);
                } else {
                }
                bool v371;
                v371 = 0l <= v342;
                bool v373;
                if (v371){
                    bool v372;
                    v372 = v342 < 32l;
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
                long v375;
                v375 = v342 * 4l;
                long v376;
                v376 = v365 + v375;
                bool v377;
                v377 = 0l <= v363;
                bool v379;
                if (v377){
                    bool v378;
                    v378 = v363 < 2l;
                    v379 = v378;
                } else {
                    v379 = false;
                }
                bool v380;
                v380 = v379 == false;
                if (v380){
                    assert("The indices should be inside the range of the dimension." && v379);
                } else {
                }
                long v381;
                v381 = v363 * 128l;
                long v382;
                v382 = v376 + v381;
                assert("Tensor range check" && 0 <= v363 && v363 < 2l);
                assert("Tensor range check" && 0 <= v365 && v365 < 4l);
                long v383;
                v383 = 4l * v363;
                long v384;
                v384 = v383 + v365;
                v355[v384] = v382;
                v365 += 1l ;
            }
            v363 += 1l ;
        }
        bool v385;
        v385 = 0l <= v343;
        bool v386;
        v386 = v385 && v344;
        bool v387;
        v387 = v386 == false;
        if (v387){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v386);
        } else {
        }
        bool v388;
        v388 = 0l <= v350;
        bool v390;
        if (v388){
            bool v389;
            v389 = v350 < 32l;
            v390 = v389;
        } else {
            v390 = false;
        }
        bool v391;
        v391 = v390 == false;
        if (v391){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v390);
        } else {
        }
        long v392;
        v392 = v350 + v343;
        float v393[8l];
        long v394;
        v394 = 0l;
        while (while_method_3(v394)){
            long v396;
            v396 = 0l;
            while (while_method_1(v396)){
                assert("Tensor range check" && 0 <= v394 && v394 < 2l);
                assert("Tensor range check" && 0 <= v396 && v396 < 4l);
                long v398;
                v398 = 4l * v394;
                long v399;
                v399 = v398 + v396;
                float v400;
                v400 = v354[v399];
                float v401;
                v401 = v400 * v400;
                assert("Tensor range check" && 0 <= v394 && v394 < 2l);
                assert("Tensor range check" && 0 <= v396 && v396 < 4l);
                v393[v399] = v401;
                v396 += 1l ;
            }
            v394 += 1l ;
        }
        float v402;
        v402 = 0.0f;
        long v403;
        v403 = 0l;
        while (while_method_3(v403)){
            long v405;
            v405 = 0l;
            while (while_method_1(v405)){
                assert("Tensor range check" && 0 <= v403 && v403 < 2l);
                assert("Tensor range check" && 0 <= v405 && v405 < 4l);
                long v407;
                v407 = 4l * v403;
                long v408;
                v408 = v407 + v405;
                float v409;
                v409 = v393[v408];
                float v410;
                v410 = v402 + v409;
                v402 = v410;
                v405 += 1l ;
            }
            v403 += 1l ;
        }
        Closure1 v411{};
        float v412;
        v412 = cooperative_groups::reduce(v346, v402, v411);
        float v413[8l];
        long v414;
        v414 = 0l;
        while (while_method_3(v414)){
            long v416;
            v416 = 0l;
            while (while_method_1(v416)){
                assert("Tensor range check" && 0 <= v414 && v414 < 2l);
                assert("Tensor range check" && 0 <= v416 && v416 < 4l);
                long v418;
                v418 = 4l * v414;
                long v419;
                v419 = v418 + v416;
                float v420;
                v420 = v393[v419];
                float v421;
                v421 = v420 / v412;
                assert("Tensor range check" && 0 <= v414 && v414 < 2l);
                assert("Tensor range check" && 0 <= v416 && v416 < 4l);
                v413[v419] = v421;
                v416 += 1l ;
            }
            v414 += 1l ;
        }
        long v422;
        v422 = 0l;
        while (while_method_3(v422)){
            assert("Tensor range check" && 0 <= v422 && v422 < 2l);
            long v424;
            v424 = 128l * v422;
            long v425;
            v425 = v424 + v353;
            assert("Tensor range check" && 0 <= v422 && v422 < 2l);
            long v426;
            v426 = 4l * v422;
            int4* v427;
            v427 = reinterpret_cast<int4*>(v413 + v426);
            int4* v428;
            v428 = reinterpret_cast<int4*>(v7 + v425);
            assert("Pointer alignment check" && (unsigned long long)(v427) % 4l == 0 && (unsigned long long)(v428) % 4l == 0);
            *v428 = *v427;
            v422 += 1l ;
        }
        v350 += 1l ;
    }
    __syncthreads();
    long v429;
    v429 = threadIdx.x;
    bool v430;
    v430 = 0l <= v429;
    bool v431;
    v431 = v430 == false;
    if (v431){
        assert("The index needs to be zero or positive." && v430);
    } else {
    }
    long v432;
    v432 = v429 % 32l;
    long v433;
    v433 = v429 / 32l;
    bool v434;
    v434 = v433 < 1l;
    bool v435;
    v435 = v434 == false;
    if (v435){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v434);
    } else {
    }
    auto & v436 = v14;
    assert("Tensor range check" && 0 <= v433 && v433 < 1l);
    assert("Tensor range check" && 0 <= v432 && v432 < 32l);
    long v437;
    v437 = 4l * v432;
    long v438;
    v438 = 256l * v433;
    long v439;
    v439 = v438 + v437;
    assert("Tensor range check" && 0 <= v433 && v433 < 1l);
    long v440;
    v440 = 0l;
    while (while_method_2(v440)){
        assert("Tensor range check" && 0 <= v440 && v440 < 32l);
        long v442;
        v442 = 256l * v440;
        long v443;
        v443 = v442 + v439;
        float v444[8l];
        long v445[8l];
        long v446;
        v446 = 0l;
        while (while_method_3(v446)){
            assert("Tensor range check" && 0 <= v446 && v446 < 2l);
            long v448;
            v448 = 4l * v446;
            assert("Tensor range check" && 0 <= v446 && v446 < 2l);
            long v449;
            v449 = 128l * v446;
            long v450;
            v450 = v449 + v443;
            int4* v451;
            v451 = reinterpret_cast<int4*>(v1 + v450);
            int4* v452;
            v452 = reinterpret_cast<int4*>(v444 + v448);
            assert("Pointer alignment check" && (unsigned long long)(v451) % 4l == 0 && (unsigned long long)(v452) % 4l == 0);
            *v452 = *v451;
            v446 += 1l ;
        }
        long v453;
        v453 = 0l;
        while (while_method_3(v453)){
            long v455;
            v455 = 0l;
            while (while_method_1(v455)){
                bool v457;
                v457 = 0l <= v455;
                bool v459;
                if (v457){
                    bool v458;
                    v458 = v455 < 4l;
                    v459 = v458;
                } else {
                    v459 = false;
                }
                bool v460;
                v460 = v459 == false;
                if (v460){
                    assert("The indices should be inside the range of the dimension." && v459);
                } else {
                }
                bool v461;
                v461 = 0l <= v432;
                bool v463;
                if (v461){
                    bool v462;
                    v462 = v432 < 32l;
                    v463 = v462;
                } else {
                    v463 = false;
                }
                bool v464;
                v464 = v463 == false;
                if (v464){
                    assert("The indices should be inside the range of the dimension." && v463);
                } else {
                }
                long v465;
                v465 = v432 * 4l;
                long v466;
                v466 = v455 + v465;
                bool v467;
                v467 = 0l <= v453;
                bool v469;
                if (v467){
                    bool v468;
                    v468 = v453 < 2l;
                    v469 = v468;
                } else {
                    v469 = false;
                }
                bool v470;
                v470 = v469 == false;
                if (v470){
                    assert("The indices should be inside the range of the dimension." && v469);
                } else {
                }
                long v471;
                v471 = v453 * 128l;
                long v472;
                v472 = v466 + v471;
                assert("Tensor range check" && 0 <= v453 && v453 < 2l);
                assert("Tensor range check" && 0 <= v455 && v455 < 4l);
                long v473;
                v473 = 4l * v453;
                long v474;
                v474 = v473 + v455;
                v445[v474] = v472;
                v455 += 1l ;
            }
            v453 += 1l ;
        }
        bool v475;
        v475 = 0l <= v433;
        bool v476;
        v476 = v475 && v434;
        bool v477;
        v477 = v476 == false;
        if (v477){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v476);
        } else {
        }
        bool v478;
        v478 = 0l <= v440;
        bool v480;
        if (v478){
            bool v479;
            v479 = v440 < 32l;
            v480 = v479;
        } else {
            v480 = false;
        }
        bool v481;
        v481 = v480 == false;
        if (v481){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v480);
        } else {
        }
        long v482;
        v482 = v440 + v433;
        float v483; long v484;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0l};
        v483 = tmp1.v0; v484 = tmp1.v1;
        long v485;
        v485 = 0l;
        while (while_method_3(v485)){
            long v487;
            v487 = 0l;
            while (while_method_1(v487)){
                assert("Tensor range check" && 0 <= v485 && v485 < 2l);
                assert("Tensor range check" && 0 <= v487 && v487 < 4l);
                long v489;
                v489 = 4l * v485;
                long v490;
                v490 = v489 + v487;
                float v491;
                v491 = v444[v490];
                long v492;
                v492 = v445[v490];
                bool v493;
                v493 = v483 > v491;
                float v494; long v495;
                if (v493){
                    v494 = v483; v495 = v484;
                } else {
                    v494 = v491; v495 = v492;
                }
                v483 = v494;
                v484 = v495;
                v487 += 1l ;
            }
            v485 += 1l ;
        }
        Closure2 v496{};
        float v497; long v498;
        Tuple1 tmp2 = cooperative_groups::reduce(v436, Tuple1{v483, v484}, v496);
        v497 = tmp2.v0; v498 = tmp2.v1;
        assert("Tensor range check" && 0 <= v440 && v440 < 32l);
        v8[v482] = v498;
        v440 += 1l ;
    }
    __syncthreads();
    long v499;
    v499 = threadIdx.x;
    bool v500;
    v500 = 0l <= v499;
    bool v501;
    v501 = v500 == false;
    if (v501){
        assert("The index needs to be zero or positive." && v500);
    } else {
    }
    long v502;
    v502 = v499 % 32l;
    long v503;
    v503 = v499 / 32l;
    bool v504;
    v504 = v503 < 1l;
    bool v505;
    v505 = v504 == false;
    if (v505){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v504);
    } else {
    }
    auto & v506 = v14;
    assert("Tensor range check" && 0 <= v503 && v503 < 1l);
    assert("Tensor range check" && 0 <= v502 && v502 < 32l);
    long v507;
    v507 = 4l * v502;
    long v508;
    v508 = 256l * v503;
    long v509;
    v509 = v508 + v507;
    assert("Tensor range check" && 0 <= v503 && v503 < 1l);
    assert("Tensor range check" && 0 <= v502 && v502 < 32l);
    long v510;
    v510 = 0l;
    while (while_method_2(v510)){
        assert("Tensor range check" && 0 <= v510 && v510 < 32l);
        long v512;
        v512 = 256l * v510;
        long v513;
        v513 = v512 + v509;
        assert("Tensor range check" && 0 <= v510 && v510 < 32l);
        float v514[8l];
        long v515[8l];
        long v516;
        v516 = 0l;
        while (while_method_3(v516)){
            assert("Tensor range check" && 0 <= v516 && v516 < 2l);
            long v518;
            v518 = 4l * v516;
            assert("Tensor range check" && 0 <= v516 && v516 < 2l);
            long v519;
            v519 = 128l * v516;
            long v520;
            v520 = v519 + v513;
            int4* v521;
            v521 = reinterpret_cast<int4*>(v1 + v520);
            int4* v522;
            v522 = reinterpret_cast<int4*>(v514 + v518);
            assert("Pointer alignment check" && (unsigned long long)(v521) % 4l == 0 && (unsigned long long)(v522) % 4l == 0);
            *v522 = *v521;
            v516 += 1l ;
        }
        long v523;
        v523 = 0l;
        while (while_method_3(v523)){
            long v525;
            v525 = 0l;
            while (while_method_1(v525)){
                bool v527;
                v527 = 0l <= v525;
                bool v529;
                if (v527){
                    bool v528;
                    v528 = v525 < 4l;
                    v529 = v528;
                } else {
                    v529 = false;
                }
                bool v530;
                v530 = v529 == false;
                if (v530){
                    assert("The indices should be inside the range of the dimension." && v529);
                } else {
                }
                bool v531;
                v531 = 0l <= v502;
                bool v533;
                if (v531){
                    bool v532;
                    v532 = v502 < 32l;
                    v533 = v532;
                } else {
                    v533 = false;
                }
                bool v534;
                v534 = v533 == false;
                if (v534){
                    assert("The indices should be inside the range of the dimension." && v533);
                } else {
                }
                long v535;
                v535 = v502 * 4l;
                long v536;
                v536 = v525 + v535;
                bool v537;
                v537 = 0l <= v523;
                bool v539;
                if (v537){
                    bool v538;
                    v538 = v523 < 2l;
                    v539 = v538;
                } else {
                    v539 = false;
                }
                bool v540;
                v540 = v539 == false;
                if (v540){
                    assert("The indices should be inside the range of the dimension." && v539);
                } else {
                }
                long v541;
                v541 = v523 * 128l;
                long v542;
                v542 = v536 + v541;
                assert("Tensor range check" && 0 <= v523 && v523 < 2l);
                assert("Tensor range check" && 0 <= v525 && v525 < 4l);
                long v543;
                v543 = 4l * v523;
                long v544;
                v544 = v543 + v525;
                v515[v544] = v542;
                v525 += 1l ;
            }
            v523 += 1l ;
        }
        bool v545;
        v545 = 0l <= v503;
        bool v546;
        v546 = v545 && v504;
        bool v547;
        v547 = v546 == false;
        if (v547){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v546);
        } else {
        }
        bool v548;
        v548 = 0l <= v510;
        bool v550;
        if (v548){
            bool v549;
            v549 = v510 < 32l;
            v550 = v549;
        } else {
            v550 = false;
        }
        bool v551;
        v551 = v550 == false;
        if (v551){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v550);
        } else {
        }
        long v552;
        v552 = v510 + v503;
        float v553;
        v553 = 0.0f;
        long v554;
        v554 = 0l;
        while (while_method_3(v554)){
            long v556;
            v556 = 0l;
            while (while_method_1(v556)){
                assert("Tensor range check" && 0 <= v554 && v554 < 2l);
                assert("Tensor range check" && 0 <= v556 && v556 < 4l);
                long v558;
                v558 = 4l * v554;
                long v559;
                v559 = v558 + v556;
                float v560;
                v560 = v514[v559];
                float v561;
                v561 = v553 + v560;
                v553 = v561;
                v556 += 1l ;
            }
            v554 += 1l ;
        }
        Closure1 v562{};
        float v563;
        v563 = cooperative_groups::reduce(v506, v553, v562);
        float v564;
        v564 = v563 / 256.0f;
        float v565[8l];
        long v566;
        v566 = 0l;
        while (while_method_3(v566)){
            long v568;
            v568 = 0l;
            while (while_method_1(v568)){
                assert("Tensor range check" && 0 <= v566 && v566 < 2l);
                assert("Tensor range check" && 0 <= v568 && v568 < 4l);
                long v570;
                v570 = 4l * v566;
                long v571;
                v571 = v570 + v568;
                float v572;
                v572 = v514[v571];
                float v573;
                v573 = v572 - v564;
                float v574;
                v574 = exp(v573);
                assert("Tensor range check" && 0 <= v566 && v566 < 2l);
                assert("Tensor range check" && 0 <= v568 && v568 < 4l);
                v565[v571] = v574;
                v568 += 1l ;
            }
            v566 += 1l ;
        }
        float v575;
        v575 = 0.0f;
        long v576;
        v576 = 0l;
        while (while_method_3(v576)){
            long v578;
            v578 = 0l;
            while (while_method_1(v578)){
                assert("Tensor range check" && 0 <= v576 && v576 < 2l);
                assert("Tensor range check" && 0 <= v578 && v578 < 4l);
                long v580;
                v580 = 4l * v576;
                long v581;
                v581 = v580 + v578;
                float v582;
                v582 = v565[v581];
                float v583;
                v583 = v575 + v582;
                v575 = v583;
                v578 += 1l ;
            }
            v576 += 1l ;
        }
        float v584;
        v584 = cooperative_groups::reduce(v506, v575, v562);
        float v585[8l];
        long v586;
        v586 = 0l;
        while (while_method_3(v586)){
            long v588;
            v588 = 0l;
            while (while_method_1(v588)){
                assert("Tensor range check" && 0 <= v586 && v586 < 2l);
                assert("Tensor range check" && 0 <= v588 && v588 < 4l);
                long v590;
                v590 = 4l * v586;
                long v591;
                v591 = v590 + v588;
                float v592;
                v592 = v565[v591];
                float v593;
                v593 = v592 / v584;
                assert("Tensor range check" && 0 <= v586 && v586 < 2l);
                assert("Tensor range check" && 0 <= v588 && v588 < 4l);
                v585[v591] = v593;
                v588 += 1l ;
            }
            v586 += 1l ;
        }
        float v594[8l];
        float v595;
        v595 = 0.0f;
        long v596;
        v596 = 0l;
        while (while_method_3(v596)){
            assert("Tensor range check" && 0 <= v596 && v596 < 2l);
            long v598;
            v598 = 4l * v596;
            assert("Tensor range check" && 0 <= v596 && v596 < 2l);
            long v599; float v600;
            Tuple0 tmp3 = Tuple0{0l, 0.0f};
            v599 = tmp3.v0; v600 = tmp3.v1;
            while (while_method_1(v599)){
                assert("Tensor range check" && 0 <= v599 && v599 < 4l);
                long v602;
                v602 = v599 + v598;
                float v603;
                v603 = v585[v602];
                float v604;
                v604 = v600 + v603;
                v600 = v604;
                v599 += 1l ;
            }
            Closure3 v605{};
            float v606;
            v606 = cooperative_groups::inclusive_scan(v506, v600, v605);
            float v607;
            v607 = v506.shfl(v606,v506.num_threads()-1);
            bool v608;
            v608 = v506.num_threads() <= 32;
            bool v609;
            v609 = v608 == false;
            if (v609){
                assert("The thread block tile in the exclusive scan has to be less than or equal 32." && v608);
            } else {
            }
            float v610;
            v610 = v506.shfl_up(v606,1);
            bool v611;
            v611 = v506.thread_rank() == 0;
            float v612;
            if (v611){
                v612 = 0.0f;
            } else {
                v612 = v610;
            }
            float v613;
            v613 = v595 + v612;
            long v614; float v615;
            Tuple0 tmp4 = Tuple0{0l, v613};
            v614 = tmp4.v0; v615 = tmp4.v1;
            while (while_method_1(v614)){
                assert("Tensor range check" && 0 <= v614 && v614 < 4l);
                long v617;
                v617 = v614 + v598;
                float v618;
                v618 = v585[v617];
                float v619;
                v619 = v615 + v618;
                assert("Tensor range check" && 0 <= v614 && v614 < 4l);
                v594[v617] = v619;
                v615 = v619;
                v614 += 1l ;
            }
            float v620;
            v620 = v595 + v607;
            v595 = v620;
            v596 += 1l ;
        }
        long v621;
        v621 = 0l;
        while (while_method_3(v621)){
            assert("Tensor range check" && 0 <= v621 && v621 < 2l);
            long v623;
            v623 = 128l * v621;
            long v624;
            v624 = v623 + v513;
            assert("Tensor range check" && 0 <= v621 && v621 < 2l);
            long v625;
            v625 = 4l * v621;
            int4* v626;
            v626 = reinterpret_cast<int4*>(v585 + v625);
            int4* v627;
            v627 = reinterpret_cast<int4*>(v5 + v624);
            assert("Pointer alignment check" && (unsigned long long)(v626) % 4l == 0 && (unsigned long long)(v627) % 4l == 0);
            *v627 = *v626;
            int4* v628;
            v628 = reinterpret_cast<int4*>(v594 + v625);
            int4* v629;
            v629 = reinterpret_cast<int4*>(v6 + v624);
            assert("Pointer alignment check" && (unsigned long long)(v628) % 4l == 0 && (unsigned long long)(v629) % 4l == 0);
            *v629 = *v628;
            v621 += 1l ;
        }
        v510 += 1l ;
    }
    __syncthreads();
    long v630;
    v630 = threadIdx.x;
    bool v631;
    v631 = 0l <= v630;
    bool v632;
    v632 = v631 == false;
    if (v632){
        assert("The index needs to be zero or positive." && v631);
    } else {
    }
    long v633;
    v633 = v630 % 32l;
    long v634;
    v634 = v630 / 32l;
    bool v635;
    v635 = v634 < 1l;
    bool v636;
    v636 = v635 == false;
    if (v636){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v635);
    } else {
    }
    auto & v637 = v14;
    assert("Tensor range check" && 0 <= v634 && v634 < 1l);
    assert("Tensor range check" && 0 <= v633 && v633 < 32l);
    long v638;
    v638 = 4l * v633;
    long v639;
    v639 = 256l * v634;
    long v640;
    v640 = v639 + v638;
    assert("Tensor range check" && 0 <= v634 && v634 < 1l);
    long v641;
    v641 = 0l;
    while (while_method_2(v641)){
        assert("Tensor range check" && 0 <= v641 && v641 < 32l);
        long v643;
        v643 = 256l * v641;
        long v644;
        v644 = v643 + v640;
        float v645[8l];
        long v646[8l];
        long v647;
        v647 = 0l;
        while (while_method_3(v647)){
            assert("Tensor range check" && 0 <= v647 && v647 < 2l);
            long v649;
            v649 = 4l * v647;
            assert("Tensor range check" && 0 <= v647 && v647 < 2l);
            long v650;
            v650 = 128l * v647;
            long v651;
            v651 = v650 + v644;
            int4* v652;
            v652 = reinterpret_cast<int4*>(v1 + v651);
            int4* v653;
            v653 = reinterpret_cast<int4*>(v645 + v649);
            assert("Pointer alignment check" && (unsigned long long)(v652) % 4l == 0 && (unsigned long long)(v653) % 4l == 0);
            *v653 = *v652;
            v647 += 1l ;
        }
        long v654;
        v654 = 0l;
        while (while_method_3(v654)){
            long v656;
            v656 = 0l;
            while (while_method_1(v656)){
                bool v658;
                v658 = 0l <= v656;
                bool v660;
                if (v658){
                    bool v659;
                    v659 = v656 < 4l;
                    v660 = v659;
                } else {
                    v660 = false;
                }
                bool v661;
                v661 = v660 == false;
                if (v661){
                    assert("The indices should be inside the range of the dimension." && v660);
                } else {
                }
                bool v662;
                v662 = 0l <= v633;
                bool v664;
                if (v662){
                    bool v663;
                    v663 = v633 < 32l;
                    v664 = v663;
                } else {
                    v664 = false;
                }
                bool v665;
                v665 = v664 == false;
                if (v665){
                    assert("The indices should be inside the range of the dimension." && v664);
                } else {
                }
                long v666;
                v666 = v633 * 4l;
                long v667;
                v667 = v656 + v666;
                bool v668;
                v668 = 0l <= v654;
                bool v670;
                if (v668){
                    bool v669;
                    v669 = v654 < 2l;
                    v670 = v669;
                } else {
                    v670 = false;
                }
                bool v671;
                v671 = v670 == false;
                if (v671){
                    assert("The indices should be inside the range of the dimension." && v670);
                } else {
                }
                long v672;
                v672 = v654 * 128l;
                long v673;
                v673 = v667 + v672;
                assert("Tensor range check" && 0 <= v654 && v654 < 2l);
                assert("Tensor range check" && 0 <= v656 && v656 < 4l);
                long v674;
                v674 = 4l * v654;
                long v675;
                v675 = v674 + v656;
                v646[v675] = v673;
                v656 += 1l ;
            }
            v654 += 1l ;
        }
        bool v676;
        v676 = 0l <= v634;
        bool v677;
        v677 = v676 && v635;
        bool v678;
        v678 = v677 == false;
        if (v678){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v677);
        } else {
        }
        bool v679;
        v679 = 0l <= v641;
        bool v681;
        if (v679){
            bool v680;
            v680 = v641 < 32l;
            v681 = v680;
        } else {
            v681 = false;
        }
        bool v682;
        v682 = v681 == false;
        if (v682){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v681);
        } else {
        }
        long v683;
        v683 = v641 + v634;
        float v684;
        v684 = 0.0f;
        long v685;
        v685 = 0l;
        while (while_method_3(v685)){
            long v687;
            v687 = 0l;
            while (while_method_1(v687)){
                assert("Tensor range check" && 0 <= v685 && v685 < 2l);
                assert("Tensor range check" && 0 <= v687 && v687 < 4l);
                long v689;
                v689 = 4l * v685;
                long v690;
                v690 = v689 + v687;
                float v691;
                v691 = v645[v690];
                float v692;
                v692 = v684 + v691;
                v684 = v692;
                v687 += 1l ;
            }
            v685 += 1l ;
        }
        Closure1 v693{};
        float v694;
        v694 = cooperative_groups::reduce(v637, v684, v693);
        float v695;
        v695 = v694 / 256.0f;
        float v696[8l];
        long v697;
        v697 = 0l;
        while (while_method_3(v697)){
            long v699;
            v699 = 0l;
            while (while_method_1(v699)){
                assert("Tensor range check" && 0 <= v697 && v697 < 2l);
                assert("Tensor range check" && 0 <= v699 && v699 < 4l);
                long v701;
                v701 = 4l * v697;
                long v702;
                v702 = v701 + v699;
                float v703;
                v703 = v645[v702];
                float v704;
                v704 = v703 - v695;
                float v705;
                v705 = exp(v704);
                assert("Tensor range check" && 0 <= v697 && v697 < 2l);
                assert("Tensor range check" && 0 <= v699 && v699 < 4l);
                v696[v702] = v705;
                v699 += 1l ;
            }
            v697 += 1l ;
        }
        float v706;
        v706 = 0.0f;
        long v707;
        v707 = 0l;
        while (while_method_3(v707)){
            long v709;
            v709 = 0l;
            while (while_method_1(v709)){
                assert("Tensor range check" && 0 <= v707 && v707 < 2l);
                assert("Tensor range check" && 0 <= v709 && v709 < 4l);
                long v711;
                v711 = 4l * v707;
                long v712;
                v712 = v711 + v709;
                float v713;
                v713 = v696[v712];
                float v714;
                v714 = v706 + v713;
                v706 = v714;
                v709 += 1l ;
            }
            v707 += 1l ;
        }
        float v715;
        v715 = cooperative_groups::reduce(v637, v706, v693);
        float v716[8l];
        long v717;
        v717 = 0l;
        while (while_method_3(v717)){
            long v719;
            v719 = 0l;
            while (while_method_1(v719)){
                assert("Tensor range check" && 0 <= v717 && v717 < 2l);
                assert("Tensor range check" && 0 <= v719 && v719 < 4l);
                long v721;
                v721 = 4l * v717;
                long v722;
                v722 = v721 + v719;
                float v723;
                v723 = v696[v722];
                float v724;
                v724 = v723 / v715;
                assert("Tensor range check" && 0 <= v717 && v717 < 2l);
                assert("Tensor range check" && 0 <= v719 && v719 < 4l);
                v716[v722] = v724;
                v719 += 1l ;
            }
            v717 += 1l ;
        }
        float v725[8l];
        float v726;
        v726 = 0.0f;
        long v727;
        v727 = 0l;
        while (while_method_3(v727)){
            assert("Tensor range check" && 0 <= v727 && v727 < 2l);
            long v729;
            v729 = 4l * v727;
            assert("Tensor range check" && 0 <= v727 && v727 < 2l);
            long v730; float v731;
            Tuple0 tmp5 = Tuple0{0l, 0.0f};
            v730 = tmp5.v0; v731 = tmp5.v1;
            while (while_method_1(v730)){
                assert("Tensor range check" && 0 <= v730 && v730 < 4l);
                long v733;
                v733 = v730 + v729;
                float v734;
                v734 = v716[v733];
                float v735;
                v735 = v731 + v734;
                v731 = v735;
                v730 += 1l ;
            }
            Closure3 v736{};
            float v737;
            v737 = cooperative_groups::inclusive_scan(v637, v731, v736);
            float v738;
            v738 = v637.shfl(v737,v637.num_threads()-1);
            bool v739;
            v739 = v637.num_threads() <= 32;
            bool v740;
            v740 = v739 == false;
            if (v740){
                assert("The thread block tile in the exclusive scan has to be less than or equal 32." && v739);
            } else {
            }
            float v741;
            v741 = v637.shfl_up(v737,1);
            bool v742;
            v742 = v637.thread_rank() == 0;
            float v743;
            if (v742){
                v743 = 0.0f;
            } else {
                v743 = v741;
            }
            float v744;
            v744 = v726 + v743;
            long v745; float v746;
            Tuple0 tmp6 = Tuple0{0l, v744};
            v745 = tmp6.v0; v746 = tmp6.v1;
            while (while_method_1(v745)){
                assert("Tensor range check" && 0 <= v745 && v745 < 4l);
                long v748;
                v748 = v745 + v729;
                float v749;
                v749 = v716[v748];
                float v750;
                v750 = v746 + v749;
                assert("Tensor range check" && 0 <= v745 && v745 < 4l);
                v725[v748] = v750;
                v746 = v750;
                v745 += 1l ;
            }
            float v751;
            v751 = v726 + v738;
            v726 = v751;
            v727 += 1l ;
        }
        assert("Tensor range check" && 0 <= v683 && v683 < 32l);
        float v752;
        v752 = v2[v683];
        float v753[8l];
        long v754;
        v754 = 0l;
        while (while_method_3(v754)){
            long v756;
            v756 = 0l;
            while (while_method_1(v756)){
                assert("Tensor range check" && 0 <= v754 && v754 < 2l);
                assert("Tensor range check" && 0 <= v756 && v756 < 4l);
                long v758;
                v758 = 4l * v754;
                long v759;
                v759 = v758 + v756;
                float v760;
                v760 = v725[v759];
                float v761;
                v761 = v760 - v752;
                assert("Tensor range check" && 0 <= v754 && v754 < 2l);
                assert("Tensor range check" && 0 <= v756 && v756 < 4l);
                v753[v759] = v761;
                v756 += 1l ;
            }
            v754 += 1l ;
        }
        float v762; long v763;
        Tuple1 tmp7 = Tuple1{-1.0f / 0.0f, 0l};
        v762 = tmp7.v0; v763 = tmp7.v1;
        long v764;
        v764 = 0l;
        while (while_method_3(v764)){
            long v766;
            v766 = 0l;
            while (while_method_1(v766)){
                assert("Tensor range check" && 0 <= v764 && v764 < 2l);
                assert("Tensor range check" && 0 <= v766 && v766 < 4l);
                long v768;
                v768 = 4l * v764;
                long v769;
                v769 = v768 + v766;
                float v770;
                v770 = v753[v769];
                long v771;
                v771 = v646[v769];
                bool v772;
                v772 = v762 >= 0.0f;
                bool v774;
                if (v772){
                    bool v773;
                    v773 = v770 >= 0.0f;
                    v774 = v773;
                } else {
                    v774 = false;
                }
                float v783; long v784;
                if (v774){
                    bool v775;
                    v775 = v762 <= v770;
                    if (v775){
                        v783 = v762; v784 = v763;
                    } else {
                        v783 = v770; v784 = v771;
                    }
                } else {
                    if (v772){
                        v783 = v762; v784 = v763;
                    } else {
                        bool v778;
                        v778 = v770 >= 0.0f;
                        if (v778){
                            v783 = v770; v784 = v771;
                        } else {
                            v783 = v762; v784 = v763;
                        }
                    }
                }
                v762 = v783;
                v763 = v784;
                v766 += 1l ;
            }
            v764 += 1l ;
        }
        Closure4 v785{};
        float v786; long v787;
        Tuple1 tmp8 = cooperative_groups::reduce(v637, Tuple1{v762, v763}, v785);
        v786 = tmp8.v0; v787 = tmp8.v1;
        assert("Tensor range check" && 0 <= v641 && v641 < 32l);
        v9[v683] = v787;
        v641 += 1l ;
    }
    __syncthreads();
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
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

options = []
options.append('--diag-suppress=550,20012,68')
options.append('--dopt=on')
options.append('--restrict')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method0(v0 : char) -> None:
    print(v0, end="")
    del v0
    return 
def method1(v0 : i32) -> bool:
    v1 = v0 < 32
    del v0
    return v1
def method2(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method3(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def method4() -> None:
    return 
def main():
    v0 = cp.arange(0,8192,1,dtype=cp.float32) # type: ignore
    v1 = v0.size
    v2 = 8192 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v6 = cp.random.uniform(size=32,dtype=cp.float32) # type: ignore
    v7 = cp.empty(1,dtype=cp.float32)
    v8 = cp.empty(8192,dtype=cp.float32)
    v9 = cp.empty(8192,dtype=cp.float32)
    v10 = cp.empty(8192,dtype=cp.float32)
    v11 = cp.empty(8192,dtype=cp.float32)
    v12 = cp.empty(32,dtype=cp.int32)
    v13 = cp.empty(32,dtype=cp.int32)
    v14 = cp.empty(8192,dtype=cp.int32)
    v15 = cp.empty(8192,dtype=cp.int32)
    v16 = cp.empty(8192,dtype=cp.int32)
    v17 = cp.empty(32,dtype=cp.int32)
    v18 = 0
    v19 = raw_module.get_function(f"entry{v18}")
    del v18
    v19.max_dynamic_shared_size_bytes = 0 
    v19((1,),(32,),(v0, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17),shared_mem=0)
    del v0, v5, v6, v7, v8, v9, v10, v11, v12, v14, v15, v16, v17, v19
    v20 = 0
    v21 = '['
    method0(v21)
    del v21
    v22 = 0
    while method1(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            method2(v26)
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
            method2(v29)
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v13[v22].item()
        method3(v31)
        del v31
        v22 += 1 
    del v13, v20, v22
    v32 = ']'
    method0(v32)
    del v32
    method4()
    print()
    return 

if __name__ == '__main__': print(main())
