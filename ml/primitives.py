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
    v1 = v0 < 32l;
    return v1;
}
__device__ inline bool while_method_1(long v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_2(long v0){
    bool v1;
    v1 = v0 < 1l;
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
        v21 = v17 % 1l;
        bool v22;
        v22 = v17 < 32l;
        bool v23;
        v23 = v22 == false;
        if (v23){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v22);
        } else {
        }
        assert("Tensor range check" && 0 <= v17 && v17 < 32l);
        assert("Tensor range check" && 0 <= v21 && v21 < 1l);
        long v24;
        v24 = 4l * v21;
        long v25;
        v25 = 4l * v17;
        long v26;
        v26 = v25 + v24;
        float v27[4l];
        int4* v28;
        v28 = reinterpret_cast<int4*>(v1 + v26);
        int4* v29;
        v29 = reinterpret_cast<int4*>(v27 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v28) % 4l == 0 && (unsigned long long)(v29) % 4l == 0);
        *v29 = *v28;
        long v30; float v31;
        Tuple0 tmp0 = Tuple0{0l, v15};
        v30 = tmp0.v0; v31 = tmp0.v1;
        while (while_method_1(v30)){
            assert("Tensor range check" && 0 <= v30 && v30 < 4l);
            float v33;
            v33 = v27[v30];
            float v34;
            v34 = v33 + v31;
            v31 = v34;
            v30 += 1l ;
        }
        v15 = v31;
        v17 += 32l ;
    }
    auto v35 = cooperative_groups::coalesced_threads();
    Closure0 v36{};
    float v37;
    v37 = cooperative_groups::reduce(v35, v15, v36);
    long v38;
    v38 = threadIdx.x;
    long v39;
    v39 = v38 / 32l;
    __shared__ float v40[1l];
    assert("Tensor range check" && 0 <= v39 && v39 < 1l);
    v40[v39] = v37;
    __syncthreads();
    long v41;
    v41 = threadIdx.x;
    long v42;
    v42 = v41 % 32l;
    bool v43;
    v43 = v39 == 0l;
    bool v45;
    if (v43){
        bool v44;
        v44 = v42 < 1l;
        v45 = v44;
    } else {
        v45 = false;
    }
    if (v45){
        auto v46 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v42 && v42 < 1l);
        float v47;
        v47 = v40[v42];
        float v48;
        v48 = cooperative_groups::reduce(v46, v47, v36);
        v3[0l] = v48;
    } else {
    }
    __syncthreads();
    long v49;
    v49 = threadIdx.x;
    bool v50;
    v50 = 0l <= v49;
    bool v51;
    v51 = v50 == false;
    if (v51){
        assert("The index needs to be zero or positive." && v50);
    } else {
    }
    long v52;
    v52 = v49 % 1l;
    bool v53;
    v53 = v49 < 32l;
    bool v54;
    v54 = v53 == false;
    if (v54){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v53);
    } else {
    }
    cooperative_groups::thread_block_tile<1l, cooperative_groups::thread_block> v55 = cooperative_groups::tiled_partition<1l>(v14);
    assert("Tensor range check" && 0 <= v49 && v49 < 32l);
    assert("Tensor range check" && 0 <= v52 && v52 < 1l);
    long v56;
    v56 = 4l * v52;
    long v57;
    v57 = 4l * v49;
    long v58;
    v58 = v57 + v56;
    assert("Tensor range check" && 0 <= v49 && v49 < 32l);
    assert("Tensor range check" && 0 <= v52 && v52 < 1l);
    long v59;
    v59 = 0l;
    while (while_method_2(v59)){
        assert("Tensor range check" && 0 <= v59 && v59 < 1l);
        long v61;
        v61 = 128l * v59;
        long v62;
        v62 = v61 + v58;
        assert("Tensor range check" && 0 <= v59 && v59 < 1l);
        long v63[4l];
        long v64[4l];
        long v65;
        v65 = 0l;
        while (while_method_2(v65)){
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            long v67;
            v67 = 4l * v65;
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            long v68;
            v68 = v67 + v62;
            int4* v69;
            v69 = reinterpret_cast<int4*>(v0 + v68);
            int4* v70;
            v70 = reinterpret_cast<int4*>(v63 + v67);
            assert("Pointer alignment check" && (unsigned long long)(v69) % 4l == 0 && (unsigned long long)(v70) % 4l == 0);
            *v70 = *v69;
            v65 += 1l ;
        }
        long v71;
        v71 = 0l;
        while (while_method_2(v71)){
            long v73;
            v73 = 0l;
            while (while_method_1(v73)){
                bool v75;
                v75 = 0l <= v73;
                bool v77;
                if (v75){
                    bool v76;
                    v76 = v73 < 4l;
                    v77 = v76;
                } else {
                    v77 = false;
                }
                bool v78;
                v78 = v77 == false;
                if (v78){
                    assert("The indices should be inside the range of the dimension." && v77);
                } else {
                }
                bool v79;
                v79 = 0l <= v52;
                bool v81;
                if (v79){
                    bool v80;
                    v80 = v52 < 1l;
                    v81 = v80;
                } else {
                    v81 = false;
                }
                bool v82;
                v82 = v81 == false;
                if (v82){
                    assert("The indices should be inside the range of the dimension." && v81);
                } else {
                }
                long v83;
                v83 = v52 * 4l;
                long v84;
                v84 = v73 + v83;
                bool v85;
                v85 = 0l <= v71;
                bool v87;
                if (v85){
                    bool v86;
                    v86 = v71 < 1l;
                    v87 = v86;
                } else {
                    v87 = false;
                }
                bool v88;
                v88 = v87 == false;
                if (v88){
                    assert("The indices should be inside the range of the dimension." && v87);
                } else {
                }
                long v89;
                v89 = v71 * 4l;
                long v90;
                v90 = v84 + v89;
                assert("Tensor range check" && 0 <= v71 && v71 < 1l);
                assert("Tensor range check" && 0 <= v73 && v73 < 4l);
                long v91;
                v91 = 4l * v71;
                long v92;
                v92 = v91 + v73;
                v64[v92] = v90;
                v73 += 1l ;
            }
            v71 += 1l ;
        }
        bool v93;
        v93 = v50 && v53;
        bool v94;
        v94 = v93 == false;
        if (v94){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v93);
        } else {
        }
        bool v95;
        v95 = 0l <= v59;
        bool v97;
        if (v95){
            bool v96;
            v96 = v59 < 1l;
            v97 = v96;
        } else {
            v97 = false;
        }
        bool v98;
        v98 = v97 == false;
        if (v98){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v97);
        } else {
        }
        long v99;
        v99 = v59 * 32l;
        long v100;
        v100 = v99 + v49;
        long v101;
        v101 = 0l;
        while (while_method_2(v101)){
            assert("Tensor range check" && 0 <= v101 && v101 < 1l);
            long v103;
            v103 = 4l * v101;
            long v104;
            v104 = v103 + v62;
            assert("Tensor range check" && 0 <= v101 && v101 < 1l);
            int4* v105;
            v105 = reinterpret_cast<int4*>(v63 + v103);
            int4* v106;
            v106 = reinterpret_cast<int4*>(v10 + v104);
            assert("Pointer alignment check" && (unsigned long long)(v105) % 4l == 0 && (unsigned long long)(v106) % 4l == 0);
            *v106 = *v105;
            v101 += 1l ;
        }
        v59 += 1l ;
    }
    __syncthreads();
    long v107;
    v107 = threadIdx.x;
    bool v108;
    v108 = 0l <= v107;
    bool v109;
    v109 = v108 == false;
    if (v109){
        assert("The index needs to be zero or positive." && v108);
    } else {
    }
    long v110;
    v110 = v107 % 1l;
    bool v111;
    v111 = v107 < 32l;
    bool v112;
    v112 = v111 == false;
    if (v112){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v111);
    } else {
    }
    cooperative_groups::thread_block_tile<1l, cooperative_groups::thread_block> v113 = cooperative_groups::tiled_partition<1l>(v14);
    assert("Tensor range check" && 0 <= v107 && v107 < 32l);
    assert("Tensor range check" && 0 <= v110 && v110 < 1l);
    long v114;
    v114 = 4l * v110;
    long v115;
    v115 = 4l * v107;
    long v116;
    v116 = v115 + v114;
    assert("Tensor range check" && 0 <= v107 && v107 < 32l);
    assert("Tensor range check" && 0 <= v110 && v110 < 1l);
    long v117;
    v117 = 0l;
    while (while_method_2(v117)){
        assert("Tensor range check" && 0 <= v117 && v117 < 1l);
        long v119;
        v119 = 128l * v117;
        long v120;
        v120 = v119 + v116;
        assert("Tensor range check" && 0 <= v117 && v117 < 1l);
        float v121[4l];
        long v122[4l];
        long v123;
        v123 = 0l;
        while (while_method_2(v123)){
            assert("Tensor range check" && 0 <= v123 && v123 < 1l);
            long v125;
            v125 = 4l * v123;
            assert("Tensor range check" && 0 <= v123 && v123 < 1l);
            long v126;
            v126 = v125 + v120;
            int4* v127;
            v127 = reinterpret_cast<int4*>(v1 + v126);
            int4* v128;
            v128 = reinterpret_cast<int4*>(v121 + v125);
            assert("Pointer alignment check" && (unsigned long long)(v127) % 4l == 0 && (unsigned long long)(v128) % 4l == 0);
            *v128 = *v127;
            v123 += 1l ;
        }
        long v129;
        v129 = 0l;
        while (while_method_2(v129)){
            long v131;
            v131 = 0l;
            while (while_method_1(v131)){
                bool v133;
                v133 = 0l <= v131;
                bool v135;
                if (v133){
                    bool v134;
                    v134 = v131 < 4l;
                    v135 = v134;
                } else {
                    v135 = false;
                }
                bool v136;
                v136 = v135 == false;
                if (v136){
                    assert("The indices should be inside the range of the dimension." && v135);
                } else {
                }
                bool v137;
                v137 = 0l <= v110;
                bool v139;
                if (v137){
                    bool v138;
                    v138 = v110 < 1l;
                    v139 = v138;
                } else {
                    v139 = false;
                }
                bool v140;
                v140 = v139 == false;
                if (v140){
                    assert("The indices should be inside the range of the dimension." && v139);
                } else {
                }
                long v141;
                v141 = v110 * 4l;
                long v142;
                v142 = v131 + v141;
                bool v143;
                v143 = 0l <= v129;
                bool v145;
                if (v143){
                    bool v144;
                    v144 = v129 < 1l;
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
                v147 = v129 * 4l;
                long v148;
                v148 = v142 + v147;
                assert("Tensor range check" && 0 <= v129 && v129 < 1l);
                assert("Tensor range check" && 0 <= v131 && v131 < 4l);
                long v149;
                v149 = 4l * v129;
                long v150;
                v150 = v149 + v131;
                v122[v150] = v148;
                v131 += 1l ;
            }
            v129 += 1l ;
        }
        bool v151;
        v151 = v108 && v111;
        bool v152;
        v152 = v151 == false;
        if (v152){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v151);
        } else {
        }
        bool v153;
        v153 = 0l <= v117;
        bool v155;
        if (v153){
            bool v154;
            v154 = v117 < 1l;
            v155 = v154;
        } else {
            v155 = false;
        }
        bool v156;
        v156 = v155 == false;
        if (v156){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v155);
        } else {
        }
        long v157;
        v157 = v117 * 32l;
        long v158;
        v158 = v157 + v107;
        long v159[4l];
        long v160[4l];
        long v161;
        v161 = 0l;
        while (while_method_2(v161)){
            long v163;
            v163 = 0l;
            while (while_method_1(v163)){
                assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                assert("Tensor range check" && 0 <= v163 && v163 < 4l);
                long v165;
                v165 = 4l * v161;
                long v166;
                v166 = v165 + v163;
                long v167;
                v167 = v122[v166];
                assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                assert("Tensor range check" && 0 <= v163 && v163 < 4l);
                v159[v166] = v158;
                v160[v166] = v167;
                v163 += 1l ;
            }
            v161 += 1l ;
        }
        long v168;
        v168 = 0l;
        while (while_method_2(v168)){
            assert("Tensor range check" && 0 <= v168 && v168 < 1l);
            long v170;
            v170 = 4l * v168;
            long v171;
            v171 = v170 + v120;
            assert("Tensor range check" && 0 <= v168 && v168 < 1l);
            int4* v172;
            v172 = reinterpret_cast<int4*>(v159 + v170);
            int4* v173;
            v173 = reinterpret_cast<int4*>(v11 + v171);
            assert("Pointer alignment check" && (unsigned long long)(v172) % 4l == 0 && (unsigned long long)(v173) % 4l == 0);
            *v173 = *v172;
            int4* v174;
            v174 = reinterpret_cast<int4*>(v160 + v170);
            int4* v175;
            v175 = reinterpret_cast<int4*>(v12 + v171);
            assert("Pointer alignment check" && (unsigned long long)(v174) % 4l == 0 && (unsigned long long)(v175) % 4l == 0);
            *v175 = *v174;
            v168 += 1l ;
        }
        v117 += 1l ;
    }
    __syncthreads();
    long v176;
    v176 = threadIdx.x;
    bool v177;
    v177 = 0l <= v176;
    bool v178;
    v178 = v177 == false;
    if (v178){
        assert("The index needs to be zero or positive." && v177);
    } else {
    }
    long v179;
    v179 = v176 % 1l;
    bool v180;
    v180 = v176 < 32l;
    bool v181;
    v181 = v180 == false;
    if (v181){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v180);
    } else {
    }
    cooperative_groups::thread_block_tile<1l, cooperative_groups::thread_block> v182 = cooperative_groups::tiled_partition<1l>(v14);
    assert("Tensor range check" && 0 <= v176 && v176 < 32l);
    assert("Tensor range check" && 0 <= v179 && v179 < 1l);
    long v183;
    v183 = 4l * v179;
    long v184;
    v184 = 4l * v176;
    long v185;
    v185 = v184 + v183;
    assert("Tensor range check" && 0 <= v176 && v176 < 32l);
    long v186;
    v186 = 0l;
    while (while_method_2(v186)){
        assert("Tensor range check" && 0 <= v186 && v186 < 1l);
        long v188;
        v188 = 128l * v186;
        long v189;
        v189 = v188 + v185;
        float v190[4l];
        long v191[4l];
        long v192;
        v192 = 0l;
        while (while_method_2(v192)){
            assert("Tensor range check" && 0 <= v192 && v192 < 1l);
            long v194;
            v194 = 4l * v192;
            assert("Tensor range check" && 0 <= v192 && v192 < 1l);
            long v195;
            v195 = v194 + v189;
            int4* v196;
            v196 = reinterpret_cast<int4*>(v1 + v195);
            int4* v197;
            v197 = reinterpret_cast<int4*>(v190 + v194);
            assert("Pointer alignment check" && (unsigned long long)(v196) % 4l == 0 && (unsigned long long)(v197) % 4l == 0);
            *v197 = *v196;
            v192 += 1l ;
        }
        long v198;
        v198 = 0l;
        while (while_method_2(v198)){
            long v200;
            v200 = 0l;
            while (while_method_1(v200)){
                bool v202;
                v202 = 0l <= v200;
                bool v204;
                if (v202){
                    bool v203;
                    v203 = v200 < 4l;
                    v204 = v203;
                } else {
                    v204 = false;
                }
                bool v205;
                v205 = v204 == false;
                if (v205){
                    assert("The indices should be inside the range of the dimension." && v204);
                } else {
                }
                bool v206;
                v206 = 0l <= v179;
                bool v208;
                if (v206){
                    bool v207;
                    v207 = v179 < 1l;
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
                long v210;
                v210 = v179 * 4l;
                long v211;
                v211 = v200 + v210;
                bool v212;
                v212 = 0l <= v198;
                bool v214;
                if (v212){
                    bool v213;
                    v213 = v198 < 1l;
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
                long v216;
                v216 = v198 * 4l;
                long v217;
                v217 = v211 + v216;
                assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                assert("Tensor range check" && 0 <= v200 && v200 < 4l);
                long v218;
                v218 = 4l * v198;
                long v219;
                v219 = v218 + v200;
                v191[v219] = v217;
                v200 += 1l ;
            }
            v198 += 1l ;
        }
        bool v220;
        v220 = v177 && v180;
        bool v221;
        v221 = v220 == false;
        if (v221){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v220);
        } else {
        }
        bool v222;
        v222 = 0l <= v186;
        bool v224;
        if (v222){
            bool v223;
            v223 = v186 < 1l;
            v224 = v223;
        } else {
            v224 = false;
        }
        bool v225;
        v225 = v224 == false;
        if (v225){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v224);
        } else {
        }
        long v226;
        v226 = v186 * 32l;
        long v227;
        v227 = v226 + v176;
        assert("Tensor range check" && 0 <= v186 && v186 < 1l);
        long v228;
        v228 = 32l * v186;
        long v229;
        v229 = v228 + v176;
        v13[v229] = v227;
        v186 += 1l ;
    }
    __syncthreads();
    long v230;
    v230 = threadIdx.x;
    bool v231;
    v231 = 0l <= v230;
    bool v232;
    v232 = v231 == false;
    if (v232){
        assert("The index needs to be zero or positive." && v231);
    } else {
    }
    long v233;
    v233 = v230 % 1l;
    bool v234;
    v234 = v230 < 32l;
    bool v235;
    v235 = v234 == false;
    if (v235){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v234);
    } else {
    }
    cooperative_groups::thread_block_tile<1l, cooperative_groups::thread_block> v236 = cooperative_groups::tiled_partition<1l>(v14);
    assert("Tensor range check" && 0 <= v230 && v230 < 32l);
    assert("Tensor range check" && 0 <= v233 && v233 < 1l);
    long v237;
    v237 = 4l * v233;
    long v238;
    v238 = 4l * v230;
    long v239;
    v239 = v238 + v237;
    assert("Tensor range check" && 0 <= v230 && v230 < 32l);
    assert("Tensor range check" && 0 <= v233 && v233 < 1l);
    long v240;
    v240 = 0l;
    while (while_method_2(v240)){
        assert("Tensor range check" && 0 <= v240 && v240 < 1l);
        long v242;
        v242 = 128l * v240;
        long v243;
        v243 = v242 + v239;
        assert("Tensor range check" && 0 <= v240 && v240 < 1l);
        float v244[4l];
        long v245[4l];
        long v246;
        v246 = 0l;
        while (while_method_2(v246)){
            assert("Tensor range check" && 0 <= v246 && v246 < 1l);
            long v248;
            v248 = 4l * v246;
            assert("Tensor range check" && 0 <= v246 && v246 < 1l);
            long v249;
            v249 = v248 + v243;
            int4* v250;
            v250 = reinterpret_cast<int4*>(v1 + v249);
            int4* v251;
            v251 = reinterpret_cast<int4*>(v244 + v248);
            assert("Pointer alignment check" && (unsigned long long)(v250) % 4l == 0 && (unsigned long long)(v251) % 4l == 0);
            *v251 = *v250;
            v246 += 1l ;
        }
        long v252;
        v252 = 0l;
        while (while_method_2(v252)){
            long v254;
            v254 = 0l;
            while (while_method_1(v254)){
                bool v256;
                v256 = 0l <= v254;
                bool v258;
                if (v256){
                    bool v257;
                    v257 = v254 < 4l;
                    v258 = v257;
                } else {
                    v258 = false;
                }
                bool v259;
                v259 = v258 == false;
                if (v259){
                    assert("The indices should be inside the range of the dimension." && v258);
                } else {
                }
                bool v260;
                v260 = 0l <= v233;
                bool v262;
                if (v260){
                    bool v261;
                    v261 = v233 < 1l;
                    v262 = v261;
                } else {
                    v262 = false;
                }
                bool v263;
                v263 = v262 == false;
                if (v263){
                    assert("The indices should be inside the range of the dimension." && v262);
                } else {
                }
                long v264;
                v264 = v233 * 4l;
                long v265;
                v265 = v254 + v264;
                bool v266;
                v266 = 0l <= v252;
                bool v268;
                if (v266){
                    bool v267;
                    v267 = v252 < 1l;
                    v268 = v267;
                } else {
                    v268 = false;
                }
                bool v269;
                v269 = v268 == false;
                if (v269){
                    assert("The indices should be inside the range of the dimension." && v268);
                } else {
                }
                long v270;
                v270 = v252 * 4l;
                long v271;
                v271 = v265 + v270;
                assert("Tensor range check" && 0 <= v252 && v252 < 1l);
                assert("Tensor range check" && 0 <= v254 && v254 < 4l);
                long v272;
                v272 = 4l * v252;
                long v273;
                v273 = v272 + v254;
                v245[v273] = v271;
                v254 += 1l ;
            }
            v252 += 1l ;
        }
        bool v274;
        v274 = v231 && v234;
        bool v275;
        v275 = v274 == false;
        if (v275){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v274);
        } else {
        }
        bool v276;
        v276 = 0l <= v240;
        bool v278;
        if (v276){
            bool v277;
            v277 = v240 < 1l;
            v278 = v277;
        } else {
            v278 = false;
        }
        bool v279;
        v279 = v278 == false;
        if (v279){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v278);
        } else {
        }
        long v280;
        v280 = v240 * 32l;
        long v281;
        v281 = v280 + v230;
        float v282;
        v282 = 0.0f;
        long v283;
        v283 = 0l;
        while (while_method_2(v283)){
            long v285;
            v285 = 0l;
            while (while_method_1(v285)){
                assert("Tensor range check" && 0 <= v283 && v283 < 1l);
                assert("Tensor range check" && 0 <= v285 && v285 < 4l);
                long v287;
                v287 = 4l * v283;
                long v288;
                v288 = v287 + v285;
                float v289;
                v289 = v244[v288];
                float v290;
                v290 = v289 + v282;
                v282 = v290;
                v285 += 1l ;
            }
            v283 += 1l ;
        }
        Closure1 v291{};
        float v292;
        v292 = cooperative_groups::reduce(v236, v282, v291);
        float v293;
        v293 = v292 / 4.0f;
        float v294[4l];
        long v295;
        v295 = 0l;
        while (while_method_2(v295)){
            long v297;
            v297 = 0l;
            while (while_method_1(v297)){
                assert("Tensor range check" && 0 <= v295 && v295 < 1l);
                assert("Tensor range check" && 0 <= v297 && v297 < 4l);
                long v299;
                v299 = 4l * v295;
                long v300;
                v300 = v299 + v297;
                float v301;
                v301 = v244[v300];
                float v302;
                v302 = v301 - v293;
                float v303;
                v303 = exp(v302);
                assert("Tensor range check" && 0 <= v295 && v295 < 1l);
                assert("Tensor range check" && 0 <= v297 && v297 < 4l);
                v294[v300] = v303;
                v297 += 1l ;
            }
            v295 += 1l ;
        }
        float v304;
        v304 = 0.0f;
        long v305;
        v305 = 0l;
        while (while_method_2(v305)){
            long v307;
            v307 = 0l;
            while (while_method_1(v307)){
                assert("Tensor range check" && 0 <= v305 && v305 < 1l);
                assert("Tensor range check" && 0 <= v307 && v307 < 4l);
                long v309;
                v309 = 4l * v305;
                long v310;
                v310 = v309 + v307;
                float v311;
                v311 = v294[v310];
                float v312;
                v312 = v311 + v304;
                v304 = v312;
                v307 += 1l ;
            }
            v305 += 1l ;
        }
        float v313;
        v313 = cooperative_groups::reduce(v236, v304, v291);
        float v314[4l];
        long v315;
        v315 = 0l;
        while (while_method_2(v315)){
            long v317;
            v317 = 0l;
            while (while_method_1(v317)){
                assert("Tensor range check" && 0 <= v315 && v315 < 1l);
                assert("Tensor range check" && 0 <= v317 && v317 < 4l);
                long v319;
                v319 = 4l * v315;
                long v320;
                v320 = v319 + v317;
                float v321;
                v321 = v294[v320];
                float v322;
                v322 = v321 / v313;
                assert("Tensor range check" && 0 <= v315 && v315 < 1l);
                assert("Tensor range check" && 0 <= v317 && v317 < 4l);
                v314[v320] = v322;
                v317 += 1l ;
            }
            v315 += 1l ;
        }
        long v323;
        v323 = 0l;
        while (while_method_2(v323)){
            assert("Tensor range check" && 0 <= v323 && v323 < 1l);
            long v325;
            v325 = 4l * v323;
            long v326;
            v326 = v325 + v243;
            assert("Tensor range check" && 0 <= v323 && v323 < 1l);
            int4* v327;
            v327 = reinterpret_cast<int4*>(v314 + v325);
            int4* v328;
            v328 = reinterpret_cast<int4*>(v4 + v326);
            assert("Pointer alignment check" && (unsigned long long)(v327) % 4l == 0 && (unsigned long long)(v328) % 4l == 0);
            *v328 = *v327;
            v323 += 1l ;
        }
        v240 += 1l ;
    }
    __syncthreads();
    long v329;
    v329 = threadIdx.x;
    bool v330;
    v330 = 0l <= v329;
    bool v331;
    v331 = v330 == false;
    if (v331){
        assert("The index needs to be zero or positive." && v330);
    } else {
    }
    long v332;
    v332 = v329 % 1l;
    bool v333;
    v333 = v329 < 32l;
    bool v334;
    v334 = v333 == false;
    if (v334){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v333);
    } else {
    }
    cooperative_groups::thread_block_tile<1l, cooperative_groups::thread_block> v335 = cooperative_groups::tiled_partition<1l>(v14);
    assert("Tensor range check" && 0 <= v329 && v329 < 32l);
    assert("Tensor range check" && 0 <= v332 && v332 < 1l);
    long v336;
    v336 = 4l * v332;
    long v337;
    v337 = 4l * v329;
    long v338;
    v338 = v337 + v336;
    assert("Tensor range check" && 0 <= v329 && v329 < 32l);
    assert("Tensor range check" && 0 <= v332 && v332 < 1l);
    long v339;
    v339 = 0l;
    while (while_method_2(v339)){
        assert("Tensor range check" && 0 <= v339 && v339 < 1l);
        long v341;
        v341 = 128l * v339;
        long v342;
        v342 = v341 + v338;
        assert("Tensor range check" && 0 <= v339 && v339 < 1l);
        float v343[4l];
        long v344[4l];
        long v345;
        v345 = 0l;
        while (while_method_2(v345)){
            assert("Tensor range check" && 0 <= v345 && v345 < 1l);
            long v347;
            v347 = 4l * v345;
            assert("Tensor range check" && 0 <= v345 && v345 < 1l);
            long v348;
            v348 = v347 + v342;
            int4* v349;
            v349 = reinterpret_cast<int4*>(v1 + v348);
            int4* v350;
            v350 = reinterpret_cast<int4*>(v343 + v347);
            assert("Pointer alignment check" && (unsigned long long)(v349) % 4l == 0 && (unsigned long long)(v350) % 4l == 0);
            *v350 = *v349;
            v345 += 1l ;
        }
        long v351;
        v351 = 0l;
        while (while_method_2(v351)){
            long v353;
            v353 = 0l;
            while (while_method_1(v353)){
                bool v355;
                v355 = 0l <= v353;
                bool v357;
                if (v355){
                    bool v356;
                    v356 = v353 < 4l;
                    v357 = v356;
                } else {
                    v357 = false;
                }
                bool v358;
                v358 = v357 == false;
                if (v358){
                    assert("The indices should be inside the range of the dimension." && v357);
                } else {
                }
                bool v359;
                v359 = 0l <= v332;
                bool v361;
                if (v359){
                    bool v360;
                    v360 = v332 < 1l;
                    v361 = v360;
                } else {
                    v361 = false;
                }
                bool v362;
                v362 = v361 == false;
                if (v362){
                    assert("The indices should be inside the range of the dimension." && v361);
                } else {
                }
                long v363;
                v363 = v332 * 4l;
                long v364;
                v364 = v353 + v363;
                bool v365;
                v365 = 0l <= v351;
                bool v367;
                if (v365){
                    bool v366;
                    v366 = v351 < 1l;
                    v367 = v366;
                } else {
                    v367 = false;
                }
                bool v368;
                v368 = v367 == false;
                if (v368){
                    assert("The indices should be inside the range of the dimension." && v367);
                } else {
                }
                long v369;
                v369 = v351 * 4l;
                long v370;
                v370 = v364 + v369;
                assert("Tensor range check" && 0 <= v351 && v351 < 1l);
                assert("Tensor range check" && 0 <= v353 && v353 < 4l);
                long v371;
                v371 = 4l * v351;
                long v372;
                v372 = v371 + v353;
                v344[v372] = v370;
                v353 += 1l ;
            }
            v351 += 1l ;
        }
        bool v373;
        v373 = v330 && v333;
        bool v374;
        v374 = v373 == false;
        if (v374){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v373);
        } else {
        }
        bool v375;
        v375 = 0l <= v339;
        bool v377;
        if (v375){
            bool v376;
            v376 = v339 < 1l;
            v377 = v376;
        } else {
            v377 = false;
        }
        bool v378;
        v378 = v377 == false;
        if (v378){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v377);
        } else {
        }
        long v379;
        v379 = v339 * 32l;
        long v380;
        v380 = v379 + v329;
        float v381[4l];
        long v382;
        v382 = 0l;
        while (while_method_2(v382)){
            long v384;
            v384 = 0l;
            while (while_method_1(v384)){
                assert("Tensor range check" && 0 <= v382 && v382 < 1l);
                assert("Tensor range check" && 0 <= v384 && v384 < 4l);
                long v386;
                v386 = 4l * v382;
                long v387;
                v387 = v386 + v384;
                float v388;
                v388 = v343[v387];
                float v389;
                v389 = v388 * v388;
                assert("Tensor range check" && 0 <= v382 && v382 < 1l);
                assert("Tensor range check" && 0 <= v384 && v384 < 4l);
                v381[v387] = v389;
                v384 += 1l ;
            }
            v382 += 1l ;
        }
        float v390;
        v390 = 0.0f;
        long v391;
        v391 = 0l;
        while (while_method_2(v391)){
            long v393;
            v393 = 0l;
            while (while_method_1(v393)){
                assert("Tensor range check" && 0 <= v391 && v391 < 1l);
                assert("Tensor range check" && 0 <= v393 && v393 < 4l);
                long v395;
                v395 = 4l * v391;
                long v396;
                v396 = v395 + v393;
                float v397;
                v397 = v381[v396];
                float v398;
                v398 = v397 + v390;
                v390 = v398;
                v393 += 1l ;
            }
            v391 += 1l ;
        }
        Closure1 v399{};
        float v400;
        v400 = cooperative_groups::reduce(v335, v390, v399);
        float v401[4l];
        long v402;
        v402 = 0l;
        while (while_method_2(v402)){
            long v404;
            v404 = 0l;
            while (while_method_1(v404)){
                assert("Tensor range check" && 0 <= v402 && v402 < 1l);
                assert("Tensor range check" && 0 <= v404 && v404 < 4l);
                long v406;
                v406 = 4l * v402;
                long v407;
                v407 = v406 + v404;
                float v408;
                v408 = v381[v407];
                float v409;
                v409 = v408 / v400;
                assert("Tensor range check" && 0 <= v402 && v402 < 1l);
                assert("Tensor range check" && 0 <= v404 && v404 < 4l);
                v401[v407] = v409;
                v404 += 1l ;
            }
            v402 += 1l ;
        }
        long v410;
        v410 = 0l;
        while (while_method_2(v410)){
            assert("Tensor range check" && 0 <= v410 && v410 < 1l);
            long v412;
            v412 = 4l * v410;
            long v413;
            v413 = v412 + v342;
            assert("Tensor range check" && 0 <= v410 && v410 < 1l);
            int4* v414;
            v414 = reinterpret_cast<int4*>(v401 + v412);
            int4* v415;
            v415 = reinterpret_cast<int4*>(v7 + v413);
            assert("Pointer alignment check" && (unsigned long long)(v414) % 4l == 0 && (unsigned long long)(v415) % 4l == 0);
            *v415 = *v414;
            v410 += 1l ;
        }
        v339 += 1l ;
    }
    __syncthreads();
    long v416;
    v416 = threadIdx.x;
    bool v417;
    v417 = 0l <= v416;
    bool v418;
    v418 = v417 == false;
    if (v418){
        assert("The index needs to be zero or positive." && v417);
    } else {
    }
    long v419;
    v419 = v416 % 1l;
    bool v420;
    v420 = v416 < 32l;
    bool v421;
    v421 = v420 == false;
    if (v421){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v420);
    } else {
    }
    cooperative_groups::thread_block_tile<1l, cooperative_groups::thread_block> v422 = cooperative_groups::tiled_partition<1l>(v14);
    assert("Tensor range check" && 0 <= v416 && v416 < 32l);
    assert("Tensor range check" && 0 <= v419 && v419 < 1l);
    long v423;
    v423 = 4l * v419;
    long v424;
    v424 = 4l * v416;
    long v425;
    v425 = v424 + v423;
    assert("Tensor range check" && 0 <= v416 && v416 < 32l);
    long v426;
    v426 = 0l;
    while (while_method_2(v426)){
        assert("Tensor range check" && 0 <= v426 && v426 < 1l);
        long v428;
        v428 = 128l * v426;
        long v429;
        v429 = v428 + v425;
        float v430[4l];
        long v431[4l];
        long v432;
        v432 = 0l;
        while (while_method_2(v432)){
            assert("Tensor range check" && 0 <= v432 && v432 < 1l);
            long v434;
            v434 = 4l * v432;
            assert("Tensor range check" && 0 <= v432 && v432 < 1l);
            long v435;
            v435 = v434 + v429;
            int4* v436;
            v436 = reinterpret_cast<int4*>(v1 + v435);
            int4* v437;
            v437 = reinterpret_cast<int4*>(v430 + v434);
            assert("Pointer alignment check" && (unsigned long long)(v436) % 4l == 0 && (unsigned long long)(v437) % 4l == 0);
            *v437 = *v436;
            v432 += 1l ;
        }
        long v438;
        v438 = 0l;
        while (while_method_2(v438)){
            long v440;
            v440 = 0l;
            while (while_method_1(v440)){
                bool v442;
                v442 = 0l <= v440;
                bool v444;
                if (v442){
                    bool v443;
                    v443 = v440 < 4l;
                    v444 = v443;
                } else {
                    v444 = false;
                }
                bool v445;
                v445 = v444 == false;
                if (v445){
                    assert("The indices should be inside the range of the dimension." && v444);
                } else {
                }
                bool v446;
                v446 = 0l <= v419;
                bool v448;
                if (v446){
                    bool v447;
                    v447 = v419 < 1l;
                    v448 = v447;
                } else {
                    v448 = false;
                }
                bool v449;
                v449 = v448 == false;
                if (v449){
                    assert("The indices should be inside the range of the dimension." && v448);
                } else {
                }
                long v450;
                v450 = v419 * 4l;
                long v451;
                v451 = v440 + v450;
                bool v452;
                v452 = 0l <= v438;
                bool v454;
                if (v452){
                    bool v453;
                    v453 = v438 < 1l;
                    v454 = v453;
                } else {
                    v454 = false;
                }
                bool v455;
                v455 = v454 == false;
                if (v455){
                    assert("The indices should be inside the range of the dimension." && v454);
                } else {
                }
                long v456;
                v456 = v438 * 4l;
                long v457;
                v457 = v451 + v456;
                assert("Tensor range check" && 0 <= v438 && v438 < 1l);
                assert("Tensor range check" && 0 <= v440 && v440 < 4l);
                long v458;
                v458 = 4l * v438;
                long v459;
                v459 = v458 + v440;
                v431[v459] = v457;
                v440 += 1l ;
            }
            v438 += 1l ;
        }
        bool v460;
        v460 = v417 && v420;
        bool v461;
        v461 = v460 == false;
        if (v461){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v460);
        } else {
        }
        bool v462;
        v462 = 0l <= v426;
        bool v464;
        if (v462){
            bool v463;
            v463 = v426 < 1l;
            v464 = v463;
        } else {
            v464 = false;
        }
        bool v465;
        v465 = v464 == false;
        if (v465){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v464);
        } else {
        }
        long v466;
        v466 = v426 * 32l;
        long v467;
        v467 = v466 + v416;
        float v468; long v469;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0l};
        v468 = tmp1.v0; v469 = tmp1.v1;
        long v470;
        v470 = 0l;
        while (while_method_2(v470)){
            long v472;
            v472 = 0l;
            while (while_method_1(v472)){
                assert("Tensor range check" && 0 <= v470 && v470 < 1l);
                assert("Tensor range check" && 0 <= v472 && v472 < 4l);
                long v474;
                v474 = 4l * v470;
                long v475;
                v475 = v474 + v472;
                float v476;
                v476 = v430[v475];
                long v477;
                v477 = v431[v475];
                bool v478;
                v478 = v476 > v468;
                float v479; long v480;
                if (v478){
                    v479 = v476; v480 = v477;
                } else {
                    v479 = v468; v480 = v469;
                }
                v468 = v479;
                v469 = v480;
                v472 += 1l ;
            }
            v470 += 1l ;
        }
        Closure2 v481{};
        float v482; long v483;
        Tuple1 tmp2 = cooperative_groups::reduce(v422, Tuple1{v468, v469}, v481);
        v482 = tmp2.v0; v483 = tmp2.v1;
        assert("Tensor range check" && 0 <= v426 && v426 < 1l);
        long v484;
        v484 = 32l * v426;
        long v485;
        v485 = v484 + v416;
        v8[v485] = v483;
        v426 += 1l ;
    }
    __syncthreads();
    long v486;
    v486 = threadIdx.x;
    bool v487;
    v487 = 0l <= v486;
    bool v488;
    v488 = v487 == false;
    if (v488){
        assert("The index needs to be zero or positive." && v487);
    } else {
    }
    long v489;
    v489 = v486 % 1l;
    bool v490;
    v490 = v486 < 32l;
    bool v491;
    v491 = v490 == false;
    if (v491){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v490);
    } else {
    }
    cooperative_groups::thread_block_tile<1l, cooperative_groups::thread_block> v492 = cooperative_groups::tiled_partition<1l>(v14);
    assert("Tensor range check" && 0 <= v486 && v486 < 32l);
    assert("Tensor range check" && 0 <= v489 && v489 < 1l);
    long v493;
    v493 = 4l * v489;
    long v494;
    v494 = 4l * v486;
    long v495;
    v495 = v494 + v493;
    assert("Tensor range check" && 0 <= v486 && v486 < 32l);
    assert("Tensor range check" && 0 <= v489 && v489 < 1l);
    long v496;
    v496 = 0l;
    while (while_method_2(v496)){
        assert("Tensor range check" && 0 <= v496 && v496 < 1l);
        long v498;
        v498 = 128l * v496;
        long v499;
        v499 = v498 + v495;
        assert("Tensor range check" && 0 <= v496 && v496 < 1l);
        float v500[4l];
        long v501[4l];
        long v502;
        v502 = 0l;
        while (while_method_2(v502)){
            assert("Tensor range check" && 0 <= v502 && v502 < 1l);
            long v504;
            v504 = 4l * v502;
            assert("Tensor range check" && 0 <= v502 && v502 < 1l);
            long v505;
            v505 = v504 + v499;
            int4* v506;
            v506 = reinterpret_cast<int4*>(v1 + v505);
            int4* v507;
            v507 = reinterpret_cast<int4*>(v500 + v504);
            assert("Pointer alignment check" && (unsigned long long)(v506) % 4l == 0 && (unsigned long long)(v507) % 4l == 0);
            *v507 = *v506;
            v502 += 1l ;
        }
        long v508;
        v508 = 0l;
        while (while_method_2(v508)){
            long v510;
            v510 = 0l;
            while (while_method_1(v510)){
                bool v512;
                v512 = 0l <= v510;
                bool v514;
                if (v512){
                    bool v513;
                    v513 = v510 < 4l;
                    v514 = v513;
                } else {
                    v514 = false;
                }
                bool v515;
                v515 = v514 == false;
                if (v515){
                    assert("The indices should be inside the range of the dimension." && v514);
                } else {
                }
                bool v516;
                v516 = 0l <= v489;
                bool v518;
                if (v516){
                    bool v517;
                    v517 = v489 < 1l;
                    v518 = v517;
                } else {
                    v518 = false;
                }
                bool v519;
                v519 = v518 == false;
                if (v519){
                    assert("The indices should be inside the range of the dimension." && v518);
                } else {
                }
                long v520;
                v520 = v489 * 4l;
                long v521;
                v521 = v510 + v520;
                bool v522;
                v522 = 0l <= v508;
                bool v524;
                if (v522){
                    bool v523;
                    v523 = v508 < 1l;
                    v524 = v523;
                } else {
                    v524 = false;
                }
                bool v525;
                v525 = v524 == false;
                if (v525){
                    assert("The indices should be inside the range of the dimension." && v524);
                } else {
                }
                long v526;
                v526 = v508 * 4l;
                long v527;
                v527 = v521 + v526;
                assert("Tensor range check" && 0 <= v508 && v508 < 1l);
                assert("Tensor range check" && 0 <= v510 && v510 < 4l);
                long v528;
                v528 = 4l * v508;
                long v529;
                v529 = v528 + v510;
                v501[v529] = v527;
                v510 += 1l ;
            }
            v508 += 1l ;
        }
        bool v530;
        v530 = v487 && v490;
        bool v531;
        v531 = v530 == false;
        if (v531){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v530);
        } else {
        }
        bool v532;
        v532 = 0l <= v496;
        bool v534;
        if (v532){
            bool v533;
            v533 = v496 < 1l;
            v534 = v533;
        } else {
            v534 = false;
        }
        bool v535;
        v535 = v534 == false;
        if (v535){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v534);
        } else {
        }
        long v536;
        v536 = v496 * 32l;
        long v537;
        v537 = v536 + v486;
        float v538;
        v538 = 0.0f;
        long v539;
        v539 = 0l;
        while (while_method_2(v539)){
            long v541;
            v541 = 0l;
            while (while_method_1(v541)){
                assert("Tensor range check" && 0 <= v539 && v539 < 1l);
                assert("Tensor range check" && 0 <= v541 && v541 < 4l);
                long v543;
                v543 = 4l * v539;
                long v544;
                v544 = v543 + v541;
                float v545;
                v545 = v500[v544];
                float v546;
                v546 = v545 + v538;
                v538 = v546;
                v541 += 1l ;
            }
            v539 += 1l ;
        }
        Closure1 v547{};
        float v548;
        v548 = cooperative_groups::reduce(v492, v538, v547);
        float v549;
        v549 = v548 / 4.0f;
        float v550[4l];
        long v551;
        v551 = 0l;
        while (while_method_2(v551)){
            long v553;
            v553 = 0l;
            while (while_method_1(v553)){
                assert("Tensor range check" && 0 <= v551 && v551 < 1l);
                assert("Tensor range check" && 0 <= v553 && v553 < 4l);
                long v555;
                v555 = 4l * v551;
                long v556;
                v556 = v555 + v553;
                float v557;
                v557 = v500[v556];
                float v558;
                v558 = v557 - v549;
                float v559;
                v559 = exp(v558);
                assert("Tensor range check" && 0 <= v551 && v551 < 1l);
                assert("Tensor range check" && 0 <= v553 && v553 < 4l);
                v550[v556] = v559;
                v553 += 1l ;
            }
            v551 += 1l ;
        }
        float v560;
        v560 = 0.0f;
        long v561;
        v561 = 0l;
        while (while_method_2(v561)){
            long v563;
            v563 = 0l;
            while (while_method_1(v563)){
                assert("Tensor range check" && 0 <= v561 && v561 < 1l);
                assert("Tensor range check" && 0 <= v563 && v563 < 4l);
                long v565;
                v565 = 4l * v561;
                long v566;
                v566 = v565 + v563;
                float v567;
                v567 = v550[v566];
                float v568;
                v568 = v567 + v560;
                v560 = v568;
                v563 += 1l ;
            }
            v561 += 1l ;
        }
        float v569;
        v569 = cooperative_groups::reduce(v492, v560, v547);
        float v570[4l];
        long v571;
        v571 = 0l;
        while (while_method_2(v571)){
            long v573;
            v573 = 0l;
            while (while_method_1(v573)){
                assert("Tensor range check" && 0 <= v571 && v571 < 1l);
                assert("Tensor range check" && 0 <= v573 && v573 < 4l);
                long v575;
                v575 = 4l * v571;
                long v576;
                v576 = v575 + v573;
                float v577;
                v577 = v550[v576];
                float v578;
                v578 = v577 / v569;
                assert("Tensor range check" && 0 <= v571 && v571 < 1l);
                assert("Tensor range check" && 0 <= v573 && v573 < 4l);
                v570[v576] = v578;
                v573 += 1l ;
            }
            v571 += 1l ;
        }
        float v579[4l];
        float v580;
        v580 = 0.0f;
        long v581;
        v581 = 0l;
        while (while_method_2(v581)){
            assert("Tensor range check" && 0 <= v581 && v581 < 1l);
            long v583;
            v583 = 4l * v581;
            assert("Tensor range check" && 0 <= v581 && v581 < 1l);
            long v584; float v585;
            Tuple0 tmp3 = Tuple0{0l, 0.0f};
            v584 = tmp3.v0; v585 = tmp3.v1;
            while (while_method_1(v584)){
                assert("Tensor range check" && 0 <= v584 && v584 < 4l);
                long v587;
                v587 = v584 + v583;
                float v588;
                v588 = v570[v587];
                float v589;
                v589 = v588 + v585;
                v585 = v589;
                v584 += 1l ;
            }
            Closure3 v590{};
            float v591;
            v591 = cooperative_groups::inclusive_scan(v492, v585, v590);
            float v592;
            v592 = v492.shfl(v591,v492.num_threads()-1);
            bool v593;
            v593 = v492.num_threads() <= 32;
            bool v594;
            v594 = v593 == false;
            if (v594){
                assert("The thread block tile in the exclusive scan has to be less than or equal 32." && v593);
            } else {
            }
            float v595;
            v595 = v492.shfl_up(v591,1);
            bool v596;
            v596 = v492.thread_rank() == 0;
            float v597;
            if (v596){
                v597 = 0.0f;
            } else {
                v597 = v595;
            }
            float v598;
            v598 = v580 + v597;
            long v599; float v600;
            Tuple0 tmp4 = Tuple0{0l, v598};
            v599 = tmp4.v0; v600 = tmp4.v1;
            while (while_method_1(v599)){
                assert("Tensor range check" && 0 <= v599 && v599 < 4l);
                long v602;
                v602 = v599 + v583;
                float v603;
                v603 = v570[v602];
                assert("Tensor range check" && 0 <= v599 && v599 < 4l);
                v579[v602] = v600;
                float v604;
                v604 = v603 + v600;
                v600 = v604;
                v599 += 1l ;
            }
            float v605;
            v605 = v580 + v592;
            v580 = v605;
            v581 += 1l ;
        }
        long v606;
        v606 = 0l;
        while (while_method_2(v606)){
            assert("Tensor range check" && 0 <= v606 && v606 < 1l);
            long v608;
            v608 = 4l * v606;
            long v609;
            v609 = v608 + v499;
            assert("Tensor range check" && 0 <= v606 && v606 < 1l);
            int4* v610;
            v610 = reinterpret_cast<int4*>(v570 + v608);
            int4* v611;
            v611 = reinterpret_cast<int4*>(v5 + v609);
            assert("Pointer alignment check" && (unsigned long long)(v610) % 4l == 0 && (unsigned long long)(v611) % 4l == 0);
            *v611 = *v610;
            int4* v612;
            v612 = reinterpret_cast<int4*>(v579 + v608);
            int4* v613;
            v613 = reinterpret_cast<int4*>(v6 + v609);
            assert("Pointer alignment check" && (unsigned long long)(v612) % 4l == 0 && (unsigned long long)(v613) % 4l == 0);
            *v613 = *v612;
            v606 += 1l ;
        }
        v496 += 1l ;
    }
    __syncthreads();
    long v614;
    v614 = threadIdx.x;
    bool v615;
    v615 = 0l <= v614;
    bool v616;
    v616 = v615 == false;
    if (v616){
        assert("The index needs to be zero or positive." && v615);
    } else {
    }
    long v617;
    v617 = v614 % 1l;
    bool v618;
    v618 = v614 < 32l;
    bool v619;
    v619 = v618 == false;
    if (v619){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v618);
    } else {
    }
    cooperative_groups::thread_block_tile<1l, cooperative_groups::thread_block> v620 = cooperative_groups::tiled_partition<1l>(v14);
    assert("Tensor range check" && 0 <= v614 && v614 < 32l);
    assert("Tensor range check" && 0 <= v617 && v617 < 1l);
    long v621;
    v621 = 4l * v617;
    long v622;
    v622 = 4l * v614;
    long v623;
    v623 = v622 + v621;
    assert("Tensor range check" && 0 <= v614 && v614 < 32l);
    long v624;
    v624 = 0l;
    while (while_method_2(v624)){
        assert("Tensor range check" && 0 <= v624 && v624 < 1l);
        long v626;
        v626 = 128l * v624;
        long v627;
        v627 = v626 + v623;
        float v628[4l];
        long v629[4l];
        long v630;
        v630 = 0l;
        while (while_method_2(v630)){
            assert("Tensor range check" && 0 <= v630 && v630 < 1l);
            long v632;
            v632 = 4l * v630;
            assert("Tensor range check" && 0 <= v630 && v630 < 1l);
            long v633;
            v633 = v632 + v627;
            int4* v634;
            v634 = reinterpret_cast<int4*>(v1 + v633);
            int4* v635;
            v635 = reinterpret_cast<int4*>(v628 + v632);
            assert("Pointer alignment check" && (unsigned long long)(v634) % 4l == 0 && (unsigned long long)(v635) % 4l == 0);
            *v635 = *v634;
            v630 += 1l ;
        }
        long v636;
        v636 = 0l;
        while (while_method_2(v636)){
            long v638;
            v638 = 0l;
            while (while_method_1(v638)){
                bool v640;
                v640 = 0l <= v638;
                bool v642;
                if (v640){
                    bool v641;
                    v641 = v638 < 4l;
                    v642 = v641;
                } else {
                    v642 = false;
                }
                bool v643;
                v643 = v642 == false;
                if (v643){
                    assert("The indices should be inside the range of the dimension." && v642);
                } else {
                }
                bool v644;
                v644 = 0l <= v617;
                bool v646;
                if (v644){
                    bool v645;
                    v645 = v617 < 1l;
                    v646 = v645;
                } else {
                    v646 = false;
                }
                bool v647;
                v647 = v646 == false;
                if (v647){
                    assert("The indices should be inside the range of the dimension." && v646);
                } else {
                }
                long v648;
                v648 = v617 * 4l;
                long v649;
                v649 = v638 + v648;
                bool v650;
                v650 = 0l <= v636;
                bool v652;
                if (v650){
                    bool v651;
                    v651 = v636 < 1l;
                    v652 = v651;
                } else {
                    v652 = false;
                }
                bool v653;
                v653 = v652 == false;
                if (v653){
                    assert("The indices should be inside the range of the dimension." && v652);
                } else {
                }
                long v654;
                v654 = v636 * 4l;
                long v655;
                v655 = v649 + v654;
                assert("Tensor range check" && 0 <= v636 && v636 < 1l);
                assert("Tensor range check" && 0 <= v638 && v638 < 4l);
                long v656;
                v656 = 4l * v636;
                long v657;
                v657 = v656 + v638;
                v629[v657] = v655;
                v638 += 1l ;
            }
            v636 += 1l ;
        }
        bool v658;
        v658 = v615 && v618;
        bool v659;
        v659 = v658 == false;
        if (v659){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v658);
        } else {
        }
        bool v660;
        v660 = 0l <= v624;
        bool v662;
        if (v660){
            bool v661;
            v661 = v624 < 1l;
            v662 = v661;
        } else {
            v662 = false;
        }
        bool v663;
        v663 = v662 == false;
        if (v663){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v662);
        } else {
        }
        long v664;
        v664 = v624 * 32l;
        long v665;
        v665 = v664 + v614;
        float v666;
        v666 = 0.0f;
        long v667;
        v667 = 0l;
        while (while_method_2(v667)){
            long v669;
            v669 = 0l;
            while (while_method_1(v669)){
                assert("Tensor range check" && 0 <= v667 && v667 < 1l);
                assert("Tensor range check" && 0 <= v669 && v669 < 4l);
                long v671;
                v671 = 4l * v667;
                long v672;
                v672 = v671 + v669;
                float v673;
                v673 = v628[v672];
                float v674;
                v674 = v673 + v666;
                v666 = v674;
                v669 += 1l ;
            }
            v667 += 1l ;
        }
        Closure1 v675{};
        float v676;
        v676 = cooperative_groups::reduce(v620, v666, v675);
        float v677;
        v677 = v676 / 4.0f;
        float v678[4l];
        long v679;
        v679 = 0l;
        while (while_method_2(v679)){
            long v681;
            v681 = 0l;
            while (while_method_1(v681)){
                assert("Tensor range check" && 0 <= v679 && v679 < 1l);
                assert("Tensor range check" && 0 <= v681 && v681 < 4l);
                long v683;
                v683 = 4l * v679;
                long v684;
                v684 = v683 + v681;
                float v685;
                v685 = v628[v684];
                float v686;
                v686 = v685 - v677;
                float v687;
                v687 = exp(v686);
                assert("Tensor range check" && 0 <= v679 && v679 < 1l);
                assert("Tensor range check" && 0 <= v681 && v681 < 4l);
                v678[v684] = v687;
                v681 += 1l ;
            }
            v679 += 1l ;
        }
        float v688;
        v688 = 0.0f;
        long v689;
        v689 = 0l;
        while (while_method_2(v689)){
            long v691;
            v691 = 0l;
            while (while_method_1(v691)){
                assert("Tensor range check" && 0 <= v689 && v689 < 1l);
                assert("Tensor range check" && 0 <= v691 && v691 < 4l);
                long v693;
                v693 = 4l * v689;
                long v694;
                v694 = v693 + v691;
                float v695;
                v695 = v678[v694];
                float v696;
                v696 = v695 + v688;
                v688 = v696;
                v691 += 1l ;
            }
            v689 += 1l ;
        }
        float v697;
        v697 = cooperative_groups::reduce(v620, v688, v675);
        float v698[4l];
        long v699;
        v699 = 0l;
        while (while_method_2(v699)){
            long v701;
            v701 = 0l;
            while (while_method_1(v701)){
                assert("Tensor range check" && 0 <= v699 && v699 < 1l);
                assert("Tensor range check" && 0 <= v701 && v701 < 4l);
                long v703;
                v703 = 4l * v699;
                long v704;
                v704 = v703 + v701;
                float v705;
                v705 = v678[v704];
                float v706;
                v706 = v705 / v697;
                assert("Tensor range check" && 0 <= v699 && v699 < 1l);
                assert("Tensor range check" && 0 <= v701 && v701 < 4l);
                v698[v704] = v706;
                v701 += 1l ;
            }
            v699 += 1l ;
        }
        float v707[4l];
        float v708;
        v708 = 0.0f;
        long v709;
        v709 = 0l;
        while (while_method_2(v709)){
            assert("Tensor range check" && 0 <= v709 && v709 < 1l);
            long v711;
            v711 = 4l * v709;
            assert("Tensor range check" && 0 <= v709 && v709 < 1l);
            long v712; float v713;
            Tuple0 tmp5 = Tuple0{0l, 0.0f};
            v712 = tmp5.v0; v713 = tmp5.v1;
            while (while_method_1(v712)){
                assert("Tensor range check" && 0 <= v712 && v712 < 4l);
                long v715;
                v715 = v712 + v711;
                float v716;
                v716 = v698[v715];
                float v717;
                v717 = v716 + v713;
                v713 = v717;
                v712 += 1l ;
            }
            Closure3 v718{};
            float v719;
            v719 = cooperative_groups::inclusive_scan(v620, v713, v718);
            float v720;
            v720 = v620.shfl(v719,v620.num_threads()-1);
            bool v721;
            v721 = v620.num_threads() <= 32;
            bool v722;
            v722 = v721 == false;
            if (v722){
                assert("The thread block tile in the exclusive scan has to be less than or equal 32." && v721);
            } else {
            }
            float v723;
            v723 = v620.shfl_up(v719,1);
            bool v724;
            v724 = v620.thread_rank() == 0;
            float v725;
            if (v724){
                v725 = 0.0f;
            } else {
                v725 = v723;
            }
            float v726;
            v726 = v708 + v725;
            long v727; float v728;
            Tuple0 tmp6 = Tuple0{0l, v726};
            v727 = tmp6.v0; v728 = tmp6.v1;
            while (while_method_1(v727)){
                assert("Tensor range check" && 0 <= v727 && v727 < 4l);
                long v730;
                v730 = v727 + v711;
                float v731;
                v731 = v698[v730];
                assert("Tensor range check" && 0 <= v727 && v727 < 4l);
                v707[v730] = v728;
                float v732;
                v732 = v731 + v728;
                v728 = v732;
                v727 += 1l ;
            }
            float v733;
            v733 = v708 + v720;
            v708 = v733;
            v709 += 1l ;
        }
        assert("Tensor range check" && 0 <= v665 && v665 < 32l);
        float v734;
        v734 = v2[v665];
        float v735[4l];
        long v736;
        v736 = 0l;
        while (while_method_2(v736)){
            long v738;
            v738 = 0l;
            while (while_method_1(v738)){
                assert("Tensor range check" && 0 <= v736 && v736 < 1l);
                assert("Tensor range check" && 0 <= v738 && v738 < 4l);
                long v740;
                v740 = 4l * v736;
                long v741;
                v741 = v740 + v738;
                float v742;
                v742 = v707[v741];
                float v743;
                v743 = v742 - v734;
                assert("Tensor range check" && 0 <= v736 && v736 < 1l);
                assert("Tensor range check" && 0 <= v738 && v738 < 4l);
                v735[v741] = v743;
                v738 += 1l ;
            }
            v736 += 1l ;
        }
        float v744; long v745;
        Tuple1 tmp7 = Tuple1{-1.0f / 0.0f, 0l};
        v744 = tmp7.v0; v745 = tmp7.v1;
        long v746;
        v746 = 0l;
        while (while_method_2(v746)){
            long v748;
            v748 = 0l;
            while (while_method_1(v748)){
                assert("Tensor range check" && 0 <= v746 && v746 < 1l);
                assert("Tensor range check" && 0 <= v748 && v748 < 4l);
                long v750;
                v750 = 4l * v746;
                long v751;
                v751 = v750 + v748;
                float v752;
                v752 = v735[v751];
                long v753;
                v753 = v629[v751];
                bool v754;
                v754 = v752 >= 0.0f;
                bool v756;
                if (v754){
                    bool v755;
                    v755 = v744 >= 0.0f;
                    v756 = v755;
                } else {
                    v756 = false;
                }
                float v765; long v766;
                if (v756){
                    bool v757;
                    v757 = v752 <= v744;
                    if (v757){
                        v765 = v752; v766 = v753;
                    } else {
                        v765 = v744; v766 = v745;
                    }
                } else {
                    if (v754){
                        v765 = v752; v766 = v753;
                    } else {
                        bool v760;
                        v760 = v744 >= 0.0f;
                        if (v760){
                            v765 = v744; v766 = v745;
                        } else {
                            v765 = v752; v766 = v753;
                        }
                    }
                }
                v744 = v765;
                v745 = v766;
                v748 += 1l ;
            }
            v746 += 1l ;
        }
        Closure4 v767{};
        float v768; long v769;
        Tuple1 tmp8 = cooperative_groups::reduce(v620, Tuple1{v744, v745}, v767);
        v768 = tmp8.v0; v769 = tmp8.v1;
        assert("Tensor range check" && 0 <= v624 && v624 < 1l);
        long v770;
        v770 = 32l * v624;
        long v771;
        v771 = v770 + v614;
        v9[v771] = v769;
        v624 += 1l ;
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
    v0 = cp.arange(0,128,1,dtype=cp.float32) # type: ignore
    v1 = v0.size
    v2 = 128 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,128,dtype=cp.float32) # type: ignore
    v6 = cp.random.uniform(size=32,dtype=cp.float32) # type: ignore
    v7 = cp.empty(1,dtype=cp.float32)
    v8 = cp.empty(128,dtype=cp.float32)
    v9 = cp.empty(128,dtype=cp.float32)
    v10 = cp.empty(128,dtype=cp.float32)
    v11 = cp.empty(128,dtype=cp.float32)
    v12 = cp.empty(32,dtype=cp.int32)
    v13 = cp.empty(32,dtype=cp.int32)
    v14 = cp.empty(128,dtype=cp.int32)
    v15 = cp.empty(128,dtype=cp.int32)
    v16 = cp.empty(128,dtype=cp.int32)
    v17 = cp.empty(32,dtype=cp.int32)
    v18 = 0
    v19 = raw_module.get_function(f"entry{v18}")
    del v18
    v19.max_dynamic_shared_size_bytes = 0 
    v19((1,),(32,),(v0, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17),shared_mem=0)
    del v0, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v19
    v20 = 0
    v21 = '['
    method0(v21)
    del v21
    v22 = 0
    while method1(v22):
        v24 = v20
        v25 = v24 >= 128
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
        v31 = v17[v22].item()
        method3(v31)
        del v31
        v22 += 1 
    del v17, v20, v22
    v32 = ']'
    method0(v32)
    del v32
    method4()
    print()
    return 

if __name__ == '__main__': print(main())
