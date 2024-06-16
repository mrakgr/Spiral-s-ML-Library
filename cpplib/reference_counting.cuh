template <typename el>
struct sptr // Shared pointer for the Spiral datatypes. They have to have the refc field inside them to work.
{
    el *base;

    __device__ sptr() : base(nullptr) {}
    __device__ sptr(el *ptr) : base(ptr) { this->base->refc++; }

    __device__ ~sptr()
    {
        if (this->base != nullptr && --this->base->refc == 0)
        {
            delete this->base;
            this->base = nullptr;
        }
    }

    __device__ sptr(sptr &x)
    {
        this->base = x.base;
        this->base->refc++;
    }

    __device__ sptr(sptr &&x)
    {
        this->base = x.base;
        x.base = nullptr;
    }

    __device__ sptr &operator=(sptr &x)
    {
        if (this->base != x.base)
        {
            delete this->base;
            this->base = x.base;
            this->base->refc++;
        }
        return *this;
    }

    __device__ sptr &operator=(sptr &&x)
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

// using default_int = int; // TODO: Don't forget to remove this.

template <typename el, default_int max_length>
struct static_array
{
    el ptr[max_length];
    el & operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < max_length);
        return this->ptr[i];
    }
};

template <typename el, default_int max_length>
struct static_array_list
{
    default_int length{0};
    el ptr[max_length];

    el & operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < this->length);
        return this->ptr[i];
    }
    void push(el & x) {
        new (&ptr[this->length++]) el{x};
        assert(this->length <= max_length);
    }
    void push(el && x) {
        new (&ptr[this->length++]) el{std::move(x)};
        assert(this->length <= max_length);
    }
    el pop() {
        --this->length;
        assert (0 <= this->length);
        auto x = ptr[this->length];
        ptr[this->length].~el();
        return x;
    }
    // Should be used only during initialization.
    void unsafe_set_length(default_int i){
        assert(0 <= i && i <= max_length);
        this->length = i;
    }
};

template <typename el>
struct dynamic_array
{
    int refc{0};
    default_int length;
    el *ptr;

    __device__ dynamic_array() = delete;
    __device__ dynamic_array(default_int l) : length(l), ptr(new el[l]) {}
    __device__ ~dynamic_array() { delete[] this->ptr; }

    el & operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < this->length);
        return this->ptr[i];
    }
};

// template <typename el>
// struct dynamic_array_list
// {
//     int refc{0};
//     default_int length{0};
//     default_int max_length;
//     el *ptr;

//     __device__ dynamic_array_list() = delete;
//     __device__ dynamic_array_list(default_int l) : max_length(l), ptr(new el[l]) {}
//     __device__ ~dynamic_array_list() { delete[] this->ptr; }

//     el & operator[](default_int i) {
//         assert("The index has to be in range." && 0 <= i && i < this->length);
//         return this->ptr[i];
//     }
//     void push(el & x) {
//         new (&ptr[this->length++]) el{x};
//         assert(this->length <= this->max_length);
//     }
//     void push(el && x) {
//         new (&ptr[this->length++]) el{std::move(x)};
//         assert(this->length <= this->max_length);
//     }
//     el pop() {
//         --this->length;
//         assert (0 <= this->length);
//         auto x = ptr[this->length];
//         ptr[this->length].~el();
//         return x;
//     }
//     // Should be used only during initialization.
//     void unsafe_set_length(default_int i){
//         assert(0 <= i && i <= this->max_length);
//         this->length = i;
//     }
// };
