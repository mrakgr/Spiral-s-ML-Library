template <typename T>
void call_destructor(T &x) { x.~T(); }
template <typename el, int dim>
struct static_array
{
    el v[dim];
};
template <typename el, int dim, typename default_int>
struct static_array_list
{
    default_int length;
    el v[dim];
};

template <typename T>
struct sptr // Shared pointer for the Spiral datatypes. They have to have the refc field inside them to work.
{
    T *base;

    __device__ sptr() : base(nullptr) {}
    __device__ sptr(T *v) : base(v) { this->base->refc++; }

    __device__ ~sptr()
    {
        if (this->base != nullptr && --this->base->refc == 0)
        {
            delete this->base;
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

template <typename T>
struct csptr : public sptr<T>
{ // Shared pointer for closures specifically.
    using sptr<T>::sptr;
    template <typename... Args>
    __device__ auto operator()(Args... args) -> decltype(this->base->operator()(args...))
    {
        return this->base->operator()(args...);
    }
};

template <typename T, typename default_int>
struct array
{
    int refc = 0;
    default_int length;
    T *ptr;

    __device__ array() = delete;
    __device__ array(int l) : length(l), ptr(new T[l]) {}
    __device__ ~array() { delete[] this->ptr; }
};