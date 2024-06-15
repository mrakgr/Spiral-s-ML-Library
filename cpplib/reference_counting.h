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
struct sptr
{
    T *base;

    sptr() : base(nullptr) {}
    sptr(T *v) : base(v) { this->base->refc++; }

    ~sptr()
    {
        if (this->base != nullptr && --this->base->refc == 0)
        {
            delete this->base;
        }
    }

    sptr(sptr &x)
    {
        this->base = x.base;
        this->base->refc++;
    }

    sptr(sptr &&x)
    {
        this->base = x.base;
        x.base = nullptr;
    }

    sptr &operator=(sptr &x)
    {
        if (this->base != x.base)
        {
            delete this->base;
            this->base = x.base;
            this->base->refc++;
        }
        return *this;
    }

    sptr &operator=(sptr &&x)
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

template <typename T, typename default_int>
struct array
{
    int refc = 0;
    default_int length;
    T *ptr;

    array() = delete;
    array(int l) : length(l), ptr(new T[l]) {}
    ~array() { delete[] this->ptr; }
};