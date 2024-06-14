template <typename T> void call_destructor(T & x) { x.~T(); }
template <typename el, int dim> struct static_array { el v[dim]; };
template <typename el, int dim, typename default_int> struct static_array_list { default_int length; el v[dim]; };

template <typename T>
struct sptr
{
    struct sptr_base {
        int refc;
        T v;

        sptr_base() = delete;
        sptr_base(T & v_) : refc(1), v(v_) {};
        sptr_base(T && v_) : refc(1), v(std::move(v_)) {};

        ~sptr_base(){ this->v.dispose(); }
    } * base;

    sptr() : base(nullptr) {}
    sptr(T & v) : base(new sptr_base(v)) {}
    sptr(T && v) : base(new sptr_base(std::move(v))) {}

    void dispose(){
        if (this->base != nullptr && --this->base->refc == 0)
        {
            delete this->base;
        }
    }

    ~sptr() { this->dispose(); }

    sptr(sptr & x) {
        this->base = x.base;
        this->base->refc++;
    }

    sptr(sptr && x) {
        this->base = x.base;
        x.base = nullptr;
    }

    sptr & operator=(sptr &x)
    {
        if (this->base != x.base){
            this->dispose();
            this->base = x.base;
            this->base->refc++;
        }
        return *this;
    }
    
    sptr & operator=(sptr &&x)
    {
        if (this->base != x.base){
            this->dispose();
            this->base = x.base;
            x.base = nullptr;
        }
        return *this;
    }
};

template <typename T, typename default_int>
struct array {
    default_int length;
    T * ptr;

    array() = delete;
    array(int l) : length(l), ptr(new T[l]) { }
    void dispose(){ delete[] this->ptr; }
};