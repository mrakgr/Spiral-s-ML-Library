// Host example that does not cause an memory error.
// I checked it using the VS Adress Sanitizer.

#include <utility>
#include <new>
#include <assert.h>
#include <stdio.h>
using default_int = int;
using default_uint = unsigned int;
template <typename el>
struct sptr // Shared pointer for the Spiral datatypes. They have to have the refc field inside them to work.
{
    el* base;

    sptr() : base(nullptr) {}
    sptr(el* ptr) : base(ptr) { this->base->refc++; }

    ~sptr()
    {
        if (this->base != nullptr && --this->base->refc == 0)
        {
            delete this->base;
            this->base = nullptr;
        }
    }

    sptr(sptr& x)
    {
        this->base = x.base;
        this->base->refc++;
    }

    sptr(sptr&& x)
    {
        this->base = x.base;
        x.base = nullptr;
    }

    sptr& operator=(sptr& x)
    {
        if (this->base != x.base)
        {
            delete this->base;
            this->base = x.base;
            this->base->refc++;
        }
        return *this;
    }

    sptr& operator=(sptr&& x)
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

template <typename el, default_int max_length>
struct static_array
{
    el ptr[max_length];
    el& operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < max_length);
        return this->ptr[i];
    }
};

template <typename el, default_int max_length>
struct static_array_list
{
    default_int length{ 0 };
    el ptr[max_length];

    el& operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < this->length);
        return this->ptr[i];
    }
    void push(el& x) {
        ptr[this->length++] = x;
        assert("The array after pushing should not be greater than max length." && this->length <= max_length);
    }
    void push(el&& x) {
        ptr[this->length++] = std::move(x);
        assert("The array after pushing should not be greater than max length." && this->length <= max_length);
    }
    el pop() {
        assert("The array before popping should be greater than 0." && 0 < this->length);
        auto x = ptr[--this->length];
        ptr[this->length].~el();
        new (&ptr[this->length]) el();
        return x;
    }
    // Should be used only during initialization.
    void unsafe_set_length(default_int i) {
        assert("The new length should be in range." && 0 <= i && i <= max_length);
        this->length = i;
    }
};

template <typename el, default_int max_length>
struct dynamic_array_list_base
{
    int refc{ 0 };
    default_int length{ 0 };
    el* ptr;

    dynamic_array_list_base() : ptr(new el[max_length]) {}
    dynamic_array_list_base(default_int l) : ptr(new el[max_length]) { this->unsafe_set_length(l); }
    ~dynamic_array_list_base() { delete[] this->ptr; }

    el& operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < this->length);
        return this->ptr[i];
    }
    void push(el& x) {
        ptr[this->length++] = x;
        assert("The array after pushing should not be greater than max length." && this->length <= max_length);
    }
    void push(el&& x) {
        ptr[this->length++] = std::move(x);
        assert("The array after pushing should not be greater than max length." && this->length <= max_length);
    }
    el pop() {
        assert("The array before popping should be greater than 0." && 0 < this->length);
        auto x = ptr[--this->length];
        ptr[this->length].~el();
        new (&ptr[this->length]) el();
        return x;
    }
    // Should be used only during initialization.
    void unsafe_set_length(default_int i) {
        assert("The new length should be in range." && 0 <= i && i <= max_length);
        this->length = i;
    }
};

template <typename el, default_int max_length>
struct dynamic_array_list
{
    sptr<dynamic_array_list_base<el, max_length>> ptr;

    dynamic_array_list() = default;
    dynamic_array_list(default_int l) : ptr(new dynamic_array_list_base<el, max_length>(l)) {}

    el& operator[](default_int i) {
        return this->ptr.base->operator[](i);
    }
    void push(el& x) {
        this->ptr.base->push(x);
    }
    void push(el&& x) {
        this->ptr.base->push(std::move(x));
    }
    el pop() {
        return this->ptr.base->pop();
    }
    // Should be used only during initialization.
    void unsafe_set_length(default_int i) {
        this->ptr.base->unsafe_set_length(i);
    }
    default_int length_() {
        return this->ptr.base->length;
    }
};

struct Union1_0 { // A_All_In
};
struct Union1_1 { // A_Call
};
struct Union1_2 { // A_Fold
};
struct Union1_3 { // A_Raise
    int v0;
    Union1_3(int t0) : v0(t0) {}
    Union1_3() = delete;
};
struct Union1 {
    union {
        Union1_0 case0; // A_All_In
        Union1_1 case1; // A_Call
        Union1_2 case2; // A_Fold
        Union1_3 case3; // A_Raise
    };
    unsigned char tag{ 255 };
    Union1() {}
    Union1(Union1_0 t) : tag(0), case0(t) {} // A_All_In
    Union1(Union1_1 t) : tag(1), case1(t) {} // A_Call
    Union1(Union1_2 t) : tag(2), case2(t) {} // A_Fold
    Union1(Union1_3 t) : tag(3), case3(t) {} // A_Raise
    Union1(Union1& x) : tag(x.tag) {
        switch (x.tag) {
        case 0: new (&this->case0) Union1_0(x.case0); break; // A_All_In
        case 1: new (&this->case1) Union1_1(x.case1); break; // A_Call
        case 2: new (&this->case2) Union1_2(x.case2); break; // A_Fold
        case 3: new (&this->case3) Union1_3(x.case3); break; // A_Raise
        }
    }
    Union1(Union1&& x) : tag(x.tag) {
        switch (x.tag) {
        case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // A_All_In
        case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // A_Call
        case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // A_Fold
        case 3: new (&this->case3) Union1_3(std::move(x.case3)); break; // A_Raise
        }
    }
    Union1& operator=(Union1& x) {
        if (this->tag == x.tag) {
            switch (x.tag) {
            case 0: this->case0 = x.case0; break; // A_All_In
            case 1: this->case1 = x.case1; break; // A_Call
            case 2: this->case2 = x.case2; break; // A_Fold
            case 3: this->case3 = x.case3; break; // A_Raise
            }
        }
        else {
            this->~Union1();
            new (this) Union1{ x };
        }
        return *this;
    }
    Union1& operator=(Union1&& x) {
        if (this->tag == x.tag) {
            switch (x.tag) {
            case 0: this->case0 = std::move(x.case0); break; // A_All_In
            case 1: this->case1 = std::move(x.case1); break; // A_Call
            case 2: this->case2 = std::move(x.case2); break; // A_Fold
            case 3: this->case3 = std::move(x.case3); break; // A_Raise
            }
        }
        else {
            this->~Union1();
            new (this) Union1{ std::move(x) };
        }
        return *this;
    }
    ~Union1() {
        switch (this->tag) {
        case 0: this->case0.~Union1_0(); break; // A_All_In
        case 1: this->case1.~Union1_1(); break; // A_Call
        case 2: this->case2.~Union1_2(); break; // A_Fold
        case 3: this->case3.~Union1_3(); break; // A_Raise
        }
        this->tag = 255;
    }
};
struct Tuple0 {
    static_array<unsigned char, 5> v0;
    char v1;
    Tuple0() = default;
    Tuple0(static_array<unsigned char, 5> t0, char t1) : v0(t0), v1(t1) {}
};
struct Union6_0 { // CommunityCardsAre
    static_array_list<unsigned char, 5> v0;
    Union6_0(static_array_list<unsigned char, 5> t0) : v0(t0) {}
    Union6_0() = delete;
};
struct Union6_1 { // Fold
    int v0;
    int v1;
    Union6_1(int t0, int t1) : v0(t0), v1(t1) {}
    Union6_1() = delete;
};
struct Union6_2 { // PlayerAction
    Union1 v1;
    int v0;
    Union6_2(int t0, Union1 t1) : v0(t0), v1(t1) {}
    Union6_2() = delete;
};
struct Union6_3 { // PlayerGotCards
    static_array<unsigned char, 2> v1;
    int v0;
    Union6_3(int t0, static_array<unsigned char, 2> t1) : v0(t0), v1(t1) {}
    Union6_3() = delete;
};
struct Union6_4 { // Showdown
    static_array<Tuple0, 2> v1;
    int v0;
    int v2;
    Union6_4(int t0, static_array<Tuple0, 2> t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    Union6_4() = delete;
};
struct Union6 {
    union {
        Union6_0 case0; // CommunityCardsAre
        Union6_1 case1; // Fold
        Union6_2 case2; // PlayerAction
        Union6_3 case3; // PlayerGotCards
        Union6_4 case4; // Showdown
    };
    unsigned char tag{ 255 };
    Union6() {}
    Union6(Union6_0 t) : tag(0), case0(t) {} // CommunityCardsAre
    Union6(Union6_1 t) : tag(1), case1(t) {} // Fold
    Union6(Union6_2 t) : tag(2), case2(t) {} // PlayerAction
    Union6(Union6_3 t) : tag(3), case3(t) {} // PlayerGotCards
    Union6(Union6_4 t) : tag(4), case4(t) {} // Showdown
    Union6(Union6& x) : tag(x.tag) {
        switch (x.tag) {
        case 0: new (&this->case0) Union6_0(x.case0); break; // CommunityCardsAre
        case 1: new (&this->case1) Union6_1(x.case1); break; // Fold
        case 2: new (&this->case2) Union6_2(x.case2); break; // PlayerAction
        case 3: new (&this->case3) Union6_3(x.case3); break; // PlayerGotCards
        case 4: new (&this->case4) Union6_4(x.case4); break; // Showdown
        }
    }
    Union6(Union6&& x) : tag(x.tag) {
        switch (x.tag) {
        case 0: new (&this->case0) Union6_0(std::move(x.case0)); break; // CommunityCardsAre
        case 1: new (&this->case1) Union6_1(std::move(x.case1)); break; // Fold
        case 2: new (&this->case2) Union6_2(std::move(x.case2)); break; // PlayerAction
        case 3: new (&this->case3) Union6_3(std::move(x.case3)); break; // PlayerGotCards
        case 4: new (&this->case4) Union6_4(std::move(x.case4)); break; // Showdown
        }
    }
    Union6& operator=(Union6& x) {
        if (this->tag == x.tag) {
            switch (x.tag) {
            case 0: this->case0 = x.case0; break; // CommunityCardsAre
            case 1: this->case1 = x.case1; break; // Fold
            case 2: this->case2 = x.case2; break; // PlayerAction
            case 3: this->case3 = x.case3; break; // PlayerGotCards
            case 4: this->case4 = x.case4; break; // Showdown
            }
        }
        else {
            this->~Union6();
            new (this) Union6{ x };
        }
        return *this;
    }
    Union6& operator=(Union6&& x) {
        if (this->tag == x.tag) {
            switch (x.tag) {
            case 0: this->case0 = std::move(x.case0); break; // CommunityCardsAre
            case 1: this->case1 = std::move(x.case1); break; // Fold
            case 2: this->case2 = std::move(x.case2); break; // PlayerAction
            case 3: this->case3 = std::move(x.case3); break; // PlayerGotCards
            case 4: this->case4 = std::move(x.case4); break; // Showdown
            }
        }
        else {
            this->~Union6();
            new (this) Union6{ std::move(x) };
        }
        return *this;
    }
    ~Union6() {
        switch (this->tag) {
        case 0: this->case0.~Union6_0(); break; // CommunityCardsAre
        case 1: this->case1.~Union6_1(); break; // Fold
        case 2: this->case2.~Union6_2(); break; // PlayerAction
        case 3: this->case3.~Union6_3(); break; // PlayerGotCards
        case 4: this->case4.~Union6_4(); break; // Showdown
        }
        this->tag = 255;
    }
};
void entry0() {
    printf("111\n");
    dynamic_array_list<Union6, 128> v10{ 0 };
    printf("222\n");
}
int main(int argc, const char* argv[])
{
    entry0();
    return 0;
}