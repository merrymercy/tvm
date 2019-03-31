#ifndef __UTIL_H__
#define __UTIL_H__

namespace Halide {

template<typename T1, typename T2>
void check_equal_shape(const Halide::Runtime::Buffer<T1> &a, const Halide::Runtime::Buffer<T2> &b) {
    if (a.dimensions() != b.dimensions()) abort();
    for (int i = 0; i < a.dimensions(); i++) {
        if (a.dim(i).min() != b.dim(i).min() ||
            a.dim(i).extent() != b.dim(i).extent()) {
            abort();
        }
    }
}

template<typename T1, typename T2>
void check_equal(const Halide::Runtime::Buffer<T1> &a, const Halide::Runtime::Buffer<T2> &b, float r_tol) {
    check_equal_shape(a, b);
    a.for_each_element([&](const int *pos) {
        if (fabs((a(pos) - b(pos)) / a(pos)) > r_tol) {
            printf("Mismatch: %f vs %f\n", (float)(a(pos)), (float)(b(pos)));
            abort();
        }
    });
}

} // namespace Halide

#endif

