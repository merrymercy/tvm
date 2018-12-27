/*
 * Test the simple auto scheduler in Halide
 *
 * g++ auto_scheduler_test.cpp *.a  -lHalide -I../include -I../tools/ -L../lib -lpthread -ldl -std=c++11
 */

#include <tuple>
#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "Halide.h"
#include "halide_benchmark.h"
#include "util.h"

#ifdef EVAL_MODE
#define EVAL_FUNC(x) x
#define EVAL_HEADER(x) x
#else
#define EVAL_FUNC(x)
#define EVAL_HEADER(x) <cstdio>
#endif

using namespace Halide;
SimpleAutoscheduleOptions cpu_options;


// norm
#include EVAL_HEADER("func_norm.h")
std::pair<double, long> test_norm(int N) {
#ifndef EVAL_MODE
    Buffer<float> A(N, N, N, "A");
    Func norm("norm");
    RDom r(A);
    norm() = cast<float>(0.0f);
    norm() += A(r.x, r.y, r.z) * A(r.x, r.y, r.z);

    simple_autoschedule(norm,
                        {}, // parameters map
                        {}, // output bounds (min, max)
                        cpu_options);

    norm.compile_to_static_library("func_norm", {A}, "func_norm");

   
    return std::make_pair(0.d, 0);
#else
    Halide::Runtime::Buffer<float> A(N, N, N);
    Halide::Runtime::Buffer<float> norm = Halide::Runtime::Buffer<float>::make_scalar();
    Halide::Runtime::Buffer<float> answer = Halide::Runtime::Buffer<float>::make_scalar();

    A.for_each_element([&](int i, int j, int k) { A(i, j, k) = (rand() % 100) * 0.01; });
    A.for_each_element([&](int i, int j, int k) { answer(0) += A(i, j, k) * A(i, j, k); });

    double min_time = Halide::Tools::benchmark([&]() { EVAL_FUNC(func_norm(A, norm)); });
    check_equal(answer, norm, 1e-1);

    return std::make_pair(min_time, 2L * N * N * N);
#endif
}


// matrix multiplication
#include EVAL_HEADER("func_matmul.h")
std::pair<double, long> test_matmul(int N, int L, int M) {
#ifndef EVAL_MODE
    Buffer<float> A(N, L, "A");
    Buffer<float> B(L, M, "B");
    Func matmul("matmul");
    Var x("x"), y("y");
    RDom k(0, L, "k");
    matmul(x, y) = 0.f;
    matmul(x, y) += A(x, k) * B(k, y);

//    simple_autoschedule(matmul,
//                        {}, // parameters map
//                        {{0, N-1},
//                         {0, M-1}}, // output bounds (min, max)
//                        cpu_options);
//    matmul.compile_to_static_library("func_matmul", {A, B}, "func_matmul");

    matmul.estimate(x, 0, N).estimate(y, 0, M);
    Pipeline p(matmul);
    p.auto_schedule(get_jit_target_from_environment());
    p.compile_to_static_library("func_matmul", {A, B}, "func_matmul");
    return std::make_pair(0.d, 0);
#else
    Halide::Runtime::Buffer<float> A(N, L);
    Halide::Runtime::Buffer<float> B(L, M);
    Halide::Runtime::Buffer<float> C(N, M);
    Halide::Runtime::Buffer<float> answer(N, M);

    A.for_each_element([&](int i, int k) { A(i, k) = (rand() % 100) * 0.01; });
    B.for_each_element([&](int k, int j) { B(k, j) = (rand() % 100) * 0.01; });

    double min_time = Halide::Tools::benchmark([&]() {
        EVAL_FUNC(func_matmul(A, B, C));
    });

    return std::make_pair(min_time, 2L * N * L * M);
#endif
}

// 3d convolution
#include EVAL_HEADER("func_conv3d.h")
std::pair<double, long> test_conv3d(int N, int CI, int D, int H, int W, int CO, int kernel_size, int stride, int padding) {
    int KD, KH, KW;
    int SD, SH, SW;
    int PD, PH, PW;
    int OD, OH, OW;

    KD = KH = KW = kernel_size;
    SD = SH = SW = stride;
    PD = PH = PW = padding;
    OD = (D + 2 * PD - KD) / SD + 1;
    OH = (H + 2 * PH - KH) / SH + 1;
    OW = (W + 2 * PW - KW) / SW + 1;

#ifndef EVAL_MODE
    Buffer<float> A(N, CI, D, H, W, "A");
    Buffer<float> B(CO, CI, KD, KH, KW, "B");

    Func conv3d("conv3d");
    RDom r(0, CI, 0, KD, 0, KH, 0, KW);
    Var n("n"), co("co"), od("od"), oh("oh"), ow("ow");
    conv3d(n, co, od, oh, ow) = 0.f;
    conv3d(n, co, od, oh, ow) += A(n, r.x, od * SD + r.y, oh * SH + r.z, ow * SW + r.w) * 
                                 B(co, r.x, r.y, r.z, r.w);

//    simple_autoschedule(conv3d,
//                        {}, // parameters map
//                        {{0, N-1}, {0, CO-1}, {0, OD-1}, {0, OH-1}, {0, OW-1}},
//                        cpu_options);
//    conv3d.compile_to_static_library("func_conv3d", {A, B}, "func_conv3d");

    conv3d.estimate(n, 0, N).estimate(co, 0, CO).estimate(od, 0, OD).estimate(oh, 0, OH).estimate(ow, 0, OW);
    Pipeline p(conv3d);
    p.auto_schedule(get_jit_target_from_environment());
    p.compile_to_static_library("func_conv3d", {A, B}, "func_conv3d");
    return std::make_pair(0.d, 0);
#else
    Halide::Runtime::Buffer<float> A(N, CI, D, H, W);
    Halide::Runtime::Buffer<float> B(CO, CI, KD, KH, KW);
    Halide::Runtime::Buffer<float> C(N, CO, OD, OH, OW);

    A.for_each_element([&](int a, int b, int c, int d, int e) { A(a, b, c, d, e) = (rand() % 100) * 0.01; });
    B.for_each_element([&](int a, int b, int c, int d, int e) { B(a, b, c, d, e) = (rand() % 100) * 0.01; });

    double min_time = Halide::Tools::benchmark([&]() {
        EVAL_FUNC(func_conv3d(A, B, C));
    });

    return std::make_pair(min_time, 2L * N* CO * OD * OH * OW * CI * KD * KH * KW);
#endif
}

// 3d pooling
#include EVAL_HEADER("func_avg_pool3d.h")
std::pair<double, long> test_avg_pool3d(int N, int CI, int D, int H, int W, int kernel_size, int stride, int padding) {
    int KD, KH, KW;
    int SD, SH, SW;
    int PD, PH, PW;
    int OD, OH, OW;

    KD = KH = KW = kernel_size;
    SD = SH = SW = stride;
    PD = PH = PW = padding;
    OD = (D + 2 * PD - KD) / SD + 1;
    OH = (H + 2 * PH - KH) / SH + 1;
    OW = (W + 2 * PW - KW) / SW + 1;

#ifndef EVAL_MODE
    Buffer<float> A(N, CI, D, H, W, "A");

    Func avg_pool3d("avg_pool3d");
    RDom r(0, KD, 0, KH, 0, KW);
    Var n("n"), ci("ci"), od("od"), oh("oh"), ow("ow");

    avg_pool3d(n, ci, od, oh, ow) += A(n, ci, od * SD + r.x, oh * SH + r.y, ow * SW + r.z); 
    avg_pool3d(n, ci, od, oh, ow) /= KD * KH * KW;

    simple_autoschedule(avg_pool3d,
                        {}, // parameters map
                        {{0, N-1}, {0, CI-1}, {0, OD-1}, {0, OH-1}, {0, OW-1}},
                        cpu_options);

    avg_pool3d.compile_to_static_library("func_avg_pool3d", {A}, "func_avg_pool3d");
    return std::make_pair(0.d, 0);
#else
    Halide::Runtime::Buffer<float> A(N, CI, D, H, W);
    Halide::Runtime::Buffer<float> C(N, CI, OD, OH, OW);

    A.for_each_element([&](int a, int b, int c, int d, int e) { A(a, b, c, d, e) = (rand() % 100) * 0.01; });

    double min_time = Halide::Tools::benchmark([&]() {
        EVAL_FUNC(func_avg_pool3d(A, C));
    });

    return std::make_pair(min_time, 2L * N * CI * OD * OH * OW * KD * KH * KW);
#endif
}

// convolution with large dilation
#include EVAL_HEADER("func_dilated_conv2d.h")
std::pair<double, long> test_dilated_conv2d(int N, int CI, int H, int W, int CO, int kernel_size, int stride, int padding, int dilation) {
    int KH, KW;
    int SH, SW;
    int PH, PW;
    int DH, DW;
    int dilated_KH, dilated_KW;
    int OH, OW;

    KH = KW = kernel_size;
    SH = SW = stride;
    PH = PW = padding;
    DH = DW = dilation;

    dilated_KH = (KH - 1) * DH + 1;
    dilated_KW = (KW - 1) * DW + 1;
    OH = (H + 2 * PH - dilated_KH) / SH + 1;
    OW = (W + 2 * PW - dilated_KW) / SW + 1;

#ifndef EVAL_MODE
    Buffer<float> A(N, CI, H, W, "A");
    Buffer<float> B(CO, CI, KH, KW, "B");

    Func dilated_conv2d("dilated_conv2d");
    RDom r(0, CI, 0, KH, 0, KW);
    Var n("n"), co("co"), od("od"), oh("oh"), ow("ow");

    dilated_conv2d(n, co, oh, ow) += A(n, r.x, oh * SH + r.y * DH, ow * SW + r.z * DW) * B(co, r.x, r.y, r.z);

//    simple_autoschedule(dilated_conv2d,
//                        {}, // parameters map
//                        {{0, N-1}, {0, CI-1}, {0, OH-1}, {0, OW-1}},
//                        cpu_options);
//    dilated_conv2d.compile_to_static_library("func_dilated_conv2d", {A, B}, "func_dilated_conv2d");

    dilated_conv2d.estimate(n, 0, N).estimate(co, 0, CO).estimate(oh, 0, OH).estimate(ow, 0, OW);
    Pipeline p(dilated_conv2d);
    p.auto_schedule(get_jit_target_from_environment());
    p.compile_to_static_library("func_dilated_conv2d", {A, B}, "func_dilated_conv2d");
    return std::make_pair(0.d, 0);
#else
    Halide::Runtime::Buffer<float> A(N, CI, H, W);
    Halide::Runtime::Buffer<float> B(CO, CI, KH, KW);
    Halide::Runtime::Buffer<float> C(N, CO, OH, OW);

    A.for_each_element([&](int a, int b, int c, int d) { A(a, b, c, d) = (rand() % 100) * 0.01; });
    B.for_each_element([&](int a, int b, int c, int d) { B(a, b, c, d) = (rand() % 100) * 0.01; });

    double min_time = Halide::Tools::benchmark([&]() {
        EVAL_FUNC(func_dilated_conv2d(A, B, C));
    });

    return std::make_pair(min_time, 2L * N * CO * OH * OW * CI * KH * KW);
#endif
}


struct TestCase {
    std::string name;
    std::function<std::pair<double, long>()> func;
};


TestCase cases[] = {
    TestCase{"norm",    std::bind(test_norm, 256)},
    TestCase{"matmul",  std::bind(test_matmul, 1024, 1024, 1024)},
    TestCase{"conv3d",  std::bind(test_conv3d, 1, 64, 16, 56, 56, 64, 3, 1, 0)},
    TestCase{"avg_pool3d", std::bind(test_avg_pool3d, 1, 2048, 7, 7, 7, 7, 1, 0)},
    TestCase{"dilated_conv2d", std::bind(test_dilated_conv2d,  1, 64, 56, 56, 64, 3, 1, 0, 16)},
};


int main() {
    for (auto test_case : cases) {
        std::stringstream ss;

        double time_cost;
        long flop;

        std::tie(time_cost, flop) = test_case.func();

        ss << "halide" << "\t"
           << test_case.name + "\t"
           << std::fixed << std::setprecision(3) << time_cost * 1e3 << "\t"
           << std::fixed << std::setprecision(2) << (flop / time_cost) / 1e9 << std::endl;

        std::string record = ss.str();

        std::cout << record;
    }
}

