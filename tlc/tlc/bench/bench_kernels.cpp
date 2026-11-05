// Бенчмарк ядер λ(A) + (λ^{-1}A)* — именно то, что меряет cli.cpp аналога 2.
//   naive : порт SpectralRadius()/Kleene() аналога 2 (степени матриц, O(n^4))
//   fast  : tlc (Карп + Флойд–Уоршелл, O(n^3))
// Матрицы генерируются так же, как в аналоге 2: a_ij = num/den, num,den ~ U{1..m},
// но с фиксированным seed (воспроизводимость). Время — медиана по reps запускам
// (steady_clock, прогрев 1 запуск). Память — пик кучи и число выделений
// (перехват operator new). Точность — относительная ошибка λ против эталона
// (Карп в long double) и max относительное расхождение звёзд.
//
// usage: bench_kernels [reps=7] [m=100] [n_list...]   -> CSV в stdout
#include <malloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <string>
#include <vector>

#include "tlc/tropical.hpp"

// ---------- учёт памяти ----------
static std::size_t g_cur = 0, g_peak = 0, g_count = 0;
void *operator new(std::size_t sz) {
    void *p = std::malloc(sz ? sz : 1);
    if (!p) throw std::bad_alloc();
    g_cur += malloc_usable_size(p);
    g_peak = std::max(g_peak, g_cur);
    ++g_count;
    return p;
}
void operator delete(void *p) noexcept {
    if (!p) return;
    g_cur -= malloc_usable_size(p);
    std::free(p);
}
void operator delete(void *p, std::size_t) noexcept { operator delete(p); }

using namespace tlc;

static Matrix gen(std::size_t n, unsigned m, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<unsigned> d(1, m);
    Matrix A(n, n, 1.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) {
            const double a = d(rng), b = d(rng);
            A(i, j) = a / b;
            A(j, i) = b / a;
        }
    return A;
}

// эталон λ: Карп в long double
static long double lambda_ref(const Matrix &A) {
    const std::size_t n = A.rows;
    const long double NEG = -std::numeric_limits<long double>::infinity();
    std::vector<long double> D((n + 1) * n, NEG);
    for (std::size_t v = 0; v < n; ++v) D[v] = 0;
    for (std::size_t k = 1; k <= n; ++k)
        for (std::size_t u = 0; u < n; ++u)
            for (std::size_t v = 0; v < n; ++v)
                D[k * n + v] = std::max(D[k * n + v], D[(k - 1) * n + u] + std::log(static_cast<long double>(A(u, v))));
    long double best = NEG;
    for (std::size_t v = 0; v < n; ++v) {
        long double w = std::numeric_limits<long double>::infinity();
        for (std::size_t k = 0; k < n; ++k) w = std::min(w, (D[n * n + v] - D[k * n + v]) / (long double)(n - k));
        best = std::max(best, w);
    }
    return std::exp(best);
}

struct Run {
    double ms_median = 0, ms_min = 0;
    std::size_t peak = 0, allocs = 0;
    double lambda = 0;
    Matrix star;
};

template <class F>
static Run measure(int reps, F &&f) {
    Run r;
    std::vector<double> t;
    f(r);  // прогрев
    for (int i = 0; i < reps; ++i) {
        g_peak = g_cur;
        const std::size_t c0 = g_count, base = g_cur;
        const auto t0 = std::chrono::steady_clock::now();
        f(r);
        const auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        r.peak = g_peak - base;
        r.allocs = g_count - c0;
    }
    std::sort(t.begin(), t.end());
    r.ms_median = t[t.size() / 2];
    r.ms_min = t.front();
    return r;
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 7;
    const unsigned m = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 100;
    std::vector<std::size_t> ns;
    for (int i = 3; i < argc; ++i) ns.push_back(static_cast<std::size_t>(std::atol(argv[i])));
    if (ns.empty()) ns = {10, 20, 40, 60, 80, 100, 120, 150, 160, 200, 300, 400};

    std::cout << "n,m,impl,median_ms,min_ms,peak_heap_bytes,allocs,lambda,lambda_rel_err,star_max_rel_diff,finite\n";
    for (std::size_t n : ns) {
        const Matrix A = gen(n, m, 1000 + n);
        const long double ref = lambda_ref(A);
        // аналог 2 прогоняет всё O(n^4); на больших n ограничиваем число повторов
        const int rn = n >= 200 ? std::max(1, reps / 3) : reps;
        Run naive = measure(rn, [&](Run &r) {
            r.lambda = spectral_radius_naive(A);
            r.star = kleene_star_naive(scale(A, 1 / r.lambda));
        });
        Run fast = measure(reps, [&](Run &r) {
            r.lambda = spectral_radius(A);
            r.star = kleene_star(scale(A, 1 / r.lambda));
        });
        double diff = 0;
        bool fin = true;
        for (std::size_t k = 0; k < fast.star.a.size(); ++k) {
            const double a = naive.star.a[k], b = fast.star.a[k];
            if (!std::isfinite(a)) { fin = false; continue; }
            diff = std::max(diff, std::fabs(a - b) / std::max(1.0, std::fabs(b)));
        }
        for (auto *r : {&naive, &fast}) {
            const bool f = r == &fast ? true : (fin && std::isfinite(naive.lambda));
            const double err = std::isfinite(r->lambda) ? std::fabs((long double)r->lambda - ref) / ref : INFINITY;
            std::cout << n << "," << m << "," << (r == &naive ? "analog2_naive" : "tlc_fast") << "," << r->ms_median
                      << "," << r->ms_min << "," << r->peak << "," << r->allocs << "," << r->lambda << "," << err
                      << "," << (r == &naive ? diff : 0.0) << "," << (f ? 1 : 0) << "\n";
        }
        std::cout.flush();
    }
}
