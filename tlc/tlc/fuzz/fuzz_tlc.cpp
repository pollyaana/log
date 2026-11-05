// Фаззинг библиотеки tlc.
// Режим 1 (по умолчанию, без libFuzzer): ./tlc_fuzz <iterations> <seed>
//   генерирует случайные (в т.ч. «плохие») матрицы и проверяет инварианты:
//   P1  λ_Karp == λ_naive                      (дифференциальный тест ядер)
//   P2  FW-звезда == сумма степеней
//   P3  каждый образующий x оптимален: max a_ij x_j/x_i == λ
//   P4  наилучшие/наихудшие векторы лежат в множестве решений и
//       max/min(best) >= max/min(любого образующего) >= max/min(worst)
//   P5  перестановка альтернатив переставляет ответ (метаморфный тест)
//   P6  весь конвейер выдаёт конечные положительные векторы
// Режим 2: сборка с -DTLC_LIBFUZZER -fsanitize=fuzzer — байты подаются в парсер и решатель.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "tlc/solver.hpp"

using namespace tlc;

#ifdef TLC_LIBFUZZER
extern "C" int LLVMFuzzerTestOneInput(const unsigned char *data, unsigned long size) {
    std::istringstream in(std::string(reinterpret_cast<const char *>(data), size));
    try {
        const auto raw = io::parse_problem(in);
        if (raw.m > 12 || raw.n > 12) return 0;
        const Problem p = Problem::from_raw(raw);
        (void)solve_log_chebyshev(p);
        (void)solve_ahp(p);
        (void)solve_gm(p);
    } catch (const std::exception &) {
    }
    return 0;
}
#else

static std::mt19937_64 rng;
static long g_fail = 0;

static void fail(const std::string &what, const Matrix &A) {
    if (++g_fail <= 10) std::cerr << "FAIL: " << what << "\n" << to_string(A, 6);
}

static double ratio(const Vector &x) {
    auto [mn, mx] = std::minmax_element(x.begin(), x.end());
    return *mx / *mn;
}

// Случайная матрица парных сравнений: обратно-симметричная (шкала Саати или
// произвольные рациональные) либо просто положительная, иногда «вырожденная».
static Matrix random_matrix(std::size_t n) {
    Matrix A(n, n, 1.0);
    const int kind = static_cast<int>(rng() % 5);
    std::uniform_int_distribution<int> saaty(1, 9);
    std::uniform_real_distribution<double> logu(-std::log(1e3), std::log(1e3));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) {
            double v;
            switch (kind) {
                case 0: v = saaty(rng); if (rng() & 1) v = 1 / v; break;   // шкала Саати
                case 1: v = static_cast<double>(saaty(rng)) / saaty(rng); break;
                case 2: v = std::exp(logu(rng)); break;                     // широкий диапазон
                case 3: v = 1.0; break;                                     // все равны (вырожденный случай)
                default: v = std::exp(logu(rng)); A(j, i) = std::exp(logu(rng)); A(i, j) = v; continue;  // не обратная
            }
            A(i, j) = v;
            A(j, i) = 1.0 / v;
        }
    return A;
}

static void check_kernels(const Matrix &A) {
    const double l1 = spectral_radius(A), l2 = spectral_radius_naive(A);
    if (!approx_equal(l1, l2, 1e-9)) fail("P1 lambda Karp " + std::to_string(l1) + " vs naive " + std::to_string(l2), A);
    const Matrix N = scale(A, 1 / l1);
    const Matrix S1 = kleene_star(N), S2 = kleene_star_naive(N);
    for (std::size_t k = 0; k < S1.a.size(); ++k)
        if (!approx_equal(S1.a[k], S2.a[k], 1e-8)) { fail("P2 Kleene star mismatch", A); break; }
    const SingleSolution s = solve_single(A);
    for (const Vector &x : s.generators)
        if (!approx_equal(approximation_error(A, x), s.lambda, 1e-8)) fail("P3 generator not optimal", A);
    if (s.generators.size() > 1) {
        const auto best = best_differentiating(s.generators), worst = worst_differentiating(A, s.lambda);
        double gmax = 0, gmin = 1e300;
        for (const Vector &g : s.generators) gmax = std::max(gmax, ratio(g)), gmin = std::min(gmin, ratio(g));
        for (const Vector &x : best) {
            if (!approx_equal(approximation_error(A, x), s.lambda, 1e-8)) fail("P4 best not optimal", A);
            if (ratio(x) < gmax * (1 - 1e-9)) fail("P4 best ratio < generator ratio", A);
        }
        for (const Vector &x : worst) {
            if (!approx_equal(approximation_error(A, x), s.lambda, 1e-8)) fail("P4 worst not optimal", A);
            if (ratio(x) > gmin * (1 + 1e-9)) fail("P4 worst ratio > generator ratio", A);
        }
        if (best.empty() || worst.empty()) fail("P4 empty set", A);
    }
}

static bool same_set(const std::vector<Vector> &a, const std::vector<Vector> &b) {
    if (a.size() != b.size()) return false;
    for (const Vector &x : a)
        if (std::none_of(b.begin(), b.end(), [&](const Vector &y) { return approx_equal(x, y, 1e-8); })) return false;
    return true;
}

static void check_pipeline(std::size_t m, std::size_t n) {
    Problem p;
    p.C = random_matrix(m);
    for (std::size_t k = 0; k < m; ++k) p.A.push_back(random_matrix(n));
    const auto r = solve_log_chebyshev(p);
    for (const auto *set : {&r.best_ratings, &r.worst_ratings, &r.best_weights, &r.worst_weights})
        for (const Vector &x : *set)
            for (double v : x)
                if (!(v > 0 && std::isfinite(v))) { fail("P6 non-positive/non-finite output", p.C); return; }
    // P5: перестановка альтернатив
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    Problem q = p;
    for (std::size_t k = 0; k < m; ++k)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) q.A[k](i, j) = p.A[k](perm[i], perm[j]);
    const auto rq = solve_log_chebyshev(q);
    auto permute = [&](std::vector<Vector> v) {
        for (Vector &x : v) {
            Vector y(n);
            for (std::size_t i = 0; i < n; ++i) y[i] = x[perm[i]];
            x = y;
        }
        return v;
    };
    if (!same_set(permute(r.best_ratings), rq.best_ratings)) fail("P5 best ratings not permutation-equivariant", p.C);
    if (!same_set(permute(r.worst_ratings), rq.worst_ratings)) fail("P5 worst ratings not permutation-equivariant", p.C);
}

int main(int argc, char **argv) {
    const long iters = argc > 1 ? std::stol(argv[1]) : 20000;
    rng.seed(argc > 2 ? std::stoull(argv[2]) : 2026);
    for (long it = 0; it < iters; ++it) {
        try {
            check_kernels(random_matrix(1 + rng() % 10));
            if (it % 4 == 0) check_pipeline(1 + rng() % 6, 1 + rng() % 6);
        } catch (const std::exception &e) {
            fail(std::string("exception: ") + e.what(), Matrix());
        }
    }
    std::cout << iters << " iterations, " << g_fail << " failures\n";
    return g_fail ? 1 : 0;
}
#endif
