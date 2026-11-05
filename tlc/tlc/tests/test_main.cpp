// Модульные тесты: контрольные примеры из Кривулин и др. (КИО, 2020, №2) + свойства.
// Без внешних зависимостей: ./tlc_tests [data_dir]
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "tlc/solver.hpp"

using namespace tlc;

static int g_fail = 0, g_checks = 0;
static std::string g_data = "data";

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(cond)) {                                                                    \
            ++g_fail;                                                                     \
            std::cerr << "  FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond "\n";     \
        }                                                                                 \
    } while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) <= (tol))

static Problem load(const std::string &f) { return Problem::from_raw(io::load_problem(g_data + "/" + f)); }

// Множество векторов (с точностью до нормировки) совпадает с ожидаемым.
static bool same_set(const std::vector<Vector> &got, std::vector<Vector> exp, double tol = 1e-9) {
    if (got.size() != exp.size()) return false;
    for (auto &e : exp) e = normalize_max(e);
    for (const Vector &g : got) {
        bool found = false;
        for (const Vector &e : exp) found = found || approx_equal(normalize_max(g), e, tol);
        if (!found) return false;
    }
    return true;
}

static void dump(const std::vector<Vector> &v) {
    for (const auto &x : v) std::cerr << "    " << to_string(x, 6) << "\n";
}
#define CHECK_SET(got, ...)                                       \
    do {                                                          \
        bool ok_ = same_set(got, __VA_ARGS__);                    \
        CHECK(ok_ && #got);                                       \
        if (!ok_) dump(got);                                      \
    } while (0)

// Каждый найденный вектор оптимален в задаче (1): ошибка равна спектральному радиусу.
static void check_optimal(const Matrix &A, const std::vector<Vector> &xs, double lambda) {
    for (const Vector &x : xs) CHECK(approx_equal(approximation_error(A, x), lambda, 1e-9));
}

// ---------------- тесты ----------------

static void test_parser() {
    CHECK_NEAR(io::parse_number("1/4").value(), 0.25, 1e-15);
    CHECK_NEAR(io::parse_number("2.5").value(), 2.5, 1e-15);
    CHECK(io::parse_number("2.5").num == 25 && io::parse_number("2.5").den == 10);
    bool threw = false;
    try { io::parse_number("-1"); } catch (const std::exception &) { threw = true; }
    CHECK(threw);
    threw = false;
    try { io::parse_number("1/0"); } catch (const std::exception &) { threw = true; }
    CHECK(threw);
    threw = false;
    try { io::parse_number("abc"); } catch (const std::exception &) { threw = true; }
    CHECK(threw);
    threw = false;
    std::istringstream bad("criteria 2\nalternatives 2\nC 1 2 1/2 1\nA 1 1 1 1\n");  // одной A не хватает
    try { io::parse_problem(bad); } catch (const std::exception &) { threw = true; }
    CHECK(threw);
    std::istringstream nonrec("criteria 1\nalternatives 2\nC 1\nA 1 2 3 1\n");
    CHECK(io::parse_problem(nonrec).warnings.size() == 1);
}

static void test_primitives() {
    Matrix A(2, 2);
    A(0, 0) = 1; A(0, 1) = 3; A(1, 0) = 1.0 / 3; A(1, 1) = 1;
    CHECK_NEAR(spectral_radius(A), 1.0, 1e-12);  // согласованная матрица: λ = 1
    Matrix B(2, 2);
    B(0, 0) = 0; B(0, 1) = 4; B(1, 0) = 9; B(1, 1) = 0;
    CHECK_NEAR(spectral_radius(B), 6.0, 1e-12);  // цикл длины 2: sqrt(36)
    CHECK_NEAR(spectral_radius(Matrix(3, 3, 0.0)), 0.0, 0.0);
    Matrix Bc = conj(B);
    CHECK_NEAR(Bc(0, 1), 1.0 / 9, 1e-15);
    CHECK_NEAR(Bc(1, 0), 1.0 / 4, 1e-15);
    // тропическая независимость: (1,1) = (1,0.5) ⊕ (0.5,1)
    CHECK(in_span({{1, 0.5}, {0.5, 1}}, {1, 1}));
    CHECK(!in_span({{1, 0.5}}, {0.5, 1}));
    CHECK(independent_columns({{1, 0.5}, {2, 1}, {0.5, 1}, {1, 1}}).size() == 2);
}

static void test_task1() {  // §4.1, картинка из задания
    const Problem p = load("task1_saaty_hu_p124.txt");
    const double lam = std::pow(2.0, 2.0 / 3), mu = 2 * std::sqrt(3.0), nu = std::pow(2.0, 11.0 / 6);

    const Vector ahp = solve_ahp(p), gm = solve_gm(p);
    const Vector ahp_paper{0.2423, 0.2485, 0.2535, 0.2557}, gm_paper{0.8747, 0.8687, 1.1483, 1.1461};
    for (int i = 0; i < 4; ++i) {
        CHECK_NEAR(ahp[i], ahp_paper[i], 1e-4);
        CHECK_NEAR(gm[i], gm_paper[i], 1e-4);
    }
    CHECK(ranking(ahp) == "A4 > A3 > A2 > A1");
    CHECK(ranking(gm) == "A3 > A4 > A1 > A2");

    const auto r = solve_log_chebyshev(p);
    CHECK_NEAR(r.criteria.lambda, lam, 1e-12);
    // (λ^{-1}C)* из статьи
    const double S[4][4] = {{1, 6, 2 / lam, 3 * lam / 2},
                            {1 / (4 * lam), 1, 1 / (3 * lam), lam / 4},
                            {0.75, 3 * lam, 1, 3 / lam},
                            {lam / 4, 4 / lam, lam / 3, 1}};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) CHECK_NEAR(r.criteria.star(i, j), S[i][j], 1e-12);
    CHECK(r.criteria.generators.size() == 2);
    CHECK_SET(r.best_weights, {{1, 1 / (4 * lam), 0.75, lam / 4}});
    CHECK_SET(r.worst_weights, {{1, 1.0 / 6, lam / 2, 2 / (3 * lam)}});
    CHECK(r.mu.size() == 1 && approx_equal(r.mu[0], mu, 1e-12));
    CHECK(r.nu.size() == 1 && approx_equal(r.nu[0], nu, 1e-12));
    CHECK_SET(r.best_ratings, {{mu / 6, 0.5, 1, 3 * mu / 16}, {mu / 6, 0.5, mu / 4, 1}});
    CHECK_SET(r.worst_ratings, {{1, nu / 4, nu / 4, 1}, {1, nu / 4, 1, 1}});
    check_optimal(p.C, r.best_weights, lam);
    check_optimal(p.C, r.worst_weights, lam);
    check_optimal(weighted_sum(p.A, r.best_weights[0]), r.best_ratings, mu);
    check_optimal(weighted_sum(p.A, r.worst_weights[0]), r.worst_ratings, nu);
}

static void test_task2() {  // §4.2
    const Problem p = load("task2_saaty_hu_p122.txt");
    const double lam = std::pow(2.0, 2.0 / 3), mu = 2 * std::sqrt(3.0), nu = std::pow(2.0, 11.0 / 6);
    const auto r = solve_log_chebyshev(p);
    CHECK_NEAR(r.criteria.lambda, lam, 1e-12);
    CHECK_SET(r.best_weights, {{1, 0.75, lam / 4, 1 / (4 * lam)}});
    CHECK_SET(r.worst_weights, {{1, lam / 2, 2 / (3 * lam), 1.0 / 6}});
    CHECK_SET(r.best_ratings, {{mu / 6, 1, mu / 6, 0.5}, {mu / 6, mu / 4, 1, 0.5}});
    CHECK_SET(r.worst_ratings, {{1, nu / 4, 1, nu / 4}, {1, 1, 1, nu / 4}});
    CHECK(ranking(solve_gm(p)) == "A2 > A1 > A4 > A3");
    CHECK(ranking(solve_ahp(p)) == "A2 > A4 > A1 > A3");
}

static void test_job() {  // §5.1
    const Problem p = load("job_saaty1977.txt");
    const double lam = std::pow(12.0, 0.25), mu = std::sqrt(45 * lam / 2);
    const auto r = solve_log_chebyshev(p);
    CHECK_NEAR(r.criteria.lambda, lam, 1e-12);
    const double l2 = lam * lam;
    CHECK_SET(r.best_weights, {{lam / 2, lam / 2, l2 / 4, 1 / (3 * lam), 1 / lam, 1},
                               {0.5, lam / 2, l2 / 4, 1 / (3 * lam), 1 / lam, 1}});
    CHECK_SET(r.worst_weights, {{lam / 2, lam / 2, l2 / 4, 3 / (5 * lam), 1 / lam, 1},
                                {l2 / 5, lam / 2, l2 / 4, 3 / (5 * lam), 1 / lam, 1}});
    for (double m : r.mu) CHECK_NEAR(m, mu, 1e-12);
    CHECK_SET(r.best_ratings, {{1, 2 * lam / mu, mu / 9}});
    CHECK_SET(r.worst_ratings, {{1, mu / 9, mu / 9}, {1, mu / 7, mu / 9}});
    const Vector ahp = solve_ahp(p), gm = solve_gm(p);
    CHECK_NEAR(ahp[0], 0.3843, 1e-4);
    CHECK_NEAR(gm[1], 1.2642, 1e-4);
}

static void test_city() {  // §5.2 — см. README: в статье есть арифметические расхождения
    const Problem p = load("city_saaty2004.txt");
    const double lam = std::pow(45.0 / 7, 0.25);
    const auto r = solve_log_chebyshev(p);
    CHECK_NEAR(r.criteria.lambda, lam, 1e-12);
    CHECK(same_set(r.best_weights, r.worst_weights));  // как и в статье: w1 = w2
    for (const Vector &x : r.best_ratings) CHECK(ranking(x) == "A1 > A3 > A2 > A4");
    const Vector gm = solve_gm(p), gm_paper{1.2652, 1.1051, 1.4876, 0.4808};
    for (int i = 0; i < 4; ++i) CHECK_NEAR(gm[i], gm_paper[i], 1e-4);
    // AHP: порядок как в статье; числа статьи (0.1757, 0.2483, 0.4598, 0.1162) не воспроизводятся
    CHECK(ranking(solve_ahp(p)) == "A3 > A2 > A1 > A4");
    // каждый рейтинг оптимален хотя бы для одной взвешенной суммы B(w1)
    for (const Vector &x : r.best_ratings) {
        bool opt = false;
        for (std::size_t t = 0; t < r.best_weights.size(); ++t)
            opt = opt || approx_equal(approximation_error(weighted_sum(p.A, r.best_weights[t]), x), r.mu[t], 1e-9);
        CHECK(opt);
    }
}

static void test_belton_gear() {  // §6: парадокс смены рангов AHP
    const Problem p1 = load("belton_gear_1.txt"), p2 = load("belton_gear_2.txt");
    const auto r1 = solve_log_chebyshev(p1), r2 = solve_log_chebyshev(p2);
    CHECK_SET(r1.best_ratings, {{1, 1, 1.0 / 9}});
    CHECK_SET(r1.worst_ratings, {{1, 1, 1}});
    CHECK_SET(r2.best_ratings, {{1, 1, 1.0 / 9, 1}});
    CHECK_SET(r2.worst_ratings, {{1, 1, 1, 1}});
    CHECK(r1.mu[0] == 9.0 || approx_equal(r1.mu[0], 9.0));
    // AHP меняет порядок A1 и A2 при добавлении копии A2, log-Cheb и GM — нет
    CHECK(ranking(solve_ahp(p1)) == "A2 > A1 > A3");
    CHECK(ranking(solve_ahp(p2)) == "A1 > A2 = A4 > A3");
    CHECK(ranking(solve_gm(p2)) == "A2 = A4 > A1 > A3");
}

static void test_fast_equals_naive() {  // ускоренные ядра дают тот же результат
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> d(1, 9);
    for (int it = 0; it < 300; ++it) {
        const std::size_t n = 1 + it % 12;
        Matrix A(n, n, 1.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < i; ++j) {
                A(i, j) = static_cast<double>(d(rng)) / d(rng);
                A(j, i) = 1.0 / A(i, j);
            }
        const double l1 = spectral_radius(A), l2 = spectral_radius_naive(A);
        CHECK(approx_equal(l1, l2, 1e-10));
        const Matrix N = scale(A, 1 / l1);
        const Matrix S1 = kleene_star(N), S2 = kleene_star_naive(N);
        for (std::size_t k = 0; k < S1.a.size(); ++k) CHECK(approx_equal(S1.a[k], S2.a[k], 1e-10));
    }
    const Problem p = load("task1_saaty_hu_p124.txt");
    Options naive;
    naive.naive_kernels = true;
    CHECK(same_set(solve_log_chebyshev(p, naive).best_ratings, solve_log_chebyshev(p).best_ratings));
}

static void test_invalid_input() {
    Problem p;
    p.C = Matrix(2, 2, 1.0);
    p.A = {Matrix(2, 2, 1.0)};  // нужно 2 матрицы
    bool threw = false;
    try { solve_log_chebyshev(p); } catch (const std::invalid_argument &) { threw = true; }
    CHECK(threw);
    p.A.push_back(Matrix(2, 2, 1.0));
    p.A[1](0, 1) = 0.0;  // неположительный элемент
    threw = false;
    try { solve_log_chebyshev(p); } catch (const std::invalid_argument &) { threw = true; }
    CHECK(threw);
    p.A[1](0, 1) = 1.0;
    const auto r = solve_log_chebyshev(p);  // всё согласовано и равно: x = (1,1)
    CHECK_SET(r.best_ratings, {{1, 1}});
}

int main(int argc, char **argv) {
    if (argc > 1) g_data = argv[1];
    const std::vector<std::pair<const char *, std::function<void()>>> tests = {
        {"parser", test_parser},           {"primitives", test_primitives}, {"task1 (§4.1)", test_task1},
        {"task2 (§4.2)", test_task2},      {"job (§5.1)", test_job},        {"city (§5.2)", test_city},
        {"belton-gear (§6)", test_belton_gear}, {"fast == naive", test_fast_equals_naive},
        {"invalid input", test_invalid_input},
    };
    for (const auto &[name, fn] : tests) {
        const int before = g_fail;
        try { fn(); } catch (const std::exception &e) { ++g_fail; std::cerr << "  EXCEPTION " << e.what() << "\n"; }
        std::cout << (g_fail == before ? "[ OK ] " : "[FAIL] ") << name << "\n";
    }
    std::cout << g_checks << " checks, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
