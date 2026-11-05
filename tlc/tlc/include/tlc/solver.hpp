// solver.hpp — многокритериальная задача парных сравнений:
//   * метод минимаксной log-чебышевской аппроксимации (Кривулин и др., 2020, §3.2,
//     формулы (3)–(12)) с наилучшими и наихудшими дифференцирующими решениями;
//   * метод анализа иерархий Т. Саати (AHP) и метод взвешенных геометрических
//     средних (GM) — для сверки с результатами статьи.
#pragma once
#include <ostream>
#include <string>
#include <vector>

#include "tlc/problem_io.hpp"
#include "tlc/tropical.hpp"

namespace tlc {

struct Problem {
    std::string name;
    Matrix C;                // m x m — сравнения критериев
    std::vector<Matrix> A;   // m матриц n x n — сравнения альтернатив
    std::size_t criteria() const { return C.rows; }
    std::size_t alternatives() const { return A.empty() ? 0 : A[0].rows; }
    static Problem from_raw(const io::RawProblem &raw);
    void validate() const;   // бросает std::invalid_argument
};

struct Options {
    bool naive_kernels = false;  // λ и A* «как в аналогах» (O(n^4)) — для бенчмарка
    double eps = kEps;
    std::ostream *trace = nullptr;  // пошаговый протокол вычислений
};

// Однокритериальная задача (1): min_x max_ij a_ij x_j / x_i.
struct SingleSolution {
    double lambda = 0.0;              // минимум ошибки = спектральный радиус
    Matrix star;                      // (λ^{-1}A)*
    std::vector<Vector> generators;   // независимые столбцы star (нормированы на max)
    bool unique() const { return generators.size() == 1; }
};

SingleSolution solve_single(const Matrix &A, const Options &opt = {});
// Формулы (4)-(5) / (9)-(10): наилучшие дифференцирующие векторы.
std::vector<Vector> best_differentiating(const std::vector<Vector> &P, const Options &opt = {});
// Формулы (6) / (12): наихудшие дифференцирующие векторы.
std::vector<Vector> worst_differentiating(const Matrix &A, double lambda, const Options &opt = {});

struct LogChebSolution {
    SingleSolution criteria;               // шаг 1
    std::vector<Vector> best_weights;      // шаг 2.1 (w1)
    std::vector<Vector> worst_weights;     // шаг 2.2 (w2)
    std::vector<double> mu, nu;            // спектральные радиусы B и D для каждого w
    std::vector<Vector> best_ratings;      // шаг 4 (x1')
    std::vector<Vector> worst_ratings;     // шаг 5 (x2')
};

LogChebSolution solve_log_chebyshev(const Problem &p, const Options &opt = {});

// Взвешенная сумма (7): ⊕_k w_k A_k
Matrix weighted_sum(const std::vector<Matrix> &A, const Vector &w);

// Классические методы для сравнения.
Vector principal_eigenvector(const Matrix &A, int max_iter = 10000, double tol = 1e-14);  // норм. по сумме
Vector solve_ahp(const Problem &p);  // ненормированная взвешенная сумма (как в статье)
Vector solve_gm(const Problem &p);   // взвешенные геометрические средние

// "A4 > A3 = A2 > A1" — по убыванию, равенство с относительной точностью eps.
std::string ranking(const Vector &x, double eps = 1e-6);

}  // namespace tlc
