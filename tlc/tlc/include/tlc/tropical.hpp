// tropical.hpp — примитивы max-алгебры (полуполе R+ с ⊕ = max, ⊗ = ×).
//
// Ключевые отличия от аналогов:
//  * spectral_radius — алгоритм Карпа в лог-шкале, O(n^3) вместо O(n^4)
//    (n тропических умножений матриц) и без переполнения a^n при больших n;
//  * kleene_star — алгоритм Флойда–Уоршелла в (max,×), O(n^3) вместо
//    повторного возведения в степени (ещё n умножений);
//  * линейная независимость столбцов проверяется точно по тропическому
//    критерию b ∈ span(P) <=> P (b^- P)^- = b, а не только по коллинеарности.
// Для сравнения оставлены «наивные» версии (как в аналогах): *_naive.
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace tlc {

using Vector = std::vector<double>;

struct Matrix {
    std::size_t rows = 0, cols = 0;
    std::vector<double> a;  // row-major
    Matrix() = default;
    Matrix(std::size_t r, std::size_t c, double v = 0.0) : rows(r), cols(c), a(r * c, v) {}
    double &operator()(std::size_t i, std::size_t j) { return a[i * cols + j]; }
    double operator()(std::size_t i, std::size_t j) const { return a[i * cols + j]; }
    static Matrix identity(std::size_t n);
    static Matrix from_columns(const std::vector<Vector> &cols);
    Vector column(std::size_t j) const;
    std::vector<Vector> columns() const;
};

constexpr double kEps = 1e-9;  // относительная точность сравнений

bool approx_equal(double x, double y, double eps = kEps);
bool approx_equal(const Vector &x, const Vector &y, double eps = kEps);

// --- базовые операции max-алгебры ---
Matrix mul(const Matrix &A, const Matrix &B);          // (AB)_ij = max_k a_ik b_kj
Vector mul(const Matrix &A, const Vector &x);
Matrix oplus(const Matrix &A, const Matrix &B);        // поэлементный max
Matrix scale(const Matrix &A, double c);
Matrix conj(const Matrix &A);                          // A^-: a^-_ij = 1/a_ji (0 -> 0)
double trace(const Matrix &A);                         // ⊕ диагонали
double max_element(const Matrix &A);                   // 1^T A 1

// --- спектральный радиус λ = ⊕_{k=1..n} tr^{1/k}(A^k) ---
double spectral_radius(const Matrix &A);               // Карп, O(n^3)
double spectral_radius_naive(const Matrix &A);         // через степени, O(n^4)

// --- звезда Клини A* = I ⊕ A ⊕ ... ⊕ A^{n-1}, требует λ(A) <= 1 ---
Matrix kleene_star(const Matrix &A);                   // Флойд–Уоршелл, O(n^3)
Matrix kleene_star_naive(const Matrix &A);             // через степени, O(n^4)

// --- линейная (не)зависимость ---
Vector normalize_max(Vector x);                        // x / max_i x_i
bool in_span(const std::vector<Vector> &gens, const Vector &b, double eps = kEps);
// Оставляет порождающие (линейно независимые) столбцы, каждый нормирован на max=1.
std::vector<Vector> independent_columns(const std::vector<Vector> &cols, double eps = kEps);

// Ошибка аппроксимации в (1): max_ij a_ij x_j / x_i. Для оптимального x равна λ(A).
double approximation_error(const Matrix &A, const Vector &x);

std::string to_string(const Matrix &A, int prec = 4);
std::string to_string(const Vector &x, int prec = 4);

}  // namespace tlc
