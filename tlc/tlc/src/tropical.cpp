#include "tlc/tropical.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tlc {

Matrix Matrix::identity(std::size_t n) {
    Matrix I(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) I(i, i) = 1.0;
    return I;
}

Matrix Matrix::from_columns(const std::vector<Vector> &cols) {
    if (cols.empty()) return {};
    Matrix M(cols[0].size(), cols.size());
    for (std::size_t j = 0; j < cols.size(); ++j)
        for (std::size_t i = 0; i < M.rows; ++i) M(i, j) = cols[j][i];
    return M;
}

Vector Matrix::column(std::size_t j) const {
    Vector v(rows);
    for (std::size_t i = 0; i < rows; ++i) v[i] = (*this)(i, j);
    return v;
}

std::vector<Vector> Matrix::columns() const {
    std::vector<Vector> c;
    c.reserve(cols);
    for (std::size_t j = 0; j < cols; ++j) c.push_back(column(j));
    return c;
}

bool approx_equal(double x, double y, double eps) {
    return std::fabs(x - y) <= eps * std::max({1.0, std::fabs(x), std::fabs(y)});
}

bool approx_equal(const Vector &x, const Vector &y, double eps) {
    if (x.size() != y.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!approx_equal(x[i], y[i], eps)) return false;
    return true;
}

static void require_square(const Matrix &A, const char *who) {
    if (A.rows != A.cols) throw std::invalid_argument(std::string(who) + ": matrix must be square");
}

Matrix mul(const Matrix &A, const Matrix &B) {
    if (A.cols != B.rows) throw std::invalid_argument("mul: size mismatch");
    Matrix R(A.rows, B.cols, 0.0);
    // порядок i-k-j: строка B читается последовательно (кэш-дружественно)
    for (std::size_t i = 0; i < A.rows; ++i)
        for (std::size_t k = 0; k < A.cols; ++k) {
            const double aik = A(i, k);
            if (aik == 0.0) continue;
            double *r = &R.a[i * R.cols];
            const double *b = &B.a[k * B.cols];
            for (std::size_t j = 0; j < B.cols; ++j) r[j] = std::max(r[j], aik * b[j]);
        }
    return R;
}

Vector mul(const Matrix &A, const Vector &x) {
    if (A.cols != x.size()) throw std::invalid_argument("mul: size mismatch");
    Vector y(A.rows, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i)
        for (std::size_t j = 0; j < A.cols; ++j) y[i] = std::max(y[i], A(i, j) * x[j]);
    return y;
}

Matrix oplus(const Matrix &A, const Matrix &B) {
    if (A.rows != B.rows || A.cols != B.cols) throw std::invalid_argument("oplus: size mismatch");
    Matrix R = A;
    for (std::size_t k = 0; k < R.a.size(); ++k) R.a[k] = std::max(R.a[k], B.a[k]);
    return R;
}

Matrix scale(const Matrix &A, double c) {
    Matrix R = A;
    for (double &v : R.a) v *= c;
    return R;
}

Matrix conj(const Matrix &A) {
    Matrix R(A.cols, A.rows, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i)
        for (std::size_t j = 0; j < A.cols; ++j)
            if (A(i, j) != 0.0) R(j, i) = 1.0 / A(i, j);
    return R;
}

double trace(const Matrix &A) {
    require_square(A, "trace");
    double t = 0.0;
    for (std::size_t i = 0; i < A.rows; ++i) t = std::max(t, A(i, i));
    return t;
}

double max_element(const Matrix &A) {
    double m = 0.0;
    for (double v : A.a) m = std::max(m, v);
    return m;
}

// λ(A) — максимальное среднее геометрическое веса цикла графа матрицы.
// Логарифмируем веса и находим максимальный средний вес цикла алгоритмом Карпа:
//   D_k(v) = max вес пути из k дуг, оканчивающегося в v (начало — любая вершина),
//   log λ = max_v min_{0<=k<n} (D_n(v) - D_k(v)) / (n - k).
double spectral_radius(const Matrix &A) {
    require_square(A, "spectral_radius");
    const std::size_t n = A.rows;
    if (n == 0) return 0.0;
    constexpr double NEG = -std::numeric_limits<double>::infinity();
    std::vector<double> L(n * n);
    for (std::size_t k = 0; k < n * n; ++k) L[k] = A.a[k] > 0.0 ? std::log(A.a[k]) : NEG;

    std::vector<double> D((n + 1) * n, NEG);
    std::fill(D.begin(), D.begin() + static_cast<std::ptrdiff_t>(n), 0.0);
    for (std::size_t k = 1; k <= n; ++k) {
        const double *prev = &D[(k - 1) * n];
        double *cur = &D[k * n];
        for (std::size_t u = 0; u < n; ++u) {
            if (prev[u] == NEG) continue;
            const double *row = &L[u * n];
            for (std::size_t v = 0; v < n; ++v)
                if (row[v] != NEG) cur[v] = std::max(cur[v], prev[u] + row[v]);
        }
    }
    double best = NEG;
    const double *Dn = &D[n * n];
    for (std::size_t v = 0; v < n; ++v) {
        if (Dn[v] == NEG) continue;
        double worst = std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < n; ++k) {
            const double dk = D[k * n + v];
            if (dk == NEG) continue;
            worst = std::min(worst, (Dn[v] - dk) / static_cast<double>(n - k));
        }
        best = std::max(best, worst);
    }
    return best == NEG ? 0.0 : std::exp(best);
}

// Прямая формула из статьи: λ = trA ⊕ tr^{1/2}(A^2) ⊕ ... ⊕ tr^{1/n}(A^n).
// Совпадает по устройству с SpectralRadius() аналога 2: n-1 умножений матриц.
double spectral_radius_naive(const Matrix &A) {
    require_square(A, "spectral_radius_naive");
    if (A.rows == 0) return 0.0;
    double r = trace(A);
    Matrix P = A;
    for (std::size_t k = 2; k <= A.rows; ++k) {
        P = mul(P, A);
        r = std::max(r, std::pow(trace(P), 1.0 / static_cast<double>(k)));
    }
    return r;
}

// Floyd–Warshall в (max, ×): S_ij = max(S_ij, S_ik S_kj). При λ(A) <= 1 циклы
// не увеличивают вес, поэтому результат равен I ⊕ A ⊕ ... ⊕ A^{n-1}.
Matrix kleene_star(const Matrix &A) {
    require_square(A, "kleene_star");
    const std::size_t n = A.rows;
    Matrix S = A;
    for (std::size_t i = 0; i < n; ++i) S(i, i) = std::max(S(i, i), 1.0);
    for (std::size_t k = 0; k < n; ++k) {
        const double *sk = &S.a[k * n];
        for (std::size_t i = 0; i < n; ++i) {
            if (i == k) continue;
            const double sik = S(i, k);
            if (sik == 0.0) continue;
            double *si = &S.a[i * n];
            for (std::size_t j = 0; j < n; ++j) si[j] = std::max(si[j], sik * sk[j]);
        }
    }
    // численная стабилизация: при λ ≈ 1 диагональ может стать 1 + O(ε)
    for (std::size_t i = 0; i < n; ++i)
        if (approx_equal(S(i, i), 1.0, 1e-12)) S(i, i) = 1.0;
    return S;
}

// Как Kleene() аналога 2: I ⊕ A ⊕ ... ⊕ A^{n-1} последовательными умножениями.
Matrix kleene_star_naive(const Matrix &A) {
    require_square(A, "kleene_star_naive");
    const std::size_t n = A.rows;
    Matrix R = Matrix::identity(n);
    Matrix P = A;
    for (std::size_t k = 1; k < n; ++k) {
        R = oplus(R, P);
        P = mul(P, A);  // последнее умножение в аналоге 2 лишнее — сохраняем ради честности замеров
    }
    if (n == 1) R = oplus(R, A);
    return R;
}

Vector normalize_max(Vector x) {
    double m = 0.0;
    for (double v : x) m = std::max(m, v);
    if (m > 0.0)
        for (double &v : x) v /= m;
    return x;
}

// b ∈ span(gens) <=> G x* = b, где x* = (b^- G)^- — наибольшее решение Gx <= b.
bool in_span(const std::vector<Vector> &gens, const Vector &b, double eps) {
    if (gens.empty()) return false;
    Vector y(b.size(), 0.0);
    for (const Vector &g : gens) {
        double x = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < b.size(); ++i)
            if (g[i] > 0.0) x = std::min(x, b[i] / g[i]);
        if (!std::isfinite(x)) continue;
        for (std::size_t i = 0; i < b.size(); ++i) y[i] = std::max(y[i], g[i] * x);
    }
    for (std::size_t i = 0; i < b.size(); ++i)
        if (!approx_equal(y[i], b[i], eps)) return false;
    return true;
}

namespace {
// b ∈ span{G_t : alive[t], t != skip}? Столбцы лежат в плоском буфере G (g x n,
// по столбцу подряд), y — рабочий буфер длины n. Без выделений памяти.
bool in_span_flat(const double *G, std::size_t g, std::size_t n, const std::vector<char> &alive,
                  std::size_t skip, const double *b, double *y, double eps) {
    std::fill(y, y + n, 0.0);
    bool any = false;
    for (std::size_t t = 0; t < g; ++t) {
        if (t == skip || !alive[t]) continue;
        const double *c = G + t * n;
        double x = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n; ++i)
            if (c[i] > 0.0) x = std::min(x, b[i] / c[i]);
        if (!std::isfinite(x)) continue;
        any = true;
        for (std::size_t i = 0; i < n; ++i) y[i] = std::max(y[i], c[i] * x);
    }
    if (!any) return false;
    for (std::size_t i = 0; i < n; ++i)
        if (!approx_equal(y[i], b[i], eps)) return false;
    return true;
}
}  // namespace

// Версия 1 строила для каждого j новый std::vector<Vector> «остальных» столбцов
// (глубокое копирование g*n чисел и g выделений на каждую проверку): по профилю
// callgrind это давало ~80% времени всего конвейера при n = 80. Здесь — плоский
// буфер и маска «живых» столбцов.
std::vector<Vector> independent_columns(const std::vector<Vector> &cols, double eps) {
    if (cols.empty()) return {};
    const std::size_t n = cols[0].size();
    std::vector<double> G;
    G.reserve(cols.size() * n);
    std::size_t g = 0;
    for (const Vector &c : cols) {  // нормировка + удаление коллинеарных (оставляем первый)
        double m = 0.0;
        for (double v : c) m = std::max(m, v);
        if (!(m > 0.0)) continue;
        const std::size_t off = G.size();
        for (double v : c) G.push_back(v / m);
        bool dup = false;
        for (std::size_t t = 0; t < g && !dup; ++t) {
            dup = true;
            for (std::size_t i = 0; i < n; ++i)
                if (!approx_equal(G[t * n + i], G[off + i], eps)) { dup = false; break; }
        }
        if (dup) G.resize(off);
        else ++g;
    }
    std::vector<char> alive(g, 1);
    std::vector<double> y(n);
    for (std::size_t j = g; j-- > 0;)  // с конца, чтобы сохранить ранние столбцы
        if (in_span_flat(G.data(), g, n, alive, j, G.data() + j * n, y.data(), eps)) alive[j] = 0;
    std::vector<Vector> out;
    for (std::size_t t = 0; t < g; ++t)
        if (alive[t]) out.emplace_back(G.begin() + static_cast<std::ptrdiff_t>(t * n),
                                       G.begin() + static_cast<std::ptrdiff_t>((t + 1) * n));
    return out;
}

double approximation_error(const Matrix &A, const Vector &x) {
    double e = 0.0;
    for (std::size_t i = 0; i < A.rows; ++i)
        for (std::size_t j = 0; j < A.cols; ++j) e = std::max(e, A(i, j) * x[j] / x[i]);
    return e;
}

std::string to_string(const Matrix &A, int prec) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec);
    for (std::size_t i = 0; i < A.rows; ++i) {
        os << "  [";
        for (std::size_t j = 0; j < A.cols; ++j) os << (j ? " " : "") << std::setw(prec + 4) << A(i, j);
        os << " ]\n";
    }
    return os.str();
}

std::string to_string(const Vector &x, int prec) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << "(";
    for (std::size_t i = 0; i < x.size(); ++i) os << (i ? ", " : "") << x[i];
    os << ")^T";
    return os.str();
}

}  // namespace tlc
