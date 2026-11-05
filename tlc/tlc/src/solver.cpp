#include "tlc/solver.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace tlc {

namespace {

Matrix to_matrix(const io::RawMatrix &r) {
    Matrix M(r.n, r.n);
    for (std::size_t i = 0; i < r.n; ++i)
        for (std::size_t j = 0; j < r.n; ++j) M(i, j) = r(i, j);
    return M;
}

void log_vecs(std::ostream *t, const std::string &title, const std::vector<Vector> &v) {
    if (!t) return;
    *t << title << " (" << v.size() << "):\n";
    for (const Vector &x : v) *t << "  " << to_string(x) << "   " << ranking(x) << "\n";
}

std::vector<Vector> union_independent(const std::vector<std::vector<Vector>> &sets, double eps) {
    std::vector<Vector> all;
    for (const auto &s : sets) all.insert(all.end(), s.begin(), s.end());
    return independent_columns(all, eps);
}

}  // namespace

Problem Problem::from_raw(const io::RawProblem &raw) {
    Problem p;
    p.name = raw.name;
    p.C = to_matrix(raw.C);
    for (const auto &a : raw.A) p.A.push_back(to_matrix(a));
    p.validate();
    return p;
}

void Problem::validate() const {
    if (C.rows == 0 || C.rows != C.cols) throw std::invalid_argument("Problem: C must be non-empty square");
    if (A.size() != C.rows) throw std::invalid_argument("Problem: number of A matrices must equal size of C");
    const std::size_t n = A[0].rows;
    for (const Matrix &M : A)
        if (M.rows != n || M.cols != n || n == 0) throw std::invalid_argument("Problem: all A_k must be n x n");
    auto positive = [](const Matrix &M) {
        return std::all_of(M.a.begin(), M.a.end(), [](double v) { return std::isfinite(v) && v > 0.0; });
    };
    if (!positive(C)) throw std::invalid_argument("Problem: C must be positive and finite");
    for (const Matrix &M : A)
        if (!positive(M)) throw std::invalid_argument("Problem: A_k must be positive and finite");
}

SingleSolution solve_single(const Matrix &A, const Options &opt) {
    SingleSolution s;
    s.lambda = opt.naive_kernels ? spectral_radius_naive(A) : spectral_radius(A);
    if (!(s.lambda > 0.0) || !std::isfinite(s.lambda))
        throw std::runtime_error("solve_single: spectral radius is not a positive finite number");
    const Matrix N = scale(A, 1.0 / s.lambda);
    s.star = opt.naive_kernels ? kleene_star_naive(N) : kleene_star(N);
    s.generators = independent_columns(s.star.columns(), opt.eps);
    return s;
}

// x' = P(I ⊕ P_lk^- P) v. Столбец j этой матрицы равен p_j ⊕ (p_lj / p_lk) p_k,
// поэтому произведения матриц не нужны: O(n·g) на пару (l, k) вместо O(n·g^2).
std::vector<Vector> best_differentiating(const std::vector<Vector> &P, const Options &opt) {
    if (P.size() <= 1) return independent_columns(P, opt.eps);
    const std::size_t n = P[0].size(), g = P.size();
    std::vector<double> ratio(g);
    for (std::size_t j = 0; j < g; ++j) {  // 1^T p_j p_j^- 1 = max p_j / min p_j
        auto [mn, mx] = std::minmax_element(P[j].begin(), P[j].end());
        ratio[j] = *mx / *mn;
    }
    const double rmax = *std::max_element(ratio.begin(), ratio.end());
    std::vector<Vector> out;
    for (std::size_t k = 0; k < g; ++k) {
        if (!approx_equal(ratio[k], rmax, opt.eps)) continue;
        const double pmin = *std::min_element(P[k].begin(), P[k].end());  // l = argmax p_ik^{-1}
        for (std::size_t l = 0; l < n; ++l) {
            if (!approx_equal(P[k][l], pmin, opt.eps)) continue;
            if (opt.trace) *opt.trace << "    pair (l,k) = (" << l + 1 << "," << k + 1 << ")\n";
            for (std::size_t j = 0; j < g; ++j) {
                const double c = P[j][l] / P[k][l];
                Vector q = P[j];
                for (std::size_t i = 0; i < n; ++i) q[i] = std::max(q[i], c * P[k][i]);
                out.push_back(std::move(q));
            }
        }
    }
    return independent_columns(out, opt.eps);
}

std::vector<Vector> worst_differentiating(const Matrix &A, double lambda, const Options &opt) {
    const Matrix N = scale(A, 1.0 / lambda);
    const Matrix S = opt.naive_kernels ? kleene_star_naive(N) : kleene_star(N);
    const double delta = max_element(S);  // Δ = 1^T (λ^{-1}A)* 1
    Matrix M = N;                         // Δ^{-1} 1 1^T ⊕ λ^{-1}A
    for (double &v : M.a) v = std::max(v, 1.0 / delta);
    const Matrix MS = opt.naive_kernels ? kleene_star_naive(M) : kleene_star(M);
    if (opt.trace) {
        *opt.trace << "    Delta = " << delta << "\n    (Delta^-1 11^T + lambda^-1 A)* =\n" << to_string(MS);
    }
    return independent_columns(MS.columns(), opt.eps);
}

Matrix weighted_sum(const std::vector<Matrix> &A, const Vector &w) {
    if (A.size() != w.size()) throw std::invalid_argument("weighted_sum: size mismatch");
    Matrix B(A[0].rows, A[0].cols, 0.0);
    for (std::size_t k = 0; k < A.size(); ++k)
        for (std::size_t t = 0; t < B.a.size(); ++t) B.a[t] = std::max(B.a[t], w[k] * A[k].a[t]);
    return B;
}

LogChebSolution solve_log_chebyshev(const Problem &p, const Options &opt) {
    p.validate();
    std::ostream *t = opt.trace;
    LogChebSolution r;

    // Шаг 1. Веса критериев w = (λ^{-1}C)* v.
    if (t) *t << "== Step 1: criteria weights ==\nC =\n" << to_string(p.C);
    r.criteria = solve_single(p.C, opt);
    if (t) {
        *t << "lambda = " << r.criteria.lambda << "\n(lambda^-1 C)* =\n" << to_string(r.criteria.star);
        log_vecs(t, "independent columns P", r.criteria.generators);
    }

    // Шаг 2. Наилучшие и наихудшие дифференцирующие веса.
    if (r.criteria.unique()) {
        r.best_weights = r.worst_weights = r.criteria.generators;
        if (t) *t << "== Step 2: weight vector is unique ==\n";
    } else {
        if (t) *t << "== Step 2.1: best differentiating weights, (4)-(5) ==\n";
        r.best_weights = best_differentiating(r.criteria.generators, opt);
        log_vecs(t, "w1", r.best_weights);
        if (t) *t << "== Step 2.2: worst differentiating weights, (6) ==\n";
        r.worst_weights = worst_differentiating(p.C, r.criteria.lambda, opt);
        log_vecs(t, "w2", r.worst_weights);
    }

    // Шаги 3-4. B = ⊕ w1_i A_i, рейтинги x1.
    std::vector<std::vector<Vector>> best_sets, worst_sets;
    for (const Vector &w : r.best_weights) {
        const Matrix B = weighted_sum(p.A, w);
        const SingleSolution s = solve_single(B, opt);
        r.mu.push_back(s.lambda);
        if (t) {
            *t << "== Step 4: B for w1 = " << to_string(w) << " ==\nB =\n" << to_string(B)
               << "mu = " << s.lambda << "\n(mu^-1 B)* =\n" << to_string(s.star);
            log_vecs(t, "independent columns Q", s.generators);
        }
        best_sets.push_back(s.unique() ? s.generators : best_differentiating(s.generators, opt));
        log_vecs(t, "best differentiating x1'", best_sets.back());
    }
    r.best_ratings = union_independent(best_sets, opt.eps);

    // Шаги 3, 5. D = ⊕ w2_i A_i, рейтинги x2.
    for (const Vector &w : r.worst_weights) {
        const Matrix D = weighted_sum(p.A, w);
        const SingleSolution s = solve_single(D, opt);
        r.nu.push_back(s.lambda);
        if (t) {
            *t << "== Step 5: D for w2 = " << to_string(w) << " ==\nD =\n" << to_string(D)
               << "nu = " << s.lambda << "\n(nu^-1 D)* =\n" << to_string(s.star);
            log_vecs(t, "independent columns", s.generators);
        }
        worst_sets.push_back(s.unique() ? s.generators : worst_differentiating(D, s.lambda, opt));
        log_vecs(t, "worst differentiating x2'", worst_sets.back());
    }
    r.worst_ratings = union_independent(worst_sets, opt.eps);
    if (t) {
        *t << "== Result ==\n";
        log_vecs(t, "best ratings", r.best_ratings);
        log_vecs(t, "worst ratings", r.worst_ratings);
    }
    return r;
}

Vector principal_eigenvector(const Matrix &A, int max_iter, double tol) {
    const std::size_t n = A.rows;
    Vector x(n, 1.0 / static_cast<double>(n)), y(n);
    for (int it = 0; it < max_iter; ++it) {
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < n; ++j) s += A(i, j) * x[j];
            y[i] = s;
        }
        const double sum = std::accumulate(y.begin(), y.end(), 0.0);
        double diff = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            y[i] /= sum;
            diff = std::max(diff, std::fabs(y[i] - x[i]));
        }
        x.swap(y);
        if (diff < tol) break;
    }
    return x;
}

Vector solve_ahp(const Problem &p) {
    p.validate();
    const Vector w = principal_eigenvector(p.C);
    Vector x(p.alternatives(), 0.0);
    for (std::size_t k = 0; k < p.A.size(); ++k) {
        const Vector xk = principal_eigenvector(p.A[k]);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += w[k] * xk[i];
    }
    return x;
}

static Vector row_geomeans(const Matrix &A) {
    Vector g(A.rows);
    for (std::size_t i = 0; i < A.rows; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < A.cols; ++j) s += std::log(A(i, j));
        g[i] = std::exp(s / static_cast<double>(A.cols));
    }
    return g;
}

Vector solve_gm(const Problem &p) {
    p.validate();
    Vector w = row_geomeans(p.C);
    const double sw = std::accumulate(w.begin(), w.end(), 0.0);
    for (double &v : w) v /= sw;
    Vector x(p.alternatives(), 0.0);  // в лог-шкале
    for (std::size_t k = 0; k < p.A.size(); ++k) {
        const Vector xk = row_geomeans(p.A[k]);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += w[k] * std::log(xk[i]);
    }
    for (double &v : x) v = std::exp(v);
    return x;
}

std::string ranking(const Vector &x, double eps) {
    std::vector<std::size_t> idx(x.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return x[a] > x[b]; });
    std::ostringstream os;
    for (std::size_t t = 0; t < idx.size(); ++t) {
        if (t) os << (approx_equal(x[idx[t - 1]], x[idx[t]], eps) ? " = " : " > ");
        os << 'A' << idx[t] + 1;
    }
    return os.str();
}

}  // namespace tlc
