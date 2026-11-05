// problem_io.hpp — общий формат входных данных для всех реализаций стенда
// (библиотека tlc, аналог 1 DecisionModel, аналог 2 MaxAlgebra cli).
// Header-only, без зависимостей, чтобы его могли подключить и аналоги.
//
// Формат файла (комментарии начинаются с '#'):
//   name task1
//   criteria 4          # m — число критериев
//   alternatives 4      # n — число альтернатив
//   C                   # m*m чисел, матрица парных сравнений критериев
//   1 4 2 2 ...
//   A                   # n*n чисел, повторяется m раз (A1..Am)
//   ...
// Числа: целые, десятичные (2.5) или дроби (1/4). Дробь сохраняется точно
// (num/den) — это нужно аналогу 2, где элементы строятся как StructA(num, den).
#pragma once
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tlc::io {

struct Rational {
    long long num = 1, den = 1;
    double value() const { return static_cast<double>(num) / static_cast<double>(den); }
};

struct RawMatrix {
    std::size_t n = 0;
    std::vector<Rational> e;  // row-major, n*n
    const Rational &at(std::size_t i, std::size_t j) const { return e[i * n + j]; }
    double operator()(std::size_t i, std::size_t j) const { return at(i, j).value(); }
};

struct RawProblem {
    std::string name = "unnamed";
    std::size_t m = 0, n = 0;   // критерии, альтернативы
    RawMatrix C;                // m x m
    std::vector<RawMatrix> A;   // m штук n x n
    std::vector<std::string> warnings;
};

inline Rational parse_number(const std::string &tok) {
    auto bad = [&] { throw std::runtime_error("problem_io: bad number '" + tok + "'"); };
    Rational r;
    auto slash = tok.find('/');
    if (slash != std::string::npos) {
        try {
            std::size_t p1 = 0, p2 = 0;
            r.num = std::stoll(tok.substr(0, slash), &p1);
            r.den = std::stoll(tok.substr(slash + 1), &p2);
            if (p1 != slash || p2 != tok.size() - slash - 1) bad();
        } catch (const std::logic_error &) { bad(); }
    } else {
        // десятичная запись -> точная дробь num / 10^k
        auto dot = tok.find('.');
        std::string digits = tok;
        long long den = 1;
        if (dot != std::string::npos) {
            std::size_t frac = tok.size() - dot - 1;
            if (frac > 12) bad();
            digits = tok.substr(0, dot) + tok.substr(dot + 1);
            for (std::size_t i = 0; i < frac; ++i) den *= 10;
        }
        try {
            std::size_t p = 0;
            r.num = std::stoll(digits, &p);
            if (p != digits.size()) bad();
        } catch (const std::logic_error &) { bad(); }
        r.den = den;
    }
    if (r.num <= 0 || r.den <= 0) throw std::runtime_error("problem_io: elements must be positive: '" + tok + "'");
    return r;
}

namespace detail {
inline std::vector<std::string> tokenize(std::istream &in) {
    std::vector<std::string> t;
    std::string line;
    while (std::getline(in, line)) {
        auto h = line.find('#');
        if (h != std::string::npos) line.resize(h);
        std::istringstream ls(line);
        std::string w;
        while (ls >> w) t.push_back(w);
    }
    return t;
}
inline RawMatrix read_matrix(const std::vector<std::string> &t, std::size_t &pos, std::size_t n, const std::string &what) {
    RawMatrix M;
    M.n = n;
    for (std::size_t k = 0; k < n * n; ++k) {
        if (pos >= t.size()) throw std::runtime_error("problem_io: matrix " + what + " is truncated");
        M.e.push_back(parse_number(t[pos++]));
    }
    return M;
}
inline void check_matrix(const RawMatrix &M, const std::string &what, std::vector<std::string> &warn) {
    for (std::size_t i = 0; i < M.n; ++i) {
        if (std::fabs(M(i, i) - 1.0) > 1e-12) warn.push_back(what + ": diagonal element " + std::to_string(i + 1) + " != 1");
        for (std::size_t j = i + 1; j < M.n; ++j)
            if (std::fabs(M(i, j) * M(j, i) - 1.0) > 1e-9)
                warn.push_back(what + ": not reciprocal at (" + std::to_string(i + 1) + "," + std::to_string(j + 1) + ")");
    }
}
}  // namespace detail

inline RawProblem parse_problem(std::istream &in) {
    auto t = detail::tokenize(in);
    RawProblem p;
    std::size_t pos = 0;
    auto need_size = [&](const char *kw) -> std::size_t {
        if (pos >= t.size()) throw std::runtime_error(std::string("problem_io: '") + kw + "' needs a number");
        long long v = std::stoll(t[pos++]);
        if (v <= 0 || v > 100000) throw std::runtime_error(std::string("problem_io: bad size for '") + kw + "'");
        return static_cast<std::size_t>(v);
    };
    bool hasC = false;
    while (pos < t.size()) {
        const std::string kw = t[pos++];
        if (kw == "name") {
            if (pos >= t.size()) throw std::runtime_error("problem_io: 'name' needs a value");
            p.name = t[pos++];
        } else if (kw == "criteria") {
            p.m = need_size("criteria");
        } else if (kw == "alternatives") {
            p.n = need_size("alternatives");
        } else if (kw == "C") {
            if (!p.m) throw std::runtime_error("problem_io: 'criteria' must precede C");
            p.C = detail::read_matrix(t, pos, p.m, "C");
            hasC = true;
        } else if (kw == "A") {
            if (!p.n) throw std::runtime_error("problem_io: 'alternatives' must precede A");
            p.A.push_back(detail::read_matrix(t, pos, p.n, "A" + std::to_string(p.A.size() + 1)));
        } else {
            throw std::runtime_error("problem_io: unknown keyword '" + kw + "'");
        }
    }
    if (!hasC) throw std::runtime_error("problem_io: matrix C is missing");
    if (p.A.size() != p.m)
        throw std::runtime_error("problem_io: expected " + std::to_string(p.m) + " matrices A, got " + std::to_string(p.A.size()));
    detail::check_matrix(p.C, "C", p.warnings);
    for (std::size_t k = 0; k < p.A.size(); ++k) detail::check_matrix(p.A[k], "A" + std::to_string(k + 1), p.warnings);
    return p;
}

inline RawProblem load_problem(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("problem_io: cannot open " + path);
    return parse_problem(f);
}

}  // namespace tlc::io
