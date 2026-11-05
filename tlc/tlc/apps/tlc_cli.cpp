// tlc — консольная утилита библиотеки.
//   tlc solve   <file> [--trace] [--naive]  решить задачу (log-Cheb + AHP + GM)
//   tlc kernels <file>                      λ и (λ^{-1}A)* для C и всех A_k (сверка с аналогом 2)
//   tlc bench   <file> <reps> [--naive]     CSV: impl,problem,reps,total_ms,us_per_call
//   tlc gen     <n> <m> <maxval> <seed>     сгенерировать случайную задачу (формат problem_io)
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "tlc/solver.hpp"

using namespace tlc;

static int usage() {
    std::cerr << "usage:\n  tlc solve <file> [--trace] [--naive]\n  tlc kernels <file>\n"
                 "  tlc bench <file> <reps> [--naive]\n  tlc gen <n_alternatives> <m_criteria> <maxval> <seed>\n";
    return 2;
}

static bool has_flag(int argc, char **argv, const char *f) {
    for (int i = 2; i < argc; ++i)
        if (std::strcmp(argv[i], f) == 0) return true;
    return false;
}

static void print_set(const char *title, const std::vector<Vector> &v) {
    std::cout << title << ":\n";
    for (const Vector &x : v) std::cout << "  " << to_string(x) << "   " << ranking(x) << "\n";
}

static int cmd_solve(const std::string &file, bool trace, bool naive) {
    const auto raw = io::load_problem(file);
    for (const auto &w : raw.warnings) std::cerr << "warning: " << w << "\n";
    const Problem p = Problem::from_raw(raw);
    Options opt;
    opt.naive_kernels = naive;
    opt.trace = trace ? &std::cout : nullptr;
    std::cout << "problem: " << p.name << "  (m=" << p.criteria() << " criteria, n=" << p.alternatives()
              << " alternatives)\n\n";

    const Vector ahp = solve_ahp(p), gm = solve_gm(p);
    std::cout << "AHP (Saaty):  x = " << to_string(ahp) << "\n  x/max = " << to_string(normalize_max(ahp))
              << "   " << ranking(ahp) << "\n";
    std::cout << "Weighted GM:  x = " << to_string(gm) << "\n  x/max = " << to_string(normalize_max(gm))
              << "   " << ranking(gm) << "\n\n";

    const auto r = solve_log_chebyshev(p, opt);
    std::cout << "Log-Chebyshev (tropical):\n  lambda = " << std::setprecision(10) << r.criteria.lambda << "\n";
    print_set("  best weights w1", r.best_weights);
    print_set("  worst weights w2", r.worst_weights);
    print_set("  best differentiating ratings", r.best_ratings);
    print_set("  worst differentiating ratings", r.worst_ratings);
    return 0;
}

static int cmd_kernels(const std::string &file) {
    const Problem p = Problem::from_raw(io::load_problem(file));
    auto dump = [](const std::string &name, const Matrix &M) {
        const SingleSolution s = solve_single(M);
        std::cout << name << ": lambda = " << std::setprecision(12) << s.lambda << "\n" << to_string(s.star, 6);
    };
    dump("C", p.C);
    for (std::size_t k = 0; k < p.A.size(); ++k) dump("A" + std::to_string(k + 1), p.A[k]);
    return 0;
}

static int cmd_bench(const std::string &file, long reps, bool naive) {
    const Problem p = Problem::from_raw(io::load_problem(file));
    Options opt;
    opt.naive_kernels = naive;
    volatile double sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < reps; ++i) sink = sink + solve_log_chebyshev(p, opt).criteria.lambda;
    auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << (naive ? "tlc_naive" : "tlc") << "," << p.name << "," << reps << "," << ms << ","
              << 1000.0 * ms / static_cast<double>(reps) << "\n";
    return 0;
}

static int cmd_gen(long n, long m, long maxval, unsigned long seed) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::uniform_int_distribution<long> d(1, maxval);
    auto mat = [&](long sz, const char *kw) {
        std::vector<std::string> e(static_cast<std::size_t>(sz * sz), "1");
        for (long i = 0; i < sz; ++i)
            for (long j = 0; j < i; ++j) {
                long a = d(rng), b = d(rng);
                e[static_cast<std::size_t>(i * sz + j)] = std::to_string(a) + "/" + std::to_string(b);
                e[static_cast<std::size_t>(j * sz + i)] = std::to_string(b) + "/" + std::to_string(a);
            }
        std::cout << kw << "\n";
        for (long i = 0; i < sz; ++i) {
            for (long j = 0; j < sz; ++j) std::cout << e[static_cast<std::size_t>(i * sz + j)] << ' ';
            std::cout << "\n";
        }
    };
    std::cout << "name random_n" << n << "_m" << m << "_v" << maxval << "_s" << seed << "\ncriteria " << m
              << "\nalternatives " << n << "\n";
    mat(m, "C");
    for (long k = 0; k < m; ++k) mat(n, "A");
    return 0;
}

int main(int argc, char **argv) try {
    if (argc < 3) return usage();
    const std::string cmd = argv[1];
    if (cmd == "solve") return cmd_solve(argv[2], has_flag(argc, argv, "--trace"), has_flag(argc, argv, "--naive"));
    if (cmd == "kernels") return cmd_kernels(argv[2]);
    if (cmd == "bench" && argc >= 4) return cmd_bench(argv[2], std::stol(argv[3]), has_flag(argc, argv, "--naive"));
    if (cmd == "gen" && argc >= 6)
        return cmd_gen(std::stol(argv[2]), std::stol(argv[3]), std::stol(argv[4]), std::stoul(argv[5]));
    return usage();
} catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
