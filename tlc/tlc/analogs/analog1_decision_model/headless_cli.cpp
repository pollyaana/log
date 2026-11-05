// headless_cli.cpp — консольный вход в аналог 1 (DecisionModel) без GUI.
// Позволяет подать любую задачу в общем формате стенда (tlc/problem_io.hpp)
// и снять время работы методов. Заменяет main.cpp (StartMenu) при сборке
// цели analog1_cli; нужен только QtCore (для QString/QVector в конструкторе).
//
//   analog1_cli solve <file>          AHP, GM и тропический метод аналога
//   analog1_cli bench <file> <reps>   CSV: impl,problem,reps,total_ms,us_per_call
#include <QString>
#include <QVector>

#include <chrono>
#include <iostream>
#include <string>

#include <decision_model.h>
#include "tlc/problem_io.hpp"

namespace {

Eigen::MatrixXd toEigen(const tlc::io::RawMatrix &r) {
	Eigen::MatrixXd M(static_cast<Eigen::Index>(r.n), static_cast<Eigen::Index>(r.n));
	for (std::size_t i = 0; i < r.n; ++i)
		for (std::size_t j = 0; j < r.n; ++j)
			M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = r(i, j);
	return M;
}

// «Представление» матрицы — строки, как их показывал бы GUI.
DecisionModel::ComparisonMatrixView toView(const tlc::io::RawMatrix &r) {
	DecisionModel::ComparisonMatrixView V(static_cast<Eigen::Index>(r.n), static_cast<Eigen::Index>(r.n));
	for (std::size_t i = 0; i < r.n; ++i)
		for (std::size_t j = 0; j < r.n; ++j) {
			const auto &q = r.at(i, j);
			V(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
					q.den == 1 ? std::to_string(q.num) : std::to_string(q.num) + "/" + std::to_string(q.den);
		}
	return V;
}

DecisionModel buildModel(const tlc::io::RawProblem &p) {
	QVector<QString> alts, crits;
	for (std::size_t i = 0; i < p.n; ++i) alts.push_back(QString("A%1").arg(i + 1));
	for (std::size_t k = 0; k < p.m; ++k) crits.push_back(QString("K%1").arg(k + 1));
	DecisionModel model(QString::fromStdString(p.name), alts, crits);
	model.setCriteriaComparisons(toEigen(p.C), toView(p.C));
	for (std::size_t k = 0; k < p.m; ++k) model.setAlternativesCompsAt(toEigen(p.A[k]), toView(p.A[k]), static_cast<int>(k));
	return model;
}

}  // namespace

int main(int argc, char **argv) try {
	if (argc < 3) {
		std::cerr << "usage: analog1_cli solve <file> | bench <file> <reps>\n";
		return 2;
	}
	const std::string cmd = argv[1];
	const auto raw = tlc::io::load_problem(argv[2]);
	DecisionModel model = buildModel(raw);

	if (cmd == "solve") {
		model.performAhpMethod();
		model.performGmMethod();
		model.performTropicalMethod();
		std::cout << "problem: " << raw.name << "\n\nAHP:\n" << model.ahpResult() << "\n\nGM:\n" << model.gmResult() << "\n\n";
		auto [best, worst] = model.tropicalResult();
		std::cout << "Tropical best:\n" << best << "\n\nTropical worst:\n" << worst << "\n";
		return 0;
	}
	if (cmd == "bench" && argc >= 4) {
		const long reps = std::stol(argv[3]);
		const auto t0 = std::chrono::steady_clock::now();
		for (long i = 0; i < reps; ++i) model.performTropicalMethod();
		const auto t1 = std::chrono::steady_clock::now();
		const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		std::cout << "analog1," << raw.name << "," << reps << "," << ms << "," << 1000.0 * ms / reps << "\n";
		return 0;
	}
	std::cerr << "unknown command\n";
	return 2;
} catch (const std::exception &e) {
	std::cerr << "error: " << e.what() << "\n";
	return 1;
}
