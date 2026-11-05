#include <decision_model.h>
#include <utils.h>
#include <algorithm>
#include <string>
#include <sstream>    // FIX(A1-3): std::stringstream использовался без include
#include <stdexcept>
#include <matrices_utils.h>

using namespace matrices_utils;

DecisionModel::DecisionModel(const QString &modelName, const QVector<QString> &alternativesNames,
							 const QVector<QString> &criteriaNames) {
	modelName_ = modelName.toStdString();
	for (const auto &alternativeName: alternativesNames) {
		addAlternativeName(alternativeName.toStdString());
	}
	for (const auto &criterionName: criteriaNames) {
		addCriteriaName(criterionName.toStdString());
		addAlternativesComparisonsMatrix();
	}
	initCriteriaComparisonsMatrix();
}

void DecisionModel::addAlternativeName(const std::string &alternative) {
	alternativesNames_.emplace_back(alternative);
}

void DecisionModel::addCriteriaName(const std::string &criteria) {
	criteriaNames_.emplace_back(criteria);
}

void DecisionModel::setDecisionName(const std::string &decisionName) {
	modelName_ = decisionName;
}

const std::string &DecisionModel::modelName() const {
	return modelName_;
}

const std::vector<std::string> &DecisionModel::criteriaNames() const {
	return criteriaNames_;
}

const std::vector<std::string> &DecisionModel::alternativesNames() const {
	return alternativesNames_;
}

void DecisionModel::setCriteriaNames(const std::vector<std::string> &criteriaNames) {
	criteriaNames_ = criteriaNames;
}

void DecisionModel::setAlternativesNames(const std::vector<std::string> &alternativesNames) {
	alternativesNames_ = alternativesNames;
}

const Eigen::MatrixXd &DecisionModel::criteriaComparisons() const {
	return criteriaComparisons_;
}

void DecisionModel::setCriteriaComparisons(const Eigen::MatrixXd &criteriaComparisons,
										   const ComparisionMatrixView &criteriaMatrixView) {
	if (criteriaComparisons.rows() != criteriaCount() || criteriaComparisons.cols() != criteriaCount())  // FIX(A1-5)
		throw std::invalid_argument("setCriteriaComparisons: matrix must be m x m (m = criteriaCount())");
	criteriaComparisons_ = criteriaComparisons;
	criteriaComparisonsMatrixView_ = criteriaMatrixView;
	criteriaComparisonMatrixIsInit_ = true;
}

const std::vector<Eigen::MatrixXd> &DecisionModel::alternativesComparisons() const {
	return alternativesComps_;
}

void DecisionModel::setAlternativesComparisons(const std::vector<Eigen::MatrixXd> &alternativesComparisons) {
	alternativesComps_ = alternativesComparisons;
}

int DecisionModel::criteriaCount() const {
	return criteriaNames_.size();
}

int DecisionModel::alternativesCount() const {
	return alternativesNames_.size();
}

bool DecisionModel::criteriaComparisonMatrixIsInit() const {
	return criteriaComparisonMatrixIsInit_;
}

const DecisionModel::ComparisonMatrixView &DecisionModel::criteriaComparisonsMatrixView() const {
	return criteriaComparisonsMatrixView_;
}

void DecisionModel::setAlternativesCompsAt(const Eigen::MatrixXd &alternativesComps,
										   const ComparisionMatrixView &alternativesCompsView, int index) {
	// FIX(A1-5): размеры не проверялись — несогласованный ввод падал глубоко внутри метода
	if (index < 0 || index >= criteriaCount())
		throw std::out_of_range("setAlternativesCompsAt: bad criterion index");
	if (alternativesComps.rows() != alternativesCount() || alternativesComps.cols() != alternativesCount())
		throw std::invalid_argument("setAlternativesCompsAt: matrix must be n x n (n = alternativesCount())");
	if (alternativesComps_.size() != static_cast<size_t>(criteriaCount())) {
		alternativesComps_.resize(criteriaCount());
		alternativesCompsViews_.resize(criteriaCount());
		alternativesCompsIsInit_.resize(criteriaCount());
	}
	alternativesComps_.at(index) = alternativesComps;
	alternativesCompsViews_.at(index) = alternativesCompsView;
	alternativesCompsIsInit_.at(index) = true;
}

const ComparisionMatrixView &DecisionModel::compsViewAt(int index) const {
	return alternativesCompsViews_.at(index);
}

const Eigen::MatrixXd &DecisionModel::compsAt(int index) const {
	return alternativesComps_.at(index);
}

bool DecisionModel::compsIsInitAt(int index) const {
	if (index >= 0 && static_cast<size_t>(index) < alternativesCompsIsInit_.size())  // FIX(A1-4): index < 0, int/size_t
		return alternativesCompsIsInit_.at(index);
	return false;
}

void DecisionModel::performTropicalMethod() {
	tropicalDecisionMethod_ = tropical_decision_method(alternativesComps_, criteriaComparisons_);
	tropicalDecisionMethod_.perform();
}

bool DecisionModel::alternativesComparisonsMatricesIsInit() const {

	return alternativesCompsIsInit_.size() == criteriaCount() &&
		   std::all_of(begin(alternativesCompsIsInit_), end(alternativesCompsIsInit_),
					   [](bool isInit) { return isInit; });
}

void DecisionModel::performAhpMethod() {
	ahpDecisionMethod_ = ahp_decision_method(alternativesComps_, criteriaComparisons_);
	ahpDecisionMethod_.perform();
}

void DecisionModel::performGmMethod() {
	gmDecisionMethod_ = gm_decision_method(alternativesComps_, criteriaComparisons_);
	gmDecisionMethod_.perform();
}

std::string DecisionModel::ahpResult() const {
	auto res = ahpDecisionMethod_.final_weights();
	std::stringstream ss;
	ss << res;
	ss << "\n\n";
	ss << modelRanking(res);
	return ss.str();
}

std::string DecisionModel::gmResult() const {
	auto res = gmDecisionMethod_.final_weights();
	std::stringstream ss;
	ss << res;
	ss << "\n\n";
	ss << modelRanking(res);
	return ss.str();
}

std::pair<std::string, std::string> DecisionModel::tropicalResult() const {
	auto [best_vectors, worst_vector] = tropicalDecisionMethod_.final_weights();
	std::stringstream ss_best, ss_worst;
	for (size_t i = 0, sz = best_vectors.size(); i < sz; ++i) {
		ss_best << best_vectors[i];
		ss_best << "\n\n";
		ss_best << modelRanking(best_vectors[i]);
		ss_best << (i + 1 < sz ? "\n\n" : "");
	}
	ss_worst << worst_vector;
	ss_worst << "\n\n";
	ss_worst << modelRanking(worst_vector);
	return {ss_best.str(), ss_worst.str()};
}

std::string DecisionModel::modelRanking(const Eigen::VectorXd &weights) {
	return modelRanking(rankModel(weights));
}

DecisionModel::ModelRanking DecisionModel::rankModel(const Eigen::VectorXd &weights) {
	ModelRanking rankings;
	for (int i = 0; i < weights.size(); ++i) {
		rankings.emplace_back(weights(i), i + 1);
	}

	std::stable_sort(begin(rankings), end(rankings),
					 [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
	return rankings;
}

std::string DecisionModel::modelRanking(const ModelRanking &modelRanking) {
	std::stringstream ss;
	for (int i = 0; i < modelRanking.size(); ++i) {
		auto [weight, alternative_n] = modelRanking.at(i);
		ss << alternativeSign + std::to_string(alternative_n);
		if (i + 1 < modelRanking.size()) {
			auto [next_weight, next_alternative_n] = modelRanking.at(i + 1);
			ss << (utils::approximately_equal(weight, next_weight) ? utils::wrap_with_spaces(equalitySign)
																   : utils::wrap_with_spaces(greaterSign));
		}
	}
	return ss.str();
}

// FIX(A1-1): в оригинале alternativesCompsIsInit_ не уменьшался, а матрицы альтернатив
// удалялись только если ВСЕ были инициализированы. После удаления критерия размеры
// векторов расходились: alternativesComparisonsMatricesIsInit() навсегда возвращал false,
// а матрицы «съезжали» на чужие критерии. Удаляем элемент из всех трёх векторов.
void DecisionModel::removeCriteria(int criteriaIndex) {
	if (criteriaIndex < 0 || criteriaIndex >= criteriaCount())
		return;
	auto eraseAt = [criteriaIndex](auto &v) {
		if (static_cast<size_t>(criteriaIndex) < v.size())
			v.erase(begin(v) + criteriaIndex);
	};
	eraseAt(alternativesComps_);
	eraseAt(alternativesCompsViews_);
	eraseAt(alternativesCompsIsInit_);
	criteriaNames_.erase(begin(criteriaNames_) + criteriaIndex);

	criteriaComparisons_ = removeRowAndColumn(criteriaComparisons_, criteriaIndex, criteriaIndex);
	criteriaComparisonsMatrixView_ =
			removeRowAndColumn(criteriaComparisonsMatrixView_, criteriaIndex, criteriaIndex);
}

// FIX(A1-2): alternativesCompsIsInit_.at(i) бросал std::out_of_range, если векторы имели
// разную длину; индекс альтернативы не проверялся (UB в removeRowAndColumn / erase).
void DecisionModel::removeAlternative(int alternativeIndex) {
	if (alternativeIndex < 0 || alternativeIndex >= alternativesCount())
		return;
	for (size_t i = 0; i < alternativesComps_.size(); ++i) {
		if (compsIsInitAt(static_cast<int>(i))) {
			auto &alt = alternativesComps_[i];
			alt = removeRowAndColumn(alt, alternativeIndex, alternativeIndex);
			if (i < alternativesCompsViews_.size()) {
				auto &altView = alternativesCompsViews_[i];
				altView = removeRowAndColumn(altView, alternativeIndex, alternativeIndex);
			}
		}
	}
	alternativesNames_.erase(begin(alternativesNames_) + alternativeIndex);
}

void DecisionModel::addAlternative(const std::string &alternativeName) {
	using namespace std::literals::string_literals;
	for (auto &alt: alternativesComps_) {
		alt = addRowAndColumn(alt, defaultMatrixValue, defaultMatrixValue);
	}
	for (auto &altView: alternativesCompsViews_) {
		altView = addRowAndColumn(altView, defaultMatrixView, defaultMatrixView);
	}
	alternativesNames_.emplace_back(alternativeName);
}

void DecisionModel::addCriteria(const std::string &criteriaName) {
	using namespace std::literals::string_literals;
	criteriaComparisons_ = addRowAndColumn(criteriaComparisons_, defaultMatrixValue, defaultMatrixValue);
	criteriaComparisonsMatrixView_ =
			addRowAndColumn(criteriaComparisonsMatrixView_, defaultMatrixView, defaultMatrixView);
	criteriaNames_.emplace_back(criteriaName);

	addAlternativesComparisonsMatrix();
}

void DecisionModel::addAlternativesComparisonsMatrix() {
	alternativesComps_.emplace_back(Eigen::MatrixXd::Ones(alternativesCount(), alternativesCount()));
	alternativesCompsViews_.emplace_back(
			Eigen::MatrixX<std::string>::Constant(alternativesCount(), alternativesCount(), defaultMatrixView));
	alternativesCompsIsInit_.emplace_back(true);
}

void DecisionModel::initCriteriaComparisonsMatrix() {
	criteriaComparisons_.setOnes(criteriaCount(), criteriaCount());
	criteriaComparisonsMatrixView_.setConstant(criteriaCount(), criteriaCount(), defaultMatrixView);
}
