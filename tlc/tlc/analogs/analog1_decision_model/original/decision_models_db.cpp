#include <decision_models_db.h>

void DecisionModelsDB::addOrUpdateModel(const std::string &modelName, const DecisionModel &decisionModel) {
	modelsDb_[modelName] = decisionModel;
}

size_t DecisionModelsDB::size() const {
	return modelsDb_.size();
}

bool DecisionModelsDB::count(const std::string &modelName) const {
	return modelsDb_.count(modelName);
}

const DecisionModel &DecisionModelsDB::model(const std::string &modelName) const {
	return modelsDb_.at(modelName);
}

DecisionModel &DecisionModelsDB::model(const std::string &modelName) {
	return modelsDb_.at(modelName);
}

void DecisionModelsDB::deleteModel(const std::string &modelName) {
	if (modelsDb_.count(modelName))
		modelsDb_.erase(modelName);
}

void DecisionModelsDB::changeModelName(const std::string &oldModelName, const std::string &newModelName) {
	if (modelsDb_.count(oldModelName)) {
		modelsDb_[newModelName] = std::move(modelsDb_[oldModelName]);
		modelsDb_.erase(oldModelName);
	}
}
