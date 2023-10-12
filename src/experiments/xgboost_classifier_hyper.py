""" Experiment configuration for XGBoost classifier hyperparameter tuning. """
from typing import Dict, Type

import numpy as np
from hyperopt import hp
from neptune import Run
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

EXPERIMENT_NAME = 'xgboost-classifier-hyperopt'
MAX_EVALS = 50
MODEL_CLASS = XGBClassifier
TAGS = ['xgboost', 'classifier', 'hyperopt-params']

SPACE = {
    'n_estimators': hp.choice('n_estimators', np.arange(50, 501, 50, dtype=int)),
    'max_depth': hp.choice('max_depth', np.arange(1, 21, dtype=int)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.5)),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'gamma': hp.uniform('gamma', 0, 1),
    'reg_alpha': hp.loguniform('reg_alpha', -6, 2),
    'reg_lambda': hp.loguniform('reg_lambda', -6, 2),
}


def objective(
    params: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_class: Type[BaseEstimator],
    run: Type[Run],
) -> float:
    """Objective function for hyperopt to minimize."""
    model = model_class(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    run['hyperopt/trial_test_accuracy'].append(test_accuracy)
    run['hyperopt/trial_train_accuracy'].append(train_accuracy)
    return -test_accuracy


EXPERIMENT_CONFIG = {
    'experiment_name': EXPERIMENT_NAME,
    'space': SPACE,
    'objective': objective,
    'max_evals': MAX_EVALS,
    'model_class': MODEL_CLASS,
    'tags': TAGS,
}
