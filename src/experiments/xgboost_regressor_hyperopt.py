import logging

import neptune
import numpy as np
from hyperopt import fmin, hp, tpe
from neptune.types import File
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from xgboost import XGBRegressor

from config.constants import (
    N_FOLDS,
    NEPTUNE_API_TOKEN,
    NEPTUNE_PROJECT_NAME,
    RANDOM_STATE,
    TARGET,
)
from src.data_export import get_data
from src.plotting_utils import plot_feature_importance, scatter_residual_analysis


def xgboost_regressor_hyperopt_experiment():
    """Runs a baseline experiment with default parameters for XGBoost regressor."""
    data = get_data()
    X, y = data.drop(TARGET, axis=1), data[TARGET]

    logging.info('Initializing Neptune')
    run = neptune.init_run(
        project=NEPTUNE_PROJECT_NAME,
        api_token=NEPTUNE_API_TOKEN,
        name='xgboost-baseline',
        tags=['xgboost', 'baseline', 'default-params'],
    )

    # Hyperopt setup
    space = {
        'n_estimators': hp.choice(
            'n_estimators', np.arange(50, 501, 50, dtype=int)
        ),  # Choose from 50 to 500 in steps of 50
        'max_depth': hp.choice(
            'max_depth', np.arange(1, 21, dtype=int)
        ),  # Choose from 1 to 20
        'learning_rate': hp.loguniform(
            'learning_rate', np.log(0.001), np.log(0.5)
        ),  # Learning rate
        'subsample': hp.uniform(
            'subsample', 0.6, 1.0
        ),  # Fraction of samples used for fitting the trees
        'colsample_bytree': hp.uniform(
            'colsample_bytree', 0.6, 1.0
        ),  # Fraction of features used for building each tree
        'min_child_weight': hp.quniform(
            'min_child_weight', 1, 10, 1
        ),  # Minimum sum of instance weight needed in a child
        'gamma': hp.uniform(
            'gamma', 0, 1
        ),  # Minimum loss reduction required to make a further partition on a leaf node
        'reg_alpha': hp.loguniform(
            'reg_alpha', -6, 2
        ),  # L1 regularization term on weights
        'reg_lambda': hp.loguniform(
            'reg_lambda', -6, 2
        ),  # L2 regularization term on weights
    }

    # ? Defining function within fuction feels not right, would like to discuss with the team.
    def objective(params: dict) -> float:
        model = XGBRegressor(**params, random_state=RANDOM_STATE)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
    )

    model = XGBRegressor(**best_params)

    logging.info('Cross-validating model')
    data[f'predicted_{TARGET}'] = (
        cross_val_predict(model, X, y, cv=N_FOLDS).round(0).astype(int),
    )

    logging.info('Visualizing error analysis')
    residual_analysis_fig = scatter_residual_analysis(data)

    logging.info('Retraining model on full dataset')
    model.fit(X, y)

    logging.info('Visualizing feature importance')
    feature_importance_fig = plot_feature_importance(model)

    logging.info('Logging metrics and artifacts')
    run['hyperparameters'] = best_params
    run['rmse'] = mean_squared_error(
        data[TARGET],
        data[f'predicted_{TARGET}'],
        squared=False,
    )
    run['mae'] = mean_absolute_error(data[TARGET], data[f'predicted_{TARGET}'])
    run['r2'] = r2_score(data[TARGET], data[f'predicted_{TARGET}'])
    run['accuracy'] = accuracy_score(data[TARGET], data[f'predicted_{TARGET}'])
    run['visuals/error_analysis'].upload(residual_analysis_fig)
    run['visuals/feature_importance'].upload(feature_importance_fig)
    run['artifacts/model'].upload(File.as_pickle(model))
    run['code/experiment_code'] = File('src/experiments/xgboost_baseline.py')

    run.stop()
    logging.info('Finished experiment')
