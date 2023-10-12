import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from hyperopt import fmin, tpe
from neptune import Run
from neptune.types import File
from neptune.utils import stringify_unsupported
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_predict, train_test_split

from config.constants import (
    N_FOLDS,
    NEPTUNE_API_TOKEN,
    NEPTUNE_PROJECT_NAME,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from src.plotting_utils import plot_feature_importance, scatter_residual_analysis

SCRIPT_PATH = str(Path(os.path.realpath(__file__)))


def log_performance(run, path, y_pred, y_test):
    run[f'metrics/{path}/rmse'] = mean_squared_error(
        y_test,
        y_pred,
        squared=False,
    )
    run[f'metrics/{path}/mae'] = mean_absolute_error(y_test, y_pred)
    run[f'metrics/{path}/r2'] = r2_score(y_test, y_pred)
    run[f'metrics/{path}/accuracy'] = accuracy_score(y_test, y_pred)


def run_experiment(
    *,
    data: pd.DataFrame,
    experiment_name: str,
    space: Optional[dict],
    objective: Callable,
    max_evals: int,
    model_class: BaseEstimator,
    config_path: str,
    tags: Optional[List[str]] = None,
    params: Optional[dict] = None,
) -> None:
    """Run an experiment using the given data and hyperparameters."""
    if space and params:
        raise ValueError('You cannot provide both space and params.')

    run = Run(
        project=NEPTUNE_PROJECT_NAME,
        api_token=NEPTUNE_API_TOKEN,
        name=experiment_name,
        tags=tags,
    )

    X, y = data.drop(TARGET, axis=1), data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE, test_size=TEST_SIZE
    )

    if space:
        params = fmin(
            fn=lambda params: objective(
                params, X_train, y_train, X_test, y_test, model_class, run
            ),
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
        )

    model = model_class(**params)

    logging.info('Fitting model..')
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Rounding and casting to int to be compatible with classification metrics
    y_train_pred = y_train_pred.round().astype(int)
    y_test_pred = y_test_pred.round().astype(int)

    logging.info('Cross-validating model..')
    data[f'predicted_{TARGET}'] = cross_val_predict(model, X, y, cv=N_FOLDS, n_jobs=-1)
    data[f'predicted_{TARGET}'] = data[f'predicted_{TARGET}'].round().astype(int)

    logging.info('Creating visuals for analysis..')
    residual_analysis_fig = scatter_residual_analysis(data)
    feature_importance_fig = plot_feature_importance(model)

    logging.info('Logging information to Neptune..')
    # Experiment metadata
    run['hyperparameters'] = params

    # Hyperopt
    if space:
        run['hyperopt/max_evals'] = max_evals
        run['hyperopt/space'] = stringify_unsupported(space)

    # Performance metrics
    log_performance(run, 'train', y_train_pred, y_train)
    log_performance(run, 'test', y_test_pred, y_test)
    log_performance(run, 'cv', data[f'predicted_{TARGET}'], data[TARGET])

    # Visuals
    run['visuals/error_analysis'].upload(residual_analysis_fig)
    run['visuals/feature_importance'].upload(feature_importance_fig)

    # Artifacts
    run['artifacts/model'].upload(File.as_pickle(model))
    run['code/experiment_code'] = File(SCRIPT_PATH)
    run['code/config'] = File(config_path)

    # Misc
    run['misc/experiment_name'] = experiment_name
    run['code/base_model'] = str(model_class)
