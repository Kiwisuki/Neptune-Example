import logging

import neptune
from neptune.types import File
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from xgboost import XGBRegressor

from config.constants import (
    MODELS_PATH,
    N_FOLDS,
    NEPTUNE_API_TOKEN,
    NEPTUNE_PROJECT_NAME,
    RANDOM_STATE,
    TARGET,
)
from src.data_export import get_data
from src.plotting_utils import plot_feature_importance, scatter_residual_analysis


def xgboost_baseline_experiment():
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

    model_params = {
        'random_state': RANDOM_STATE,
    }

    model = XGBRegressor(**model_params)

    logging.info('Cross-validating model')
    data[f'predicted_{TARGET}'] = (
        cross_val_predict(model, X, y, cv=N_FOLDS).round(0).astype(int)
    )

    logging.info('Visualizing error analysis')
    residual_analysis_fig = scatter_residual_analysis(data)

    logging.info('Retraining model on full dataset')
    model.fit(X, y)

    logging.info('Visualizing feature importance')
    feature_importance_fig = plot_feature_importance(model)

    logging.info('Logging metrics and artifacts')
    run['rmse'] = mean_squared_error(
        data[TARGET],
        data[f'predicted_{TARGET}'],
        squared=False,
    )
    run['mae'] = mean_absolute_error(data[TARGET], data[f'predicted_{TARGET}'])
    run['r2'] = r2_score(data[TARGET], data[f'predicted_{TARGET}'])
    run['visuals/error_analysis'].upload(residual_analysis_fig)
    run['visuals/feature_importance'].upload(feature_importance_fig)
    run['artifacts/model'].upload(File.as_pickle(model))
    run['code/experiment_code'] = File('src/experiments/xgboost_baseline.py')

    run.stop()
    logging.info('Finished experiment')
