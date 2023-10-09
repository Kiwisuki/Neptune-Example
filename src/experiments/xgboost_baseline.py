from xgboost import XGBRegressor
import neptune
from src.data_export import get_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.plotting_utils import scatter_residual_analysis, plot_feature_importance
from sklearn.model_selection import cross_val_predict
import logging
from neptune.types import File

from config.constants import RANDOM_STATE, NEPTUNE_PROJECT_NAME, NEPTUNE_API_TOKEN, TARGET, N_FOLDS, MODELS_PATH

def xgboost_baseline_experiment():
    """Runs a baseline experiment with default parameters for XGBoost regressor."""
    data = get_data()
    X, y = data.drop(TARGET, axis=1), data[TARGET]
    

    logging.info('Initializing Neptune')
    run = neptune.init_run(project=NEPTUNE_PROJECT_NAME,
                           api_token=NEPTUNE_API_TOKEN,
                           name='xgboost-baseline',
                            tags=['xgboost', 'baseline', 'default-params'])
    
    model_params = {
        'random_state': RANDOM_STATE,
    }

    model = XGBRegressor(**model_params)

    logging.info('Cross-validating model')
    data[f'predicted_{TARGET}'] = cross_val_predict(model, X, y, cv=N_FOLDS)

    # Since we are predicting a discrete value, we need to round the predictions
    data[f'predicted_{TARGET}'] = data[f'predicted_{TARGET}'].round(0).astype(int)

    run['rmse'] = mean_squared_error(data[TARGET], data[f'predicted_{TARGET}'], squared=False)
    run['mae'] = mean_absolute_error(data[TARGET], data[f'predicted_{TARGET}'])
    run['r2'] = r2_score(data[TARGET], data[f'predicted_{TARGET}'])

    logging.info('Visualizing error analysis')
    fig = scatter_residual_analysis(data)
    run['visuals/error_analysis'].upload(fig)

    logging.info('Retraining model on full dataset')
    model.fit(X, y)

    logging.info('Visualizing feature importance')
    fig = plot_feature_importance(model)
    run['visuals/feature_importance'].upload(fig)


    run['artifacts/model'].upload(File.as_pickle(model))
    run['code/experiment_code'] = File('src/experiments/xgboost_baseline.py')

    run.stop()
    logging.info('Finished experiment')



