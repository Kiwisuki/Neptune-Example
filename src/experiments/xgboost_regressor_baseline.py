""" Experiment configuration for XGBoost classifier baseline. """
from pathlib import Path

from xgboost import XGBRegressor

SCRIPT_PATH = str(Path.resolve(Path(__file__)))

EXPERIMENT_NAME = 'xgboost-regressor-baseline'
MAX_EVALS = 50
MODEL_CLASS = XGBRegressor
TAGS = ['xgboost', 'regressor', 'baseline']
PARAMS = {'verbosity': 0}

EXPERIMENT_CONFIG = {
    'experiment_name': EXPERIMENT_NAME,
    'space': None,
    'objective': None,
    'max_evals': MAX_EVALS,
    'model_class': MODEL_CLASS,
    'tags': TAGS,
    'params': PARAMS,
    'config_path': SCRIPT_PATH,
}
