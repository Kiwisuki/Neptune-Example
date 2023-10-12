import logging
import warnings

from config.constants import ADJUST_CONSTANT, TARGET
from src.data_export import get_data
from src.experiments.run_template import run_experiment
from src.experiments.xgboost_classifier_baseline import (
    EXPERIMENT_CONFIG as XGB_BASE_CONFIG,
)
from src.experiments.xgboost_classifier_hyper import (
    EXPERIMENT_CONFIG as XGB_HYPER_CONFIG,
)
from src.experiments.xgboost_regressor_baseline import (
    EXPERIMENT_CONFIG as XGB_REGRESSOR_BASE_CONFIG,
)
from src.experiments.xgboost_regressor_hyper import (
    EXPERIMENT_CONFIG as XGB_REGRESSOR_HYPER_CONFIG,
)

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%m-%d %H:%M:%S',
)

if __name__ == '__main__':
    data = get_data()
    # Adjust target column so it starts from 0
    data[TARGET] = data[TARGET] - ADJUST_CONSTANT
    run_experiment(data=data, **XGB_HYPER_CONFIG)
    run_experiment(data=data, **XGB_BASE_CONFIG)
    run_experiment(data=data, **XGB_REGRESSOR_BASE_CONFIG)
    run_experiment(data=data, **XGB_REGRESSOR_HYPER_CONFIG)
