import logging
import warnings

from src.data_export import get_data
from src.experiments.xgboost_classifier_baseline import (
    xgboost_classifier_baseline_experiment,
)
from src.experiments.xgboost_regressor_baseline import (
    xgboost_regressor_baseline_experiment,
)
from src.experiments.xgboost_regressor_hyperopt import (
    xgboost_regressor_hyperopt_experiment,
)

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%m-%d %H:%M:%S',
)

if __name__ == '__main__':
    logging.info('Starting baseline experiment')
    # xgboost_regressor_baseline_experiment()
    logging.info('Starting hyperopt regressor experiment')
    # xgboost_regressor_hyperopt_experiment()
    logging.info('Starting classifier experiment')
    xgboost_classifier_baseline_experiment()
