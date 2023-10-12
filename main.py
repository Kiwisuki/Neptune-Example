import logging
import warnings

from config.constants import ADJUST_CONSTANT, TARGET
from src.data_export import get_data
from src.experiments.run_template import run_experiment
from src.experiments.xgboost_classifier_hyper import EXPERIMENT_CONFIG

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%m-%d %H:%M:%S',
)

if __name__ == '__main__':
    data = get_data()
    # Adjust target column so it starts from 0
    data[TARGET] = data[TARGET] - ADJUST_CONSTANT
    run_experiment(data=data, **EXPERIMENT_CONFIG)
