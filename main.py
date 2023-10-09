import logging
import warnings

from src.data_export import get_data
from src.experiments.xgboost_baseline import xgboost_baseline_experiment

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%m-%d %H:%M:%S',
)

if __name__ == '__main__':
    xgboost_baseline_experiment()
