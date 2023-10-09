import os
from pathlib import Path

# Data constants
DATASET_PATH = Path('data/WineQT.csv')
TARGET = 'quality'

# Neptune constants
NEPTUNE_PROJECT_NAME = 'marius.arlauskas/Neptune-Demo'
NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')

# Modelling constants
RANDOM_STATE = 42
N_FOLDS = 5
MODELS_PATH = Path('models')
MAX_EVALS = 30

OPTIMIZATION_PARAMETERS = [
    'n_estimators',
    'max_depth',
    'learning_rate',
    'subsample',
    'colsample_bytree',
    'min_child_weight',
    'gamma',
    'reg_alpha',
    'reg_lambda',
]

CLASSIFIER_REMAP_CONSTANT = 3
