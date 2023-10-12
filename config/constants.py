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

ADJUST_CONSTANT = 3

TEST_SIZE = 0.2
