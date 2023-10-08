import pandas as pd

from config.constants import DATASET_PATH


def get_data():
    """Reads the dataset from the csv file and returns it as a pandas DataFrame."""
    return pd.read_csv(DATASET_PATH, index_col=-1)
