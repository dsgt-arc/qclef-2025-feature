import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads dataset from CSV."""
    return pd.read_csv(file_path)

def preprocess_data(data, target_column='target'):
    """Prepares data for training: standardization and train-test split."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)
