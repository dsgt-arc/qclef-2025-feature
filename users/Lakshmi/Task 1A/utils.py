import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads dataset from CSV."""
    return pd.read_csv(file_path)

def load_letor_data(filepath):
    """Loads a LETOR-formatted dataset and properly parses the columns."""
    df = pd.read_csv(filepath, sep=" ", header=None, skipinitialspace=True, engine="python")

    # Remove empty columns created due to extra spaces at the end of each line
    df = df.dropna(axis=1, how="all")

    # Rename the first column as 'label' (target)
    df.rename(columns={0: "label"}, inplace=True)

    # Extract query IDs (which are in the second column in 'qid:<number>' format)
    df["qid"] = df[1].str.split(":").str[1].astype(int)

    # Drop the original qid column (column index 1)
    df.drop(columns=[1], inplace=True)

    # Convert feature columns to numeric values (handling "index:value" format)
    feature_columns = {}
    for i in range(2, df.shape[1]):  # Features start from column index 2
        sample_value = str(df[i].dropna().iloc[0])  # Convert to string to check format
        if ":" in sample_value:  
            feature_index, feature_value = sample_value.split(":")
            feature_columns[i] = int(feature_index)  # Extract feature index
            df[i] = df[i].astype(str).str.split(":").str[1].astype(float)  # Extract feature values
        else:
            feature_columns[i] = i  # Keep default index if no ":" format

    # Rename feature columns
    df.rename(columns=feature_columns, inplace=True)
    df.columns = df.columns.astype(str)
    return df


def preprocess_data(data, target_column='target'):
    """Prepares data for training: standardization and train-test split."""
    data = data.select_dtypes(include=["number"])
    qids = data["qid"].values
    data = data.drop(columns=["qid"], errors="ignore")
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Standardize features
    X = StandardScaler().fit_transform(X)

    # Split the data while preserving qids
    X_train, X_test, y_train, y_test, qids_train, qids_test = train_test_split(
        X, y, qids, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, qids_train, qids_test
