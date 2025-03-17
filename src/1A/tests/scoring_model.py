import lightgbm as lgb
import numpy as np
from sklearn.metrics import ndcg_score

def train_lambdamart(X_train, y_train, group_sizes):
    """
    Trains a LambdaMART model using LightGBM.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Feature matrix for training.
        y_train (pd.Series or np.ndarray): Relevance scores for training.
        group_sizes (list): List of group sizes for ranking task (e.g., number of documents per query).
    
    Returns:
        model (LightGBM model): Trained LambdaMART model.
    """
    # Create a dataset for LightGBM with ranking information
    train_data = lgb.Dataset(X_train, label=y_train, group=group_sizes)
    
    # Set parameters for LambdaMART (LightGBM specific)
    params = {
        'objective': 'lambdarank',  # LambdaMART objective
        'metric': 'ndcg',           # Optimize for NDCG
        'boosting_type': 'gbdt',    # GBDT boosting
        'num_leaves': 31,           # Number of leaves in one tree
        'learning_rate': 0.05,      # Learning rate
        'num_iterations': 100       # Number of boosting iterations
    }
    
    # Train the model
    model = lgb.train(params, train_data)
    return model

def evaluate_ndcg(model, X_test, y_test, k=10):
    """
    Evaluates the model's performance using nDCG@k.
    
    Args:
        model (LightGBM model): Trained LambdaMART model.
        X_test (pd.DataFrame or np.ndarray): Feature matrix for testing.
        y_test (pd.Series or np.ndarray): Relevance scores for testing.
        group_sizes (list): List of group sizes for ranking task (e.g., number of documents per query).
        k (int): Rank position for nDCG (default is 10).
    
    Returns:
        ndcg (float): nDCG@k score for the test set.
    """
    # Predict the scores for the test data
    y_pred = model.predict(X_test)
    
    # Calculate nDCG at k (default 10)
    ndcg = ndcg_score([y_test], [y_pred], k=k)
    return ndcg

def calc_ndcg_score(X_train_mq, X_test_mq, y_train_mq, y_test_mq, qids_train, qids_test, selected_features):
    # Select only the columns that match the selected feature names
    X_train_selected = X_train_mq[selected_features].values  # Convert to NumPy array
    X_test_selected = X_test_mq[selected_features].values  # Convert to NumPy array

    y_train_mq = np.array(y_train_mq)

    # Sort data based on query IDs
    sorted_indices = np.argsort(qids_train)
    X_train_selected = X_train_selected[sorted_indices]
    y_train_mq = y_train_mq[sorted_indices]
    qids_train = np.array(qids_train)[sorted_indices]

    # Compute group sizes for ranking
    unique_qids, group_sizes = np.unique(qids_train, return_counts=True)

    # Train LambdaMART and get predictions
    model = train_lambdamart(X_train_selected, y_train_mq, group_sizes)

    # Compute nDCG@10
    ndcg_score = evaluate_ndcg(model, X_test_selected, y_test_mq)
    return ndcg_score