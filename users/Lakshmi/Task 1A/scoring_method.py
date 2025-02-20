import numpy as np
import lightgbm as lgb
from sklearn.metrics import ndcg_score

def train_lambdamart(X_train, y_train, qids_train, X_test, y_test, qids_test):
    """Trains a LambdaMART model using LightGBM and makes predictions."""

    # Convert qids to group format required by LightGBM
    train_group = np.bincount(qids_train).tolist()
    test_group = np.bincount(qids_test).tolist()

    model = lgb.LGBMRanker(
        boosting_type='gbdt',
        objective='lambdarank',
        metric='ndcg',
        n_estimators=100,  # Adjust as needed
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42
    )

    # Fit the model
    model.fit(
        X_train, y_train,
        group=train_group,
        eval_set=[(X_test, y_test)],
        eval_group=[test_group],
        eval_at=[10],
        callbacks=[lgb.early_stopping(10)]
    )

    return model.predict(X_test)

def evaluate_ndcg(y_true, y_pred):
    """Computes nDCG@10 for the given predictions."""
    return ndcg_score([y_true], [y_pred], k=10)
