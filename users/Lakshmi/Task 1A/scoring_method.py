import numpy as np
from sklearn.metrics import ndcg_score
from pyltr.models.lambdamart import LambdaMART

def train_lambdamart(X_train, y_train, X_test, y_test):
    """Trains a LambdaMART model and makes predictions."""
    model = LambdaMART()
    model.fit(X_train, y_train)
    return model.predict(X_test)

def evaluate_ndcg(y_true, y_pred):
    """Computes nDCG@10 for the given predictions."""
    return ndcg_score([y_true], [y_pred], k=10)
