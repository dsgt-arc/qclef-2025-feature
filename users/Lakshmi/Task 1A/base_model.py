import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def select_features_rfe(X, y, num_features=10):
    """Performs feature selection using Recursive Feature Elimination (RFE)."""
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=num_features)
    fit = rfe.fit(X, y)
    return fit.support_  # Boolean mask of selected features
