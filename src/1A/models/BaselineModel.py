import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def select_features_rfe(X, y, num_features=10):
    """Performs feature selection using Recursive Feature Elimination (RFE)."""
    model = LogisticRegression(max_iter=10000)
    rfe = RFE(model, n_features_to_select=num_features)
    fit = rfe.fit(X, y)
    
    # Get feature names from the DataFrame's columns if X is a pandas DataFrame
    feature_names = X.columns if hasattr(X, 'columns') else [f'feature{i+1}' for i in range(X.shape[1])]
    
    # Create dictionary of selected features with 1 for selected and 0 for not selected
    # selected_features = {feature: 1 if selected else 0 
    #                      for feature, selected in zip(feature_names, fit.support_)}

    # Create a list of selected features
    selected_features = [feature for feature, selected in zip(feature_names, fit.support_) if selected]
    
    return selected_features