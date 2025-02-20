import numpy as np
from utils import load_data, load_letor_data, preprocess_data
from sa_csa import simulated_annealing_csa
from base_model import select_features_rfe
from scoring_method import train_lambdamart, evaluate_ndcg

# Load and preprocess MQ2007 dataset
mq2007_data = load_letor_data("data/train.txt")
print(mq2007_data.columns)
print(mq2007_data.head())
df = mq2007_data.drop(columns=["#docid", "inc", "prob"], errors="ignore") 
X_train_mq, X_test_mq, y_train_mq, y_test_mq, qids_train, qids_test = preprocess_data(mq2007_data, target_column="label")

# Define different feature selection methods
feature_selection_methods = {
    "RFE": lambda X, y: select_features_rfe(X, y),
    "Simulated Annealing CSA": lambda X, y: simulated_annealing_csa(X, y),
    "Simulated Annealing MI": lambda X, y: simulated_annealing_csa(X, y)
    # Add more methods here as needed
}

results = {}

# Apply each feature selection method and evaluate performance
for method_name, feature_selector in feature_selection_methods.items():
    print(f"Running feature selection using {method_name}...")
    
    # Select features
    selected_features = feature_selector(X_train_mq, y_train_mq)
    
    # If the method returns a boolean mask (RFE), use it as is
    # If it returns a binary array (SA), convert it to boolean
    if isinstance(selected_features, np.ndarray) and selected_features.dtype == np.int_:
        selected_features = selected_features == 1

    X_train_selected = X_train_mq[:, selected_features]
    X_test_selected = X_test_mq[:, selected_features]
    y_train_mq = np.array(y_train_mq)
    sorted_indices = np.argsort(qids_train)
    X_train_selected = X_train_selected[sorted_indices]
    y_train_mq = y_train_mq[sorted_indices]
    qids_train = qids_train[sorted_indices]

    # Train LambdaMART and get predictions
    predictions = train_lambdamart(X_train_selected, y_train_mq, qids_train, X_test_selected, y_test_mq, qids_test)

    # Compute nDCG@10
    ndcg_score = evaluate_ndcg(y_test_mq, predictions)
    results[method_name] = ndcg_score
    print(f"nDCG@10 for {method_name}: {ndcg_score}")

# Print final comparison
print("\n=== Feature Selection Method Comparison ===")
for method, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{method}: nDCG@10 = {score}")

# Best-performing method
best_method = max(results, key=results.get)
print(f"\nBest feature selection method: {best_method} with nDCG@10 = {results[best_method]}")
