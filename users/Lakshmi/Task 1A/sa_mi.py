import numpy as np
import random
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import log_loss

def compute_cross_entropy(X_subset, y):
    """ Computes cross-entropy loss using a simple Logistic Regression model. """
    model = LogisticRegression()
    probs = cross_val_predict(model, X_subset, y, cv=3, method='predict_proba')
    return log_loss(y, probs)

def simulated_annealing_mi_ce(X, y, max_iter=1000, initial_temp=1.0, cooling_rate=0.99, lambda_1=0.5, lambda_2=0.5):
    """ Simulated Annealing for feature selection using Mutual Information (MI) and Cross-Entropy (CE). """
    n_features = X.shape[1]

    # Compute Mutual Information scores for each feature
    mi_scores = mutual_info_classif(X, y)
    
    # Start with a random subset of features
    current_solution = np.random.choice([0, 1], size=n_features)
    
    # Compute initial energy (negative MI + cross-entropy loss)
    selected_features = np.where(current_solution == 1)[0]
    if len(selected_features) == 0:
        return np.zeros(n_features)  # Avoid empty feature set
    
    current_energy = -lambda_1 * np.sum(mi_scores[selected_features]) + lambda_2 * compute_cross_entropy(X[:, selected_features], y)
    
    best_solution = current_solution.copy()
    best_energy = current_energy

    temp = initial_temp

    for _ in range(max_iter):
        # Generate a neighbor by flipping one random feature (add/remove a feature)
        neighbor = current_solution.copy()
        flip_index = np.random.randint(n_features)
        neighbor[flip_index] = 1 - neighbor[flip_index]

        # Ensure we don’t get an empty selection
        selected_features = np.where(neighbor == 1)[0]
        if len(selected_features) == 0:
            continue

        # Compute new energy
        neighbor_energy = -lambda_1 * np.sum(mi_scores[selected_features]) + lambda_2 * compute_cross_entropy(X[:, selected_features], y)

        # Acceptance criteria (Metropolis criterion)
        if neighbor_energy < current_energy:  # Minimize energy function
            current_solution = neighbor
            current_energy = neighbor_energy
        else:
            # Accept with probability based on temperature
            delta = neighbor_energy - current_energy
            probability = np.exp(-delta / temp)
            if np.random.rand() < probability:
                current_solution = neighbor
                current_energy = neighbor_energy

        # Update best solution
        if current_energy < best_energy:
            best_solution = current_solution
            best_energy = current_energy

        # Reduce temperature
        temp *= cooling_rate

        # Stop if temperature is very low
        if temp < 1e-4:
            break

    return best_solution
