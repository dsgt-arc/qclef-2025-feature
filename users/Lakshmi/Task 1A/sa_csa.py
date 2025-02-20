import numpy as np
import random
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def evaluate_solution(solution, X, y):
    """Evaluates a solution using cross-validation accuracy."""
    selected_features = np.where(solution == 1)[0]
    if len(selected_features) == 0:
        return 0  # Avoid selecting no features at all
    X_selected = X[:, selected_features]
    model = LogisticRegression(max_iter=1000)
    score = np.mean(cross_val_score(model, X_selected, y, cv=3))
    return score

def mutate_solution(solution):
    """Randomly flips a feature selection bit."""
    index = random.randint(0, len(solution) - 1)
    new_solution = solution.copy()
    new_solution[index] = 1 - new_solution[index]
    return new_solution

def accept_solution(current_score, new_score, temp):
    """Acceptance probability for simulated annealing."""
    if new_score > current_score:
        return True
    return np.exp((new_score - current_score) / temp) > random.random()

def simulated_annealing_csa(X, y, initial_temp=1000, final_temp=1, alpha=0.95):
    """Applies simulated annealing for feature selection."""
    current_solution = np.random.choice([0, 1], size=X.shape[1])
    current_score = evaluate_solution(current_solution, X, y)
    temp = initial_temp
    
    while temp > final_temp:
        new_solution = mutate_solution(current_solution)
        new_score = evaluate_solution(new_solution, X, y)
        
        if accept_solution(current_score, new_score, temp):
            current_solution = new_solution
            current_score = new_score
            
        temp *= alpha
    
    return current_solution
