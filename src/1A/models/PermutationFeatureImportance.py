import dimod
import numpy as np
import lightgbm as lgb
from sklearn.metrics import ndcg_score
from sklearn.base import clone
from models.MutualInformation import conditional_mutual_information, prob, maximum_energy_delta
import pandas as pd
import itertools

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

class PermutationFeatureImportance:
    def __init__(self, X, y, qids):
        self.qids = qids
        self.X = X
        self.y = y
        self.BQM = None
        # self.model = model
        self.feature_importances_ = None
        self.selected_features = None
    
    def _compute_ndcg(self, model, X, y):
        """
        Computes 1 - NDCG@10 to reformulate as a minimization problem
        
        Args:
        X: Feature matrix.
        y: Data labels.
    
        Returns:
        1 - ndcg (float): 1 - nDCG@k score for the test set.
        """
        y_pred = model.predict(X)
        # y = y.reshape(1, -1)
        # y_pred = y_pred.reshape(1, -1)
        return 1 - ndcg_score([y], [y_pred], k=10)

    def _permute_feature(self, feature_idx):
        """
        Creates a copy of X with one feature shuffled.
        
        Args:
        feature_idx: index of the feature to be permuted.
    
        Returns:
        X_perm: A copy of the feature matrix with one feature shuffled.
        """
        X_perm = self.X.copy()
        X_perm_array = X_perm.iloc[:, feature_idx].values #Turn the column into an array
        np.random.shuffle(X_perm_array)
        X_perm.iloc[:, feature_idx] = X_perm_array #Add the shuffled column back into the matrix
        return X_perm
    
    def fit(self, k=10):
        """
        Computes feature importances using permutation importance and simulated annealing.
        
        Args:
        X: Feature matrix.
        y: Data labels.
        """
        #Train model clone on original data
        y_train = np.array(self.y)
        X_train = self.X.values

        # Sort data based on query IDs
        sorted_indices = np.argsort(self.qids)
        X_train = X_train[sorted_indices]
        y_train = y_train[sorted_indices]
        qids_train = np.array(self.qids)[sorted_indices]

        # Compute group sizes for ranking
        unique_qids, group_sizes = np.unique(qids_train, return_counts=True)

        # Train LambdaMART and get predictions
        model = train_lambdamart(X_train, y_train, group_sizes)
        
        #Compute original model error
        e_orig = self._compute_ndcg(model, self.X, self.y)
        
        #Compute change in loss for each feature to determine feature importance
        feature_importances = []
        num_features = self.X.shape[1]
        for j in range(num_features):
            X_perm = self._permute_feature(j)
            e_perm_j = self._compute_ndcg(model, X_perm, self.y)
            FI_j = e_perm_j - e_orig
            feature_importances.append((j, FI_j))
        
        self.feature_importances_ = dict(feature_importances)

        self.BQM = self.form_bqm(k)

        #Use simulated annealing to optimize feature ranking
        return self._optimize_feature_ranking()
    
    def form_bqm(self, k):
        """Builds the BQM according to Permutation Feature Importance and
        Conditional Mutual Information"""
        BQM = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        feature_labels = list(self.X.columns)

        # Compute diagonal elements (PFI)
        for j in range(len(feature_labels)):
            BQM.add_variable(feature_labels[j], - self.feature_importances_.get(j))

        # Compute off-diagonal elements (CMI)
        for (f1_idx, f2_idx) in itertools.combinations(range(len(self.feature_importances_.keys())), 2):
            f1_label, f2_label = feature_labels[f1_idx], feature_labels[f2_idx]

            # Compute Conditional Mutual Information
            cmi = conditional_mutual_information(
                prob(pd.concat([self.y, self.X[[f1_label, f2_label]]], axis=1).values), 1, 2
            )

            BQM.add_interaction(f1_label, f2_label, -cmi)

        penalty = maximum_energy_delta(BQM)
        kbqm = dimod.generators.combinations(BQM.variables, k, strength=penalty)
        kbqm.update(BQM)

        return kbqm

    def _optimize_feature_ranking(self):
        """
        Uses Simulated Annealing to find an optimal ranking of features.
        
        Returns:
        self.selected (list): A list of column indices for the selected features
        """
        
        sampler = dimod.SimulatedAnnealingSampler()
        sampleset = sampler.sample(self.BQM, num_reads=100)

        #Extract feature ranking from SA results
        best_sample = sampleset.first.sample
        
        self.selected_features = [int(feat)-1 for feat, selected in best_sample.items() if selected == 1]

        return self.selected_features
    
    def get_selected_features(self):
        """        
        Returns:
            selected_features (list): A list of the indices for the selected features
            X_reduced (DataFrame): The dataset with only the selected features.
        """
        
        selected_features = self.selected_features

        X_reduced = self.X.iloc[:, selected_features]

        return selected_features, X_reduced
    
    def calc_ndcg_score(self, X, X_test, y, y_test, qids_train, qids_test):
        
        selected_features, X_train_reduced = self.get_selected_features()

        # Sort data based on query IDs
        sorted_indices = np.argsort(qids_train)
        qids_train = np.array(qids_train)[sorted_indices]

        # Compute group sizes for ranking
        unique_qids, group_sizes = np.unique(qids_train, return_counts=True)

        # Train LambdaMART and get predictions
        model = train_lambdamart(X_train_reduced, y, group_sizes)

        y_pred = model.predict(X_test.iloc[:, selected_features])
        
        # Calculate nDCG at k (default 10)
        ndcg = ndcg_score([y_test], [y_pred], k=10)
        return ndcg
