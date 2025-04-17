import pandas as pd
from Evaluation.Evaluator import EvaluatorHoldout
import scipy.sparse as sps
import numpy as np
from sklearn.model_selection import train_test_split
import itertools, multiprocessing
from functools import partial
import gc
from sklearn.feature_selection import mutual_info_classif
import dimod
from neal import SimulatedAnnealingSampler
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _from_df_to_sparse(URM_df, n_users, n_items):
    URM_sps = sps.csr_matrix((np.ones(len(URM_df)),
                              (URM_df["UserID"].values, URM_df["ItemID"].values)),
                             shape=(n_users, n_items))
    return URM_sps

def compute_feature_relevance(ICM, URM_train, n_features):
    item_labels = np.any(URM_train.toarray(), axis=0).astype(int)
    mi_scores = mutual_info_classif(ICM.toarray(), item_labels, discrete_features=False)
    logger.info(f"Mutual information scores: min={mi_scores.min():.4f}, max={mi_scores.max():.4f}, mean={mi_scores.mean():.4f}")
    mi_scores = mi_scores / np.max(mi_scores + 1e-10)
    return mi_scores

def compute_feature_correlations(ICM, batch_size=1000):
    n_features = ICM.shape[1]
    ICM = ICM.tocsr()
    ICM_normalized = ICM / (sps.linalg.norm(ICM, axis=0) + 1e-10)
    ICM_normalized = ICM_normalized.tocsr()
    correlation = sps.lil_matrix((n_features, n_features))  # Use LIL for efficient updates
    for start in range(0, n_features, batch_size):
        end = min(start + batch_size, n_features)
        batch = ICM_normalized[:, start:end]
        correlation[start:end, :] = batch.T.dot(ICM_normalized).tolil()
    correlation = sps.csr_matrix(np.abs(correlation))
    correlation = correlation - sps.diags(correlation.diagonal())  # Set diagonal to 0 efficiently
    return correlation

def _run_experiment(hyperparameters, BQM, ICM, URM_train, evaluator_validation):
    k_largest, lambda_red, gamma_sparsity = hyperparameters
    logger.info(f"Running experiment with k_largest={k_largest}, lambda_red={lambda_red:.2E}, gamma_sparsity={gamma_sparsity:.2E}")

    try:
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(BQM, num_reads=1000, beta_range=(0.1, 10.0))
        selected_dict = sampleset.first.sample
        selected_features = [key for key, value in selected_dict.items() if value == 1]

        if not selected_features:
            logger.warning("No features selected. Returning NDCG=0.")
            return k_largest, lambda_red, gamma_sparsity, 0.0, 0, selected_dict

        if len(selected_features) > k_largest:
            logger.info(f"Selected {len(selected_features)} features, capping at {k_largest}")
            selected_features = selected_features[:k_largest]
            selected_dict = {i: 1 if i in selected_features else 0 for i in range(ICM.shape[1])}

        ICM_selected = ICM[:, selected_features]
        recommender = ItemKNNCBFRecommender(URM_train, ICM_selected)
        recommender.fit(topK=100, shrink=5, similarity='cosine', normalize=True)  # Adjusted parameters

        result_df, _ = evaluator_validation.evaluateRecommender(recommender)
        NDCG = result_df.loc[10, "NDCG"]
        n_selected = len(selected_features)

        logger.info(f"k_largest={k_largest}, lambda_red={lambda_red:.2E}, gamma_sparsity={gamma_sparsity:.2E}: NDCG@10={NDCG:.4f}, selected={n_selected}")

        return k_largest, lambda_red, gamma_sparsity, NDCG, n_selected, selected_dict

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        return k_largest, lambda_red, gamma_sparsity, 0.0, 0, {}

if __name__ == "__main__":
    n_features = 500
    URM_all_df = pd.read_csv("URM.csv")
    ICM_df = pd.read_csv("500_ICM.csv")

    n_users = URM_all_df["UserID"].max() + 1
    n_items = ICM_df["ItemID"].max() + 1

    URM_all = _from_df_to_sparse(URM_all_df, n_users, n_items)
    ICM = sps.csr_matrix((ICM_df["Value"].values,
                          (ICM_df["ItemID"].values, ICM_df["FeatureID"].values)),
                         shape=(n_items, n_features))

    logger.info(f"URM shape: {URM_all.shape}, sparsity: {URM_all.nnz / (n_users * n_items):.4f}")
    logger.info(f"ICM shape: {ICM.shape}, sparsity: {ICM.nnz / (n_items * n_features):.4f}")

    URM_train_df, URM_validation_df = train_test_split(URM_all_df, test_size=0.20)
    URM_train = _from_df_to_sparse(URM_train_df, n_users, n_items)
    URM_validation = _from_df_to_sparse(URM_validation_df, n_users, n_items)
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])

    recommender_full = ItemKNNCBFRecommender(URM_train, ICM)
    recommender_full.fit(topK=100, shrink=5, similarity='cosine', normalize=True)
    result_df_full, _ = evaluator_validation.evaluateRecommender(recommender_full)
    baseline_NDCG = result_df_full.loc[10, "NDCG"]
    logger.info(f"Baseline NDCG@10 with all {n_features} features: {baseline_NDCG:.4f}")
    gc.collect()

    relevance_scores = compute_feature_relevance(ICM, URM_train, n_features)
    correlation_matrix = compute_feature_correlations(ICM)

    k_largest_list = list(range(50, 201, 10))  # Reduced range
    lambda_red_list = [1e-3, 1e-2, 1e-1]
    gamma_sparsity_list = [1e-3, 1e-2, 1e-1]
    hyperparameter_list = list(itertools.product(k_largest_list, lambda_red_list, gamma_sparsity_list))

    def create_BQM(relevance_scores, correlation_matrix, lambda_red, gamma_sparsity):
        Q = np.zeros((n_features, n_features))
        np.fill_diagonal(Q, -relevance_scores + gamma_sparsity)
        Q += lambda_red * correlation_matrix.toarray()
        # Updated for newer dimod API
        BQM = dimod.BinaryQuadraticModel(Q, dimod.BINARY)
        BQM.normalize()
        return BQM

    try:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        result_list = []
        for k_largest, lambda_red, gamma_sparsity in hyperparameter_list:
            BQM = create_BQM(relevance_scores, correlation_matrix, lambda_red, gamma_sparsity)
            result = _run_experiment((k_largest, lambda_red, gamma_sparsity), BQM, ICM, URM_train, evaluator_validation)
            result_list.append(result)
        pool.close()
        pool.join()
    except Exception as e:
        logger.error(f"Multiprocessing failed: {e}. Falling back to single-threaded execution.")
        result_list = []
        for k_largest, lambda_red, gamma_sparsity in hyperparameter_list:
            BQM = create_BQM(relevance_scores, correlation_matrix, lambda_red, gamma_sparsity)
            result = _run_experiment((k_largest, lambda_red, gamma_sparsity), BQM, ICM, URM_train, evaluator_validation)
            result_list.append(result)

    best_configuration = None
    best_selected_dict = None
    best_NDCG = None

    for i, data in enumerate(result_list):
        k_largest, lambda_red, gamma_sparsity, NDCG, n_selected, selected_dict = data
        logger.info(f"Processed {i+1}/{len(hyperparameter_list)}: k_largest={k_largest}, lambda_red={lambda_red:.2E}, gamma_sparsity={gamma_sparsity:.2E}, NDCG={NDCG:.4f}")
        if best_NDCG is None or best_NDCG < NDCG:
            best_NDCG = NDCG
            best_configuration = (k_largest, lambda_red, gamma_sparsity)
            best_selected_dict = selected_dict

    k_largest, lambda_red, gamma_sparsity = best_configuration
    logger.info(f"Optimal configuration: k_largest={k_largest}, lambda_red={lambda_red:.2E}, gamma_sparsity={gamma_sparsity:.2E}, NDCG@10={best_NDCG:.4f}")
    logger.info(f"Improvement over baseline: {best_NDCG - baseline_NDCG:.4f}")

    selected_features = [key for key, value in best_selected_dict.items() if value == 1]
    if len(selected_features) > 1:
        selected_ICM = ICM[:, selected_features]
        corr_matrix = np.corrcoef(selected_ICM.toarray(), rowvar=False)
        to_remove = set()
        for i in range(len(selected_features)):
            for j in range(i + 1, len(selected_features)):
                if abs(corr_matrix[i, j]) > 0.8:
                    to_remove.add(selected_features[j])
        selected_features = [f for f in selected_features if f not in to_remove]

    file_name = "{task}_{dataset}_{method}_{groupname}_{submissionID}.txt".format(
        task="1B",
        dataset="{}_ICM".format(n_features),
        method="CQFS",
        groupname="example-group",
        submissionID="000"
    )
    
    with open(file_name, 'w') as f:
        f.write("\n".join(map(str, selected_features)))