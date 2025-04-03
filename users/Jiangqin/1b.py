import pandas as pd
from Evaluation.Evaluator import EvaluatorHoldout
import scipy.sparse as sps
import numpy as np
from sklearn.model_selection import train_test_split
import itertools, multiprocessing
from functools import partial
import gc

import dimod
from neal import SimulatedAnnealingSampler
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.DataIO import DataIO

from run_experiment import _run_experiment

def _from_df_to_sparse(URM_df, n_users, n_items):
    URM_sps = sps.csr_matrix((np.ones(len(URM_df)),
                              (URM_df["UserID"].values, URM_df["ItemID"].values)),
                             shape=(n_users, n_items))
    return URM_sps

def compute_similarity_in_batches(matrix, batch_size=1000):
    n_items = matrix.shape[0]
    similarity = sps.csr_matrix((n_items, n_items))
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        batch = matrix[start:end]
        similarity[start:end, :] = batch.dot(matrix.T)
    return similarity

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

    URM_train_df, URM_validation_df = train_test_split(URM_all_df, test_size=0.20)
    URM_train = _from_df_to_sparse(URM_train_df, n_users, n_items)
    URM_validation = _from_df_to_sparse(URM_validation_df, n_users, n_items)
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])

    recommender_full = ItemKNNCBFRecommender(URM_train, ICM)
    recommender_full.fit(shrink=5, topK=100, similarity='cosine')
    result_df_full, _ = evaluator_validation.evaluateRecommender(recommender_full)
    baseline_NDCG = result_df_full.loc[10, "NDCG"]
    print(f"Baseline NDCG@10 with all {n_features} features: {baseline_NDCG:.4f}")
    gc.collect()

    similarity_collaborative = compute_similarity_in_batches(URM_train.T)
    similarity_content = compute_similarity_in_batches(ICM)
    similarity_collaborative_bin = similarity_collaborative.astype(bool)
    similarity_content_bin = similarity_content.astype(bool)
    Keep = similarity_collaborative_bin.multiply(similarity_content_bin)

    FPM = ICM.T.dot(Keep).dot(ICM)
    BQM = dimod.BinaryQuadraticModel(FPM.toarray(), "BINARY")
    BQM.normalize()
    gc.collect()

    # Convert itertools.product to a list for len() support
    # hyperparameter_list = list(itertools.product(range(75, n_features, 25), [1e-3, 1e-1, 1, 10]))
    hyperparameter_list = list(itertools.product(range(75, n_features, 5), 
                                        [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
    _run_experiment_partial = partial(_run_experiment,
                                      BQM=BQM,
                                      ICM=ICM,
                                      URM_train=URM_train,
                                      evaluator_validation=evaluator_validation)

    try:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        result_list = pool.map(_run_experiment_partial, hyperparameter_list, chunksize=1)
        pool.close()
        pool.join()
    except Exception as e:
        print(f"Multiprocessing failed: {e}. Falling back to single-threaded execution.")
        result_list = [_run_experiment_partial(hp) for hp in hyperparameter_list]

    best_configuration = None
    best_selected_dict = None
    best_NDCG = None

    for i, data in enumerate(result_list):
        k_largest, penalty, NDCG, n_selected, selected_dict = data
        print(f"Processed {i+1}/{len(hyperparameter_list)}: k_largest {k_largest}, penalty {penalty}")
        if best_NDCG is None or best_NDCG < NDCG:
            best_NDCG = NDCG
            best_configuration = (k_largest, penalty)
            best_selected_dict = selected_dict

    k_largest, penalty = best_configuration
    print(f"Optimal configuration: k_largest {k_largest}, penalty {penalty}, NDCG@10 {best_NDCG:.4f}")
    print(f"Improvement over baseline: {best_NDCG - baseline_NDCG:.4f}")

    selected_features = [key for key, value in best_selected_dict.items() if value == 1]
    file_name = "{task}_{dataset}_{method}_{groupname}_{submissionID}.txt".format(
        task="1B",
        dataset="{}_ICM".format(n_features),
        method="SA",
        groupname="example-group",
        submissionID="000"
    )
    
    with open(file_name, 'w') as f:
        f.write("\n".join(map(str, selected_features)))