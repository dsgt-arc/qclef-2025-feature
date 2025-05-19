from models.PermutationFeatureImportance import PermutationFeatureImportance
import pandas as pd
import os
from models.MutualInformation import mi_bqm_with_penalty
from neal import SimulatedAnnealingSampler
from tests.scoring_model import calc_ndcg_score
import numpy as np
from sklearn.metrics import ndcg_score
import xgboost as xgb
from collections import Counter
from collections import defaultdict
import random
from joblib import Parallel, delayed

def get_data(dataset):
    if dataset == "MQ2007":
        X_train = pd.read_csv("d:/repos/qclef-2025-feature/src/1A/data/mq2007_train.csv")
        X_test = pd.read_csv("d:/repos/qclef-2025-feature/src/1A/data/mq2007_dev.csv")

        y_train = X_train['relevance']
        y_test = X_test['relevance']
        qids_train = X_train['query_id']
        qids_test = X_test['query_id']
        X_train = X_train.drop(columns=['relevance', 'query_id'])
        X_test = X_test.drop(columns=['relevance', 'query_id'])

    elif dataset=="ISTELLA":
        X_train = pd.read_parquet("d:/repos/qclef-2025-feature/src/1A/data/ISTELLA_train.parquet")
        X_test = pd.read_parquet("d:/repos/qclef-2025-feature/src/1A/data/ISTELLA_dev.parquet")

        y_train = X_train['relevance']
        y_test = X_test['relevance']
        qids_train = X_train['query_id']
        qids_test = X_test['query_id']
        X_train = X_train.drop(columns=['relevance', 'query_id'])
        X_test = X_test.drop(columns=['relevance', 'query_id'])
        
    else:
        raise ValueError("Invalid input for dataset")

    return X_train, y_train, qids_train, X_test, y_test, qids_test

def calc_ndcg_score_xgb(X_train, X_test, y_train, y_test, qids_train, qids_test, selected_features):
    # Select only the columns that match the selected feature names
    X_train_selected = X_train[selected_features].values  # Convert to NumPy array
    X_test_selected = X_test[selected_features].values  # Convert to NumPy array

    y_train = np.array(y_train)

    # Sort data based on query IDs
    sorted_indices = np.argsort(qids_train)
    X_train_selected = X_train_selected[sorted_indices]
    y_train = y_train[sorted_indices]
    qids_train = np.array(qids_train)[sorted_indices]

    # Compute group sizes for ranking
    unique_qids, group_sizes = np.unique(qids_train, return_counts=True)

    # Train LambdaMART and get predictions
    Pfi_model = PermutationFeatureImportance(X_train, y_train, qids_train)
    model = Pfi_model.train_xgboost_surrogate(X_train_selected, y_train, group_sizes)

    # Compute nDCG@10
    y_pred = model.predict(xgb.DMatrix(X_test_selected))
    
    ndcg = ndcg_score([y_test], [y_pred], k=10)
    return ndcg

def get_kbqm_pfi(X_train, y_train, qids_train, num_features, _type):
    print("Generating kbqm for Permutation Feature Importance")
    Pfi_model = PermutationFeatureImportance(X_train, y_train, qids_train)
    feature_importances = Pfi_model.fit(k=num_features, _type=_type)
    kbqm = Pfi_model.BQM

    return kbqm

def get_kbqm_mi(X_train, y_train, num_features):
    print("Generating kbqm for Mutual Information")
    kbqm = mi_bqm_with_penalty(X_train, y_train, num_features)

    return kbqm

def run_SA(kbqm, run_name, num_reads):
    print("Running the SA method")
    sampler_SA = SimulatedAnnealingSampler()
    response_SA = sampler_SA.sample(kbqm, num_reads=num_reads, seed=42)
    
    selected_features = response_SA.first.sample
    return selected_features

def write_submission(selected_features, submissionID):
    filename = f"{filepath}{'1A'}_{dataset}_{method}_{'ds-at-gt-qclef'}_{submissionID}.txt"
            
    with open(filename, 'w') as file:
        # Write the keys
        for feature in selected_features:
            file.write(f"{feature}\n")

def run_submission(submission_type, method, num_features, run_name, num_reads, dataset, _type="cpfi"):
    X_train, y_train, qids_train, X_test, y_test, qids_test = get_data(dataset)

    if submission_type=="mi":
        kbqm = get_kbqm_mi(X_train, y_train, num_features)
    elif submission_type=="pfi":
        kbqm = get_kbqm_pfi(X_train, y_train, qids_train, num_features, _type)
    else:
        raise ValueError("Invalid submission type")
        return
    
    if method=="SA":
        selected_features, problem_id = run_SA(kbqm, run_name, num_reads)
    else:
        raise ValueError("Invalid method")

    selected_feature_names = [feature for feature, selected in selected_features.items() if selected == 1]
    ndcg_score_sa  = calc_ndcg_score_xgb(X_train, X_test, y_train, y_test, qids_train, qids_test, selected_feature_names)
    print(f"nDCG score for {run_name}: {ndcg_score_sa}")

    write_submission(selected_features, submissionID=run_name, problem_id=problem_id)

def _process_batch(i, batch, X_train, y_train, qids_train, submission_type, method, num_features, run_name, num_reads, _type):
    print(f"\n Running batch {i+1}: {batch}")
    batch_feature_names = [X_train.columns[idx] for idx in batch]
    X_batch = X_train[batch_feature_names]

    X_batch = X_batch.replace([np.inf, -np.inf], np.nan)
    X_batch = X_batch.clip(lower=-1e10, upper=1e10)
    X_batch = X_batch.fillna(0)

    # Train and select features using current batch
    if submission_type=="mi":
        kbqm = get_kbqm_mi(X_batch, y_train, num_features)
    elif submission_type=="pfi":
        kbqm = get_kbqm_pfi(X_batch, y_train, qids_train, num_features, _type)
    else:
        raise ValueError("Invalid submission type")
        
    if method=="SA":
        selected_features = run_SA(kbqm, run_name, num_reads)
    else:
        raise ValueError("Invalid method")

    return selected_features


def run_batched_submission(submission_type, method, batch_size, feature_reps, num_features, run_name, num_reads, dataset, _type="cpfi"):
    """
    Runs a batched submission and writes the results file to the filepath.

    Parameters:
        submission_type (str): Thhis can either be "mi" for Mutual Information or "pfi" for Permutation Feature Importance.
        method (str): either "SA" or "QA".
        batch_size (int): Size of the batches. Must be greater than num_features.
        feature_reps (int):  How many times each feature should appear in total throughout the various batches.
        num_features (int): Number of features to be chosen from each batch. Must bbe less than batch_size.
        run_name (str): Run name, used for the results filename.
        num_reads (int): Num reads when running SA/QA.
        dataset (str): Can be either "MQ2007" or "ISTELLA".
        _type (str): To be used when submission_type="pfi" to determine whhat should go on the off-diagonals. Can have a 
            value of "cmi" for Conditional Mutual Information or cpfi for Conditional Permutation Feature Importance.
    """
    X_train, y_train, qids_train, X_test, y_test, qids_test = get_data(dataset)
    total_n_features = len(X_train.columns)
    n_batches = int((total_n_features/batch_size)*feature_reps)
    batches, batch_appearance_count = create_balanced_random_batches(n_features = total_n_features, batch_size=batch_size, n_batches=n_batches)

    print(f"\n Starting the run for {n_batches} batches.")
    results = Parallel(n_jobs=-1)(
        delayed(_process_batch)(
            i, batch, X_train, y_train, qids_train, submission_type, method, num_features, run_name, num_reads, _type
        )
        for i, batch in enumerate(batches)
    )
    print("batches done")

    vote_counter = defaultdict(int)
    for selected_features in results:
        for feat_name, selected in selected_features.items():
            if selected:
                vote_counter[feat_name] += 1
    print("vote counter done")

    if submission_type == "pfi":
        print("starting global")
        Pfi_model_global = PermutationFeatureImportance(X_train, y_train, qids_train)
        global_importances = Pfi_model_global._get_feature_importances()
        normalized_votes = {feat: ((vote_counter[feat] / batch_appearance_count[int(feat)-1]), global_importances[0][int(feat)-1]) for feat in vote_counter}
        sorted_features = sorted(normalized_votes.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
        print("global done")
    else:
        normalized_votes = {feat: (vote_counter[feat] / batch_appearance_count[int(feat)-1]) for feat in vote_counter}
        sorted_features = sorted(normalized_votes.items(), key=lambda x: (x[1]), reverse=True)    

    selected_feature_names = [feat for feat, _ in sorted_features[:num_features]]

    print("start scoring")
    ndcg_score_sa  = calc_ndcg_score_xgb(X_train, X_test, y_train, y_test, qids_train, qids_test, selected_feature_names)
    print(f"nDCG score for {run_name}: {ndcg_score_sa}")

    write_submission(selected_feature_names, submissionID=run_name)

def create_balanced_random_batches(n_features, batch_size, n_batches, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    total_slots = n_batches * batch_size
    avg_repeats = total_slots // n_features
    extra = total_slots % n_features  # remainder features to distribute

    # Create initial feature list with balanced repeats
    feature_pool = list(range(n_features)) * avg_repeats
    feature_pool += random.sample(range(n_features), extra)  # distribute extras randomly

    random.shuffle(feature_pool)

    batches = []
    used_indices = set()

    for _ in range(n_batches):
        batch = set()
        while len(batch) < batch_size:
            if not feature_pool:
                # Replenish with random features (not necessarily unique from previous pool)
                feature_pool = random.choices(range(n_features), k=batch_size * 2)
                random.shuffle(feature_pool)

            feat = feature_pool.pop()
            batch.add(feat)

        batches.append(sorted(list(batch)))

    flat = [feat for batch in batches for feat in batch]

    return batches, Counter(flat)

#--------Submission-------------

filepath = 'd:/repos/qclef-2025-feature/src/1A/outputs/'
dataset = 'ISTELLA'
method = 'SA'
submission_type = 'pfi'
num_features=[120]
num_reads=1000

for features in num_features:
    run_name = f'pfi-k-{features}'
    run_batched_submission(submission_type=submission_type, method=method, batch_size = 150, feature_reps = 5, num_features=features, run_name=run_name, num_reads=num_reads, dataset=dataset, _type="cmi")