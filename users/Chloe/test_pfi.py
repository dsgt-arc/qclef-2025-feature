from models.PermutationFeatureImportance import PermutationFeatureImportance
import numpy as np
import pandas as pd
import sys
import os
import datetime
import time
import dimod
from models.BaselineModel import select_features_rfe
from models.MutualInformation import mi_bqm_with_penalty
from tests.scoring_model import train_lambdamart, evaluate_ndcg, calc_ndcg_score

mq2007_train = pd.read_csv("MQ2007_train.csv")
mq2007_test = pd.read_csv("MQ2007_dev.csv")

results_path = "res.txt"

X_train_mq = mq2007_train.drop(columns=['relevance', 'query_id'])
X_test_mq = mq2007_test.drop(columns=['relevance', 'query_id'])
y_train_mq = mq2007_train['relevance']
y_test_mq = mq2007_test['relevance']
qids_train = mq2007_train['query_id']
qids_test = mq2007_test['query_id']

# Running the Baseline
# print("Running the baseline method: RFE")
# start_time = time.time()
# selected_features_baseline = select_features_rfe(X_train_mq, y_train_mq, 10)
# end_time = time.time()
# execution_time = end_time - start_time
# # print("Selected features using baseline RFE:")
# print(selected_features_baseline)
# ndcg_score_baseline = calc_ndcg_score(X_train_mq, X_test_mq, y_train_mq, y_test_mq, qids_train, qids_test, selected_features_baseline)
# print("nDCG score for baseline: ", ndcg_score_baseline)
# with open(results_path, "a") as f:  # 'a' mode appends instead of overwriting
#     f.write("Results for baseline method: RFE")
#     f.write(f"Results recorded at: {datetime.datetime.now()}\n")
#     f.write(f"Execution Time: {execution_time:.2f} seconds\n")  # Log execution time
#     f.write(f"nDCG Score: {ndcg_score_baseline:.4f}\n\n")
#     f.write("Selected Features:\n")
#     f.write(", ".join(selected_features_baseline) + https://m365.cloud.microsoft/launch/visio?auth=2"\n\n")
#     f.write("-" * 50 + "\n")  # Separator for clarity

start_time = time.time()
# Simulated annealing
print("Running the SA method: Permutation Feature Importance")
Pfi_model = PermutationFeatureImportance(X_train_mq, y_train_mq, qids_train)
feature_importances = Pfi_model.fit(k=10)
selected_features_SA, X_reduced = Pfi_model.get_selected_features()
X_test_reduced = X_test_mq.iloc[:, selected_features_SA]
num_reads = 5000

# For running SA locally
end_time = time.time()
execution_time = end_time - start_time 
# print("Selected features using SA mutual information: ")
print(selected_features_SA)
ndcg_score_sa = Pfi_model.calc_ndcg_score(X_train_mq, X_test_mq, y_train_mq, y_test_mq, qids_train, qids_test)
print("nDCG score for SA: ", ndcg_score_sa)
with open(results_path, "a") as f:  # 'a' mode appends instead of overwriting
    f.write("Results for Simulated Annealing: Permutation Feature Importance")
    f.write(f"Results recorded at: {datetime.datetime.now()}\n")
    f.write(f"Execution Time: {execution_time:.2f} seconds\n")  # Log execution time
    f.write(f"nDCG Score: {ndcg_score_sa:.4f}\n\n")
    f.write("Selected Features:\n")
    f.write(", ".join([str(x) for x in selected_features_SA]) + "\n\n")
    f.write("-" * 50 + "\n")  # Separator for clarity