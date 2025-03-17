import numpy as np
import pandas as pd
import sys
import os
import datetime
import time
import dimod
from models.BaselineModel import select_features_rfe
from models.MutualInformation import mi_bqm_with_penalty
from scoring_model import train_lambdamart, evaluate_ndcg, calc_ndcg_score
from neal import SimulatedAnnealingSampler
# from qclef import qa_access as qa
# import dwave.inspector
# from dwave.system.samplers import DWaveSampler
# from dwave.system.composites import EmbeddingComposite




train_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "MQ2007_train.csv")
val_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "MQ2007_dev.csv")
results_path = os.path.join(os.path.dirname(__file__), "..", "results", "1A_MQ2007_MI.txt") 

mq2007_train = pd.read_csv(train_data_path)
mq2007_test = pd.read_csv(val_data_path)

X_train_mq = mq2007_train.drop(columns=['relevance', 'query_id'])
X_test_mq = mq2007_test.drop(columns=['relevance', 'query_id'])
y_train_mq = mq2007_train['relevance']
y_test_mq = mq2007_test['relevance']
qids_train = mq2007_train['query_id']
qids_test = mq2007_test['query_id']

# Running the Baseline
print("Running the baseline method: RFE")
start_time = time.time()
selected_features_baseline = select_features_rfe(X_train_mq, y_train_mq, 10)
end_time = time.time()
execution_time = end_time - start_time
# print("Selected features using baseline RFE:")
# print(selected_features_baseline)
ndcg_score_baseline = calc_ndcg_score(X_train_mq, X_test_mq, y_train_mq, y_test_mq, qids_train, qids_test, selected_features_baseline)
print("nDCG score for baseline: ", ndcg_score_baseline)
with open(results_path, "a") as f:  # 'a' mode appends instead of overwriting
    f.write("Results for baseline method: RFE")
    f.write(f"Results recorded at: {datetime.datetime.now()}\n")
    f.write(f"Execution Time: {execution_time:.2f} seconds\n")  # Log execution time
    f.write(f"nDCG Score: {ndcg_score_baseline:.4f}\n\n")
    f.write("Selected Features:\n")
    f.write(", ".join(selected_features_baseline) + "\n\n")
    f.write("-" * 50 + "\n")  # Separator for clarity

start_time = time.time()
# Get the QUBO
kbqm = mi_bqm_with_penalty(X_train_mq, y_train_mq, 10)

# Simulated annealing
print("Running the SA method: Mutual Information")
sampler_SA = SimulatedAnnealingSampler()
num_reads = 5000

# # For running in the qclef environment
response_SA=qa.submit(sampler,
    SimulatedAnnealingSampler.sample,
    kbqm,
    label='1A - MQ2007 - MI Feature Selection - SA',
    num_reads=num_reads
)

# For running SA locally
# response_SA = sampler_SA.sample(kbqm, num_reads=num_reads, seed=42)
selected_features_SA = response_SA.first.sample
end_time = time.time()
execution_time = end_time - start_time 
# Convert the dict to a list of selected feature names
selected_features_SA = [feature for feature, selected in selected_features_SA.items() if selected == 1]
# print("Selected features using SA mutual information: ")
# print(selected_features_SA)
ndcg_score_sa = calc_ndcg_score(X_train_mq, X_test_mq, y_train_mq, y_test_mq, qids_train, qids_test, selected_features_SA)
print("nDCG score for SA: ", ndcg_score_sa)
with open(results_path, "a") as f:  # 'a' mode appends instead of overwriting
    f.write("Results for Simulated Annealing: Mutual Information")
    f.write(f"Results recorded at: {datetime.datetime.now()}\n")
    f.write(f"Execution Time: {execution_time:.2f} seconds\n")  # Log execution time
    f.write(f"nDCG Score: {ndcg_score_sa:.4f}\n\n")
    f.write("Selected Features:\n")
    f.write(", ".join(selected_features_SA) + "\n\n")
    f.write("-" * 50 + "\n")  # Separator for clarity
# # Quantum Annealing
start_time = time.time()
sampler_QA=EmbeddingComposite(DWaveSampler())

response_QA=qa.submit(sampler_QA,
    EmbeddingComposite.sample,
    kbqm,
    label='1A - MQ2007 - MI Feature Selection - QA',
    num_reads=num_reads
)

selected_features_QA = response_QA.first.sample
end_time = time.time()
execution_time = end_time - start_time
selected_features_QA = [feature for feature, selected in selected_features_QA.items() if selected == 1]

# Calculate nDCG score
ndcg_score_qa = calc_ndcg_score(X_train_mq, X_test_mq, y_train_mq, y_test_mq, qids_train, qids_test, selected_features_QA)
print("nDCG score for SA: ", ndcg_score_qa)
with open(results_path, "a") as f:  # 'a' mode appends instead of overwriting
    f.write("Results for Simulated Annealing: Mutual Information")
    f.write(f"Results recorded at: {datetime.datetime.now()}\n")
    f.write(f"Execution Time: {execution_time:.2f} seconds\n")  # Log execution time
    f.write(f"nDCG Score: {ndcg_score_qa:.4f}\n\n")
    f.write("Selected Features:\n")
    f.write(", ".join(selected_features_QA) + "\n\n")
    f.write("-" * 50 + "\n")  # Separator for clarity


# Prepare for submission
# TODO
