from models.PermutationFeatureImportance import PermutationFeatureImportance
import pandas as pd
import os
from models.MutualInformation import mi_bqm_with_penalty
from neal import SimulatedAnnealingSampler
from tests.scoring_model import calc_ndcg_score
from qclef import qa_access as qa
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

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

def get_kbqm_pfi(X_train, y_train, qids_train, num_features, type):
    print("Generating kbqm for Permutation Feature Importance")
    Pfi_model = PermutationFeatureImportance(X_train, y_train, qids_train)
    feature_importances = Pfi_model.fit(k=num_features, type=type)
    kbqm = Pfi_model.BQM

    return kbqm

def get_kbqm_mi(X_train, y_train, num_features):
    print("Generating kbqm for Mutual Information")
    kbqm = mi_bqm_with_penalty(X_train, y_train, num_features)

    return kbqm

def run_SA(kbqm, run_name, num_reads, env="qclef"):
    print("Running the SA method")
    if env=="local":
        sampler_SA = SimulatedAnnealingSampler()
        response_SA = sampler_SA.sample(kbqm, num_reads=num_reads, seed=42)
    else:
        sampler_SA = SimulatedAnnealingSampler()

        response_SA=qa.submit(sampler_SA,
            SimulatedAnnealingSampler.sample,
            kbqm,
            label=f'1A - MQ2007 - {run_name} - SA',
            num_reads=num_reads
        )
    
    selected_features = response_SA.first.sample
    return selected_features

def run_QA(kbqm, run_name, num_reads):
    print("Running the QA method")
    sampler_QA=EmbeddingComposite(DWaveSampler())

    response_QA=qa.submit(sampler_QA,
        EmbeddingComposite.sample,
        kbqm,
        label=f'1A - MQ2007 - {run_name} - QA',
        num_reads=num_reads
    )

    selected_features = response_QA.first.sample
    return selected_features

def write_submission(selected_features, submissionID):
    filename = f"{filepath}{'1A'}_{dataset}_{method}_{'ds-at-gt-qclef'}_{submissionID}.txt"
            
    with open(filename, 'w') as file:
        # Write the keys
        for feature, selected in selected_features.items():
            if selected:
                file.write(f"{feature}\n")

def run_submission(submission_type, method, num_features, run_name, num_reads, env="qclef", type="cpfi"):
    if submission_type=="mi":
        kbqm = get_kbqm_mi(X_train_mq, y_train_mq, num_features)
    elif submission_type=="pfi":
        kbqm = get_kbqm_pfi(X_train_mq, y_train_mq, qids_train, num_features, type)
    else:
        print("Invalid submission type")
        return
    
    if method=="SA":
        selected_features = run_SA(kbqm, run_name, num_reads, env)
    elif method=="QA":
        selected_features = run_QA(kbqm, run_name, num_reads)
    else:
        print("Invalid method")

    if env=="local":
        selected_feature_names = [feature for feature, selected in selected_features.items() if selected == 1]
        ndcg_score_sa = calc_ndcg_score(X_train_mq, X_test_mq, y_train_mq, y_test_mq, qids_train, qids_test, selected_feature_names)
        print(f"nDCG score for {run_name}: {ndcg_score_sa}")

    write_submission(selected_features, submissionID=run_name)


#--------Submission-------------

filepath = 'team_workspace/1A/outputs/'
dataset = 'MQ2007'
method = 'SA'
submission_type = 'pfi'
num_features=[5, 10, 15, 20, 25, 30, 35, 40]
num_reads=1000

for features in num_features:
    run_name = f'pfi-k-{features}-cmi'
    run_submission(submission_type=submission_type, method=method, num_features=features, run_name=run_name, num_reads=num_reads, env="local", type="cmi")