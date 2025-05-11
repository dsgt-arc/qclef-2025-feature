from models.PermutationFeatureImportance import PermutationFeatureImportance
import pandas as pd
import os
from models.MutualInformation import mi_bqm_with_penalty
from neal import SimulatedAnnealingSampler
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

def get_kbqm_pfi(X_train, y_train, qids_train, num_features):
    Pfi_model = PermutationFeatureImportance(X_train, y_train, qids_train)
    feature_importances = Pfi_model.fit(k=num_features)
    kbqm = Pfi_model.BQM

    return kbqm

def get_kbqm_mi(X_train, y_train, num_features):
    kbqm = mi_bqm_with_penalty(X_train, y_train, num_features)

    return kbqm

def run_SA(kbqm, run_name, num_reads):
    sampler_SA = SimulatedAnnealingSampler()

    response_SA=qa.submit(sampler_SA,
        SimulatedAnnealingSampler.sample,
        kbqm,
        label=f'1A - MQ2007 - {run_name} - SA',
        num_reads=num_reads
    )

def run_QA(kbqm, run_name, num_reads):
    print("Running the QA method: Permutation Feature Importance")
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
        for feature in selected_features:
            file.write(f"{feature}\n")

def run_submission(submission_type, method, num_features, run_name, num_reads):
    if submission_type=="mi":
        kbqm = get_kbqm_mi(X_train_mq, y_train_mq, num_features)
    elif submission_type=="pfi":
        kbqm = get_kbqm_pfi(X_train_mq, y_train_mq, qids_train, num_features)
    else:
        print("Invalid submission type")
        return
    
    if method=="SA":
        selected_features = run_SA(kbqm, run_name, num_reads)
    elif method=="QA":
        selected_features = run_QA(kbqm, run_name, num_reads)

    write_submission(selected_features, submissionID=run_name)


#--------Submission-------------

filepath = 'team_workspace/1A/outputs/'
dataset = 'MQ2007'
method = 'SA'
submission_type = 'mi'
run_name = 'mii-10-features'
num_features=10
num_reads=5000

run_submission(submission_type=submission_type, method=method, num_features=num_features, run_name=run_name, num_reads=num_reads)