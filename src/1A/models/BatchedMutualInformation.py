import os
import numpy as np
import pandas as pd
import dimod
import itertools
from multiprocessing import Pool

def shannon_entropy(p):
    p = p.flatten()
    return -sum(pi * np.log2(pi) for pi in p if pi)

def conditional_shannon_entropy(p, *conditional_indices):
    axis = tuple(i for i in np.arange(len(p.shape)) if i not in conditional_indices)
    return shannon_entropy(p) - shannon_entropy(np.sum(p, axis=axis))

def mutual_information(prob, j):
    return shannon_entropy(np.sum(prob, axis=j)) - conditional_shannon_entropy(prob, j)

def conditional_mutual_information(p, j, *conditional_indices):
    marginal_conditional_indices = [i - 1 if i > j else i for i in conditional_indices]
    return (conditional_shannon_entropy(np.sum(p, axis=j), *marginal_conditional_indices)
            - conditional_shannon_entropy(p, j, *conditional_indices))

def prob(dataset, max_bins=100):
    num_rows, num_columns = dataset.shape
    bins = [min(len(np.unique(dataset[:, ci])), max_bins) for ci in range(num_columns)]
    prob, _ = np.histogramdd(dataset, bins)
    return prob / np.sum(prob)

def compute_mi_for_batch(batch):
    X_train, y_train = batch
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    for f_label in X_train.columns:
        f_MI = mutual_information(prob(pd.concat([y_train, X_train[f_label]], axis=1).values), 1)
        bqm.add_variable(f_label, -f_MI)
    return bqm

def compute_cmi_for_batch(batch):
    X_train, y_train = batch
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    for f1_label, f2_label in itertools.combinations(X_train.columns, 2):
        f_CMI = conditional_mutual_information(prob(pd.concat([y_train, X_train[f1_label], X_train[f2_label]], axis=1).values), 1, 2)
        bqm.add_interaction(f1_label, f2_label, -f_CMI)
    return bqm

def mutual_information_bqm_parallel(X_train, y_train, num_batches=10):
    feature_batches = np.array_split(X_train.columns, num_batches)
    batches = [(X_train[batch], y_train) for batch in feature_batches]
    
    with Pool() as pool:
        mi_bqms = pool.map(compute_mi_for_batch, batches)
        cmi_bqms = pool.map(compute_cmi_for_batch, batches)
    
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    for bq in mi_bqms + cmi_bqms:
        bqm.update(bq)
    
    return bqm

def maximum_energy_delta(bqm):
    return max(abs(bqm.get_linear(i)) + sum(abs(bqm.get_quadratic(i, j)) for j in bqm.iter_neighbors(i)) for i in bqm.variables)

def mi_bqm_with_penalty_parallel(X_train, y_train, k, num_batches=10):
    bqm = mutual_information_bqm_parallel(X_train, y_train, num_batches)
    penalty = maximum_energy_delta(bqm)
    kbqm = dimod.generators.combinations(bqm.variables, k, strength=penalty)
    kbqm.update(bqm)
    return kbqm