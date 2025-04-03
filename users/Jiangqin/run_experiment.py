#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/05/2024

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import dimod
from neal import SimulatedAnnealingSampler
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

def _run_experiment(hyperparameters, BQM, ICM, URM_train, evaluator_validation):
    k_largest, penalty = hyperparameters

    BQM_k = dimod.generators.combinations(BQM.num_variables, k_largest) * penalty
    BQM_k.update(BQM)

    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(BQM_k, num_reads=200)
    selected_dict = sampleset.first.sample
    selected_features = [key for key, value in selected_dict.items() if value == 1]

    if not selected_features:
        return k_largest, penalty, 0.0, 0, selected_dict

    ICM_selected = ICM[:, selected_features]
    recommender = ItemKNNCBFRecommender(URM_train, ICM_selected)
    recommender.fit(topK=100, shrink=5, similarity='cosine', normalize=True)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender)
    n_selected = len(selected_features)

    print(f"k_largest {k_largest}, penalty {penalty:.2E}: NDCG@10 is {result_df.loc[10, 'NDCG']:.4f}, selected {n_selected}")

    return k_largest, penalty, result_df.loc[10, "NDCG"], n_selected, selected_dict