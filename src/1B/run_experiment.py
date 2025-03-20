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

def _filter_ICM(ICM_all, selection_dict):

    selected_flag = np.zeros(ICM_all.shape[1], dtype=bool)

    for key, value in selection_dict.items():
        if value == 1:
            selected_flag[int(key)] = True

    selected_ICM = ICM_all.tocsc()[:,selected_flag].tocsr()

    return selected_ICM, selected_flag.sum()



def _run_experiment(hyperparameters, BQM, ICM, URM_train, evaluator_validation):

    k_largest, penalty = hyperparameters

    BQM_k = dimod.generators.combinations(BQM.num_variables, k_largest)*penalty
    BQM_k.update(BQM)

    sampler = SimulatedAnnealingSampler()

    sampleset = sampler.sample(BQM_k, num_reads = 1000)
    ICM_selected, n_selected = _filter_ICM(ICM, sampleset.first.sample)

    if n_selected == 0:
        return k_largest, penalty, 0.0, n_selected, sampleset.first.sample

    recommender_instance = ItemKNNCBFRecommender(URM_train, ICM_selected)
    recommender_instance.fit(topK = 100, shrink = 5, similarity = 'cosine', normalize = True)

    result_df, result_string = evaluator_validation.evaluateRecommender(recommender_instance)

    print("k_largest {}, penalty {:.2E}: NDCG@10 is {:.4f}, selected {}".format(k_largest, penalty, result_df.loc[10, "NDCG"], n_selected))

    return k_largest, penalty, result_df.loc[10, "NDCG"], n_selected, sampleset.first.sample

