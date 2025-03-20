import numpy as np
import pandas as pd
import math
import dimod
import itertools

def shannon_entropy(p):
    """Shannon entropy H(X) is the negative sum of P(X)log(P(X)) for probability
    distribution P(X).
    """
    p = p.flatten()
    return -sum(pi*np.log2(pi) for pi in p if pi)


def conditional_shannon_entropy(p, *conditional_indices):
    """Conditional Shannon entropy H(X|Y) = H(X,Y) - H(Y)."""

    # Sanity check on validity of conditional_indices.  In particular,
    # try to trap issues in which dimensions have been removed from
    # probability table through marginalization, but
    # conditional_indices were not updated accordingly.
    assert(all(ci < p.ndim for ci in conditional_indices))

    axis = tuple(i for i in np.arange(len(p.shape))
                 if i not in conditional_indices)

    return shannon_entropy(p) - shannon_entropy(np.sum(p, axis=axis))


def mutual_information(prob, j):
    """Mutual information between variables X and variable Y.
    Calculated as I(X; Y) = H(X) - H(X|Y)."""

    return (shannon_entropy(np.sum(prob, axis=j))
            - conditional_shannon_entropy(prob, j))


def conditional_mutual_information(p, j, *conditional_indices):
    """Mutual information between variables X and variable Y conditional on variable Z.
    Calculated as I(X;Y|Z) = H(X|Z) - H(X|Y,Z)"""

    # Compute an updated version of the conditional indices for use
    # when the probability table is marginalized over dimension j.
    # This marginalization removes one dimension, so any conditional
    # indices pointing to dimensions after this one must be adjusted
    # accordingly.
    marginal_conditional_indices = [i-1 if i > j else i for i in conditional_indices]

    return (conditional_shannon_entropy(np.sum(p, axis=j), *marginal_conditional_indices)
            - conditional_shannon_entropy(p, j, *conditional_indices))


def prob(dataset, max_bins=100):
    """Joint probability distribution P(X) for the given data."""

    # bin by the number of different values per feature
    num_rows, num_columns = dataset.shape
    bins = [min(len(np.unique(dataset[:, ci])), max_bins) for ci in range(num_columns)]

    prob, _ = np.histogramdd(dataset, bins)
    return prob / np.sum(prob)


def mutual_information_bqm(X_train, y_train):
    """Builds the BQM according to Mutual Information and
    Conditional Mutual Information"""
    BQM = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    for f_label in X_train.columns:
        f_MI = mutual_information(prob(pd.concat([y_train, X_train[f_label]], axis=1).values), 1)

        BQM.add_variable(f_label, -f_MI)

    for f1_label, f2_label in itertools.permutations(X_train.columns, 2):
        f_CMI = conditional_mutual_information(prob(pd.concat([y_train, X_train[f1_label], X_train[f2_label]], axis=1).values), 1, 2)

        BQM.add_interaction(f1_label, f2_label, -f_CMI)

    return BQM


def maximum_energy_delta(bqm):
    """Compute conservative bound on maximum change in energy when flipping a single variable"""
    return max(abs(bqm.get_linear(i))
               + sum(abs(bqm.get_quadratic(i,j))
                     for j in bqm.iter_neighbors(i))
               for i in iter(bqm.variables))

def mi_bqm_with_penalty(X_train, y_train, k):
    bqm = mutual_information_bqm(X_train, y_train)
    penalty = maximum_energy_delta(bqm)
    kbqm = dimod.generators.combinations(bqm.variables, k, strength=penalty)
    kbqm.update(bqm)
    return kbqm

