"""
Create a named analysis folder
- network_data.pkl
- trajectory_i.pkl
- model_accuracy.json
  - { trajectory_id : [...]}
- model_score.json
  - { trajectory_id : [...]}

- model_top_accuracy.json
  {trajectory_id : top_accuracy}

- model_top_score.json
  {trajectory_id : top_score}

trajectory_groups.json
 - {group_id : [1],
    group_id : [457, 9}

- trajectory_group_top_accuracy.json 
  - {trajectory_id : [top_accuracy]}
- trajectory_group_top_score.json 
  - {trajectory_id : [top_score]}



Network Generating Function
- Give N nodes and an edge frequency of pi_T, generate a PPI network.
1. Generate edges at a frequency of pi_T
2. Select M bait.
3. Calculate number of paths (i, j) all pairs of distance <= max_path_length
4. Remove all edges not connected to one of the bait
5. Calculate pi_obs (the observed edge frequency
6. If a pair of proteins do not interact, generate AP-MS pearson R from the null distribution
7. If a pair of proteins interact generate AP-MS pearson R from the causal distribution
8. Gather into a similarity matrix
9. Return the true_network, bait_indices, and pearson R similarity matrix.

Visualize the Synthetic Data
- Observe the distribution

Set Priors
- Set common sense priors on pi_obs, mu, and sigma

Run A MCMC chains for B steps
- Initialize different starting positions
- Index trajectories
- Save model scores

Calculate the top scoring model and most accurate model for each trajectory
- O(A x B) 
- Build the score dictionary

Plot accuracy score correlation for a representative subset
- Select V accuracy, score pairs 
- Calculate pearson R

Define trajectory groups and save

Index combinations of trajectories from 1 to A
- For each trajectory group calculate the top scoring model and most accurate model
- Calculate combinations without replacement 

Naming Conventions
"""

import jax
import jax.numpy as jnp
import numpy as np
import math
import _model_variations as mv
import pickle as pkl
import xarray as xr
import numpyro
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_value,
    init_to_uniform,
    )
import click
import pandas as pd
import matplotlib.pyplot as plt
from typing import List 
from itertools import combinations, product
from pathlib import Path


_test_key = jax.random.PRNGKey(0)

def undirected_weighted_adjacency_rng(key, N, edge_prob, Adtype=jnp.float32):
    """
    key : a jax random prng key
    """
    A = jax.random.bernoulli(key, edge_prob, shape=(N, N)).astype(Adtype)
    A = jnp.tril(A, k=-1)
    A = A + A.T
    A = A.at[jnp.diag_indices(N)].set(0)
    return A

def adjacency_and_distance_rng(key, N: int, edge_prob: float, max_path_length: int):
    """
    key : jax rng key
    N : number of nodes
    edge_prob: proportion of edges
    n_bait : int, the number of bait proteins
    max_path_length: int, the maximal path length to calculate in the distance matrix 
    """
    assert max_path_length > 2
    key, k2 = jax.random.split(key)
    A = undirected_weighted_adjacency_rng(key, N, edge_prob) 
    # Shortest paths up to 10
    f = jax.tree_util.Partial(mv.shortest_paths_up_to_N, N=max_path_length)
    f = jax.jit(f)
    D = f(A)
    diag_indices = jnp.diag_indices(N)
    #assert jnp.sum(A[diag_indices]) == 0, A[diag_indices]
    # Assign the self distances to 0
    D = D.at[diag_indices].set(0)
    assert jnp.alltrue(A == A.T), A
    return A, D 

def bait_list_rng(key, N, n_bait):
    """
    Given a number of nodes in the network N and the number of bait,
    randomly generate bait indices.
    """
    bait_indices = jax.random.choice(key, N, shape=(n_bait,), replace=False)
    return bait_indices

def jax_pairwise_matrix_2_xarray(M, row_dim="preyu", col_dim="preyv"):
    """
    Converts a 2 dimensional jax or numpy matrix to a DataArray
    with named coordinates
    """
    N, _ = M.shape
    M = xr.DataArray(M, coords = {row_dim : jnp.arange(N),
                                  col_dim : jnp.arange(N)})
    return M


def get_connected_network_from_distance(A, D, d_crit, bait_idx_array,
                                        convert_to_xr=True):
    """
    Given a distance matrix D and a list of bait proteins baits:
    remove all prey that are not within a distance d_crit to any bait protein 
    """
    # Function expects self distances to be 0, otherwise a bait would be removed from
    # itself
    N, _ = A.shape
    if convert_to_xr:
        A = jax_pairwise_matrix_2_xarray(A)
        D = jax_pairwise_matrix_2_xarray(D)
    diag_indices = jnp.diag_indices(N)
    assert jnp.sum(D.values[diag_indices]) == 0
    # Select Prey distances
    B  = D.values[bait_idx_array, :]
    # Update the bait such that the distance to self is 0
    mask = B <= d_crit 
    # Take the sum of the columns such that if any prey has a distance le d_crit
    # The prey is kept in the network
    prey_idx = np.sum(mask, axis=0) > 0
    Aconnected = A.sel(preyu=prey_idx, preyv=prey_idx)
    return Aconnected, prey_idx

def get_bait_prey_network(rng_key, n_prey, n_bait: int, d_crit,
                          edge_prob=0.2, max_path_length=21):
    """
    A jax PRNGKey
    n_bait : the number of unique bait types
    d_crit : The maximal distance to at which prey are not considered connected
    to the bait
    edge_prob : the independant edge probability
    max_path_length : the maximal length of paths for distance matrix calculation
    """
    assert n_prey > n_bait , "There must be more prey types than bait types"
    keys = jax.random.split(rng_key, 2) 
    A, D = adjacency_and_distance_rng(keys[0], n_prey, edge_prob, max_path_length)  
    bait_idx_array = bait_list_rng(keys[1], n_prey, n_bait)
    Aconnected, prey_idx = get_connected_network_from_distance(A, D, d_crit,
                           bait_idx_array)
    return Aconnected, bait_idx_array, prey_idx
     
def data_from_network_model14_rng(rng_key, A, mu=0.23, sigma=0.21, ntest=3005):
    """
    Assume a binary matrix input A 
    Given an adjacency matrix  
    mu : 0.23
    sigma : 0.21
    Assumptions : Same number of conditions
    Params:
    1. Load in the network
    2. Count the number of edges
    3. Generate that many samples from the causal distribution
    4. Generate the remaining using the null 
    """
    k1, k2 = jax.random.split(rng_key)
    A = A.astype(bool)
    causal = jax.random.normal(k1, A.shape) * sigma + mu  
    # Load in null data
    with open("shuffled_apms_correlation_matrix.pkl", "rb") as f:
        shuffled_apms_correlation_matrix = pkl.load(f)
    shuffled_apms_correlation_matrix = shuffled_apms_correlation_matrix[0:ntest, 0:ntest]
    flattened_shuffled_apms = mv.matrix2flat(
            jnp.array(shuffled_apms_correlation_matrix, dtype=jnp.float32))
    null_dist = mv.Histogram(flattened_shuffled_apms, bins=1000)
    Null = null_dist.sample(k2, sample_shape=A.shape)
    Samples = jnp.where(A, causal, Null) 
    Samples = jnp.tril(Samples, k=-1)
    Samples = Samples + Samples.T
    return Samples

def composite_connectivity_benchmark(analysis_name, Ns: List[int], Cms: List[int], num_samples=2000, num_warmup=500):
    """
    1. Vary the network size N and the size of the smallest composite 
    2. Run inference
    3. Calculate the total posterior probability mass of solutions with a edge
       weight above Cm_N - 1
    4. Plot and save results 
    """
    assert len(Ns) == len(Cms), "lists aren't of equal size"
    n = len(Ns)
    x = np.array(Cms) 
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    y_med = np.zeros(n)
    y_std = np.zeros(n)
    y_med_null = np.zeros(n)
    y_std_null = np.zeros(n)
    for i, cm in enumerate(Cms):
        cm = np.arange(cm)
        N = Ns[i]
        key = jax.random.PRNGKey(13)
        nuts = NUTS(mv.model20_test6f)
        mcmc = MCMC(nuts,num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(key, N, cm, extra_fields=('potential_energy',))
        samples = mcmc.get_samples()
        sum_of_edges = np.sum(samples['e'], axis=1)
        med = np.mean(sum_of_edges) 
        std = np.std(sum_of_edges)
        pp = np.sum(sum_of_edges >= len(cm)-1) / num_samples
        nuts = NUTS(mv.model20_test6f_null)
        mcmc = MCMC(nuts,num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(key, N, cm, extra_fields=('potential_energy',))
        samples = mcmc.get_samples()
        sum_of_edges = np.sum(samples['e'], axis=1)
        pp_null = np.sum(sum_of_edges >= len(cm)-1) / num_samples
        med_null = np.mean(sum_of_edges) 
        std_null = np.std(sum_of_edges)
        y1[i] = pp
        y2[i] == pp_null
        y_med[i] = med
        y_std[i] = std
        y_med_null[i] = med_null
        y_std_null[i] = std_null
    df = pd.DataFrame({"N": Ns, "Cms": Cms, "y1": y1, "y_null": y2, "z_med": y_med,
                       "z_std": y_std, "z_null_med": y_med_null, "z_std_null": y_std_null})
    if ".tsv" not in analysis_name:
        analysis_name = analysis_name + ".tsv"
    df.to_csv(analysis_name, sep="\t")
    _composite_connectivity_benchmark_plot(df)

def _composite_connectivity_benchmark_plot(df):
    fig, ax = plt.subplots()
    ax.plot(df['Cms'], df['y1'], 'k.', label="Composite")
    ax.plot(df['Cms'], df['y_null'], 'r.', label="Null")
    ax.set_xlabel("N Composite")
    ax.set_ylabel("Posterior probability Sum of Edge Weights >= x-1")
    ax.legend()
    plt.savefig(analysis_name.strip(".tsv") + ".png", dpi=300)
    plt.close()
    fig, ax = plt.subplots()
    ax.errorbar(df['Cms'], df['z_med'], yerr=df['z_std'], fmt='k.', label="Composite")
    ax.errorbar(df['Cms'], df['z_null_med'], yerr=df['z_std_null'], fmt='r.', label="Null")
    ax.set_xlabel("N Composite")
    ax.set_ylabel("Sum of Edge Weights")
    ax.legend()
    plt.savefig(analysis_name.strip(".tsv") + "_sum.png", dpi=300)
    plt.close()


def run_multiple_benchmarks(N=None,
                            edge_prob=None,
                            n_bait=3,
                            scores=None,
                            overwrite_existing=False,
                            suffix="",
                            remove_dirs=False,
                            static_seed=0,
                            d_crit=15,
                            max_path_length = 17,
                            ):
    """
    The solutions must be the same
    N: 3, 10, 50, 100, 500 
    Edge Density: 0.04, 0.13, 0.48, 0.97 
    Baits : 3
    Likelihood Only
    Prior Only
    Prior & Likelihood
    Single Composite
    - Composite Hierarchy
    - Overlapping vs non overlapping 
    For the chosen amount of sampling X we calculate
    1. Top Accuracy per chain +/- s.d 
    2. Top precision per chain +/- s.d 
    3. Accuracy of average network, precision of average network
    Generate the synthetic data 
    Default Parameters Values
    d_crit 
    """
    assert d_crit < max_path_length - 1
    assert n_bait > 0
    if N is None:
        N = [4, 9, 50, 100, 500]
    for i in N:
        assert i > 2
    if edge_prob is None:
        edge_prob = [0.04, 0.13, 0.48, 0.97]
    for i in edge_prob:
        assert i < 1
        assert i > 0
    if scores is None:
        scores = ["bp_lp", "lbh_ll", "lbh_ll__bp_lp"]
    all_tests = list(product(N, edge_prob, scores))
    paths = []
    for test in all_tests:
        path_str = str(test[0]) + "_" + str(test[1]) + "_" + str(test[2]) + f"_{n_bait}" + suffix
        path = Path(path_str)
        paths.append(path)
        if not path.is_dir():
            path.mkdir()
        if remove_dirs:
            path.rmdir()
    static_key = jax.random.PRNGKey(static_seed)
    # Generate the synthetic data for each example
    for i, test in enumerate(all_tests):
        #1 Generate the ground truth network
        N, prob, score_key = test
        Aref, bait_idx, prey_idx = get_bait_prey_network(static_key,N,n_bait,d_crit,prob, max_path_length)   
        #2 Generate the ground truth data
        data = data_from_network_model14_rng(static_key, Aref) 
    return Aref, bait_idx, prey_idx, data

