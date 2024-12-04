"""
Run a standard analysis on the output of mcmc sampling.
Generate figures in a directory.

1. Load in trajectory 
2. Calculate Rhat
3. Check chain convergances by eye
4. Load in references
5. Calculate AUC for references

MCMCM Fields
- potential energy

Coordinates are transformed to an unconstrained space
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import pandas as pd
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_value,
    init_to_uniform,
    )
import click
import pickle as pkl
from pathlib import Path
import time
from typing import NamedTuple
import math
import sys
import xarray as xr
import _model_variations
from numpyro.infer.util import (
    log_density
        )
from _model_variations import (
        matrix2flat,
        flat2matrix
        )
import _BSASA_functions
import flyplot
import sklearn
import os

class MCMCPlotData(NamedTuple):
    xlabel: str
    ylabel: str
    fmt: str
    title: str
   
__default_mcmc_plot_data = MCMCPlotData(
        xlabel = 'step',
        ylabel = 'auc',
        fmt = 'k.',
        title = 'direct_benchmark')

def load(fpath):
    with open(fpath, 'rb') as f: dat = pkl.load(f)
    return dat

def load_direct_benchmark(path="direct_benchmark.pkl"):
    return load(path)

def log_density_from_samples(model, samples: dict, nsamples: int, model_args = (None,), model_kwargs = None):
    """
    Params:
      model : Numpyro model
      samples : a dictionary keyed by site name 
      nsamples : int - number of samples
      model_args : tuple 
      model_kwargs : dict 
    """
    if model_kwargs is None:
        model_kwargs = {}
    log_prob_array = jnp.zeros(nsamples)
    for i in range(nsamples):
        param_dict = {key: samples[key][i, ...] for key in samples}
        lp_, _ = log_density(model, model_args, model_kwargs, param_dict) 
        log_prob_array = log_prob_array.at[i].set(lp_)
    return log_prob_array

def auc_from_samples(samples: dict, nsamples, auc_func = "model12"):
    """
    Params:
      samples : a dictionary keyed by site name 
      nsamples : int - number of samples
    """
    if auc_func == "model12":
        direct_matrix = load_direct_benchmark().reference.matrix
        auc_func = partial(auc_from_sample_model12, direct_matrix=direct_matrix) 
    auc_array = jnp.zeros(nsamples)
    for i in range(nsamples):
        param_dict = {key: samples[key][i, ...] for key in samples}
        auc = auc_func(param_dict)
        auc_array = auc_array.at[i].set(auc)
    return auc_array

def auc_from_sample_model12(param_dict, direct_matrix, nnodes=1879):
    """
    Params
      param_dict : keyed by sample site
      direct_matrix : reference xarray
      cos_sim_matrix : label xarray
      nnodes : number of nodes in network
    """
    # Truncate the matrices to the smaller one
    cos_sim_matrix = load_model12_cos_sim_matrix()
    a1, a2 = direct_matrix.shape
    b1, b2 = cos_sim_matrix.shape
    assert a1 == a2
    assert b1 == b2
    if a1 > b1:
        # Truncate direct to cos sim matrix
        direct_matrix = direct_matrix.sel(preyu=cos_sim_matrix.preyu, preyv=cos_sim_matrix.preyv)
    elif b1 > a1:
        cos_sim_matrix = cos_sim_matrix.sel(preyu=direct_matrix.preyu, preyv=direct_matrix.preyv)
    else:
        ...
    n_possible_edges = math.comb(nnodes, 2)
    weight_array = weights2xarray(param_dict['w'], nnodes, cos_sim_matrix)
    direct_matrix = direct_matrix.sel(preyu=weight_array.preyu, preyv=weight_array.preyv)
    direct_matrix.values = direct_matrix > 0
    n_edges = np.sum(np.tril(direct_matrix, k=-1))
    thresholds = np.arange(0, 1, 0.01)
    pps, tps = _BSASA_functions.pp_tp_from_pairwise_prediction_matrix_and_ref(
        direct_matrix,
        weight_array,
        thresholds)
    pps = np.array(pps)
    tps = np.array(tps)
    tpr = tps / n_edges
    ppr = pps / n_possible_edges
    auc = sklearn.metrics.auc(
        x = ppr,
        y = tpr,
        )
    return auc

def trajectory_data2data_array(trajectory, nnodes):
    """
    """
    nsamples, nedges = trajectory.shape   
    data_array_shape = (nsamples, nnodes, nnodes)
    data_array = jnp.zeros(data_array_shape)
    tril_a, tril_b = jnp.tril_indices(nnodes, k=-1) 
    for i in range(nsamples):
        data_array = data_array.at[i, tril_a, tril_b].set(trajectory[i, :]) 
        L = data_array[i, :, :]
        A = L + L.T
        data_array = data_array.at[i, :, :].set(A)
    return data_array

def calculate_auc(trajectory_data_array, reference_matrix):
    draws, nnodes1, nnodes2 = trajectory_data_array.shape
    auc_vector = jnp.zeros(draws) 

def get_trajectory_auc(edgelist_trajectory, direct_benchmark, selector):
    ...


def load_model12_cos_sim_matrix():
    """
    Load in the cos similarity matrix used in model12
    """
    with open("direct_benchmark.pkl", "rb") as f:
        cos_sim_matrix = pkl.load(f).prediction.cosine_similarity.matrix
    with open("tensor_saint_scores.pkl", "rb") as f:
        tensor_saint_scores = pkl.load(f)
    prey_sel = ((tensor_saint_scores > 0).sum(dim = ["bait", "condition"]) > 0)
    prey_sel = prey_sel.sortby("preyu")
    cos_sim_matrix = cos_sim_matrix.sortby("preyu") 
    cos_sim_matrix = cos_sim_matrix.sortby("preyv")
    prey_sel = prey_sel.sel(preyu=cos_sim_matrix.preyu)
    prey_selv = xr.DataArray(prey_sel.values, coords={"preyv": prey_sel.preyu.values})
    cos_sim_matrix = cos_sim_matrix.sel(preyu=prey_sel, preyv=prey_selv)
    return cos_sim_matrix

def save_mcmc_auc_plot(auc_array):
    fig, ax = plt.subplots()

def save_simple_scatter(ydata, xlabel, ylabel, title, savename, scatter_kwargs = None, dpi=300):
    if scatter_kwargs is None:
        scatter_kwargs = {}
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(ydata)), np.array(ydata), **scatter_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig(savename, dpi=dpi)


def weights2xarray(w, nnodes, cos_sim_matrix):
    """
    Converts the sample site 'w' to a pairwise matrix
    in xarray format. For use in conjuction with model12 from _model_variations.py
    """
    tril_indices = jnp.tril_indices(nnodes, k=-1)
    A = jnp.zeros((nnodes, nnodes))
    A = A.at[tril_indices].set(w)
    A = A + A.T  # non-negative weighted adjacency matrix with 0 diagonal.
    A = np.array(A)
    weight_data = xr.DataArray(A, coords=cos_sim_matrix.coords)
    return weight_data

def load_flattened_reference():
    direct = load( "direct_benchmark.pkl")   
    cocomplex = load("cocomplex_benchmark.pkl")
    direct_reference = matrix2flat(direct.reference.matrix.values) # same method as data loading 
    cocomplex_reference = matrix2flat(cocomplex.reference.matrix.values) 
    return direct_reference, cocomplex_reference

def get_auc_model14(trajectory_path: str, n_thresholds: int):
    """
    n_thresholds : the number of thresholds to interpolate between 0 and 1
      the actual number of threhoslds is n_thresholds + 1
    """
    thresholds = np.arange(0, 1, 1/n_thresholds)
    N_expected_nodes = 3005
    N_expected_edges = math.comb(N_expected_nodes, 2)
    tril_indices = np.tril_indices(N_expected_nodes, k=-1)
    # Are the nodes in the correct order?
    # Let's check
    apms_data_path = "../table1.csv"
    apms_ids = pd.read_csv(apms_data_path)
    # 1 to 1 mapping
    reference_preyu2uid = {r['PreyGene'].removesuffix('_HUMAN'): r['UniprotId'] for i,r in apms_ids.iterrows()}
    spectral_count_xarray = load("spectral_count_xarray.pkl")
    stacked_spectral_counts_array = spectral_count_xarray.stack(y=['AP', 'condition', 'bait', 'rep'])
    # Remove nodes where all counts are 0
    empty_nodes = np.sum(stacked_spectral_counts_array != 0, axis=1) == 0
    stacked_spectral_counts_array = stacked_spectral_counts_array.isel(preyu=~empty_nodes)
    apms_ids = [reference_preyu2uid[k] for k in stacked_spectral_counts_array.preyu.values]
    traj = load(trajectory_path)
    model14_data = load("xr_apms_correlation_matrix.pkl")
    assert apms_ids == list(model14_data.uid_preyu.values) 
    # The data loaded in for model 14 matches the ids in referce_prey2uid
    # And can be mapped back to Gene Identifiers
    # Model14 data was flattened using the following command
    # 
    #apms_tril_indices = jnp.tril_indices(n, k=-1)
    #flattened_apms_similarity_scores = matrix2flat(
    #        jnp.array(apms_correlation_matrix.values, dtype=jnp.float32)) 
    direct = load( "direct_benchmark.pkl")   
    cocomplex = load("cocomplex_benchmark.pkl")
    # Check the id mapping
    assert [reference_preyu2uid[k] for k in direct.reference.matrix.preyu.values] == list(model14_data.uid_preyu.values) 
    assert [reference_preyu2uid[k] for k in cocomplex.reference.matrix.preyu.values] == list(model14_data.uid_preyu.values) 
    # ids pass
    direct_reference = matrix2flat(direct.reference.matrix.values) # same method as data loading 
    del direct
    cocomplex_reference = matrix2flat(cocomplex.reference.matrix.values) 
    del cocomplex
    assert direct_reference.shape == (N_expected_edges,)
    assert cocomplex_reference.shape == (N_expected_edges,)
    nsamples, nedges = traj['samples']['pT'].shape 
    direct_auc_array = np.zeros(nsamples) 
    cocomplex_auc_array = np.zeros(nsamples) 
    for samples_idx, edge_list in enumerate(traj['samples']['pT']):
        direct_xs = np.zeros(len(thresholds))
        direct_ys= np.zeros(len(thresholds))
        cocomplex_xs = np.zeros(len(thresholds))
        cocomplex_ys= np.zeros(len(thresholds))
        for i, threshold in enumerate(thresholds):
            direct_tpr, direct_ppr = roc_analysis(edge_list, threshold, direct_reference, N_expected_edges)
            direct_xs[i] = direct_ppr
            direct_ys[i] = direct_tpr
            cocomplex_tpr, cocomplex_ppr = roc_analysis(edge_list, threshold, cocomplex_reference, N_expected_edges)
            cocomplex_xs[i] = cocomplex_tpr
        direct_auc = sklearn.metrics.auc(direct_xs, direct_ys)
        direct_auc_array[samples_idx] = direct_auc
        cocomplex_auc = sklearn.metrics.auc(cocomplex_xs, cocomplex_ys)
        cocomplex_auc_array[samples_idx] = cocomplex_auc
    return direct_auc_array, cocomplex_auc_array

def generate_roc_curve(thresholds, reference, prediction):
    xs = np.zeros(len(thresholds))
    ys = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        direct_tpr, direct_ppr = roc_analysis(prediction, threshold, reference, len(reference))
        xs[i] = direct_ppr
        ys[i] = direct_tpr
    return xs, ys

    

def roc_analysis(pred_array, threshold, reference, N_expected_edges):
    binary_predictions = pred_array >= threshold
    positive_predictions = np.sum(binary_predictions)
    tp = binary_predictions * reference 
    tpr = np.sum(tp) / np.sum(reference)
    ppr = np.sum(positive_predictions) / N_expected_edges
    return tpr, ppr 
    
def calc_accuracy(edge_samples, reference, metric="auc_roc"):
    n, m = edge_samples.shape
    edge_samples = np.array(edge_samples, dtype=np.float32)
    reference = np.array(reference, dtype=np.float32)
    accuracy_array = np.zeros(n)

    if metric == "auc_roc":
        f = sklearn.metrics.roc_auc_score
    elif metric == "ap":
        f = sklearn.metrics.average_precision_score
    else:
        raise NotImplementedError
    for i in range(n):
        edges = edge_samples[i, :]
        assert edges.shape == reference.shape, (edges.shape, reference.shape)
        accuracy = f(reference, edges)
        accuracy_array[i] = accuracy
    return accuracy_array

def calc_log_density(edge_samples, model, model_data):
    n, m = edge_samples.shape
    log_density_array = np.zeros(n)
    for i in range(n):
        edges = edge_samples[i, :]
        log_density, _ = numpyro.infer.util.log_density(
                model, (model_data,), {},
                {'pT' : edges})
        log_density_array[i] = log_density
    return log_density_array


def model14_traj2analysis(traj_path, model, model_data, reference=None,):
    """
    traj : str to a trajectory pickle file. e.g, 0_model14_0.pkl
    Output:
      saves an analysis_file. 0_model14_0_analysis.pkl
    """
    with open(traj_path, "rb") as f:
        d = pkl.load(f)
    score_array = np.array(d['extra_fields']['potential_energy'])
    edge_samples = d['samples']['pT']
    if reference is not None:
        accuracy_array = calc_accuracy(edge_samples, reference) 
        log_density_array = calc_log_density(edge_samples, model, model_data)
    return {"potential_energy" : score_array,
            "accuracy" : accuracy_array,
            "log_density": log_density_array}

def model14_ap_score(traj_path, model, model_data, reference=None):
    with open(traj_path, "rb") as f:
        d = pkl.load(f)
    score_array = np.array(d['extra_fields']['potential_energy'])
    edge_samples = d['samples']['pT']
    if reference is not None:
        ap = calc_accuracy(edge_samples, reference, metric='ap')
    return {"AP" : ap} 
           
           



@click.command()
@click.option("--trajectory-name" ,help="prefix to the trajectory pickle file")
@click.option("--analysis-name" ,help="a name for the analysis")
@click.option("--model-name")
@click.option("--fig-dpi", type=int, default=300)
@click.option("--dir-name", type=str)
def main(trajectory_name, analysis_name, fig_dpi, model_name):
    _main(trajectory_name, analysis_name, fig_dpi, model_name, dir_name)

def _main(trajectory_name, analysis_name, fig_dpi, model_name, dir_name):
    prefix = analysis_name + "__" + trajectory_name
    save_path = Path(dir_name) / prefix

    with open(trajectory_name + ".pkl", "rb") as f:
        d = pkl.load(f)
    energy = np.array(d['extra_fields']['potential_energy']) 
    plt.hist(energy, bins=50)
    plt.xlabel("score")
    plt.ylabel("frequency")
    plt.savefig(str(save_path) + "_potential_energy_hist.png", dpi=fig_dpi)
    plt.close()

    x = np.arange(len(energy))
    plt.plot(x, energy, 'k.')
    plt.ylabel("score")
    plt.xlabel("step")
    plt.savefig(str(save_path) + "_caterpillar.png", dpi=fig_dpi)
    
if __name__ == "__main__":
    main()
