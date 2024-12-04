"""
Given an file of samples write figures visualing samples to an output directory
"""
import sklearn
import sklearn.metrics
import click
from functools import partial
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle as pkl
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib as mpl
import networkx as nx
import scipy as sp
import sklearn 
import time

import logging

import data_io
import _model_variations as mv
import tpr_ppr
import undirected_edge_list as uel
import merge_analysis as merge 
import generate_benchmark_figures as gbf

import scipy as sp

logger = logging.getLogger(__name__)
# Globals
hist_range = (-1, 1)

@click.command()
@click.option("--o", type=str, help="output directory")
@click.option("--i", type=str, help="input file")
@click.option("--mode", type=str, default="cullin")
def main(o, i, mode):
    _main(o, i, mode)

_base_style = ""
_corr_rc =   {"image.cmap" : "coolwarm"} 
_matrix_rc = {"image.cmap" : "hot"}
_spec_rc =   {"image.cmap" : "gist_gray"} 
_histogram_rc = ""
_scatter_plot_rc = ""
_caterpillar_rc = ""

_dev = True 

def rc_context(rc = None, fname = None):
    """
    Decorator maker
    """
    def rc_decorator(func):
        def wrapper(*args, **kwargs):
            with mpl.rc_context(rc = rc, fname = fname):
                return func(*args, **kwargs)
        return wrapper
    return rc_decorator

def input_load(i, fbasename, suffix):
    with open(i / f"{fbasename}{suffix}.pkl", "rb") as f:
        return pkl.load(f)


def run_multichain_specific_plots(x, model_data, suffix="", save=None, o = None, do_animate = False):
    """
    Params:
      x : { 
      CHAIN       
      }
    Types of plots to run

    Iteration scatter w/ chain std.
    """
    def caterrplot(y, xlabel, ylabel, title, savename, alpha = 0.2, save=save):
        fig, ax = plt.subplots()
        nchains, n_iter = y.shape
        x = np.arange(n_iter)
        ax.errorbar(x, np.mean(y, axis=0), yerr=np.std(y, axis=0), fmt='o', alpha=alpha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)
    start_time = time.time()    
    merged_results_two_subsets_scores_hist_and_test(x, save=save, o = o, suffix=suffix, min_val = 131_000, max_val = 134_000)   
    end_time = time.time()
    logging.info(f"Time to run merged_results_two_subsets_scores_hist_and_test: {end_time - start_time}")

    # Plot of improving scores

    ef = x["extra_fields"]
    logging.info("calculating best score per chain based on amount of sampling")
    start_time = time.time()
    results_dict = best_score_per_chain_based_on_amount_of_sampling(ef["potential_energy"])
    errplot_from_av_std_dict(results_dict, "amount of sampling", "best score", "Best score per chain", "best_score_per_chain" + suffix, save=save)
    end_time = time.time()
    logging.info(f"Time to run best_score_per_chain_based_on_amount_of_sampling: {end_time - start_time}")
    
    start_time = time.time()
    pdb_ppi_direct = data_io.get_pdb_ppi_predict_direct_reference()
    direct_ij = align_reference_to_model(model_data, pdb_ppi_direct, mode="cullin")
    rng_key = jax.random.PRNGKey(0)
    shuff_direct_ij = jax.random.permutation(rng_key, direct_ij)

    pdb_ppi_costructure = data_io.get_pdb_ppi_predict_cocomplex_reference()
    costructure_ij = align_reference_to_model(model_data, pdb_ppi_costructure, mode="cullin")
    end_time = time.time()
    logging.info(f"Time to run align_reference_to_model: {end_time - start_time}")
    
    # optional None or synthetic network
    synthetic_ij = None if "synthetic_network" not in model_data else model_data["synthetic_network"]

    plot_a_b_roc(x, direct_ij, o = o, save=save, suffix="_direct" + suffix)
    plot_a_b_roc(x, costructure_ij, o = o, save=save, suffix="_costructure" + suffix)
    plot_a_b_roc(x, synthetic_ij, o = o, save=save, suffix="_synthetic" + suffix)

    plot_humap_saint_inm_roc(x, model_data, direct_ij, save=save, o = o, suffix="_direct" + suffix)
    plot_humap_saint_inm_roc(x, model_data, costructure_ij, save=save, o = o, suffix="_costructure" + suffix)
    plot_humap_saint_inm_roc(x, model_data, synthetic_ij, save=save, o = o, suffix="_synthetic" + suffix)

    #plot_sliding_window_roc(x, ef, direct_ij, save=save, window_size = 100, suffix="_direct" + suffix)
    #plot_sliding_window_roc(x, ef, direct_ij, save=save, window_size = 50, suffix="_direct" + suffix)
    #plot_sliding_window_roc(x, ef, direct_ij, save=save, suffix="_direct" + suffix)
    #plot_sliding_window_roc(x, ef, direct_ij, save=save, window_size = 15, suffix="_direct" + suffix)
    #plot_sliding_window_roc(x, ef, direct_ij, save=save, window_size = 10, suffix="_direct" + suffix)
    #plot_sliding_window_roc(x, ef, direct_ij, save=save, window_size = 5, suffix="_direct" + suffix)
    #plot_sliding_window_roc(x, ef, direct_ij, save=save, window_size = 2, suffix="_direct" + suffix)
    #plot_per_frame_roc(x, ef, direct_ij, save=save, suffix="_direct" + suffix) 
    plot_roc_as_an_amount_of_sampling(x, direct_ij, save=save, suffix="_direct" + suffix)
    #plot_sliding_window_roc(x, ef, costructure_ij, save=save, suffix="_costructure" + suffix)
    #plot_per_frame_roc(x, ef, costructure_ij, save=save, suffix="_costructure" + suffix)
    plot_roc_as_an_amount_of_sampling(x, costructure_ij, save=save, suffix="_costructure" + suffix)
    plot_roc_as_an_amount_of_sampling(x, synthetic_ij, save=save, suffix="_synthetic" + suffix)
    
    if do_animate:
        animate_modeling_run_frames(mv.Z2A(x["samples"]["z"]) > 0.5, model_data = model_data, save=save, o=o, suffix = suffix)
    # Plot ROC as a function of increased sampling

    def score_vs_plot(rng_key, stat_mat, score_mat, ylabel, title, save, suffix):
        n_chains, n_iter = stat_mat.shape
        nvalues = n_chains * n_iter
        indices = jax.random.permutation(rng_key, jnp.arange(nvalues))[:100_000]
        scores_subsample = np.array(jnp.ravel(score_mat)[indices])
        stat_subsample =   np.array(jnp.ravel(stat_mat )[indices])

        fig, ax = plt.subplots()
        ax.plot(scores_subsample, stat_subsample, 'k.', alpha=0.2)
        ax.set_xlabel("score")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(title + suffix)

    def per_chain_amount_of_sampling(stat_mat, transform_fn = lambda x: x, every=100):
        """
        state_mat : (n_chains, n_iter) 
        transform_fn : F: S -> A 
        """
        n_chains, n_iter = stat_mat.shape
        every = np.concatenate([np.array([1]), np.arange(every, n_iter, every)])  
        
        out = np.zeros((len(every), 2)) 
        for i, N in enumerate(every):
            temp = transform_fn(stat_mat[:, 0:N])
            mean = np.mean(temp, axis=0)
            std = np.std(temp, axis=0)
            out[i, 0] = mean 
            out[i, 1] = std
        return out, every
    
    def plot_per_chain_amount_of_sampling(every, out, xlabel, ylabel, title, save, suffix):
        fig, ax = plt.subplots()
        ax.errorbar(every, out[:, 0], yerr=out[:, 1], fmt='.', capsize=2, alpha=0.2, color="k")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(title + suffix)

    def run_plots(x, ef, refij, save, o, jf_tp, jf_tn, M, suffix):
        if refij is None:
            return
        start_time = time.time()

        # calculate positive predictions
        aij_mat = mv.Z2A(x['samples']['z']) > 0.5
        n_chain, n_iter, M = aij_mat.shape

        mean_aij = np.mean(aij_mat, axis=(0, 1))
        roc_curve(refij, mean_aij, save=save, o=o, suffix=suffix)
        prc_curve(refij, mean_aij, save=save, o=o, suffix=suffix)
        del mean_aij

        score_vs_auprc_plot(x, ef, refij, save, o, suffix=suffix)

        n_positives_mat = jnp.sum(aij_mat, axis=2)

        tp_mat = jf_tp(aij_mat, refij)
        out, every = per_chain_amount_of_sampling(tp_mat, transform_fn=lambda x: jnp.max(x, axis=1))
        plot_per_chain_amount_of_sampling(every, out, "amount of sampling", "TPs", "Top TPs per chain", save, suffix)
        fp_mat = n_positives_mat - tp_mat 
        out, every = per_chain_amount_of_sampling(fp_mat, transform_fn=lambda x: jnp.min(x, axis=1))
        plot_per_chain_amount_of_sampling(every, out, "amount of sampling", "FPs", "Bottom FPs per chain", save, suffix)
        ppv_mat = tp_mat / (tp_mat + fp_mat)
        out, every = per_chain_amount_of_sampling(ppv_mat, transform_fn=lambda x: jnp.max(x, axis=1))
        plot_per_chain_amount_of_sampling(every, out, "amount of sampling", "PPV", "Top PPV per chain", save, suffix)

        score_mat = ef["potential_energy"]
        assert score_mat.shape == ppv_mat.shape
        rng_key = jax.random.PRNGKey(0)
        rng_key, key = jax.random.split(rng_key)
        score_vs_plot(rng_key, tp_mat, score_mat, "TPs", "TPs vs score", save, suffix)
        rng_key, key = jax.random.split(key)
        score_vs_plot(rng_key, fp_mat, score_mat, "FPs", "FPs vs score", save, suffix)
        rng_key, key = jax.random.split(key)
        score_vs_plot(rng_key, ppv_mat, score_mat, "PPV", "PPV vs score", save, suffix)
        del fp_mat
        del tp_mat
        del ppv_mat
        n_negatives_mat = M - n_positives_mat 
        del n_positives_mat

        tn_mat = jf_tn(aij_mat, refij)
        out, every = per_chain_amount_of_sampling(tn_mat, transform_fn=lambda x: jnp.max(x, axis=1))
        fn_mat = n_negatives_mat - tn_mat
        out, every = per_chain_amount_of_sampling(fn_mat, transform_fn=lambda x: jnp.min(x, axis=1))
        plot_per_chain_amount_of_sampling(every, out, "amount of sampling", "Bottom FNs", "Bottom FNs per chain", save, suffix)
        del n_negatives_mat
        rng_key, key = jax.random.split(key)
        score_vs_plot(rng_key, tn_mat, score_mat, "TNs", "TNs vs score", save, suffix)
        rng_key, key = jax.random.split(key)
        score_vs_plot(rng_key, fn_mat, score_mat, "FNs", "FNs vs score", save, suffix)
    
    jf_tp = jax.jit(lambda x,y :tp_per_iteration_vectorized_3d(x, y).T)
    jf_tn = jax.jit(lambda x,y :tn_per_iteration_vectorized_3d(x, y).T)
    M = model_data["M"]

    run_plots(x, ef, direct_ij, save, o,  jf_tp, jf_tn, M, suffix="_direct" + suffix)
    run_plots(x, ef, costructure_ij, save, o,  jf_tp, jf_tn, M, suffix="_costructure" + suffix)
    run_plots(x, ef, synthetic_ij, save, o,  jf_tp, jf_tn, M, suffix="_synthetic" + suffix)
    run_plots(x, ef, shuff_direct_ij, save, o, jf_tp, jf_tn, M, suffix="_shuff_direct" + suffix)
    shuff_costructure_ij = jax.random.permutation(rng_key, costructure_ij)
    run_plots(x, ef, shuff_costructure_ij, save, o, jf_tp, jf_tn, M, suffix="_shuff_costructure" + suffix)

    #start_time = time.time()
    #score_vs_ppv_plot(jax.random.PRNGKey(0), x["samples"], ef, direct_ij, save=save, o=o, suffix="_direct_" + suffix)
    #score_vs_tp_plot( jax.random.PRNGKey(0), x["samples"], ef, direct_ij, save=save, o=o, suffix="_direct_" + suffix)
    #score_vs_fp_plot( jax.random.PRNGKey(0), x["samples"], ef, direct_ij, save=save, o=o, suffix="_direct_" + suffix)
    #score_vs_tn_plot( jax.random.PRNGKey(0), x["samples"], ef, direct_ij, save=save, o=o, suffix="_direct_" + suffix)
    #score_vs_fn_plot( jax.random.PRNGKey(0), x["samples"], ef, direct_ij, save=save, o=o, suffix="_direct_" + suffix)
    #end_time = time.time()
    #logging.info(f"Time to run score_vs_X_plot on direct: {end_time - start_time}")
    
    #start_time = time.time()
    #score_vs_ppv_plot(jax.random.PRNGKey(0), x["samples"], ef, costructure_ij, save=save, o=o, suffix="_co_structure" + suffix)
    #score_vs_tp_plot( jax.random.PRNGKey(0), x["samples"], ef, costructure_ij, save=save, o=o, suffix="_co_structure" + suffix)
    #score_vs_fp_plot( jax.random.PRNGKey(0), x["samples"], ef, costructure_ij, save=save, o=o, suffix="_co_structure" + suffix)
    #score_vs_tn_plot( jax.random.PRNGKey(0), x["samples"], ef, costructure_ij, save=save, o=o, suffix="_co_structure" + suffix)
    #score_vs_fn_plot( jax.random.PRNGKey(0), x["samples"], ef, costructure_ij, save=save, o=o, suffix="_co_structure" + suffix)
    #end_time = time.time()
    #logging.info(f"Time to run score_vs_X_plot on costructure: {end_time - start_time}")


    #logging.info("calculating ppv per chain based on amount of sampling")
    #start_time = time.time()
    #ppv_results_dict = ppv_per_chain_based_on_amount_of_sampling(x['samples']['z'] > 0.5, direct_ij, amount_of_sampling_list = None) 
    #errplot_from_av_std_dict(ppv_results_dict, "amount of sampling", "ppv", "PPV per chain", "archive_ppv_per_chain" + suffix, save=save)
    #end_time = time.time()
    #logging.info(f"Time to run ppv_per_chain_based_on_amount_of_sampling: {end_time - start_time}")
    #start_time = time.time()
    #top_ppv_dict = top_ppv_per_chain_based_on_amount_of_sampling(x['samples']['z'] > 0.5, direct_ij, amount_of_sampling_list = None)
    #errplot_from_av_std_dict(top_ppv_dict, "amount of sampling", "top ppv", "Top PPV per chain", "archive_top_ppv_per_chain" + suffix, save=save)
    #end_time = time.time()
    #logging.info(f"Time to run top_ppv_per_chain_based_on_amount_of_sampling: {end_time - start_time}")

    start_time = time.time()
    k = 0
    for key in ef:
        val = ef[key]
        caterrplot(val, "iteration", key, key, f"{key}_caterpill" + suffix)    
        k += 1
    end_time = time.time()
    logging.info(f"Time to run {k} caterrplots: {end_time - start_time}")

def roc_curve(y_true, y_score, save=None, o=None, suffix=""):
    thresholds = np.linspace(0, 1, 1000)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)

    auc = sklearn.metrics.roc_auc_score(y_true = y_true, y_score = y_score)
    
    n_true = np.sum(y_true)
    n_pred = len(y_score)
    fig, ax = plt.subplots()
    ax.set_xlabel(f"FPR - N pos pred = {n_pred}")
    ax.set_ylabel(f"TPR - N={n_true}")
    ax.plot(fpr, tpr, 'k-', alpha=0.2)
    ax.text(0.6, 0.6, f"AUC={auc}", transform=plt.gca().transAxes)
    save("roc_curve" + suffix)

def prc_curve(y_true, y_score, save=None, o=None, suffix=""):
    thresholds = np.linspace(0, 1, 1000)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score)
    auc = sklearn.metrics.average_precision_score(y_true = y_true, y_score = y_score)
    
    n_true = np.sum(y_true)
    n_pred = len(y_score)
    fig, ax = plt.subplots()
    ax.set_xlabel(f"Recall - N pos pred = {n_pred}")
    ax.set_ylabel(f"Precision - N={n_true}")
    ax.plot(recall, precision, 'k-', alpha=0.2)
    ax.text(0.6, 0.6, f"AUC={auc}", transform=plt.gca().transAxes)
    save("pr_curve" + suffix)




def score_vs_auprc_plot(x, ef, refij, save=None, o=None, suffix="", n_score_bins = 100, n_models_per_bin = 5):
    # 1. Save every score : [(chain_idx, iteration_idx)] in a dictionary  
    scores = ef["potential_energy"]
    n_chains, n_iter = scores.shape

    score_dict = {}
    for chain_idx in range(n_chains):
        for iteration_idx in range(n_iter):
            score = float(scores[chain_idx, iteration_idx])
            score_dict[score] = [chain_idx, iteration_idx]

    # 2. Create N bins for the scores, {mean_score : [coords]}
    min_score = np.min(scores)
    max_score = np.max(scores)
    def n_th_bin(score, min_score, max_score, nbins):
        return int((score - min_score) / (max_score - min_score) * (nbins-1))

    score_bins = defaultdict(list)
    for score, coord in score_dict.items():
        bin_idx = n_th_bin(score, min_score, max_score, n_score_bins)
        score_bins[bin_idx].append((coord, score))

    # 3. Select N models per bin, calculate the average model in the bin
    
    mean_score_average_model_dict = {}
    for bin_idx, coord_score_lst in score_bins.items():
        coords, scores = zip(*coord_score_lst)
        n_models = len(coords)
        if n_models <= 2:
            continue
        elif n_models < n_models_per_bin:
            ...
        else:
            coords = coords[:n_models_per_bin]
        models = []
        for i, coord in enumerate(coords):
            model = mv.Z2A(x["samples"]["z"][coord[0], coord[1]])
            models.append(model)
        models = np.array(models)
        average_model = np.mean(models, axis=0)
        mean_score = np.mean(scores)
        mean_score_average_model_dict[mean_score] = average_model

    # 4. Calculate the auroc for the average model
    score_results = []
    for mean_score, average_model in mean_score_average_model_dict.items():
        roc_score = sklearn.metrics.roc_auc_score(y_true = refij, y_score = average_model)
        prc_score = sklearn.metrics.average_precision_score(y_true = refij, y_score = average_model)
        score_results.append((mean_score, roc_score, prc_score))
    
    score_results = np.array(score_results)
    # 5. Plot the auroc vs the mean score
    fig, ax = plt.subplots()
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Mean score")
    ax.plot(score_results[:, 1], score_results[:, 0], 'k.', alpha=0.2)
    save("auroc_vs_mean_score" + suffix)

    fig, ax = plt.subplots()
    ax.set_xlabel("AUPRC")
    ax.set_ylabel("Mean score")
    ax.plot(score_results[:, 2], score_results[:, 0], 'k.', alpha=0.2)
    save("auprc_vs_mean_score" + suffix)

    
def ppv(aij, refij):
    """
    Positive predictive value
    """
    n_tp = jnp.sum((aij == 1) &  (refij == 1))
    n_fp = jnp.sum((aij == 1) & (refij == 0))
    return n_tp / (n_tp + n_fp)

def ppv_per_iteration(aij_mat, refij):
    """
    Return the positive predictive value for each iteration
    """
    N, _ = aij_mat.shape
    output = jnp.zeros(N)
    for i in range(N):
        output = output.at[i].set(ppv(aij_mat[i, :], refij))
    return output

def ppv_per_iteration_vectorized(aij_mat, refij):
    ppv_vectorized = jax.vmap(ppv, in_axes=(0, None))
    output = ppv_vectorized(aij_mat, refij)
    return output

def ppv_per_iteration_vectorized_3d(aij_3dim, refij):
    f1 = jax.vmap(ppv, in_axes=(0, None))
    f2 = jax.vmap(f1, in_axes=(1, None))
    output = f2(aij_3dim, refij)
    return output

def tp_per_iteration_vectorized_3d(aij_3dim, refij):
    f1 = jax.vmap(n_tp, in_axes=(0, None))
    f2 = jax.vmap(f1, in_axes=(1, None))
    output = f2(aij_3dim, refij)
    return output

def fp_per_iteration_vectorized_3d(aij_3dim, refij):
    f1 = jax.vmap(n_fp, in_axes=(0, None))
    f2 = jax.vmap(f1, in_axes=(1, None))
    output = f2(aij_3dim, refij)
    return output

def tn_per_iteration_vectorized_3d(aij_3dim, refij):
    f1 = jax.vmap(n_tn, in_axes=(0, None))
    f2 = jax.vmap(f1, in_axes=(1, None))
    output = f2(aij_3dim, refij)
    return output

def fn_per_iteration_vectorized_3d(aij_3dim, refij):
    f1 = jax.vmap(n_fn, in_axes=(0, None))
    f2 = jax.vmap(f1, in_axes=(1, None))
    output = f2(aij_3dim, refij)
    return output

def min_score_vectorized(score_mat):
    """
    score_mat : (chain, iteration) 
    """
    return jnp.min(score_mat, axis=1)

def metric_as_a_function_of_amount_of_sampling_per_chain(x, metricf, amount_of_sampling_list):
    """
    x : (CHAIN, ITERATION, ...) 
    metricf : 
    """
    M = len(amount_of_sampling_list)
    out = jnp.zeros((M, 2))
    for i, N in enumerate(amount_of_sampling_list):
        temp = x[:, 0:N, ...] # select up to N iterations per chain
        metric = metricf(temp)
        if metric.ndim > 0:
            av = jnp.mean(metric, axis=0) # average over chains
            std = jnp.std(metric, axis=0) # std over chains
        else:
            av = metric
            std = 0
        out = out.at[i, :].set([av, std])
    return dict(out = out,
                amount_of_sampling_list = amount_of_sampling_list)


_STD_AMOUNT_OF_SAMPING = [1] + list(np.arange(10, 2_000, 10)) 
_STD_AMOUNT_OF_SAMPLING2 = [1,10, 100, 500,] + list(np.arange(1_000, 20_000, 1_000))
_CHAIN_DIM = 0
_ITER_DIM = 1
_AV_DIM = 0
_STD_DIM = 1

def best_score_per_chain_based_on_amount_of_sampling(
        x, amount_of_sampling_list = None):
    if x.ndim == 2:
        n_chains, n_iter = x.shape
    elif x.ndim == 3:
        n_chains, n_iter, _ = x.shape
    else:
        raise ValueError("Only 2 or 3 dimensions are supported")
    if amount_of_sampling_list is None:
        if n_iter > 2_000:
            amount_of_sampling_list = [1] + list(np.arange(10, n_iter, 1_000)) 
        else: 
            amount_of_sampling_list = _STD_AMOUNT_OF_SAMPING 
    results = metric_as_a_function_of_amount_of_sampling_per_chain(x, lambda x: jnp.min(x, axis=_ITER_DIM), amount_of_sampling_list)
    return results

def top_ppv_per_chain_based_on_amount_of_sampling(
        x, refij, amount_of_sampling_list = None):
    if x.ndim == 3:
        nchains, n_iter, vdim = x.shape
    elif x.ndim == 4:
        nchains, n_iter, N, _ = x.shape
    else:
        raise ValueError("Only 3 or 4 dimensions are supported")
    if amount_of_sampling_list is None:
        if n_iter > 2_000:
            amount_of_sampling_list = [1] + list(np.arange(10, n_iter, 1_000)) 
        else:
            amount_of_sampling_list = _STD_AMOUNT_OF_SAMPING
    results = metric_as_a_function_of_amount_of_sampling_per_chain(x, lambda x: dim_aware_max(ppv_per_iteration_vectorized(x, refij)), amount_of_sampling_list)
    return results

def sliding_window_roc(aij_mat, ef, refij, window_size = 25, rseed = 1024, every=2):
    rng_key = jax.random.PRNGKey(rseed)
    assert aij_mat.ndim == 3
    n_chains, n_iter, vdim = aij_mat.shape
    total_samples = n_chains * n_iter

    keys = jax.random.split(rng_key, 6)
    shuff_ij1 = jax.random.permutation(keys[1], refij)

    n_steps = n_iter - window_size
    aucs = []
    shuff_aucs = []
    mean_scores = []
    scores = ef["potential_energy"]

    for i in range(n_steps):
        if i % every != 0:
            continue
        pred_per_chain = np.mean(aij_mat[:, i:i+window_size, :], axis=1)
        av_score_per_chain = np.mean(scores[:, i:i+window_size], axis=1)
        for j in range(n_chains):
            auc = sklearn.metrics.roc_auc_score(y_true = refij, y_score = pred_per_chain[j, :])
            shuff_auc = sklearn.metrics.roc_auc_score(y_true = shuff_ij1, y_score = pred_per_chain[j, :])
            aucs.append(auc)
            shuff_aucs.append(shuff_auc)
            mean_scores.append(av_score_per_chain[j])
    return dict(
        aucs = np.array(aucs),
        mean_scores = np.array(mean_scores),
        shuff_aucs = np.array(shuff_aucs),
    )

def per_frame_roc(aij_mat, ef, refij, rseed=512, every=100):
    rng_key = jax.random.PRNGKey(rseed)
    assert aij_mat.ndim == 3
    n_chains, n_iter, vdim = aij_mat.shape
    total_samples = n_chains * n_iter

    keys = jax.random.split(rng_key, 6)
    shuff_ij1 = jax.random.permutation(keys[1], refij)

    aucs = []
    shuff_aucs = []
    scores_lst = []
    scores = ef["potential_energy"]
    for i in range(n_iter * n_chains):
        if i % 100 != 0:
            continue
        model = aij_mat.reshape(-1, aij_mat.shape[-1])[i, :]
        score = scores.reshape(-1)[i]
        auc = sklearn.metrics.roc_auc_score(y_true = refij, y_score = model)
        aucs.append(auc)
        scores_lst.append(score)
    return dict(
        aucs = np.array(aucs),
        scores = np.array(scores_lst),
    )

def plot_degree_plots(x, save = None, suffix = ""): 
    aij_mat = mv.Z2A(x["samples"]["z"]) > 0.5

    n_chains, n_iter, N, N2 = aij_mat.shape
    assert N == N2, (N, N2)

    degree_mat = np.sum(aij_mat, axis=3)

    # Plot degree over all chains
    fig, ax = plt.subplots()
    ax.hist(np.ravel(degree_mat), bins=100)
    ax.set_xlabel("Degree per iteration per node")
    ax.set_ylabel("Count")
    save("degree_hist" + suffix)

    weighted_degree_mat = np.mean(aij_mat, axis=1) # Average network per chain
    weighted_degree_mat = np.sum(weighted_degree_mat, axis=2)

    fig, ax = plt.subplots()
    ax.hist(np.ravel(weighted_degree_mat), bins=100)
    ax.set_xlabel("Weighted degree")
    ax.set_ylabel("Count")
    save("weighted_degree_hist" + suffix)


def roc_as_an_amount_of_sampling(aij_mat, refij, amount_of_sampling_list = None, every = 1, n_bootstraps = 3, rseed = 2048):
    """
    
    """
    rng_key = jax.random.PRNGKey(rseed)
    assert aij_mat.ndim == 3
    n_chains, n_iter, vdim = aij_mat.shape
    total_samples = n_chains * n_iter
    
    if amount_of_sampling_list is None:
        amount_of_sampling_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18,] + list(np.arange(20, n_iter, every)) 
    av_aucs = []
    av_shuff_aucs = []
    std_aucs = []
    std_shuff_aucs = []
    keys = jax.random.split(rng_key, 6)
    shuff_ij1 = jax.random.permutation(keys[1], refij)
    shuff_ij2 = jax.random.permutation(keys[2], refij)
    shuff_ij3 = jax.random.permutation(keys[3], refij)
    shuff_ij4 = jax.random.permutation(keys[4], refij)
    shuff_ij5 = jax.random.permutation(keys[5], refij)

    rng_key = keys[0]

    for nsamples in amount_of_sampling_list:
        aucs = []
        shuff_aucs = []
        rng_key, key = jax.random.split(rng_key)
        for i in range(n_bootstraps):
            rng_key, key = jax.random.split(rng_key)
            indices = jax.random.permutation(key, total_samples)[:nsamples]
            pred = jnp.mean(aij_mat.reshape(-1, aij_mat.shape[-1])[indices, :], axis=0)
            #fpr, tpr, thresholds = sklearn.metrics.roc_curve(refij, pred)
            #shuff_fpr, shuff_tpr, shuff_thresholds = sklearn.metrics.roc_curve(shuff_ij, pred)
            auc = sklearn.metrics.roc_auc_score(y_true = refij, y_score = pred)
            shuff_auc1 = sklearn.metrics.roc_auc_score(y_true = shuff_ij1, y_score = pred)
            shuff_auc2 = sklearn.metrics.roc_auc_score(y_true = shuff_ij2, y_score = pred)
            shuff_auc3 = sklearn.metrics.roc_auc_score(y_true = shuff_ij3, y_score = pred)
            shuff_auc4 = sklearn.metrics.roc_auc_score(y_true = shuff_ij4, y_score = pred)
            shuff_auc5 = sklearn.metrics.roc_auc_score(y_true = shuff_ij5, y_score = pred)

            aucs.append(auc)
            shuff_aucs = shuff_aucs + [shuff_auc1, shuff_auc2, shuff_auc3, shuff_auc4, shuff_auc5]
        av_auc = np.mean(aucs)
        av_shuff_auc = np.mean(shuff_aucs)
        std_auc = np.std(aucs)
        std_shuff_auc = np.std(shuff_aucs)

        av_aucs.append(av_auc)
        av_shuff_aucs.append(av_shuff_auc)
        std_aucs.append(std_auc)
        std_shuff_aucs.append(std_shuff_auc)
    return dict(
        amount_of_sampling_list = amount_of_sampling_list,
        av_aucs = av_aucs,
        av_shuff_aucs = av_shuff_aucs, 
        std_aucs = std_aucs,
        std_shuff_aucs = std_shuff_aucs,
    )

def plot_roc_as_an_amount_of_sampling(x, refij, save=None, suffix=""):
    if refij is None:
        return
    aij_mat = mv.Z2A(x['samples']['z']) > 0.5
    plot_xy_data = roc_as_an_amount_of_sampling(aij_mat, refij) 

    fig, ax = plt.subplots()
    ax.errorbar(plot_xy_data["amount_of_sampling_list"], plot_xy_data["av_aucs"], yerr=plot_xy_data["std_aucs"], fmt='.', capsize=2, alpha=0.2, label="AUC")
    ax.errorbar(plot_xy_data["amount_of_sampling_list"], plot_xy_data["av_shuff_aucs"], yerr=plot_xy_data["std_shuff_aucs"], fmt='.', capsize=2, alpha=0.2, label="Shuffled AUC")
    ax.set_xlabel("N models")
    ax.set_ylabel("ROC - AUC")
    ax.legend()

    save("roc_as_amount_of_sampling" + suffix)

def plot_per_frame_roc(x, ef, refij, save=None, suffix=""):
    aij_mat = mv.Z2A(x['samples']['z']) > 0.5
    plot_xy_data = per_frame_roc(aij_mat, ef, refij)

    fig, ax = plt.subplots()
    ax.plot(plot_xy_data["aucs"], plot_xy_data["scores"], ".", alpha=0.2)
    ax.set_xlabel("AUC")
    ax.set_ylabel("Score")
    ax.set_title("AUC per model")
    save("per_frame_roc" + suffix)

def plot_a_b_roc(x, refij, o, save=None, suffix="", alpha=0.6):
    """
    Plot ROC of A, B, and A+B 
    """
    if refij is None:
        return
    aij_mat = mv.Z2A(x["samples"]["z"]) > 0.5
    n_chains, n_iter, M = aij_mat.shape
    # flatten the first two dimensions
    aij_mat_flat = aij_mat.reshape(-1, M)
    indices = jax.random.permutation(jax.random.PRNGKey(0), jnp.arange(n_chains * n_iter))
    midpoint = len(indices) // 2
    av_a_models = np.mean(aij_mat_flat[indices[:midpoint], :], axis=0)
    av_b_models = np.mean(aij_mat_flat[indices[midpoint:], :], axis=0)
    av_ab_models = np.mean(aij_mat_flat, axis=0)

    representative_model1 = np.array(aij_mat_flat[0, :])
    representative_modelm = np.array(aij_mat_flat[midpoint, :])
    representative_modelm2 = np.array(aij_mat_flat[-1, :])

    models = dict(a=av_a_models, b=av_b_models, ab=av_ab_models, n1=representative_model1, n2=representative_modelm, n3=representative_modelm2)
    fig, ax = plt.subplots()
    for key, model in models.items():
        auc = sklearn.metrics.roc_auc_score(y_true = refij, y_score = model)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(refij, model)
        ax.plot(fpr, tpr, alpha=alpha, label=f"{key} - AUC={round(auc, 2)}")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    save(f"ab_roc" + suffix)

    n_ref = np.sum(refij)
    df = pd.DataFrame({"ref type" : [suffix], "n ref" : [n_ref], "auc" : [auc]})
    df.to_csv(o / ("ab_roc_table" + suffix + ".tsv"), sep="\t", index=False)

def filter_by_nodes(df, node_lst, a_col, b_col):
    df = df[df[a_col].isin(node_lst) & df[b_col].isin(node_lst)]
    return df

def build_expected_edges(node_lst):
    edge_dict = {}
    for i, a in enumerate(node_lst):
        for j in range(i+1, len(node_lst)):
            b = node_lst[j]
            edge_dict[frozenset((a, b))] = 1
    return edge_dict

def build_new_edge_dict(uref, expected_edges):
    uref._build_edge_dict()
    new_edge_dict = {}
    for edge in expected_edges:
        if edge in uref._edge_dict:
            new_edge_dict[edge] = uref._edge_dict[edge]
        else:
            new_edge_dict[edge] = 0.
    return new_edge_dict

def edge_dict2df(edge_dict, acolname="auid", bcolname="buid"):
    a = []
    b = []
    w = []
    for edge, weight in edge_dict.items():
        edge = list(edge)
        a.append(edge[0])
        b.append(edge[1])
        w.append(weight)
    return pd.DataFrame({acolname: a, bcolname: b, "w": w})

def get_buffered_humap_prediction_df(model_data, humap2_str_path="../data/processed/references/humap2_ppis_all.tsv"):
    humap2_all_pred = pd.read_csv(humap2_str_path, sep="\t")
    name2uid = gbf.get_cullin_reindexer()
    nodes = [name2uid[name] for name in model_data["name2node_idx"]] 
    humap2_all_pred = filter_by_nodes(humap2_all_pred, nodes, "auid", "buid")
    # Add zeros to edge pairs that were not guessed
    u = uel.UndirectedEdgeList()
    u.update_from_df(humap2_all_pred, "auid", "buid", edge_value_colname="w", multi_edge_value_merge_strategy="max")
    new_edge_dict = build_new_edge_dict(u, build_expected_edges(nodes))
    return edge_dict2df(new_edge_dict)

def get_buffered_saint_prediction_df(model_data):
    saint_scores = data_io.get_cullin_saint_scores_edgelist()
    name2uid = gbf.get_cullin_reindexer()
    nodes = [name2uid[name] for name in model_data["name2node_idx"]] 
    saint_scores = filter_by_nodes(saint_scores, nodes, "auid", "buid")

    
    return saint_scores
        

def get_and_align_humap_prediction(model_data, humap2_str_path="../data/processed/references/humap2_ppis_all.tsv"):
    humap2_buffered_df = get_buffered_humap_prediction_df(model_data, humap2_str_path)
    u = uel.UndirectedEdgeList()
    u.update_from_df(humap2_buffered_df, "auid", "buid", edge_value_colname="w", multi_edge_value_merge_strategy="max")
    assert u.nedges == model_data["M"], (len(u._edge_dict), model_data["M"])
    aligned_prediction = align_prediction_to_model(model_data, u)
    return aligned_prediction 

def get_and_align_saint_prediction(model_data, o):
    saint_max_bait_prey_buffered_df = get_maximal_bait_prey_composite_scores(model_data, o)
    u = uel.UndirectedEdgeList()
    u.update_from_df(saint_max_bait_prey_buffered_df, "auid", "buid", edge_value_colname="w", multi_edge_value_merge_strategy="max")
    assert u.nedges == model_data["M"], (len(u._edge_dict), model_data["M"])
    aligned_prediction = align_prediction_to_model(model_data, u)
    return aligned_prediction 

def get_maximal_bait_prey_composite_scores(model_data, o):
    composite_table_df = pd.read_csv(o / "composite_table.tsv", sep="\t")
    reindexer = data_io.get_cullin_reindexer()
    uids = [reindexer[x] for x in model_data["name2node_idx"]]

    bait_prey_scores = defaultdict(float) 
    for i, r in composite_table_df.iterrows():
        bait_uid = reindexer[r["Bait"]]
        prey_uid = reindexer[r["Prey"]]

        if bait_uid in uids and prey_uid in uids:
            dict_score = bait_prey_scores[(bait_uid, prey_uid)]
            current_score = r["MSscore"]
            if current_score > dict_score:
                bait_prey_scores[(bait_uid, prey_uid)] = current_score 
    
    bait_uid = []
    prey_uid = []
    scores = []
    for (b, p), s in bait_prey_scores.items():
        bait_uid.append(b)
        prey_uid.append(p)
        scores.append(s)

    df = pd.DataFrame({"auid": bait_uid, "buid": prey_uid, "w": scores}) 
    u = uel.UndirectedEdgeList()
    u.update_from_df(df, "auid", "buid", edge_value_colname="w", multi_edge_value_merge_strategy="max")
    new_edge_dict = build_new_edge_dict(u, build_expected_edges(uids))
    return edge_dict2df(new_edge_dict)
    
def get_and_align_HuRI_predictions(x, model_data):
    huri_all_pred = pd.read_csv("../data/processed/references/HuRI_reference.tsv", sep="\t")
    nodes = list(model_data["node_name2uid"].keys())
    huri_all_pred = filter_by_nodes(huri_all_pred, nodes, "auid", "buid")

def get_norm_and_deg_reweight(aij, model_data):
    assert aij.ndim == 1
    N = model_data["N"]
    aij = mv.flat2matrix(aij, n=N)
    N, N2 = aij.shape
    assert N == N2, (N, N2)
    min_val = np.min(aij)
    max_val = np.max(aij)
    aij = (aij - min_val) / (max_val - min_val)
    weighted_degree = np.sum(aij, axis=1)
    weighted_degree = weighted_degree  / N
    compliment = 1 - weighted_degree
    aij = aij * compliment
    aij = aij * compliment.T
    aij = mv.matrix2flat(aij)
    return aij

def plot_humap_saint_inm_roc(x, model_data, refij, save=None, o=None, suffix="", decimals=2, alpha=0.6):
    if refij is None:
        return
    humap_pred = get_and_align_humap_prediction(model_data)
    saint_pred = get_and_align_saint_prediction(model_data, o)
    assert x["samples"]["z"].ndim == 3, x["samples"]["z"].shape
    av_aij_mat = np.mean(mv.Z2A(x["samples"]["z"]) > 0.5, axis=(0, 1))
    norm_and_deg_reweight = get_norm_and_deg_reweight(av_aij_mat, model_data)
     
    hfpr, htpr, hthresholds = sklearn.metrics.roc_curve(refij, humap_pred)
    sfpr, stpr, _ = sklearn.metrics.roc_curve(refij, saint_pred)
    afpr, atpr, _ = sklearn.metrics.roc_curve(refij, av_aij_mat)
    nfpr, ntpr, _ = sklearn.metrics.roc_curve(refij, norm_and_deg_reweight)

    hauc = sklearn.metrics.roc_auc_score(y_true = refij, y_score = humap_pred)
    sauc = sklearn.metrics.roc_auc_score(y_true = refij, y_score = saint_pred)
    aauc = sklearn.metrics.roc_auc_score(y_true = refij, y_score = av_aij_mat)
    nauc = sklearn.metrics.roc_auc_score(y_true = refij, y_score = norm_and_deg_reweight)

    fig, ax = plt.subplots()
    ax.plot(hfpr, htpr, alpha=alpha, label=f"HuMAP 2.0 - AUC={hauc:.{decimals}f}")
    ax.plot(sfpr, stpr, alpha=alpha, label=f"SAINT     - AUC={sauc:.{decimals}f}")
    ax.plot(afpr, atpr, alpha=alpha, label=f"INM       - AUC={aauc:.{decimals}f}")
    ax.plot(nfpr, ntpr, alpha=alpha, label=f"INM (norm and deg reweighted) - AUC={nauc:.{decimals}f}")

    ax.legend()
    save("humap_roc" + suffix)


def plot_sliding_window_roc(x, ef, refij, window_size = 25, save=None, suffix=""):
    assert isinstance(window_size, int)
    aij_mat = mv.Z2A(x['samples']['z']) > 0.5
    plot_xy_data = sliding_window_roc(aij_mat, ef, refij, window_size = window_size)
    fig, ax = plt.subplots()
    ax.set_title(f"Sliding window ({window_size}) ROC")
    ax.plot(plot_xy_data["shuff_aucs"], plot_xy_data["mean_scores"], ".", alpha=0.8, label="Shuffled reference")
    ax.plot(plot_xy_data["aucs"], plot_xy_data["mean_scores"],  ".", alpha=0.2, label=f"{suffix} reference")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Mean score")
    ax.legend()
    save(f"sliding_window_roc_{window_size}" + suffix)

    fig, ax = plt.subplots()
    ax.set_title(f"Sliding window ({window_size}) ROC")
    ax.plot(plot_xy_data["shuff_aucs"], np.log10(plot_xy_data["mean_scores"]), ".", alpha=0.8,  label="Shuffled reference")
    ax.plot(plot_xy_data["aucs"],       np.log10(plot_xy_data["mean_scores"]),  ".", alpha=0.2, label=f"{suffix} reference")
    ax.set_xlabel("AUC")
    ax.set_ylabel("log10 Mean score")
    ax.legend()
    save(f"sliding_window_roc_{window_size}_log10" + suffix)

    

def animate_modeling_run_frames(aij_mat, model_data, save=None, o=None, cmap="cividis", interval=10, suffix=""):
    fig, ax = plt.subplots(figsize=(12, 12))
    N = model_data["N"]
    if aij_mat.ndim == 3:
        # select a representative chain
        a, b, c = aij_mat.shape
        assert b != c, f"{(b, c)}should be chain, iter, M"
        aij_mat = aij_mat[0, :, :]
    n_iter, M = aij_mat.shape 

    nframe = n_iter
    
    mat = ax.matshow(mv.flat2matrix(aij_mat[0, :], n=N), cmap=cmap)

    def update(frame):
        m = mv.flat2matrix(aij_mat[frame, :], n=N)
        mat.set_data(m)
    
    ani = animation.FuncAnimation(fig, update, frames=nframe, repeat=False, interval = interval)
    ani.save(o / f"representative_modeling_run{suffix}.mp4")
    #ani.save(o / f"representative_modeling_run{suffix}.gif", writer="pillow")

    
def dim_aware_max(x):
    if x.ndim == 1:
        return jnp.max(x)
    elif x.ndim == 2:
        return jnp.max(x, axis=1)
    else:
        raise ValueError("Only 1 or 2 dimensions are supported")

def ppv_per_chain_based_on_amount_of_sampling(
        x, refij, amount_of_sampling_list = None):
    if x.ndim == 2:
        n_chains, n_iter = x.shape
    elif x.ndim == 3:
        n_chains, n_iter, _ = x.shape
    if amount_of_sampling_list is None:
        if n_iter > 2_000:
            amount_of_sampling_list = [1] + list(np.arange(10, n_iter, 1_000)) 
        else:
            amount_of_sampling_list = _STD_AMOUNT_OF_SAMPING
    results = metric_as_a_function_of_amount_of_sampling_per_chain(x, lambda x: ppv_per_iteration_vectorized(x, refij), amount_of_sampling_list)
    return results

def score_vs_X_plot(rng_key, samples, ef, refij, Y, title, ylabel, prefix, save=None, o=None, N=100_000, suffix=""):
    scores = ef["potential_energy"]
    assert scores.ndim == 2
    n_chains, n_iter = scores.shape
    
    Nsamples = n_chains * n_iter

    indices = jax.random.permutation(rng_key, jnp.arange(Nsamples))[:N]

    aij_mat = mv.Z2A(samples['z']) > 0.5
    
    Y_f = jax.jit(Y)
    y = Y_f(aij_mat, refij)

    assert scores.shape == y.shape, (scores.shape, y.shape)

    scores_subsample = jnp.ravel(scores)[indices]

    y_subsample = jnp.ravel(y)[indices]

    fig, ax = plt.subplots()
    ax.plot(scores_subsample, y_subsample, 'k.', alpha=0.2)
    ax.set_xlabel("score")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save(prefix + suffix)


def score_vs_ppv_plot(rng_key, samples, ef, refij, save=None, o=None, N=100_000, suffix=""):
    """
    Plot the score vs ppv for each chain
    """

    scores = ef["potential_energy"]
    assert scores.ndim == 2
    n_chains, n_iter = scores.shape
    
    Nsamples = n_chains * n_iter

    indices = jax.random.permutation(rng_key, jnp.arange(Nsamples))[:N]

    aij_mat = mv.Z2A(samples['z']) > 0.5
    ppv_f = jax.jit(ppv_per_iteration_vectorized_3d)
    ppv = ppv_f(aij_mat, refij).T

    assert scores.shape == ppv.shape, (scores.shape, ppv.shape)

    scores_subsample = jnp.ravel(scores)[indices]
    ppv_subsample = jnp.ravel(ppv)[indices]

    fig, ax = plt.subplots()
    ax.plot(scores_subsample, ppv_subsample, 'k.', alpha=0.2)
    ax.set_xlabel("score")
    ax.set_ylabel("ppv")
    ax.set_title("Score vs PPV")
    save("score_vs_ppv" + suffix)

score_vs_ppv_plot = partial(score_vs_X_plot, Y=lambda x, y : ppv_per_iteration_vectorized_3d(x, y).T, title="Score vs PPV", ylabel="ppv", prefix="score_vs_ppv",)
score_vs_tp_plot = partial(score_vs_X_plot, Y=lambda x, y: tp_per_iteration_vectorized_3d(x, y).T, title="Score vs TPs", ylabel="TPs", prefix="score_vs_tp",)
score_vs_fp_plot = partial(score_vs_X_plot, Y=lambda x, y: fp_per_iteration_vectorized_3d(x, y).T, title="Score vs FPs", ylabel="FPs", prefix="score_vs_fp",)
score_vs_tn_plot = partial(score_vs_X_plot, Y=lambda x, y: tn_per_iteration_vectorized_3d(x, y).T, title="Score vs TNs", ylabel="TNs", prefix="score_vs_tn",)
score_vs_fn_plot = partial(score_vs_X_plot, Y=lambda x, y: fn_per_iteration_vectorized_3d(x, y).T, title="Score vs FNs", ylabel="FNs", prefix="score_vs_fn",)

def errplot_from_av_std_dict(av_std_dict, xlabel, ylabel, title, savename, save=None, alpha=0.2):
    fig, ax = plt.subplots()
    x = av_std_dict['amount_of_sampling_list']
    y = av_std_dict['out']
    ax.errorbar(x, y[:, _AV_DIM], yerr=y[:, _STD_DIM], fmt='.', alpha=alpha, capsize=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save(savename)
    


def n_tp(aij, refij):
    """
    Return the values of the array where both are 1
    Boolean arrays
    """
    return jnp.sum((aij == 1) &  (refij == 1))

def n_fp(aij, refij):
    return jnp.sum((aij == 1) & (refij == 0))

def n_tn(aij, refij):
    return jnp.sum((aij == 0) & (refij == 0))

def n_fn(aij, refij):
    return jnp.sum((aij == 0) & (refij == 1))

def n_edge(aij):
    return np.sum(aij > 0.5)

def merged_results_two_subsets_scores_hist_and_test(merged_results, save=None, o = None, suffix="", min_val=None, max_val=None):
    scores = merged_results["extra_fields"]["potential_energy"]
    
    nchains, n_iter = scores.shape
    scores_flat = np.ravel(scores)
    scores_flat = np.array(jax.random.permutation(jax.random.PRNGKey(0), scores_flat))
    N_scores = len(scores_flat)
    midpoint = N_scores // 2
    scores1 = scores_flat[:midpoint] 
    scores2 = scores_flat[midpoint:]
    assert not np.all(scores1 == scores2)

    test_results = sp.stats.ks_2samp(scores1, scores2)

    # write a csv file for the two sample test results
    pd.DataFrame(test_results._asdict(), index = [0],).to_csv(o / "ks_2samp_scores_results.tsv", sep="\t")

    med = np.median(scores_flat)
    sigma = np.std(scores_flat)

    if (min_val is None) and (max_val is None):
        n_sigma = 1. 
        max_val = med + n_sigma * sigma
        min_val = med - n_sigma * sigma
    hist_range = (min_val, max_val)

    fig, ax = plt.subplots()
    ax.hist(scores1, bins=100, alpha=0.5, label=f"subset 1 N = {len(scores1)}", range=hist_range)
    ax.hist(scores2, bins=100, alpha=0.5, label=f"subset 2 N = {len(scores2)}", range=hist_range)
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    plt.legend()
    D = "{:.3f}".format(test_results.statistic)
    pval = "{:.3f}".format(test_results.pvalue)

    s = f"Two-sample KS test\nD={D}\np-value={pval}"
    plt.text(0.6, 0.6, s, transform=plt.gca().transAxes)
    save("scores_2_hist" + suffix)

def _loop(aij_mat, refij, f):
    N, _ = aij_mat.shape
    output = np.zeros(N)
    for i in range(N):
        output[i] = f(aij_mat[i, :], refij)
    return output

n_tps = partial(_loop, f=n_tp)
n_fps = partial(_loop, f=n_fp)
n_edges = partial(_loop, f=lambda x, _: n_edge(x), refij=None)

def n_tps_from_samples(samples, refij):
    aij_mat = mv.Z2A(samples['z'])
    aij_mat = aij_mat > 0.5
    return n_tps(aij_mat, refij)

def n_fps_from_samples(samples, refij):
    aij_mat = mv.Z2A(samples['z'])
    aij_mat = aij_mat > 0.5
    return n_fps(aij_mat, refij)

def postprocess_samples(i, fbasename, merge=False):
    """
    Merge argument
    - For a single chain - the concatentation runs along the 0th axis
      concatentating the warmup and non warmup samples.
      (SAMPLES, VARIABLES ...)
    - For multiple chains - the concatenation runs along the 1st axis
      because the 0th axis is the chain axis. 
    """
    if merge:
        def concatf(a, b):
            return np.concatenate([a, b], axis=1)
    else:
        def concatf(a, b):
            return np.concatenate([a, b], axis=0)

    # Get the model data
    warmup_out_dict = input_load(i, fbasename, "_warmup_samples") 
    sample_out_dict = input_load(i, fbasename, "")

    def concat_warmup_samples(w, s):
        ws = w['samples']
        ss = s['samples']

        we = w['extra_fields']
        se = s['extra_fields']

        # concatenate along the first axis with warmup first 
        os = {}
        oe = {}
        found_keys = []
        for key in ws:
            os[key] = concatf(ws[key], ss[key])
            found_keys.append(key)
        for key in we:
            oe[key] = concatf(we[key], se[key])
            found_keys.append(key)
        found_keys = list(set(found_keys))

        out = {"samples" : os, "extra_fields" : oe}

        # add keys not in warmup
        for key in se:
            if key not in found_keys:
                out['extra_fields'][key] = se[key]
        for key in ss:
            if key not in found_keys:
                out['samples'][key] = ss[key]
        for key in s:
            if key not in found_keys:
                if key not in ("samples", "extra_fields"):
                    out[key] = s[key]
        return out 

    results = concat_warmup_samples(warmup_out_dict, sample_out_dict) 
    return results


def align_reference_to_model(model_data, uref, mode="cullin",):
    """
    Order edges in the reference according to the model.
    """
    assert isinstance(uref, uel.UndirectedEdgeList)
    n_possible_edges = model_data['M']
    node_idx2name = model_data["node_idx2name"]
    n_nodes = model_data['N']
    # uref is in UID but model is in names. Reindex
    if mode == "cullin":
        reindexer = {val : key for key, val in data_io.get_cullin_reindexer().items()}
        uref.reindex(reindexer, enforce_coverage = False) 
    else:
        raise NotImplementedError("Only cullin has been implemented")

    uref._build_edge_dict()
    reflist_out = np.zeros(n_possible_edges, dtype=np.int64)
    for k in range(n_possible_edges):
        i, j = mv.ij_from(k, n_nodes)
        i_node_name = node_idx2name[i]
        j_node_name = node_idx2name[j]
        edge = frozenset([i_node_name, j_node_name])
        if edge in uref._edge_dict:
            reflist_out[k] = 1
        k += 1
    assert j == n_nodes -1
    return reflist_out

def align_prediction_to_model(model_data, upred, mode="cullin"):
    assert isinstance(upred, uel.UndirectedEdgeList)
    assert np.sum(upred.edge_values) > 10 
    n_possible_edges = model_data['M']
    n_nodes = model_data['N']
    if mode == "cullin":
        reindexer = data_io.get_cullin_reindexer()

    edge_dict = {}

    for idx, a_node in enumerate(upred.a_nodes):
        b_node = upred.b_nodes[idx]
        edge_weight = upred.edge_values[idx]  
        edge_dict[frozenset((a_node, b_node))] = edge_weight

    assert n_possible_edges == len(edge_dict) 

    reflist_out = np.zeros(n_possible_edges, dtype=np.float64)
    for k in range(n_possible_edges):
        i, j = mv.ij_from(k, n_nodes)
        i_node_name = model_data["node_idx2name"][i]
        j_node_name = model_data["node_idx2name"][j]
        i_node_uid = reindexer[i_node_name]
        j_node_uid = reindexer[j_node_name]
        edge_ = frozenset((i_node_uid, j_node_uid))
        assert edge_ in edge_dict
        edge_weight = edge_dict[edge_]
        reflist_out[k] = edge_weight
    assert np.sum(reflist_out) > 10
    return reflist_out

def bootstrap_samples(rng_key, refij, n_boostraps = 100):
    n = refij.shape[0]
    output = jnp.zeros((n_boostraps, n))
    for i in range(n_boostraps):
        rng_key, key = jax.random.split(rng_key)
        shuff_ij = np.array(jax.random.permutation(key, refij))
        output = output.at[i, :].set(shuff_ij)
    return output

def run_multi_chain_version_of_single_chain_plots(x, model_data, save, o, suffix=""):
    # Unpack
    samples = x['samples']
    ef = x['extra_fields']

    nchains, n_iter, M = samples['z'].shape
    a = mv.Z2A(samples['z']) 
    mean = np.mean(a, axis=0) 

    a_typed = np.array(a > 0.5)

    # generate controls for a single chain
    control_ij = a_typed[0, 0, :] 
    control_last_ij = a_typed[0, -1, :]

    #direct_tps = np.array(n_tps(a_typed, direct_ij))
    #direct_fps = np.array(n_fps(a_typed, direct_ij))

    rng_key = jax.random.PRNGKey(122387)
    direct_bootstrap_reference = bootstrap_samples(rng_key, direct_ij)
    n_boostraps, n_true = direct_bootstrap_reference.shape

    prev_bootstrap_direct_tps = None 
    for i in range(n_boostraps):
        bootstrap_direct_tps = np.array(n_tps(a_typed, direct_bootstrap_reference[i, :]))
        if prev_bootstrap_direct_tps is None:
            prev_bootstrap_direct_tps = bootstrap_direct_tps
        else:
            prev_bootstrap_direct_tps = np.concatenate([prev_bootstrap_direct_tps, bootstrap_direct_tps])

    prev_bootstrap_direct_tps = np.ravel(prev_bootstrap_direct_tps) 
    direct_mu_tps = np.mean(direct_tps)
    direct_sigma_tps = np.mean(direct_tps)

    boot_mu_tps = np.mean(prev_bootstrap_direct_tps)
    boot_mu_sigma_tps = np.std(prev_bootstrap_direct_tps)

    # Make an error bar plot
    fig, ax = plt.subplots()
    x = [0, 1]
    y = [direct_mu_tps, boot_mu_tps]
    yerr = [direct_sigma_tps, boot_mu_sigma_tps]
    ax.errorbar(x, y, yerr=yerr, fmt='o')
    ax.set_title("Quantification of direct TPs vs bootstrapped reference")
    ax.set_ylabel("TPs")
    labels = ["direct", "bootstrap"]
    plt.xticks(x, labels)
    save("direct_vs_bootstrap_tps")

    plt.hist(prev_bootstrap_direct_tps, bins = 100, alpha=0.5, density=True, label="bootstrap direct TPs")
    plt.hist(direct_tps, alpha=0.5, bins = 100, density=True, label="direct TPs")
    plt.xlabel("direct TPs")
    plt.ylabel("density")
    plt.legend()
    save("direct_vs_bootstrap_tps_hist")

    costructure_tps = np.array(n_tps(a_typed, costructure_ij))
    costructure_fps = np.array(n_fps(a_typed, costructure_ij))

    control_first_tps = np.array(n_tps(a_typed, control_ij))
    control_first_fps = np.array(n_fps(a_typed, control_ij))

    control_last_tps = np.array(n_tps(a_typed, control_last_ij))
    control_last_fps = np.array(n_fps(a_typed, control_last_ij))

    shuff_direct_tps = np.array(n_tps(a_typed, shuff_direct_ij))
    shuff_direct_fps = np.array(n_fps(a_typed, shuff_direct_ij))

    shuff_costructure_tps = np.array(n_tps(a_typed, shuff_costructure_ij))
    shuff_costructure_fps = np.array(n_fps(a_typed, shuff_costructure_ij))
    
    n_total_edges = np.array(n_edges(a_typed))
    score = np.array(ef['potential_energy'])
    
    plot_plot(direct_tps,       score, "k.", "direct tp edges",       "score", "direct_tp_vs_score" + suffix) 
    plot_plot(direct_fps,       score, "k.", "direct fp edges",       "score", "direct_fp_vs_score" + suffix) 
    
    plot_plot(costructure_tps,  score, "k.", "costructure tp edges",  "score", "costructure_tp_vs_score" + suffix) 
    plot_plot(costructure_fps,  score, "k.", "costructure fp edges",  "score", "costructure_fp_vs_score" + suffix) 

    plot_plot(control_first_tps,      score, "k.", "CONTROL FIRST position tp edges", "score", "control_tp_vs_score" + suffix) 
    plot_plot(control_first_fps,      score, "k.", "CONTROL FIRST position fp edges", "score", "control_fp_vs_score" + suffix) 

    plot_plot(control_last_tps,      score, "k.", "CONTROL LAST position tp edges", "score", "control_last_tp_vs_score" + suffix) 
    plot_plot(control_last_fps,      score, "k.", "CONTROL LAST position fp edges", "score", "control_last_fp_vs_score" + suffix) 

    plot_plot(shuff_direct_tps, score, "k.", "shuff direct tp edges", "score", "shuff_direct_tp_vs_score" + suffix) 
    plot_plot(shuff_direct_fps, score, "k.", "shuff direct fp edges", "score", "shuff_direct_fp_vs_score" + suffix) 

    plot_plot(shuff_costructure_tps, score, "k.", "shuff costructure tp edges", "score", "shuff_costructure_tp_vs_score" + suffix) 
    plot_plot(shuff_costructure_fps, score, "k.", "shuff costructure fp edges", "score", "shuff_costructure_fp_vs_score" + suffix) 

    plot_plot(n_total_edges,    score, "k.", "N edges", "score", "n_edges_vs_score" + suffix)

    for key, val in dict(
            direct_tps = direct_tps,
            direct_fps = direct_fps,
            costructure_tps = costructure_tps,
            costructure_fps = costructure_fps,
            control_first_tps = control_first_tps,
            control_first_fps = control_first_fps,
            control_last_tps = control_last_tps,
            control_last_fps = control_last_fps,
            shuff_direct_tps = shuff_direct_tps,
            shuff_direct_fps = shuff_direct_fps,
            shuff_costructure_tps = shuff_costructure_tps,
            shuff_costructure_fps = shuff_costructure_fps,
            n_total_edges = n_total_edges,
            ).items():
        plot_caterpillar(
            y = val,
            xlabel = "iteration",
            ylabel = key,
            title = key,
            savename = f"{key}_caterpill" + suffix,)


    # Plot a table of the average value of every composite N
    if "new_composite_dict_norm_approx" in model_data:
        save_composite_table(model_data, o, samples)
    # Plot a graph of composite data satisfaction

    # Plot the average adjacency matrix  

    # Plot a histogram of edges

    plot_histogram(
            x = mean,
            xlabel = "mean edge value",
            ylabel = "count",
            savename = "edge_score_hist" + suffix,
            hist_range = (0, 1),)

    var = np.var(a, axis=0) 
    A = mv.flat2matrix(mean, N)
    V = mv.flat2matrix(var, N)
    plot_matrix(
            matrix = mv.flat2matrix(a[0, :] , N), 
            xlabel = "node",
            title = "Initial value",
            savename = "initial_value" + suffix,
            )
    plot_matrix(
            matrix = np.array(A),
            xlabel = "node",
            title = "Average edge score",
            savename = "mean_adjacency" + suffix,)

    # Energy histogram
    energy = np.array(ef["energy"])
    plot_histogram(
            x = energy,
            xlabel = "energy",
            ylabel = "count",
            title = "H(q,p) = U(q) + K(p)", 
            hist_range = None,
            savename = "energy" + suffix,)
    
    x_cater = np.arange(len(ef['energy'])) 
    # Energy Caterpillar
    plot_caterpillar(
            y = energy, 
            xlabel = "iteration",
            ylabel = "energy",
            title = "H(q,p) = U(q) + K(p)",
            savename = "energy_caterpill" + suffix,)

    # Mean acceptance histogram
    plot_histogram(
            x = np.array(ef['mean_accept_prob']),
            xlabel = "mean acceptance probability",
            ylabel = "count",
            title = None,
            hist_range = (0.5, 1.),
            savename = "mean_accept_prob" + suffix,)

    # potential energy 
    pe = np.array(ef['potential_energy'])

    plot_histogram(
            x = pe,
            xlabel = "potential energy",
            ylabel = "count",
            title = "U",
            hist_range = None,
            savename = "potential_energy" + suffix,)

    plot_caterpillar(
            y = pe,
            xlabel = "iteration",
            ylabel = "potential energy",
            title = "Score",
            savename = "potential_energy_caterpill" + suffix,)

    ke = energy - pe
    plot_histogram(
            x = ke,
            xlabel = "kinetic energy",
            ylabel = "count",
            title = "k",
            hist_range = None,
            savename = "kinetic_energy" + suffix,)
    
    plot_caterpillar(
            y = ke,
            xlabel = "iteration",
            ylabel = "kinetic energy",
            title = "k",
            savename = "kinetic_energy_caterpill" + suffix,)
    
    plot_histogram(
            x = np.array(ef['diverging']),
            xlabel = "diverging",
            ylabel = "count",
            savename = "diverging" + suffix,)
    
    plot_histogram(
            x = np.array(ef['accept_prob']),
            xlabel = "acceptance probability",
            ylabel = "count",
            savename = "accept_prob" + suffix,)

    for key in ("accept_prob", "mean_accept_prob", "diverging", "num_steps",
                "r_score", "sij_score", "z_score", "n_edges_score", "grad_r_score", "grad_z_score",
                "degree_score"):
        plot_stuff = False
        if (key in ef):  
            d = ef
            plot_stuff = True
        elif (key in samples):
            d = samples
            plot_stuff = True
        
        if plot_stuff:
            val = d[key]
            plot_caterpillar(
                    y = val,
                    xlabel = "iteration",
                    ylabel = key,
                    title = key,
                    savename = f"{key}_caterpill" + suffix,)

    # Plot the hist
    """
    # Animation 
    fig, ax = plt.subplots()
    mat = ax.matshow(np.zeros((N, N)))
    
    def animate(i):
        a_i = mv.flat2matrix(a[i, :], N)
        mat = ax.matshow(np.array(a_i))
        return mat,

    nsamples, nedges = samples['z'].shape 
    ani = animation.FuncAnimation(fig, animate, repeat=True,
       frames=nsamples, interval=50)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15,
                                     metadata=dict(artist='Me'),
                                     bitrate=1800)

    ani.save('edge_matrix.gif', writer=writer)
    """
    plot_matrix(
            matrix = np.array(V),
            title = "Edge variance",
            xlabel = "node",
            savename = "var_adjacency" + suffix,)

    # Mean Variance correlation plot
    plot_plot(
            x = mean,
            y = var,
            fmt = 'k.',
            xlabel = "Edge mean",
            ylabel = "Edge variance",
            savename = "mean_var_scatter" + suffix,)

    # Distribution of u 
    if 'u' in samples:
        u = samples['u']
        plot_histogram(
                x = u,
                xlabel = "$u$",
                ylabel = "count",
                title = "Nuisance variable $u$",
                hist_range = None, #(0, 0.005),
                savename = "hist_u" + suffix,) 
    w = []
    a = []
    b = []
    node_idx2name = model_data['node_idx2name']

    viral_proteins = ["vifprotein", "tatprotein", "nefprotein", "gagprotein",
                      "polpolyprotein"]

    for i in range(N):
        for j in range(0, i):
            w.append(A[i, j])
            a.append(node_idx2name[i])
            b.append(node_idx2name[j])
    
    df = pd.DataFrame({'a': a, 'b': b, 'w': w})
    base_name = str(o / i_.stem)
    df.to_csv(base_name + f"edge_score_table{suffix}.tsv", sep="\t", index=None)


    
def _main(o, i, mode, merge = False):
    if merge:
        i = Path(str(i.parent) + "_merged") / "merged_results" 
        o = i.parent
    logger = logging.getLogger(__name__)
    logger.info("Enter generate_sampling_figures")
    logging.info(f"Params")
    logging.info(f"    i:{i}")
    logging.info(f"    o:{o}")
    logging.info(f" mode:{mode}")

    i = Path(i)
    o = Path(o)

    def save(name):
        plt.savefig(str(o / i.stem) + f"_{name}_300.png", dpi=300)
        plt.savefig(str(o / i.stem) + f"_{name}_1200.png", dpi=1200)
        plt.close()
    
    @rc_context(rc = _spec_rc)
    def plot_spec_count_table(table, xlabel, ylabel, savename, title):
        fig, ax = plt.subplots()
        plt.matshow(np.array(table))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.colorbar()
        save(savename)

    @rc_context(rc = _corr_rc) 
    def plot_correlation_matrix(matrix, xlabel, savename, title, ylabel = None):
        if ylabel is None:
            ylabel = xlabel
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix, vmin=-1, vmax=1)
        plt.colorbar(shrink = 0.95, mappable = cax, ax = ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)

    @rc_context(rc = _matrix_rc) 
    def plot_matrix(matrix, xlabel, savename, title, ylabel = None):
        if ylabel is None:
            ylabel = xlabel
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix, vmin=0, vmax=1)
        plt.colorbar(shrink = 0.95, mappable = cax, ax = ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)

    @rc_context(rc = _matrix_rc)
    def plot_binary_matrix(matrix):
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix, vmin=0, vmax=1)
        plt.colorbar(shrink = 0.95, mappable = cax, ax = ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)


    def plot_histogram(x, xlabel, ylabel, savename, title = None, bins = 100, hist_range=(-1, 1),
                       text = None, textx = None, texty = None,): 
        if title is None:
            N = len(x)
            title = f"N={N}    {bins} bins"
        fig, ax = plt.subplots()
        plt.hist(x, bins = bins, range = hist_range)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if text:
            plt.text(textx, texty, text)
        save(savename)

    def plot_caterpillar(y, xlabel, ylabel, title, savename, alpha = 0.2):
        fig, ax = plt.subplots()
        x = np.arange(len(y))
        ax.plot(x, y, alpha = alpha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)

    def plot_plot(x, y, fmt, xlabel, ylabel, savename, title = None, alpha = 0.2):
        fig, ax = plt.subplots()
        ax.plot(x, y, fmt, alpha=alpha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(savename)

    def model_data_plots(model_data):
        # Correlation matrix
        plot_correlation_matrix(
                matrix = model_data['corr'],
                xlabel = 'node',
                savename = 'corr',
                title = "$R$",)
    
        # Correlation matrix
        plot_correlation_matrix(
                matrix = model_data['shuff_corr'],
                xlabel = "node",
                savename = "shuff_corr",
                title = "$R^{0}$",)
    
        # Composite dict - skip
    
        # Matrix plot of shuffled spec table
        plot_spec_count_table(
                table = model_data['selected_shuff_spec_table'],
                xlabel = "condition",
                ylabel = "node",
                savename = "selected_shuff_spec_table",
                title = "Shuffled spectral counts"),
    
        # Matrix plot of spec table
        plot_spec_count_table(
                table = model_data['selected_spec_table'],
                xlabel = "condition",
                ylabel = "node",
                savename = "selected_spec_table",
                title = "Spectral counts",)
    
        # lower_edge_prob - skip
        # uppder_edge_prob - skip
        # thresholds - skip
    
        # Matrix plot of correlation matrix after shuffling 
       # plot_correlation_matrix(
               # matrix = model_data['shuff_corr_all'],
              #  xlabel = "node",
              #  savename = "shuff_corr_all",
              #  title = "$R^0$",) 
        
        # name2node_idx - skip
        # node_idx2name - skip
    
        # N skip
        # M skip
    
        # Histogram of profile correlations 
        ###
        #plot_histogram(
               # x = model_data['apms_corr_flat'],
               # title = "AP-MS profile correlations",
               # xlabel = "$R_{ij}$",
               # ylabel = "count",
                #savename = "apms_corr_flat")
    
        # Histogram of Null all
        #n_null_bins = model_data['n_null_bins']
        #R0 = model_data['apms_shuff_corr_all_flat']
        #plot_histogram(
               # x = R0, 
               # title = "AP-MS permuted correlations",
               # xlabel = "$R_{ij}^{0}$",
              #  ylabel = "count",
              #  bins = n_null_bins,
              #  text = f"N {len(R0)}\nbins : {n_null_bins}",
              #  textx = 0.1,
              #  texty = 0.5,
              #  savename = "apms_shuff_corr_all_flat")
    
        # Histogram of Null from selected columns 
       # R0 = model_data['apms_shuff_corr_flat']
       # plot_histogram(
              #  x = R0, 
               # title = "AP-MS permuted correlations",
               # xlabel = "$R_{ij}^{0}$",
               # ylabel = "count",
              #  bins = n_null_bins,
              #  text = f"N {len(R0)}\nbins : {n_null_bins}",
               # textx = 0.1,
              #  texty = 0.5,
              #  savename = "apms_shuff_corr_flat")
    
        # Plot the spec_table
        spec_table = pd.read_csv(str(o / "spec_table.tsv"), sep="\t", index_col=0)
        plot_spec_count_table(
                table = spec_table,
                xlabel = "node",
                ylabel = "condition",
                title = "Spectral counts",
                savename = "spec_table",)

        # Conditionally plot the saint pair score edgelist table
        conditional_keys = ("saint_max_pair_score_edgelist",)
        N = model_data["N"]
        M = model_data["M"]
        for cond_key in conditional_keys:
            if cond_key in model_data:
                cond_val = model_data[cond_key]
                if cond_key == "saint_max_pair_score_edgelist":
                    cond_mat = mv.flat2matrix(cond_val, n=N)
                    plot_matrix(cond_mat,
                                xlabel = "node",
                                title = "Saint max pair score",
                                savename = "saint_max_pair_score")


    def single_chain_sampling_analysis():

        # Get the model data
        with open(str(i.parent / i.stem) + "_model_data.pkl", "rb") as f:
            model_data = pkl.load(f)
        i_ = i
    
        if not _dev:
            model_data_plots(model_data = model_data)
        # Get concatenated samples

        direct_ref = data_io.get_pdb_ppi_predict_direct_reference()
        costructure_ref = data_io.get_pdb_ppi_predict_cocomplex_reference()

        rng_key = jax.random.PRNGKey(122387)
        direct_ij = align_reference_to_model(model_data, direct_ref, mode=mode) 
        costructure_ij = align_reference_to_model(model_data, costructure_ref, mode=mode)

        rng_key, key  = jax.random.split(rng_key)
        shuff_costructure_ij = np.array(jax.random.permutation(key, costructure_ij))

        rng_key, key  = jax.random.split(rng_key)
        shuff_direct_ij = np.array(jax.random.permutation(key, direct_ij))

        x : dict = postprocess_samples(i.parent, fbasename=i.stem) 

        N = model_data['N']
        x = optional_flatten(x)
        with open(i, "rb") as f:
            x2 = pkl.load(f)
        x2 = optional_flatten(x2)

        def run_plots(x, suffix=""):
            # Unpack
            samples = x['samples']
            ef = x['extra_fields']
            nsamples, M = samples['z'].shape
            a = jax.nn.sigmoid((samples['z']-0.5)*1_000)
            mean = np.mean(a, axis=0) 
            a_typed = np.array(a > 0.5)

            control_ij = a_typed[0, :]
            control_last_ij = a_typed[-1, :]

            direct_tps = np.array(n_tps(a_typed, direct_ij))
            direct_fps = np.array(n_fps(a_typed, direct_ij))

            rng_key = jax.random.PRNGKey(122387)
            direct_bootstrap_reference = bootstrap_samples(rng_key, direct_ij)
            n_boostraps, n_true = direct_bootstrap_reference.shape

            prev_bootstrap_direct_tps = None 
            for i in range(n_boostraps):
                bootstrap_direct_tps = np.array(n_tps(a_typed, direct_bootstrap_reference[i, :]))
                if prev_bootstrap_direct_tps is None:
                    prev_bootstrap_direct_tps = bootstrap_direct_tps
                else:
                    prev_bootstrap_direct_tps = np.concatenate([prev_bootstrap_direct_tps, bootstrap_direct_tps])

            prev_bootstrap_direct_tps = np.ravel(prev_bootstrap_direct_tps) 
            direct_mu_tps = np.mean(direct_tps)
            direct_sigma_tps = np.mean(direct_tps)

            boot_mu_tps = np.mean(prev_bootstrap_direct_tps)
            boot_mu_sigma_tps = np.std(prev_bootstrap_direct_tps)

            # Make an error bar plot
            fig, ax = plt.subplots()
            x = [0, 1]
            y = [direct_mu_tps, boot_mu_tps]
            yerr = [direct_sigma_tps, boot_mu_sigma_tps]
            ax.errorbar(x, y, yerr=yerr, fmt='o')
            ax.set_title("Quantification of direct TPs vs bootstrapped reference")
            ax.set_ylabel("TPs")
            labels = ["direct", "bootstrap"]
            plt.xticks(x, labels)
            save("direct_vs_bootstrap_tps")

            plt.hist(prev_bootstrap_direct_tps, bins = 100, alpha=0.5, density=True, label="bootstrap direct TPs")
            plt.hist(direct_tps, alpha=0.5, bins = 100, density=True, label="direct TPs")
            plt.xlabel("direct TPs")
            plt.ylabel("density")
            plt.legend()
            save("direct_vs_bootstrap_tps_hist")

            costructure_tps = np.array(n_tps(a_typed, costructure_ij))
            costructure_fps = np.array(n_fps(a_typed, costructure_ij))

            control_first_tps = np.array(n_tps(a_typed, control_ij))
            control_first_fps = np.array(n_fps(a_typed, control_ij))

            control_last_tps = np.array(n_tps(a_typed, control_last_ij))
            control_last_fps = np.array(n_fps(a_typed, control_last_ij))

            shuff_direct_tps = np.array(n_tps(a_typed, shuff_direct_ij))
            shuff_direct_fps = np.array(n_fps(a_typed, shuff_direct_ij))

            shuff_costructure_tps = np.array(n_tps(a_typed, shuff_costructure_ij))
            shuff_costructure_fps = np.array(n_fps(a_typed, shuff_costructure_ij))
            
            n_total_edges = np.array(n_edges(a_typed))
            score = np.array(ef['potential_energy'])
            
            plot_plot(direct_tps,       score, "k.", "direct tp edges",       "score", "direct_tp_vs_score" + suffix) 
            plot_plot(direct_fps,       score, "k.", "direct fp edges",       "score", "direct_fp_vs_score" + suffix) 
            
            plot_plot(costructure_tps,  score, "k.", "costructure tp edges",  "score", "costructure_tp_vs_score" + suffix) 
            plot_plot(costructure_fps,  score, "k.", "costructure fp edges",  "score", "costructure_fp_vs_score" + suffix) 

            plot_plot(control_first_tps,      score, "k.", "CONTROL FIRST position tp edges", "score", "control_tp_vs_score" + suffix) 
            plot_plot(control_first_fps,      score, "k.", "CONTROL FIRST position fp edges", "score", "control_fp_vs_score" + suffix) 

            plot_plot(control_last_tps,      score, "k.", "CONTROL LAST position tp edges", "score", "control_last_tp_vs_score" + suffix) 
            plot_plot(control_last_fps,      score, "k.", "CONTROL LAST position fp edges", "score", "control_last_fp_vs_score" + suffix) 

            plot_plot(shuff_direct_tps, score, "k.", "shuff direct tp edges", "score", "shuff_direct_tp_vs_score" + suffix) 
            plot_plot(shuff_direct_fps, score, "k.", "shuff direct fp edges", "score", "shuff_direct_fp_vs_score" + suffix) 

            plot_plot(shuff_costructure_tps, score, "k.", "shuff costructure tp edges", "score", "shuff_costructure_tp_vs_score" + suffix) 
            plot_plot(shuff_costructure_fps, score, "k.", "shuff costructure fp edges", "score", "shuff_costructure_fp_vs_score" + suffix) 

            plot_plot(n_total_edges,    score, "k.", "N edges", "score", "n_edges_vs_score" + suffix)

            for key, val in dict(
                    direct_tps = direct_tps,
                    direct_fps = direct_fps,
                    costructure_tps = costructure_tps,
                    costructure_fps = costructure_fps,
                    control_first_tps = control_first_tps,
                    control_first_fps = control_first_fps,
                    control_last_tps = control_last_tps,
                    control_last_fps = control_last_fps,
                    shuff_direct_tps = shuff_direct_tps,
                    shuff_direct_fps = shuff_direct_fps,
                    shuff_costructure_tps = shuff_costructure_tps,
                    shuff_costructure_fps = shuff_costructure_fps,
                    n_total_edges = n_total_edges,
                    ).items():
                plot_caterpillar(
                    y = val,
                    xlabel = "iteration",
                    ylabel = key,
                    title = key,
                    savename = f"{key}_caterpill" + suffix,)


            # Plot a table of the average value of every composite N
            if "new_composite_dict_norm_approx" in model_data:
                save_composite_table(model_data, o, samples)
            # Plot a graph of composite data satisfaction

            # Plot the average adjacency matrix  

            # Plot a histogram of edges

            plot_histogram(
                    x = mean,
                    xlabel = "mean edge value",
                    ylabel = "count",
                    savename = "edge_score_hist" + suffix,
                    hist_range = (0, 1),)

            var = np.var(a, axis=0) 
            A = mv.flat2matrix(mean, N)
            V = mv.flat2matrix(var, N)
            plot_matrix(
                    matrix = mv.flat2matrix(a[0, :] , N), 
                    xlabel = "node",
                    title = "Initial value",
                    savename = "initial_value" + suffix,
                    )
            plot_matrix(
                    matrix = np.array(A),
                    xlabel = "node",
                    title = "Average edge score",
                    savename = "mean_adjacency" + suffix,)

            # Energy histogram
            energy = np.array(ef["energy"])
            plot_histogram(
                    x = energy,
                    xlabel = "energy",
                    ylabel = "count",
                    title = "H(q,p) = U(q) + K(p)", 
                    hist_range = None,
                    savename = "energy" + suffix,)
            
            x_cater = np.arange(len(ef['energy'])) 
            # Energy Caterpillar
            plot_caterpillar(
                    y = energy, 
                    xlabel = "iteration",
                    ylabel = "energy",
                    title = "H(q,p) = U(q) + K(p)",
                    savename = "energy_caterpill" + suffix,)

            # Mean acceptance histogram
            plot_histogram(
                    x = np.array(ef['mean_accept_prob']),
                    xlabel = "mean acceptance probability",
                    ylabel = "count",
                    title = None,
                    hist_range = (0.5, 1.),
                    savename = "mean_accept_prob" + suffix,)

            # potential energy 
            pe = np.array(ef['potential_energy'])

            plot_histogram(
                    x = pe,
                    xlabel = "potential energy",
                    ylabel = "count",
                    title = "U",
                    hist_range = None,
                    savename = "potential_energy" + suffix,)

            plot_caterpillar(
                    y = pe,
                    xlabel = "iteration",
                    ylabel = "potential energy",
                    title = "Score",
                    savename = "potential_energy_caterpill" + suffix,)

            ke = energy - pe
            plot_histogram(
                    x = ke,
                    xlabel = "kinetic energy",
                    ylabel = "count",
                    title = "k",
                    hist_range = None,
                    savename = "kinetic_energy" + suffix,)
            
            plot_caterpillar(
                    y = ke,
                    xlabel = "iteration",
                    ylabel = "kinetic energy",
                    title = "k",
                    savename = "kinetic_energy_caterpill" + suffix,)
            
            plot_histogram(
                    x = np.array(ef['diverging']),
                    xlabel = "diverging",
                    ylabel = "count",
                    savename = "diverging" + suffix,)
            
            plot_histogram(
                    x = np.array(ef['accept_prob']),
                    xlabel = "acceptance probability",
                    ylabel = "count",
                    savename = "accept_prob" + suffix,)

            for key in ("accept_prob", "mean_accept_prob", "diverging", "num_steps",
                    "r_score", "sij_score", "z_score", "n_edges_score", "grad_r_score", "grad_z_score",
                    "degree_score", "s_score"):
                plot_stuff = False
                if (key in ef):  
                    d = ef
                    plot_stuff = True
                elif (key in samples):
                    d = samples
                    plot_stuff = True
                
                if plot_stuff:
                    val = d[key]
                    plot_caterpillar(
                            y = val,
                            xlabel = "iteration",
                            ylabel = key,
                            title = key,
                            savename = f"{key}_caterpill" + suffix,)

            # Plot the hist
            """
            # Animation 
            fig, ax = plt.subplots()
            mat = ax.matshow(np.zeros((N, N)))
            
            def animate(i):
                a_i = mv.flat2matrix(a[i, :], N)
                mat = ax.matshow(np.array(a_i))
                return mat,

            nsamples, nedges = samples['z'].shape 
            ani = animation.FuncAnimation(fig, animate, repeat=True,
               frames=nsamples, interval=50)

            # To save the animation using Pillow as a gif
            writer = animation.PillowWriter(fps=15,
                                             metadata=dict(artist='Me'),
                                             bitrate=1800)

            ani.save('edge_matrix.gif', writer=writer)
            """
            plot_matrix(
                    matrix = np.array(V),
                    title = "Edge variance",
                    xlabel = "node",
                    savename = "var_adjacency" + suffix,)

            # Mean Variance correlation plot
            plot_plot(
                    x = mean,
                    y = var,
                    fmt = 'k.',
                    xlabel = "Edge mean",
                    ylabel = "Edge variance",
                    savename = "mean_var_scatter" + suffix,)

            # Distribution of u 
            if 'u' in samples:
                u = samples['u']
                plot_histogram(
                        x = u,
                        xlabel = "$u$",
                        ylabel = "count",
                        title = "Nuisance variable $u$",
                        hist_range = None, #(0, 0.005),
                        savename = "hist_u" + suffix,) 
            w = []
            a = []
            b = []
            node_idx2name = model_data['node_idx2name']

            viral_proteins = ["vifprotein", "tatprotein", "nefprotein", "gagprotein",
                              "polpolyprotein"]

            for i in range(N):
                for j in range(0, i):
                    w.append(A[i, j])
                    a.append(node_idx2name[i])
                    b.append(node_idx2name[j])
            
            df = pd.DataFrame({'a': a, 'b': b, 'w': w})
            base_name = str(o / i_.stem)
            df.to_csv(base_name + f"edge_score_table{suffix}.tsv", sep="\t", index=None)

        run_plots(x, suffix="_w_warmup")
        run_plots(x2)
        model_data_plots(model_data=model_data)
        # Write an edge list table of scores

        # Analysis

        # Count the number of Viral-Viral interactions 

        n_direct_ps = np.sum(direct_ij)
        n_costructure_ps = np.sum(costructure_ij)

        df = pd.DataFrame({"n_positives": [n_direct_ps, n_costructure_ps]}, index=["direct", "costructure"])
        df.to_csv(str(o / "summary.tsv"), sep="\t")
    
    def multi_chain_sampling_analysis(i=i, o=o):
        logging.basicConfig(level=logging.INFO,
                            output = o / "multi_chain_sampling_analysis.log",
                            format = "%(levelname)s:%(module)s:%(funcName)s:%(lineno)s:MSG:%(message)s",)
        # Reset i and o to reference the merged directory
        logging.info("multi_chain_sampling_analysis")
        # Load the model data
        with open(str(i.parent) + "/merged_results_model_data.pkl", "rb") as f:
            model_data = pkl.load(f)
        i_ = i
        
        if not _dev:
            model_data_plots(model_data)

        direct_ref = data_io.get_pdb_ppi_predict_direct_reference()
        costructure_ref = data_io.get_pdb_ppi_predict_cocomplex_reference()

        direct_ij = align_reference_to_model(model_data, direct_ref, mode=mode) 
        costructure_ij = align_reference_to_model(model_data, costructure_ref, mode=mode)

        rng_key = jax.random.PRNGKey(122387)
        rng_key, key  = jax.random.split(rng_key)

        shuff_costructure_ij = np.array(jax.random.permutation(key, costructure_ij))

        rng_key, key  = jax.random.split(rng_key)
        shuff_direct_ij = np.array(jax.random.permutation(key, direct_ij))
        start_time = time.time()
        x : dict = postprocess_samples(i.parent, fbasename=i.stem, merge=True) 
        end_time = time.time()
        logging.info(f"Time to load samples {end_time - start_time}")
        
        N = model_data['N']
        start_time = time.time()
        plot_degree_plots(x, save=save)
        x = optional_flatten(x)
        end_time = time.time()
        logging.info(f"Time to flatten samples {end_time - start_time}")
        
        start_time = time.time()
        multi_chain_validate_shapes(x, model_data)
        end_time = time.time()
        logging.info(f"Time to validate shapes {end_time - start_time}")
        
        fplot = partial(
            run_multichain_specific_plots,
            model_data = model_data,
            save=save, o = o)
        fplot(x, suffix="_w_warmup")
        
        del x
        with open(str(i) + ".pkl", "rb") as f:
            x2 = pkl.load(f)
        x2 = optional_flatten(x2)
        #multi_chain_validate_shapes(x2, model_data)
        fplot(x2)

        gplot = partial(
            run_multi_chain_version_of_single_chain_plots,
            model_data = model_data,
            save = save,
            o = o)
                
        #gplot(x = x, suffix = "_w_warmup")
        #gplot(x2)

    if not merge: 
        single_chain_sampling_analysis()
    else:
        multi_chain_sampling_analysis()
    #return samples




def multi_chain_validate_shapes(x, model_data):
    samples = x['samples']
    ef = x['extra_fields']

    n_chains, n_iter, M = samples['z'].shape
    assert M == model_data['M'], (M, model_data["M"])
    for key, value in ef.items():
        assert value.shape == (n_chains, n_iter), (key, value.shape)


def save_composite_table(model_data, o, samples):
    Ns = []
    means = []
    stds = []
    nmax = []
    for i in range(1000):
        key = f"c{i}_N"
        if key in samples:
            Ns.append(key)
            vals = samples[key]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            nmax.append(model_data['new_composite_dict_norm_approx'][i]['N'])
    df = pd.DataFrame({"Nmax" : nmax, "av" : means, "std" : stds}, index=Ns) 
    outpath = str(o / "composite_N.tsv")
    df.to_csv(outpath, sep="\t")


def optional_flatten(x):
    """
    If z is in matrix form, flatten it
    """
    samples = x["samples"]
    ef = x["extra_fields"]
    temp = {}
    flatf = jax.jit(mv.matrix2flat)
    flattened_lst = []
    for key, val in samples.items():
        if val.ndim < 2:
            continue 
        shape = val.shape
        if shape[-1] == shape[-2]:
            if val.ndim == 3: # (iter, N, N)
                for matrix in val:
                    flattend = flatf(matrix)
                    flattened_lst.append(flattend)
                temp[key] = np.array(flattened_lst)
            elif val.ndim == 4: #(chain, iter, N, N)
                n_chains, n_iter, N, _ = val.shape
                output = np.zeros((n_chains, n_iter, N*(N-1)//2))
                for i in range(n_chains):
                    for j in range(n_iter):
                        output[i, j, :] = flatf(val[i, j, :, :])
                temp[key] = output
            else:
                raise ValueError(f"Unknown shape {shape} for key {key}")
    for key in temp:
        x["samples"][key] = temp[key]
    return x


def remove_nans(x):
    """
    Remove iterations with NaNs 
    """
    samples = x["samples"]
    nans_at_samples = np.isnan(samples['z']).any(axis=2)
    nans_at_scores = np.isnan(x['extra_fields']['potential_energy'])
    

def networkx_graph_from_aij(aij_mat, model_data, threshold = 0.9):
    nchains, niter, M = aij_mat.shape

    G = nx.Graph()
    average_network = np.mean(aij_mat, axis=(0, 1))
    N = model_data["N"]
    nodeidx2name = model_data["node_idx2name"]
    for k in range(len(average_network)):
        weight = float(average_network[k])
        if weight > threshold:
            i, j = mv.flat2ij(k, N)
            u = nodeidx2name[i]
            v = nodeidx2name[j]
            G.add_edge(u, v, weight=float(average_network[k]))
    return G


def plot_degree_network(G, node_name):

    ...


_example = {
    "i": "../cullin_run0/0_model23_ll_lp_13.pkl", "o": "../cullin_run0/"}

if __name__ == "__main__":
    main()
