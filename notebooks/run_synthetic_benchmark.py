# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # SyntheticBenchmark50
# Do a synthetic benchmark with a 50 node network and model14
# ## -1
# 1. Imports
# 2. Make a directory for the outputs of this notebook if it doesn't exist
# 3. Set up logging
# 4. Global variables
# ## 0 Ground Truth Generation
# 1. Choose 3 bait and 50 node
# 2. Generate a network of direct protein-protein interactions
# 3. Ensure everything is connected to the bait
# ## 1 Data Generation
# 1. Generate profile similarities given the network
# ## 2 Fitting
# 1. Generate M chains of N samples each, saving to files
# 2. Ensure the score is saved
# 3. Optionally check sampling diagnostics
# ## 3 Analysis
# 1. Calculate accuracy for every position - (True accuracy metric)
# 2. Save the top accuracy
# 3. Save the top scoring model
# 2. Accuracy score precision
# ## 4 Increased Sampling
# 1. Calculate a dictionary of increasing combinations of the M chains
# ## 5 Plots
# 1. Plot accuracy score correlation for a representative trajectory
# 2. Plot top accuracy top score calculation across all trajectories
# 3. Show top accuracy as a function of increased sampling
# 4. Show top score as a function of increased sampling
# 5. Finish
#
# ## Scores Used in Numpyro Models
# ### Terms
# - target density   : $\pi (x) $
# - Hamiltonian      : $H(x, w) = -\log \pi (x) + V(w)$
# - potential energy : $U(x) = -\log \pi(x) $
# - kinematic energy : $V(w)$
# - log density      : $\log \pi(x)$
# - state            : (x, w)
# - p(w)             : 
#
# ### Implementation
# - potential_energy expects unconstrained inputs (done under the hood)
#
#

# +
## -1.1 Imports
from collections import defaultdict
import datetime
import importlib
from itertools import combinations
import jax
import jax.numpy as jnp
import json
import logging
import math
import networkit as nk
import numpy as np
import numpyro
from pathlib import Path
import pickle as pkl
from functools import partial
import pandas as pd
import sklearn
import seaborn as sns
import scipy as sp
from types import SimpleNamespace
import matplotlib.pyplot as plt
# Custom modules
import synthetic_benchmark
import _model_variations
import analyze_mcmc_output

# Reload custom modules for devlopment
importlib.reload(synthetic_benchmark)
importlib.reload(_model_variations)
importlib.reload(analyze_mcmc_output)

# +
def synthetic_benchmark_fn(analysis_name : str,
                        n_prey : int,
                        n_bait : int,
                        d_crit : int,
                        dir_name : str,
                        rseed : int,
                        edge_probability : float,
                        num_warmup=500,
                        num_samples=1000,
                        m_chains=400,
                        fig_dpi=300,
                        model_name="model14",
                        initial_position = None,
                        fit_up_to = 100,
                        analyze_next_N = 100,
                        generate_cartoon_figures = True,
                        n_successive_trajectories_to_analyze = 100,
                        num_bootstraps = 50,   
                        init_to_value_fp = None,
                        ):
    """
    Make a directoy, generate all figures
    Params:
      analysis_name : A name for the analysis
      n_prey : the number of prey types
      n_bait : the number of bait types
      d_crit : The maximal distance to search for shortest paths
      dir_name : a path to a directory for the analysis
      rseed : the seed for the random number generator
      edge_probability : the independant edge probability for the ground truth network
      num_warmup : number of warmup samples
      num_samples : number of MCMC samples for the NUTS sampler
      fig_dpi : the dpi for figures
      model_name : the name of the model to use for inference
      fit_up_to : Fit up to the next N trajectories
      analyze_next_N: Get log density, accuracy, precision, for the next N trajectories
      initial_position : instead of random intialization begin at a specific position
      overwrite_directory : overwrite everything
      init_to_value_fp : use an initial position when sampling if the file path is set  
    """
    
    # Make a directory
    assert n_prey > 2
    dir_path = Path(dir_name)
    if not dir_path.is_dir():
        dir_path.mkdir()
        
    # Set up logging
    time = datetime.datetime.now()
    log_filename = dir_path / f"{analysis_name}.log"
    # Set up the logger
    logging.basicConfig(filename=str(log_filename), encoding="utf-8", level=logging.DEBUG)
    logging.info("synthetic_benchmark()")
    logging.info("Parameters")
    logging.info(f"analysis_name:{analysis_name}")
    logging.info(f"n_prey:{n_prey}")
    logging.info(f"dir_name:{str(dir_name)}")
    logging.info(f"rseed:{rseed}")
    logging.info(f"edge_probability{edge_probability}")
    logging.info(f"num_warmup:{num_warmup}")
    logging.info(f"num_samples:{num_samples}")
    logging.info(f"m_chains:{m_chains}")
    logging.info(f"fig_dpi:{fig_dpi}")
    logging.info(f"model_name:{model_name}")
    
    # Make the PRNGKey
    key = jax.random.PRNGKey(rseed)
    
    key, keygen = jax.random.split(key)
    edge_probability = float(edge_probability)
    A, bait_idx, prey_idx = synthetic_benchmark.get_bait_prey_network(
            key,
            n_prey=n_prey,
            n_bait=n_bait,
            d_crit=d_crit,
            edge_prob=edge_probability,)
    
    # Plot the ground truth network
    A.plot() # xarray
    plt.hlines(np.array(bait_idx), -1, 50, label="Bait", alpha=0.8, color='r')
    plt.legend()
    plt.savefig(str(dir_path / "GroundTruth.png"), dpi=fig_dpi)
    plt.close()
    
    # Calculate the prey distance matrix
    D = _model_variations.shortest_paths_up_to_23(A.values)
    
    # 
    plt.matshow(D)
    plt.colorbar(shrink=0.8, label="Shortest path")
    plt.title("Ground Truth Distance Matrix")
    plt.savefig(str(dir_path / "GroundTruthDistanceMatrix.png"), dpi=fig_dpi)
    plt.close()
    
    Gtruth = adjacency2graph(A, weighted=False)
    
    # Plot the connected components
    nk.plot.connectedComponentsSizes(Gtruth)
    plt.title("connected component sizes") # A single connected component
    plt.savefig(str(dir_path / "ConnectedComponents.png"), dpi=fig_dpi)
    plt.close()
    
    # 1.1 Generate similarities
    key, keygen = jax.random.split(keygen)
    data = synthetic_benchmark.data_from_network_model14_rng(key, A.values)
    
    # Plot the similarities
    plt.matshow(data)
    plt.colorbar(shrink=0.8, label="Profile similarity")
    plt.savefig(str(dir_path / "SyntheticDataMatrix.png"), dpi=fig_dpi)
    plt.close()
    
    # Plot the Ground truth contact map
    plt.matshow(A.values, cmap="binary")
    plt.colorbar(shrink=0.8, label="Edge weight")
    plt.savefig(str(dir_path / "GroundTruthMatplot.png"), dpi=fig_dpi)
    plt.close()
    
    # Flatten the inputs for modeling
    reference_flat = _model_variations.matrix2flat(A.values) # use this method to flatten and unflatten matrices
    # so that indices are preserved
    
    data_flat =  _model_variations.matrix2flat(data)
    assert len(data_flat) == math.comb(n_prey, 2)
    
    # Model 14 is the mixture model
    # Model 15 has beta priors, not much different

    _model_variations.save_model22_ll_lp_data(
            save_dir,
            N,
            flattened_apms_similarity_scores,
            flattened_shuffled_apms,
            lower_edge_prob_bound = 0.0,
            upper_edge_prob_bound = 1.0,
            z2edge_slope = 1000,
            composites = [],
            BAIT_PREY_SLOPE)
    
    model14_names = ("model14", "model15")

    if model_name in model14_names:
        model_data = _model_variations.model14_data_getter()
        model_data['flattened_apms_similarity_scores'] = data_flat
        model = _model_variations.model14
    elif model_name == "model22_ll_lp":
        model_data = _model_variations.model22_ll_lp_data_getter(dir_name)
    else:
        raise ValueError(f"Unknown model name {model_name}")
    # Plot the reference and causal edges
    fig, ax = plt.subplots()
    n_true = int(np.sum(reference_flat))
    n_false = len(reference_flat) - n_true
    plt.hist(np.array(data_flat[reference_flat == 0]), label=f"False ({n_false})", bins=100, alpha=0.5)
    plt.hist(np.array(data_flat[reference_flat == 1]), label=f"True  ({n_true})", bins=100, alpha=0.5)
    plt.xlabel("Profile Similarity (Pearson R)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(str(dir_path / "SynthDataDistributions.png"), dpi=fig_dpi)
    plt.close()
    
    # Make the inital position relative to the analysis directory
    if initial_position is not None:
        initial_position = str(dir_path / initial_position)
    # Fit the model m_chain times writing to the file system
    fit(model_name = model_name,
        m_chains = m_chains,
        dir_path = dir_path,
        data_flat = data_flat,
        num_warmup = num_warmup,
        num_samples = num_samples,
        fit_up_to = fit_up_to,
        initial_position = initial_position)
    # Load in the first trajectory
    example_path = dir_path / f"0_{model_name}_0.pkl"
    # Example trajectory
    example_samples = _model_variations.load(str(example_path))
    
    if model_name in model14_names:
        analysis_dict = analyze_mcmc_output.model14_traj2analysis(
                str(example_path), 
                model_data=model_data,
                model=model,
                reference=np.array(reference_flat))
    else:
        raise ValueError(f"Unkown model name {model_name}")
    logging.info("Analyzing samples")        
    # Analyze samples
    analyze_samples(m_chains = m_chains,
                    dir_path = dir_path,
                    model_name = model_name,
                    model_data = model_data,
                    model = model,
                    reference_flat = reference_flat,
                    analyze_next_N = analyze_next_N,)
    
    if generate_cartoon_figures:
        # Generate Cartoon figures
        a = (0.0, 1)
        b = (0.5, 0.5)
        
        plt.errorbar(-1, 1, fmt='r^', alpha=0.5, label="Network A", xerr=0.05, capsize=3)
        plt.errorbar(1, 1, fmt='bo', alpha=0.5, label="Network B", xerr=0.05, capsize=3)
        plt.errorbar(0, 1, fmt='gx', alpha=0.5, label="Differential Network", xerr=0.1, capsize=3)
        plt.title("Low Uncertainty Regime")
        plt.xlabel("Network Model")
        plt.ylabel("Probability")
        plt.legend()
        plt.xlim(-2, 2)
        plt.savefig(f"{str(dir_path)}/LowUncertaintyCartoon.png", dpi=fig_dpi)
        
        # Cartoon
        a = (0.0, 1)
        b = (0.5, 0.5)
        
        plt.errorbar(-1, 1, fmt='r^', alpha=0.5, label="Network A", xerr=0.4, capsize=3)
        plt.errorbar(1, 1, fmt='bo', alpha=0.5, label="Network B", xerr=0.23, capsize=3)
        plt.errorbar(0, 1, fmt='gx', alpha=0.5, label="Differential Network", xerr=0.7, capsize=3)
        plt.title("High Uncertainty Regime")
        plt.xlabel("Network Model")
        plt.ylabel("Probability")
        plt.legend()
        plt.xlim(-2, 2)
        plt.savefig(f"{str(dir_path)}/HighUncertaintyCartoon.png", dpi=fig_dpi)
        plt.close()

    # Accuracy score correlation for an example trajectory
    average_network = accuracy_plot(analysis_dict = analysis_dict,
                                    samples = example_samples,
                                    model = model,
                                    model_data = model_data,
                                    model_name = model_name,
                                    reference_flat = reference_flat,
                                    dir_path = dir_path,
                                    fig_dpi = fig_dpi,
                                    calc_pearson_r = True,
                                    decoys = None,)
                                    
    # Why are the accuracy and precision in different modules?
    # Should place all benchmarking funcitons in a single module 
    average_precision = analyze_mcmc_output.model14_ap_score(
            traj_path = str(example_path), 
            model = model,
            model_data = model_data,
            reference = np.array(reference_flat))
       
    precision_plot(
            analysis_dict = analysis_dict,
            average_precision = average_precision,
            reference_flat = reference_flat,
            average_network = average_network,
            model_name = "model14",
            model = model,
            model_data = model_data,
            dir_path = dir_path,
            fig_dpi = fig_dpi)
    # decoy_1 = reference_flat * 0.21

    # Plot example score distributions for two edges
    fig, ax = plt.subplots()
    scores = analysis_dict['log_density'] * -1
    plt.plot(-scores)
    plt.xlabel("MCMC Step")
    plt.ylabel("Score (-log density)")
    plt.savefig(str(dir_path / "ExampleCatterpilliar.png"), dpi=fig_dpi)
    plt.close()
    
    fig, ax = plt.subplots()
    plt.hist(-scores, bins=30)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(str(dir_path / "ExampleScoreDistribution.png"), dpi=fig_dpi)
    plt.close()

    fig, ax = plt.subplots()
    # Select the first true and false edges
    true_edge_idx = 0
    false_edge_idx = 0
    for i in range(n_prey):
        if reference_flat[i] == 1:
            break
    true_edge_idx = i
    for i in range(n_prey):
        if reference_flat[i] == 0:
            break
    false_edge_idx = i


    plt.hist(example_samples['samples']['pT'][:, false_edge_idx], bins=100, label="False Edge")
    plt.hist(example_samples['samples']['pT'][:, true_edge_idx], bins=100, alpha=0.5, label="True Edge")
    plt.title("Representative Distributions for two edges")
    plt.xlabel("$\pi_T$")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(str(dir_path / "ExampleTwoEdges.png"), dpi=fig_dpi)
    plt.close()

    #
    acc_mat, scr_mat = get_top_accuracy_and_top_score_matrices_from_dir(
            dir_path = dir_path,
            traj_idx_arr = n_successive_trajectories_to_analyze,
            n = num_samples,
            model_name = model_name)

    results = calc_top_score_top_accuracy(
            rng_key = key,
            accuracy_matrix = acc_mat,
            score_matrix = scr_mat,
            k = num_bootstraps,
            J = n_successive_trajectories_to_analyze)
    top_accuracy_top_score_figure(
            dir_path,
            results,
            n_successive_trajectories_to_analyze,
            fig_dpi)
            
def _():
    """
    1. For k trajectories get successive trajectory groups
    2. Get the distribution of top scores and top accuracies
       2a. Get the top scoring model
       2b. Get the accuracy of the top scoring model
    3. plot the average top score and std
    4. plot the average top accuracy and std  
    5. plot the average accuracy of top scoring model and std 
    """

def fit(
        m_chains,
        dir_path: Path,
        data_flat,
        num_warmup,
        num_samples,
        model_name="model14",
        fit_up_to = 100,
        initial_position = None):
    logging.info("BEGIN sampling")
    k = 0
    warmup_savename = "0" + "_" + model_name + "_" + "hmc_warmup.pkl"
    warmup_savepath = dir_path / warmup_savename 
    for i in range(m_chains):
        k += 1
        if k >= fit_up_to:
            break
        if warmup_savepath.is_file():
            load_warmup = True
            save_warmup = False
        else:
            load_warmup = False
            save_warmup = True
        savename = dir_path / f"chain_{i}.pkl"
        bool_savename = f"{str(dir_path)}/0_{model_name}_{i}.pkl"
        if not Path(bool_savename).is_file():
            logging.info(f"BEGIN sampling {savename}")
            _model_variations._main(
            model_id="0",
            model_name=model_name,
            rseed=i,
            model_data=data_flat,
            num_warmup=num_warmup,
            num_samples=num_samples,
            include_potential_energy=True,
            progress_bar=True,
            save_dir=str(dir_path),
            include_mean_accept_prob=False,
            include_extra_fields=True,
            save_warmup = save_warmup,
            load_warmup = load_warmup,
            initial_position = initial_position)
            logging.info(f"END sampling {savename}")
    logging.info("END sampling")

## 2.1 Fitting
def _2_1_fit(model_name="model14"):
    for i in range(nb.m_chains):
        savename = Path("SyntheticBenchmark50/") / f"chain_{i}.pkl"
        bool_savename = f"SyntheticBenchmark50/0_{model_name}_{i}.pkl"
        if not Path(bool_savename).is_file():
            logging.info(f"BEGIN sampling {savename}")
            _model_variations._main(
            model_id="0",
            model_name=model_name,
            rseed=i,
            model_data=data_flat,
            num_warmup=nb.num_warmup,
            num_samples=nb.num_samples,
            include_potential_energy=True,
            progress_bar=True,
            save_dir="SyntheticBenchmark50",
            include_mean_accept_prob=False,
            include_extra_fields=True)
            logging.info(f"END sampling {savename}")
        else:
            logging.info(f"SKIP sampling {savename} exists.")  
            
def accuracy_plot(analysis_dict, samples, model, model_data, model_name,
                 reference_flat, dir_path, fig_dpi, calc_pearson_r=True, decoys=None):
    """
    Some would say this function does too many things
    1. Plots score vs accuracy
    2. Calculates an average network and plots
    3. Calculates score and accuracy for true network
    4. Optionally include one or more decoy networks
    5. Optionally does linear regression
    6. Saves the plots
    7. Returns the average network from step 2
    """
    fig, ax = plt.subplots()
    # Place scores and accuracy in arrays
    scores = -np.array(analysis_dict['log_density'])
    accuracy = np.array(analysis_dict['accuracy'])
    # Plot
    plt.plot(scores, accuracy, 'ro', alpha=0.1,
        label="$i$th Network")
    plt.xlabel("Score (-log density)")
    plt.ylabel("Accuracy Score (AUC ROC)")
    plt.title("Accuracy Score Correlation for a Representative Trajectory")
    # Calculate the average network, score, and accuracy
    average_network = np.mean(samples['samples']['pT'], axis=0)
    average_network_accuracy = sklearn.metrics.roc_auc_score(reference_flat, average_network)
    log_density_func = partial(numpyro.infer.util.log_density,
                model=model, model_args=(model_data,), model_kwargs={},)
    average_network_log_density, _ = log_density_func(params={"pT": average_network})
    # Plot
    plt.plot(-average_network_log_density, average_network_accuracy, 'b+',
            label="Average Network")
    # Calculate the true network, score, and accuracy
    true_network_accuracy = sklearn.metrics.roc_auc_score(reference_flat, reference_flat)
    true_network_log_density, _ = log_density_func(params={"pT": np.clip(reference_flat, 0.001, 0.999)})
    # Plot
    plt.plot(-true_network_log_density, true_network_accuracy, 'g^', label="True Network")
    # Score of the true network
    #_model_variations.model14
    # Get all scores for linear regression
    all_scores = -np.array(list(analysis_dict['log_density']) + [average_network_log_density] + [true_network_log_density])
    all_accuracy = np.array(list(analysis_dict['accuracy']) + [average_network_accuracy] + [true_network_accuracy])
    assert np.alltrue(np.isnan(all_scores)==False), "All scores has NaN"
    assert np.alltrue(np.isnan(all_accuracy)==False), "All accuracy has Nan"
    # Optionally include a decoy and plot
    if decoys is not None:
        for decoy in decoys:
            log_dens, _ = log_density_func(params={"pT": decoy})
            print(log_dens)
            acc = sklearn.metrics.roc_auc_score(reference_flat, decoy)
            plt.plot(log_dens, acc, 'kx', label='decoy')
    # Optionally do linear regression
    if calc_pearson_r:
        lin_reg = sp.stats.pearsonr(all_scores, all_accuracy)
        plt.text(-1200, 0.9, f"Pearson R {round(lin_reg.statistic, 3)}")
    plt.ylim(0.5, 1.05)
    #sns.pairplot(pd.DataFrame(analysis_dict))
    save_path = dir_path / f"AccuracyScoreCorrelation_{model_name}"
    # Save
    if decoys is None:
        save_name = str(save_path) + ".png"
    else:
        save_name = str(save_path) + "_decoy.png"
    plt.legend()
    plt.savefig(save_name, dpi=fig_dpi)
    plt.close()
    # Return average network
    return average_network

def analyze_samples(m_chains,
                    dir_path,
                    model_name,
                    model_data,
                    model,
                    reference_flat,
                    analyze_next_N):
    analysis_dict = {}
    k = 0
    for i in range(m_chains):
        # Load in the trajectory
        save_path = dir_path / f"0_{model_name}_{i}_analysis.pkl"
        if not save_path.is_file():
            k += 1
            if k >= analyze_next_N:
                break
            logging.info(f"WRITE analysis for {str(save_path)}")

            traj_path = dir_path / f"0_{model_name}_{i}.pkl"
            if not traj_path.is_file():
                logging.info(f"{str(traj_path)} is not a file. Skipping")
                continue
            #d = _model_variations.load(path)

            # Calculate sampling analysis for each trajectory
            if model_name in ("model14", "model15"):
                analysis_dict = analyze_mcmc_output.model14_traj2analysis(
                        traj_path = str(traj_path), 
                        model = model,
                        model_data = model_data, 
                        reference = np.array(reference_flat))
            else:
                raise NotImplementedError
            # do the sampling analysis
              # calc top score
              # calc accuracy array
              # get the top accuracy

              # 0_model14_0 : {top_score, accuracy_array, top_accuracy, potential_energy}
            with open(str(save_path), "wb") as f:
                pkl.dump(analysis_dict, f)
    return analysis_dict

## 3.1 Sampling Analysis
def _3_1():
    for i in range(nb.m_chains):
        # Load in the trajectory
        save_path = Path(f"SyntheticBenchmark50/0_model14_{i}_analysis.pkl")
        if not save_path.is_file():
            logging.info(f"WRITE analysis for {str(save_path)}")

            path = Path("SyntheticBenchmark50/") / f"0_model14_{i}.pkl"
            #d = _model_variations.load(path)

            # Calculate sampling analysis for each trajectory
            analysis_dict = analyze_mcmc_output.model14_traj2analysis(str(path), 
                    model_data=model_data, model=model, reference=np.array(reference_flat))
            # do the sampling analysis
              # calc top score
              # calc accuracy array
              # get the top accuracy

              # 0_model14_0 : {top_score, accuracy_array, top_accuracy, potential_energy}
            with open(str(save_path), "wb") as f:
                pkl.dump(analysis_dict, f)
        else:
            logging.info(f"SKIP Writing analysis for {str(save_path)}")
    return analysis_dict
    
def adjacency2graph(A, weighted=False):
    n, m = A.shape
    G = nk.Graph(n, weighted=weighted)
    for u in range(n):
        for v in range(u+1, m):
            w = A[u, v]
            if weighted:
                G.addEdge(u, v, w)
            else:
                if w > 0:
                    G.addEdge(u, v, w)
    return G

def precision_plot(analysis_dict, average_precision, reference_flat, average_network, model_name, model, model_data, dir_path, fig_dpi,):
    scores = -np.array(analysis_dict['log_density'])
    #accuracy = np.array(analysis_dict['accuracy'])
    fig, ax = plt.subplots()
    plt.plot(scores,
            average_precision['AP'],
            'ro',
            alpha=0.1,
            label="$i$th Network")
    plt.xlabel("Score (-log density)")
    plt.ylabel("Average Precision Score")
    plt.title("Precision Score Correlation for a Representative Trajectory")

    #average_network = np.mean(_0_model14_0['samples']['pT'], axis=0)
    #average_network_accuracy = sklearn.metrics.roc_auc_score(reference_flat, average_network)
    
    if model_name == "model14":
        log_density_func = partial(numpyro.infer.util.log_density,
                    model=model, model_args=(model_data,), model_kwargs={},)
    else:
        raise NotImplementedError
    average_network_log_density, _ = log_density_func(params={"pT": average_network})
    average_network_precision = sklearn.metrics.average_precision_score(reference_flat, average_network)
    plt.plot(-average_network_log_density, average_network_precision, 'b+',
            label="Average Network")

    true_network_precision = sklearn.metrics.average_precision_score(reference_flat,
                                                         reference_flat)
    true_network_log_density, _ = log_density_func(params={"pT": np.clip(reference_flat, 0.001, 0.999)})
    plt.plot(-true_network_log_density, true_network_precision,
            'g^', label="True Network")
    plt.legend()

    # Score of the true network

    all_scores = -np.array(list(analysis_dict['log_density']) + [average_network_log_density] + [true_network_log_density])
    all_accuracy = np.array(list(average_precision['AP']) + [average_network_precision] + [true_network_precision])

    lin_reg = sp.stats.pearsonr(all_scores, all_accuracy)
    plt.text(-1200, 0.9, f"Pearson R {round(lin_reg.statistic, 3)}")
    plt.ylim(0, 1.05)
    #sns.pairplot(pd.DataFrame(analysis_dict))
    plt.savefig(str(dir_path / f"PrecisionScoreCorrelation_{model_name}.png"), dpi=fig_dpi)
    plt.close()

## 0.1 Ground Truth Generation
def _0_1_ground_truth_generation():
    A, bait_idx, prey_idx = synthetic_benchmark.get_bait_prey_network(nb.keys[0], n_prey=50,
                                                                     n_bait=3, d_crit=20)
    return A, bait_idx, prey_idx

def _2_1_fit(model_name="model14"):
    for i in range(nb.m_chains):
        savename = Path("SyntheticBenchmark50/") / f"chain_{i}.pkl"
        bool_savename = f"SyntheticBenchmark50/0_{model_name}_{i}.pkl"
        if not Path(bool_savename).is_file():
            logging.info(f"BEGIN sampling {savename}")
            _model_variations._main(
            model_id="0",
            model_name=model_name,
            rseed=i,
            model_data=data_flat,
            num_warmup=nb.num_warmup,
            num_samples=nb.num_samples,
            include_potential_energy=True,
            progress_bar=True,
            save_dir="SyntheticBenchmark50",
            include_mean_accept_prob=False,
            include_extra_fields=True)
            logging.info(f"END sampling {savename}")
        else:
            logging.info(f"SKIP sampling {savename} exists.")


def calculate_chain_combinations(key, n_chains):
    # indexed by combo id
    combos = {}
    for k in range(n_chains):
        key, k1 = jax.random.split(key)
        chains = jax.random.choice(k1, n_chains, replace=False,
                                     shape=(k+1,))
        chains = np.array(chains)
        combos[k] = chains
    return combos
    
def get_top_accuracy_and_top_score_matrices_from_dir(dir_path: Path,
                                                     traj_idx_arr,
                                                     n,
                                                     model_name: str,
                                                     model_id=0):
    """
    traj_idx_arr : int or array
        if an int trajectory indices as if arange(traj_idx_arr)
        else traj_idx_arr are the indices
    m : the number of trajectories int or array
    n : the number of samples per trajectory 
    """
    if isinstance(traj_idx_arr, int):
        traj_idx_arr = np.arange(traj_idx_arr)
    m = len(traj_idx_arr)
    shape = (m, n)
    acc_mat = np.zeros(shape)
    scr_mat = np.zeros(shape)
    # Read in each file and populate the respective matrices
    for traj_idx in traj_idx_arr:
        fp = dir_path / f"{model_id}_{model_name}_{traj_idx}_analysis.pkl"
        assert fp.is_file(), f"{str(fp)} is not a file"
        fp = str(fp)
        with open(fp, "rb") as f:
            d = pkl.load(f)
        acc = d['accuracy']
        scr = d['log_density'] * -1
        acc_mat[traj_idx, :] = acc
        scr_mat[traj_idx, :] = scr
    return acc_mat, scr_mat

def calc_top_score_top_accuracy(
    rng_key,
    accuracy_matrix,
    score_matrix,
    k,
    J,
    ):
    """
    Given two m x n matrices 
    - J a 1-d trajectory group array. The elements of J are the number of trajectories 
      or int. If J is an int J as arange(1, J + 1)
    - k the total number of bootstrap samples
    - Total m number of trajectories
    - Total n number of MCMC samples per trajectory
    Calculate succesive statistics over size k groups
    Answer matrix is (l, 6)
    - l is the length of J, the number of groups 
    - the 6 columns are
    - 0 av_top_scr
    - 1 sd_top_scr
    - 2 av_top_acc
    - 3 sd_top_acc
    - 4 av_top_mod
    - 5 sd_top_mod
    """
    if isinstance(J, int):
        J = np.arange(1, J + 1)
    m, n = accuracy_matrix.shape
    assert score_matrix.shape == (m, n)
    # No Nans or infs in inputs 
    assert np.alltrue(~np.isinf(accuracy_matrix))
    assert np.alltrue(~np.isinf(score_matrix))
    assert np.alltrue(~np.isnan(accuracy_matrix))
    assert np.alltrue(~np.isnan(score_matrix))
    # index array
    top_accuracy = np.zeros((len(J), k))
    top_scores = np.zeros((len(J), k))
    acc_of_tsm = np.zeros((len(J), k))
    key_gen = rng_key
    for j_idx, n_trajectories in enumerate(J):
        assert n_trajectories > 0, "Must have positive number of trajectories"
        # select k groups of n trajectories without replacement
        for k_i in range(k):
            rng_key, key_gen  = jax.random.split(key_gen)
            traj_indices = jax.random.choice(rng_key, m, shape=(n_trajectories,), replace=False)
            traj_indices = np.array(traj_indices)
            # Get the group
            acc_group = np.ravel(accuracy_matrix[traj_indices, :])
            score_group = np.ravel(score_matrix[traj_indices, :])
            # Calculate some statistics for the group
            top_acc = np.max(acc_group)
            top_score = np.min(score_group)
            sel = score_group == top_score
            acc_of_top_scr_model = acc_group[sel]
            if np.sum(sel) > 1:
                acc_of_top_scr_model = acc_of_top_scr_model[0]
            top_accuracy[j_idx, k_i] = top_acc
            top_scores[j_idx, k_i] = top_score
            acc_of_tsm[j_idx, k_i] = acc_of_top_scr_model
    av_top_acc = np.mean(top_accuracy, axis=1)
    sd_top_acc = np.std (top_accuracy, axis=1)
    av_top_scr = np.mean(top_scores, axis=1)
    sd_top_scr = np.std (top_scores, axis=1)
    av_top_acc_of_tsm = np.mean(acc_of_tsm, axis=1)
    sd_top_acc_of_tsm = np.std (acc_of_tsm, axis=1) 
    return { "top_acc_av" : av_top_acc,
             "top_scr_av" : av_top_scr,
             "top_mod_av" : av_top_acc_of_tsm,
             "top_acc_sd" : sd_top_acc,
             "top_scr_sd" : sd_top_scr,
             "top_mod_sd" : sd_top_acc_of_tsm,}

def top_accuracy_top_score_figure(dir_path,
        top_score_top_accuracy_results,
        J,
        fig_dpi):
    """
    """
    r = top_score_top_accuracy_results
    if isinstance(J, int):
        J = np.arange(1, J + 1)
    fig, ax = plt.subplots()
    plt.errorbar(J, r['top_acc_av'], yerr=r['top_acc_sd'], label="top accuray", alpha=0.5)  
    plt.errorbar(J, r['top_mod_av'], yerr=r['top_mod_sd'], label="Accuracy of top scoring model",
                 alpha=0.5) 
    plt.ylabel("Accuracy (AUC ROC)")
    plt.xlabel("N trajectories")
    plt.legend()
    savename = str(dir_path / "BootstrapAccuracy.png")
    plt.savefig(savename, dpi=fig_dpi)
    plt.close()
    fig, ax = plt.subplots()
    plt.errorbar(J, r['top_scr_av'], yerr=r['top_scr_sd'], label="Top score", alpha=0.5)
    plt.legend()
    plt.ylabel("Score")
    plt.xlabel("N trajectories")
    savename = str(dir_path / "BoostrapScore.png")
    plt.savefig(savename, dpi=fig_dpi)
     
def errplot(x, y, yerr):
    plt.errorbar(x, y, yerr, fmt='.')
    plt.xlabel("Sample group")

    
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
                            num_warmup = 500,
                            num_samples = 1000,
                            m_chains = 50,
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
    scorekey2model_name = {"bp_lp" : None,
                           "lbh_ll" : None,
                           "lbh_ll__bp_lp" : None }
    for i, path in enumerate(paths):
        model_name = scorekey2model_name[scores[i]]    
        synthetic_benchmark_fn(
                analysis_name = str(path),
                n_prey = N[i],
                n_bait = n_bait,
                d_crit = d_crit,
                dir_name = str(path),
                rseed = static_seed,
                edge_probability = edge_prob[i],
                num_warmup = num_warmup,
                num_samples = num_samples,
                m_chains = m_chains, 
                model_name = model_name) 

