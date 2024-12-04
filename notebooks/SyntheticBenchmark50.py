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
def synthetic_benchmark(analysis_name : str,
                        n_prey : int,
                        n_bait : int,
                        d_crit : int
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
                        analyze_next_N = 100
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
    logging.basicConfig(filename=str(log_filename), encoding="utf-8", level=logging.DEBUG, format='%(asctime)s %(message)s')
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
    A, bait_idx, prey_idx = synthetic_benchmark.get_bait_prey_network(key, n_prey=n_prey,
                                                                     n_bait=n_bait, d_crit=d_crit)
    
    # Plot the ground truth network
    A.plot() # xarray
    plt.hlines(np.array(bait_idx), -1, 50, label="Bait", alpha=0.8, color='r')
    plt.legend()
    plt.savefig(str(dir_path / "GroundTruth.png"), dpi=fig_dpi)
    
    # Calculate the prey distance matrix
    D = _model_variations.shortest_paths_up_to_23(A.values)
    
    # 
    plt.matshow(D)
    plt.colorbar(shrink=0.8, label="Shortest path")
    plt.title("Ground Truth Distance Matrix")
    plt.savefig(str(dir_path / "GroundTruthDistanceMatrix.png", dpi=fig_dpi))
    
    Gtruth = adjacency2graph(A, weighted=False)
    
    # Plot the connected components
    nk.plot.connectedComponentsSizes(G)
    plt.title("connected component sizes") # A single connected component
    
    # 1.1 Generate similarities
    key, keygen = jax.random.PRNGKey(keygen)
    data = synthetic_benchmark.data_from_network_model14_rng(key, A.values)
    
    # Plot the similarities
    plt.matshow(data)
    plt.colorbar(shrink=0.8, label="Profile similarity")
    plt.savefig(str(dir_path / "SyntheticDataMatrix.png"), dpi=fig_dpi.dpi)
    
    # Plot the Ground truth contact map
    plt.matshow(A.values, cmap="binary")
    plt.colorbar(shrink=0.8, label="Edge weight")
    plt.savefig(str(dir_path / "GroundTruthMatplot.png"), dpi=fig_dpi)
    
    # Flatten the inputs for modeling
    
    reference_flat = _model_variations.matrix2flat(A.values) # use this method to flatten and unflatten matrices
    # so that indices are preserved
    
    data_flat =  _model_variations.matrix2flat(data)
    assert len(data_flat) == math.comb(n_prey, 2)
    
    # Model 14 is the mixture model
    # Model 15 has beta priors, not much different
    
    model14_names = ("model14", "model15")

    if model_name in model14_names:
        model_data = _model_variations.model14_data_getter()
        model_data['flattened_apms_similarity_scores'] = data_flat
        model = _model_variations.model14
    else:
        raise ValueError(f"Unknown model name {model_name}")
    
    # Plot the reference and causal edges
    n_true = int(np.sum(reference_flat))
    n_false = len(reference_flat) - n_true
    plt.hist(np.array(data_flat[reference_flat == 0]), label=f"False ({n_false})", bins=100, alpha=0.5)
    plt.hist(np.array(data_flat[reference_flat == 1]), label=f"True  ({n_true})", bins=100, alpha=0.5)
    plt.xlabel("Profile Similarity (Pearson R)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(str(dir_path / "SynthDataDistributions.png"), dpi=fig_dpi)
    
    # Fit the model m_chain times writing to the file system
    fit(model_name = model_name
        m_chains = m_chains,
        dir_path = dir_path,
        data_flat = data_flat,
        num_warmup = num_warmup,
        num_samples = num_samples)
    
    # Load in the first trajectory
    path = dir_path / f"0_{model_name}_0.pkl"
    _0_model14_0 = _model_variations.load(str(path))
    
    if model_name in model14_names:
        analysis_dict = analyze_mcmc_output.model14_traj2analysis(
                str(path), 
                model_data=model_data,
                model=model,
                reference=np.array(reference_flat))
        
    # Analyze samples
    analyze_samples(m_chains = m_chains,
                    dir_path = dir_path,
                    model_name = model_name,
                    model_data = model_data,
                    model = model,
                    reference_flat = reference_flat,
                    analyze_next_N = analyze_next_N,)
    
    
    else:
        raise ValueError(f"Unkown model name {model_name}")

        
## 2.1 Fitting
def fit(model_name="model14",
        m_chains,
        dir_path: Path,
        data_flat,
        num_warmup,
        num_samples,
        fit_up_to = 100):
    logging.info("BEGIN sampling")
    k = 0
    for i in range(m_chains):
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
            include_extra_fields=True)
            logging.info(f"END sampling {savename}")
            k += 1
            if k >= fit_up_to:
                break
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

            
def analyze_samples(m_chains,
                    dir_path,
                    model_name,
                    model_data,
                    model,
                    reference_flat,
                    analyze_next_N):
    k = 0
    for i in range(m_chains):
        # Load in the trajectory
        save_path = dir_path / f"0_{model_name}_{i}_analysis.pkl"
        if not save_path.is_file():
            k += 1
            logging.info(f"WRITE analysis for {str(save_path)}")

            traj_path = dir_path / f"0_{model_name}_{i}.pkl"
            #d = _model_variations.load(path)

            # Calculate sampling analysis for each trajectory
            if model_name in ("model14", "model15"):
            analysis_dict = analyze_mcmc_output.model14_traj2analysis(str(traj_path), 
                    model_data=model_data, model=model, reference=np.array(reference_flat))
            else:
                raise NotImplementedError
            # do the sampling analysis
              # calc top score
              # calc accuracy array
              # get the top accuracy

              # 0_model14_0 : {top_score, accuracy_array, top_accuracy, potential_energy}
            with open(str(save_path), "wb") as f:
                pkl.dump(analysis_dict, f)
            if k >= analyze_next_N:
                break
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


# -

## -1.2 Make a directory if it doesn't exist
nb_write_path = Path("SyntheticBenchmark50")
if not nb_write_path.is_dir():
    nb_write_path.mkdir()

## -1.3
# Start a new log file to write to
time = datetime.datetime.now()
log_filename = nb_write_path / f"{time.year}_{time.month}_{time.day}_{time.hour}_{time.minute}_SyntheticBenchmark50.log"
# Set up the logger
logging.basicConfig(filename=str(log_filename), encoding="utf-8", level=logging.DEBUG)
logging.info(f"SyntheticBenchmark50 Notebook Begin at {str(time)}")

## -1.4
nb = SimpleNamespace()
nb.rseed = 13
nb.rkey = jax.random.PRNGKey(nb.rseed)
nb.n_keys = 3
nb.keys = jax.random.split(nb.rkey, nb.n_keys)
nb.dev_key = jax.random.PRNGKey(22)
nb.n_prey = 50
nb.tril_idx = np.tril_indices(nb.n_prey)
nb.num_samples = 1000
nb.m_chains = 1000
nb.num_warmup = 500
nb.dpi = 300
nb.dir_path = Path("SyntheticBenchmark50/")


# +
## 0.1 Ground Truth Generation
def _0_1_ground_truth_generation():
    A, bait_idx, prey_idx = synthetic_benchmark.get_bait_prey_network(nb.keys[0], n_prey=50,
                                                                     n_bait=3, d_crit=20)
    return A, bait_idx, prey_idx

A, bait_idx, prey_idx = _0_1_ground_truth_generation()
# -

bait_idx

A.plot()
plt.hlines(np.array(bait_idx), -1, 50, label="Bait", alpha=0.8, color='r')
plt.legend()
plt.savefig(str(nb.dir_path / "GroundTruth.png"), dpi=nb.dpi)

D = _model_variations.shortest_paths_up_to_23(A.values)

plt.matshow(D)
plt.colorbar(shrink=0.8, label="Shortest path")

G = adjacency2graph(A)

nk.plot.connectedComponentsSizes(G)
plt.title("connected component sizes") # A single connected component

# 1.1 Generate similarities
data = synthetic_benchmark.data_from_network_model14_rng(nb.keys[1], A.values)

plt.matshow(data)
plt.colorbar(shrink=0.8, label="Profile similarity")
plt.savefig(str(nb.dir_path / "SyntheticDataMatrix.png"), dpi=nb.dpi)

plt.matshow(A.values, cmap="binary")
plt.colorbar(shrink=0.8, label="Edge weight")
plt.savefig(str(nb.dir_path / "GroundTruthMatplot.png"), dpi=nb.dpi)

# $\pi_{T_{uv}} \sim U(0, 1)$
#
#
# $y_{u,v} \sim p(y_{u,v} | \pi_{T_{u,v}})$

# +
reference_flat = _model_variations.matrix2flat(A.values)

data_flat =  _model_variations.matrix2flat(data)
assert len(data_flat) == math.comb(nb.n_prey, 2)

model_data = _model_variations.model14_data_getter()
model_data['flattened_apms_similarity_scores'] = data_flat
model = _model_variations.model14
# -

model_data.keys()

n_true = int(np.sum(reference_flat))
n_false = len(reference_flat) - n_true
plt.hist(np.array(data_flat[reference_flat == 0]), label=f"False ({n_false})", bins=100, alpha=0.5)
plt.hist(np.array(data_flat[reference_flat == 1]), label=f"True  ({n_true})", bins=100, alpha=0.5)
plt.xlabel("Profile Similarity (Pearson R)")
plt.ylabel("Frequency")
plt.legend()
plt.savefig(str(nb.dir_path / "SynthDataDistributions.png"), dpi=nb.dpi)


# +
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

_2_1_fit()
# -

path = Path("SyntheticBenchmark50") / f"0_model14_0.pkl"
_0_model14_0 = _model_variations.load(str(path))
analysis_dict = analyze_mcmc_output.model14_traj2analysis(str(path), 
        model_data=model_data, model=model, reference=np.array(reference_flat))

# +

_3_1()

# +
# Cartoon
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
plt.savefig("SyntheticBenchmark50/LowUncertaintyCartoon.png", dpi=300)

# +
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
plt.savefig("SyntheticBenchmark50/HighUncertaintyCartoon.png", dpi=300)

# +
# Expect to see low scores correlated with high accuracy
# This would make a negative slope

path = Path("SyntheticBenchmark50") / f"0_model14_0.pkl"
_0_model14_0 = _model_variations.load(str(path))
analysis_dict = analyze_mcmc_output.model14_traj2analysis(str(path), 
        model_data=model_data, model=model, reference=np.array(reference_flat))



# +
decoy_1 = reference_flat * 0.21
def accuracy_plot(analysis_dict, samples, model, model_data, model_name,
                 reference_flat, calc_pearson_r=True, decoys=None):
    scores = -np.array(analysis_dict['log_density'])
    accuracy = np.array(analysis_dict['accuracy'])
    plt.plot(scores, accuracy, 'ro', alpha=0.1,
        label="$i$th Network")
    plt.xlabel("Score (-log density)")
    plt.ylabel("Accuracy Score (AUC ROC)")
    plt.title("Accuracy Score Correlation for a Representative Trajectory")
    average_network = np.mean(samples['samples']['pT'], axis=0)
    average_network_accuracy = sklearn.metrics.roc_auc_score(reference_flat, average_network)

    log_density_func = partial(numpyro.infer.util.log_density,
                model=model, model_args=(model_data,), model_kwargs={},)
    average_network_log_density, _ = log_density_func(params={"pT": average_network})
    plt.plot(-average_network_log_density, average_network_accuracy, 'b+',
            label="Average Network")

    true_network_accuracy = sklearn.metrics.roc_auc_score(reference_flat, reference_flat)
    true_network_log_density, _ = log_density_func(params={"pT": np.clip(reference_flat, 0.001, 0.999)})
    
    plt.plot(-true_network_log_density, true_network_accuracy, 'g^', label="True Network")
    

    # Score of the true network
    #_model_variations.model14

    all_scores = -np.array(list(analysis_dict['log_density']) + [average_network_log_density] + [true_network_log_density])
    all_accuracy = np.array(list(analysis_dict['accuracy']) + [average_network_accuracy] + [true_network_accuracy])
    
    assert np.alltrue(np.isnan(all_scores)==False), "All scores has NaN"
    assert np.alltrue(np.isnan(all_accuracy)==False), "All accuracy has Nan"
    if decoys is not None:
        for decoy in decoys:
            
            log_dens, _ = log_density_func(params={"pT": decoy})
            print(log_dens)
            acc = sklearn.metrics.roc_auc_score(reference_flat, decoy)
            plt.plot(log_dens, acc, 'kx', label='decoy')
    
    if calc_pearson_r:
        lin_reg = sp.stats.pearsonr(all_scores, all_accuracy)
        plt.text(-1200, 0.9, f"Pearson R {round(lin_reg.statistic, 3)}")
    plt.ylim(0.5, 1.05)
    #sns.pairplot(pd.DataFrame(analysis_dict))
    

    save_path = nb.dir_path / f"AccuracyScoreCorrelation_{model_name}"
    if decoys is None:
        save_name = str(save_path) + ".png"
    else:
        save_name = str(save_path) + "_decoy.png"
    
    
    plt.savefig(save_name, dpi=nb.dpi)
    plt.legend()
    return average_network
    
average_network = accuracy_plot(analysis_dict, _0_model14_0, model, model_data, "model14", reference_flat,
                               decoys=[decoy_1])
# -

average_precision = analyze_mcmc_output.model14_ap_score(str(path), 
        model_data=model_data, model=model, reference=np.array(reference_flat))


# +
def precision_plot(analysis_dict, scores, average_precision, reference_flat, average_network, model_name):
    #scores = -np.array(analysis_dict['log_density'])
    #accuracy = np.array(analysis_dict['accuracy'])
    plt.plot(scores, average_precision['AP'], 'ro', alpha=0.1,
            label="$i$th Network")
    plt.xlabel("Score (-log density)")
    plt.ylabel("Average Precision Score")
    plt.title("Precision Score Correlation for a Representative Trajectory")

    #average_network = np.mean(_0_model14_0['samples']['pT'], axis=0)
    #average_network_accuracy = sklearn.metrics.roc_auc_score(reference_flat, average_network)

    #model14_log_density_func = partial(numpyro.infer.util.log_density,
                #model=model, model_args=(model_data,), model_kwargs={},)

    #average_network_log_density, _ = model14_log_density_func(params={"pT": average_network})
    average_network_precision = sklearn.metrics.average_precision_score(reference_flat, average_network)
    plt.plot(-average_network_log_density, average_network_precision, 'b+',
            label="Average Network")

    true_network_precision = sklearn.metrics.average_precision_score(reference_flat,
                                                         reference_flat)
    plt.plot(-true_network_log_density, true_network_precision,
            'g^', label="True Network")
    plt.legend()

    # Score of the true network
    _model_variations.model14

    all_scores = -np.array(list(analysis_dict['log_density']) + [average_network_log_density] + [true_network_log_density])
    all_accuracy = np.array(list(average_precision['AP']) + [average_network_accuracy] + [true_network_accuracy])

    lin_reg = sp.stats.pearsonr(all_scores, all_accuracy)
    plt.text(-1200, 0.9, f"Pearson R {round(lin_reg.statistic, 3)}")
    plt.ylim(0, 1.05)
    #sns.pairplot(pd.DataFrame(analysis_dict))
    plt.savefig(str(nb.dir_path / f"PrecisionScoreCorrelation_{model_name}.png"), dpi=nb.dpi)
    
precision_plot(analysis_dict, scores, average_precision, reference_flat, average_network, "model14")
# -

# Conclusions
# - The samples are linearly correlated with accuracy
# - The average network is more accurate than and better scoring than any individual network model. i.e., the average network satisfies the data better than any individual.
# - The true network has a lower log density
#

plt.plot(-scores)
plt.xlabel("MCMC Step")
plt.ylabel("Score (-log density)")
plt.savefig(str(nb.dir_path / "ExampleCatterpilliar.png"), dpi=nb.dpi)

plt.hist(-scores, bins=30)
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.savefig(str(nb.dir_path / "ExampleScoreDistribution.png"), dpi=nb.dpi)
plt.show()

plt.hist(_0_model14_0['samples']['pT'][:, 0], bins=100, label="False Edge")
plt.hist(_0_model14_0['samples']['pT'][:, 2], bins=100, alpha=0.5, label="True Edge")
plt.title("Representative Distributions for two edges")
plt.xlabel("$\pi_T$")
plt.ylabel("Frequency")
plt.legend()
plt.savefig(str(nb.dir_path / "ExampleTwoEdges.png"), dpi=nb.dpi)
plt.show()


# Conclusion:
# - The model score is well correlated to accuracy on this synthetic example.
# - The average network achieves an accuracy of 0.87
#
# Is this a funciton of sampling?
#

# +
## 4. Calculate a dictionary of increasing combinations of the M chains
def calculate_chain_combinations(n_chains):
    # indexed by combo id
    combos = {}
    key = nb.keys[2]
    for k in range(n_chains):
        key, k1 = jax.random.split(key)
        chains = jax.random.choice(k1, n_chains, replace=False,
                                     shape=(k+1,))
        chains = np.array(chains)
        combos[k] = chains
    return combos
            
N = 398   
combos = calculate_chain_combinations(N)

def calc_top_score_top_accuracy(chain_combo_dict):
    
    top_combo_scores = defaultdict(list) # - log dens
    top_combo_accuracy = defaultdict(list)
    for combo_id, chains in chain_combo_dict.items():
        for chain_idx in chains:
            # load in the analysis
            chain_path = f"SyntheticBenchmark50/0_model14_{chain_idx}_analysis.pkl"
            chain_analysis = _model_variations.load(chain_path)
            max_log_density = np.max(chain_analysis['log_density'])
            max_accuracy = np.max(chain_analysis['accuracy'])
            top_combo_scores[combo_id].append(-max_log_density)
            top_combo_accuracy[combo_id].append(max_accuracy)
    return top_combo_scores, top_combo_accuracy

top_combo_scores, top_combo_accuracy = calc_top_score_top_accuracy(combos)

# +
w = 0.
tip_top_combo_scores = np.array(
    [np.min(top_combo_scores[k]) for k in range(len(top_combo_scores))])
tip_top_combo_hpdi = np.array(
    [numpyro.diagnostics.hpdi(top_combo_scores[k], w) for k in range(len(top_combo_scores))])

tip_top_combo_accuracy = np.array(
[np.max(top_combo_accuracy[k]) for k in range(len(top_combo_accuracy))])

tip_top_combo_accuracy_hpdi = np.array(
[numpyro.diagnostics.hpdi(top_combo_accuracy[k]) for k in range(len(top_combo_accuracy))])


# +
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Sample group')
ax1.set_ylabel("Best Score", color=color)


t = np.arange(N)
#ax1.plot(t, tip_top_combo_scores, 'o', alpha=0.5, label="Top Score")
ax1.errorbar(t, tip_top_combo_scores, fmt='.', alpha=0.5, label="Top score")
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
ax2.set_ylabel("Top Accuracy", color=color)  # we already handled the x-label with ax1
ax2.plot(t, tip_top_combo_accuracy, '.', color=color, alpha=0.5, label="Accuracy")
ax2.tick_params(axis='y')

plt.title("Network Score and Accuracy as a function of increased sampling")
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("SyntheticBenchmark50/TopAccuracyScore.png", dpi=nb.dpi)
plt.show()

# -

plt.plot(tip_top_combo_scores)
plt.xlabel("Sample Group")
plt.ylabel("Top score")

plt.plot(tip_top_accuracy_scores)
plt.ylabel("Top accuracy")
plt.xlabel("Sample group")


def errplot(x, y, yerr):
    plt.errorbar(x, y, yerr, fmt='.')
    plt.xlabel("Sample group")


# +
yerr = np.abs(tip_top_combo_scores - tip_top_combo_hpdi.T)
errplot(np.arange(len(tip_top_combo_scores)), tip_top_combo_scores, yerr)

#plt.xlabel("Sample Group")
plt.ylabel("Top score (- log density)")
# -

tip_to

yerr = np.abs(tip_top_combo_accuracy - tip_top_combo_accuracy_hpdi.T)
errplot(np.arange(len(tip_top_accuracy_scores)), tip_top_accuracy_scores, yerr)
plt.ylabel("Acuracy (AUC ROC)")

v = tip_top_combo_accuracy_hpdi.T
plt.vlines(np.arange(48), v[0, :], v[1, :])
plt.plot(np.arange(48), tip_top_accuracy_scores, 'r.')
plt.plot()

# ?plt.vlines

plt.errorbar(np.arange(len(tip_top_combo_accuracy)),
    tip_top_combo_accuracy, yerr=yerr)
plt.xlabel("Sample Group")
plt.ylabel("Top accuracy")

# +
# Same Analysis as before but with model 15 and a beta prior
# -

import numpyro.distributions as dist

x = np.arange(0, 1, 0.01)
y = np.exp(dist.Beta(0.5, 0.5).log_prob(x))  # 1.5, 3
plt.plot(x, y)
plt.xlabel("$\pi_T$")
plt.ylabel("Probability Density")

# $\pi_{T_{uv}} \sim \text{Beta}(\alpha, \beta)$
#
#
# $y_{u,v} \sim p(y_{u,v} | \pi_{T_{u,v}})$
#
# $ \alpha = 0.5, \beta = 0.5$

_2_1_fit(model_name="model15")

# +
path = Path("SyntheticBenchmark50") / f"0_model15_0.pkl"
model15 = _model_variations.model15


_0_model15_0 = _model_variations.load(str(path))
analysis_dict_0_model15_0 = analyze_mcmc_output.model14_traj2analysis(str(path), 
        model_data=model_data, model=model15, reference=np.array(reference_flat))
# -

average_model15_network = accuracy_plot(analysis_dict_0_model15_0, _0_model15_0,
        model15, model_data, "model15", reference_flat, calc_pearson_r=True)

average_precision = analyze_mcmc_output.model14_ap_score(str(path), 
        model_data=model_data, model=model, reference=np.array(reference_flat))

_0_model15_0.keys()

precision_plot(analysis_dict_0_model15_0, scores, average_precision, reference_flat, average_network, "model15")

numpyro.infer.util.log_density(model15, (model_data,), {}, {"pT": reference_flat})

numpyro.infer.util.log_density(model15, (model_data,), {}, {"pT": np.clip(reference_flat, 0.00001, 0.99999)})

# ?np.clip

dist.Uniform(0, 1).support
