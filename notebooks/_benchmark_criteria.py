from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.integrate import simpson
import scipy.stats

def tp_from_edge_id_lists(ref, pred) -> int:
    """Returns the number of shared elements between the
       two lists"""
    return len(set(ref).intersection(set(pred)))

def tp_at_thresholds(ref, pred_table):
    """
    Compute the true positives between ref and 
    pred_table['edge_id'] at the designated score thresholds
    given by pred_table.

    Returns a list of true positives at each prediction threshold
    pred_table: pd.DataFrame with |edge_id | score|
    """
    scores = sorted(set(pred_table['score']))
    tp_at_k = 0
    tp_at_k_plus_1 = 0
    # True positives
    tps = np.zeros(len(scores), dtype=int)
    # Positive predictions
    pps = np.zeros(len(scores), dtype=int)
    pps_at_k = 0
    pps_at_k_plus_1 = 0
    
    for k_plus_1, score in enumerate(scores):
        sel = pred_table['score'] == score
        subframe = pred_table.loc[sel]
        pred = subframe['edge_id'].values
        tp = tp_from_edge_id_lists(ref, pred)
        tp_at_k_plus_1 = tp + tp_at_k
        pps_at_k_plus_1 = len(pred) + pps_at_k
        tps[k_plus_1] = tp_at_k_plus_1
        pps[k_plus_1] = pps_at_k_plus_1
        tp_at_k = tp_at_k_plus_1
        pps_at_k = pps_at_k_plus_1
    df = pd.DataFrame({"threshold": scores, "tp": tps, "pp": pps})
    return df

def init_hypergeo_null(size_interaction_space, n_ref_edges, pps):
    """Initialize a Hypergoemetric distribution to use as a null model
    of randomly selecting edges from a population of edges, some of which     are 'positive' or 'true' classes."""
    return scipy.stats.hypergeom(M=size_interaction_space,
       n=n_ref_edges) 

def null_expectation(hyper_geom_dist, pps):
    """Calculate the average number of edges for a given number of
    guesses (positive predictions"""
    return hyper_geom_dist.mean(N=pps)

def scale_pred_df_to_rates(pred_df, ref, size_interaction_space):
    """Add rates (true positive rate and positive predictive rate) to
    the dataframe. tpr is the ratio of true positives to the number of positve cases. Positive predictive rate is the number of guesses to the size
    of the interaction space (or number of possible guesses)."""
    pred_df.loc[:, "tpr"] = pred_df["tp"].values / len(ref)
    pred_df.loc[:, "ppr"] = pred_df["pp"].values / size_interaction_space
    return pred_df

def auc_tp_pps(pred_df, col1="tpr", col2="ppr"):
    """Calculate the area under the Y: tpr, X: ppr curve using
    Simpsons rule (scipy default params)"""
    return simpson(y=pred_df[col1].values, x=pred_df[col2].values)

# Some example funcs for testing
if __name__ == "__main__":
    example_ref = np.array(range(0, 3023)) * 10
    example_pred = np.array(range(0, int(1e5))) * 100 
    example_scores = example_pred 
    example_pred_table = pd.DataFrame({"edge_id": example_pred,
        "score": example_scores})
    
