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

# ## Accuracy metrics

import generate_benchmark_figures as gbf
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
from pathlib import Path
import tpr_ppr

# +
# PDB Reference
direct_ref = gbf.get_pdb_ppi_predict_direct_reference()
co_structure_ref = gbf.get_pdb_ppi_predict_cocomplex_reference()
indirect_ref = co_structure_ref.edge_identity_difference(direct_ref)




def n_true_pos(upred, uref):
    return len(upred.edge_identity_intersection(uref))


def n_pdb_missing(upred):
    """
    If the edge isn't co structure it is missing
    """
    in_pdb = upred.edge_identity_intersection(co_structure_ref)
    return len(set(upred._edge_dict.keys()) - in_pdb)

def threshold(x, threshold):
    a = []
    b = []
    w = []
    for i, edge_value in enumerate(x.edge_values):
        if edge_value >= threshold:
            a.append(x.a_nodes[i])
            b.append(x.b_nodes[i])
            w.append(edge_value)
    u = gbf.UndirectedEdgeList()
    u.update_from_df(
        pd.DataFrame({"auid": a, "buid": b, "w": w}),
        multi_edge_value_merge_strategy = "max",
        edge_value_colname="w")
    return u

# Predictors
humap_hc = gbf.get_humap_high_reference()
humap_med = gbf.get_humap_medium_reference()
saint_pred_all = gbf.get_cullin_saint_scores_edgelist()
saint_pred_05 = threshold(saint_pred, 0.5)
saint_pred_07 = threshold(saint_pred, 0.7)
# -

inm_all = gbf.model23_results2edge_list(Path("../results/se_sr_all_20k_r16/"), fbasename="0_model23_se_sr_16")

inm_mock = gbf.model23_results2edge_list(Path("../results/se_sr_mock_ctrl_20k/"), fbasename="0_model23_se_sr_13")

inm_mock_09 = threshold(inm_mock, 0.9)

inm_mock_099 = threshold(inm_mock, 0.99)

inm_09 = threshold(inm_all, 0.9)

inm_099 = threshold(inm_all, 0.99)

inm_low_prior_1 = gbf.model23_results2edge_list(
    Path("../results/se_sr_low_prior_1_all_100k/"),
    fbasename = "0_model23_se_sr_13")

inm_low_prior_1_099 = threshold(inm_low_prior_1, threshold=0.99)

inm_low_prior_2_mock = gbf.model23_results2edge_list(
    Path("../results/se_sr_low_prior_2_uniform_mock_20k/"),
    fbasename = "0_model23_se_sr_13"
)

inm_low_prior_2_mock_099 = threshold(
    inm_low_prior_2_mock, 0.99)

# +
# Binary Comparison
# -

model23_results2edge_list(model_output_dirpath, fbasename)


# +
# Plot Direct vs missing edges
def plotf(x, fmt, pkwargs):
    plt.plot(n_pdb_missing(x), n_true_pos(x, direct_ref), fmt, **pkwargs)

plotf(humap_hc, 'k.', pkwargs = dict(label="hu MAP 2.0 HC"))
plotf(humap_med, 'k^', pkwargs = dict(label="hu MAP 2.0 MC"))
plotf(saint_pred_all, 'b.', pkwargs = dict(label = "SAINT All"))
plotf(saint_pred_05, 'bo', pkwargs = dict(label = "SAINT 0.5"))
plotf(saint_pred_07, 'b^', pkwargs = dict(label = "SAINT 0.7"))
plotf(inm_all, 'g.', pkwargs = dict(label = "Av INM All"))
plotf(inm_09, 'go', pkwargs = dict(label = "Av INM 0.9"))
plotf(inm_099, 'g^', pkwargs = dict(label = "Av INM 0.99"))
plotf(inm_low_prior_1_099, 'r^', pkwargs = dict(label = "inm low prior 1 0.99"))
plotf(inm_low_prior_2_mock_099, 'rx', pkwargs = dict(label = "inm low prior 2 0.99"))

plt.xlabel("N false (missing) positives")
plt.ylabel("N true (direct) positives")
plt.legend()


# +
# Plot indirect vs missing edges
def plotf(x, fmt, pkwargs):
    plt.plot(n_pdb_missing(x), n_true_pos(x, indirect_ref), fmt, **pkwargs)

plotf(humap_hc, 'k.', pkwargs = dict(label="hu MAP 2.0 HC"))
plotf(humap_med, 'k^', pkwargs = dict(label="hu MAP 2.0 MC"))
plotf(saint_pred_all, 'b.', pkwargs = dict(label = "SAINT All"))
plotf(saint_pred_05, 'bo', pkwargs = dict(label = "SAINT 0.5"))
plotf(saint_pred_07, 'b^', pkwargs = dict(label = "SAINT 0.7"))
plotf(inm_all, 'g.', pkwargs = dict(label = "Av INM All"))
plotf(inm_09, 'go', pkwargs = dict(label = "Av INM 0.9"))
plotf(inm_099, 'g^', pkwargs = dict(label = "Av INM 0.99"))
plotf(inm_low_prior_1_099, 'r^', pkwargs = dict(label = "inm low prior 1 0.99"))
plotf(inm_low_prior_2_mock_099, 'rx', pkwargs = dict(label = "inm low prior 2 0.99"))


plt.xlabel("N false (missing) positives")
plt.ylabel("N true (indirect) positives")
plt.legend()


# +
# Plot indirect vs missing edges
def plotf(x, fmt, pkwargs):
    plt.plot(n_pdb_missing(x), n_true_pos(x, co_structure_ref), fmt, **pkwargs)

plotf(humap_hc, 'k.', pkwargs = dict(label="hu MAP 2.0 0.94"))
plotf(humap_med, 'k^', pkwargs = dict(label="hu MAP 2.0 0.49"))
plotf(saint_pred_all, 'b.', pkwargs = dict(label = "SAINT All"))
plotf(saint_pred_05, 'bo', pkwargs = dict(label = "SAINT 0.5"))
plotf(saint_pred_07, 'b^', pkwargs = dict(label = "SAINT 0.7"))
#plotf(inm_all, 'g.', pkwargs = dict(label = "Av INM All"))
#plotf(inm_09, 'go', pkwargs = dict(label = "INM all 0.9"))
#plotf(inm_mock_099, 'y^', pkwargs = dict(label = "INM mock 0.99"))
plotf(inm_mock_099, 'g^', pkwargs = dict(label = "INM 0.99"))
plotf(inm_low_prior_1_099, 'r^', pkwargs = dict(label = "INM low prior 1 mock 0.99"))
#plotf(inm_low_prior_2_mock_099, 'rx', pkwargs = dict(label = "INM low prior 2 mock 0.99"))


plt.xlabel("N false (missing) positives")
plt.ylabel("N true (co-structure) positives")
plt.legend()
plt.savefig("AndrejsFigure2", dpi=300)
# -



# +
# Plot indirect vs missing edges
def plotf(x, fmt, pkwargs):
    plt.plot(n_pdb_missing(x), n_true_pos(x, co_structure_ref), fmt, **pkwargs)

plotf(humap_hc, 'k.', pkwargs = dict(label="hu MAP 2.0 HC"))
plotf(humap_med, 'k^', pkwargs = dict(label="hu MAP 2.0 MC"))
plotf(saint_pred_all, 'b.', pkwargs = dict(label = "SAINT All"))
plotf(saint_pred_05, 'bo', pkwargs = dict(label = "SAINT 0.5"))
plotf(saint_pred_07, 'b^', pkwargs = dict(label = "SAINT 0.7"))
#plotf(inm_all, 'g.', pkwargs = dict(label = "Av INM All"))
#plotf(inm_09, 'go', pkwargs = dict(label = "INM all 0.9"))
plotf(inm_mock_099, 'y^', pkwargs = dict(label = "INM mock 0.99"))
plotf(inm_099, 'g^', pkwargs = dict(label = "INM all 0.99"))
plotf(inm_low_prior_1_099, 'r^', pkwargs = dict(label = "INM low prior 1 mock 0.99"))
plotf(inm_low_prior_2_mock_099, 'rx', pkwargs = dict(label = "INM low prior 2 mock 0.99"))

plt.xlabel("N false (missing) positives")
plt.ylabel("N true (co-structure) positives")
#plt.ylim(0, 160)
#plt.legend()

# +
# Plot indirect vs missing edges
def plotf(x, fmt, pkwargs):
    plt.plot(n_pdb_missing(x), n_true_pos(x, direct_ref), fmt, **pkwargs)

plotf(humap_hc, 'k.', pkwargs = dict(label="hu MAP 2.0 HC"))
plotf(humap_med, 'k^', pkwargs = dict(label="hu MAP 2.0 MC"))
plotf(saint_pred_all, 'b.', pkwargs = dict(label = "SAINT All"))
plotf(saint_pred_05, 'bo', pkwargs = dict(label = "SAINT 0.5"))
plotf(saint_pred_07, 'b^', pkwargs = dict(label = "SAINT 0.7"))
#plotf(inm_all, 'g.', pkwargs = dict(label = "Av INM All"))
#plotf(inm_09, 'go', pkwargs = dict(label = "INM all 0.9"))
plotf(inm_mock_099, 'y^', pkwargs = dict(label = "INM mock 0.99"))
plotf(inm_099, 'g^', pkwargs = dict(label = "INM all 0.99"))
plotf(inm_low_prior_1_099, 'r^', pkwargs = dict(label = "INM low prior 1 mock 0.99"))
plotf(inm_low_prior_2_mock_099, 'rx', pkwargs = dict(label = "INM low prior 2 mock 0.99"))


plt.xlabel("N false (missing) positives")
plt.ylabel("N true (direct) positives")
#plt.ylim(0, 160)
#plt.legend()

# +
# Now take into account coverage of reference
INM_predicted_nodes = inm_all.get_node_list()
humap_hc_at_INM_predicted_nodes = humap_hc.node_select(INM_predicted_nodes)
humap_med_at_INM_predicted_nodes = humap_med.node_select(INM_predicted_nodes)
saint_pred_all_at_INM_predicted_nodes = saint_pred_all.node_select(INM_predicted_nodes)

saint_pred_all_at_INM_predicted_nodes_05 = threshold(saint_pred_all_at_INM_predicted_nodes, 0.5)
saint_pred_all_at_INM_predicted_nodes_07 = threshold(saint_pred_all_at_INM_predicted_nodes, 0.7)
# -

humap_all = gbf.get_humap_all_reference()

# +
# Get the INM Network models at the low prior and plot them



# +
# Plot indirect vs missing edges
def plotf(x, fmt, pkwargs):
    plt.plot(n_pdb_missing(x), n_true_pos(x, co_structure_ref), fmt, **pkwargs)

plotf(humap_hc_at_INM_predicted_nodes, 'k.', pkwargs = dict(label="hu MAP 2.0 0.94"))
plotf(humap_med_at_INM_predicted_nodes, 'k^', pkwargs = dict(label="hu MAP 2.0 0.5"))
plotf(saint_pred_all_at_INM_predicted_nodes, 'b.', pkwargs = dict(label = "SAINT All"))
plotf(saint_pred_all_at_INM_predicted_nodes_05, 'bo', pkwargs = dict(label = "SAINT 0.49"))
plotf(saint_pred_all_at_INM_predicted_nodes_07, 'b^', pkwargs = dict(label = "SAINT 0.7"))
#plotf(inm_all, 'g.', pkwargs = dict(label = "Av INM All"))
#plotf(inm_09, 'go', pkwargs = dict(label = "Av INM 0.9"))
plotf(inm_mock_099, 'g^', pkwargs = dict(label = "INM 0.99"))
#plotf(inm_mock_099, 'r^', pkwargs = dict(label = "Av INM mock 0.99"))


plotf(inm_low_prior_1_099, 'r^', pkwargs = dict(label = "INM low prior 1 0.99"))
#plotf(inm_low_prior_2_mock_099, 'rx', pkwargs = dict(label = "inm low prior 2 0.99"))
#plotf(saint_pred_all_at_INM_predicted_nodes_05, )


plt.xlabel("N false (missing) positives")
plt.ylabel("N true (co-structure) positives")
plt.ylim(0, 160)
plt.legend()
plt.savefig("AndrejsFigure_300.png", dpi=300)


# +
# Direct vs indirect
def ftemp(x, fmt, pkwargs):
    
    plt.plot(n_true_pos(x, indirect_ref), 
         n_true_pos(x, direct_ref), fmt, **pkwargs)
    
ftemp(humap_hc_at_INM_predicted_nodes, 'k.', dict(label = "hu MAP 2.0 0.94"))
ftemp(humap_med_at_INM_predicted_nodes, 'k^', dict(label= "hu MAP 2.0 0.49"))
ftemp(saint_pred_all_at_INM_predicted_nodes, 'b.', dict(label = "SAINT All"))
ftemp(saint_pred_all_at_INM_predicted_nodes_05, 'bo', dict(label = "SAINT 0.5"))
ftemp(saint_pred_all_at_INM_predicted_nodes_07, 'b^', dict(label = "SAINT 0.7"))
ftemp(inm_low_prior_2_mock_099, 'r^', dict(label="INM"))
plt.xlabel("N true positive (indirect)")
plt.ylabel("N true positive (direct)")
plt.legend()
plt.savefig("DirectvsIndirect_300.png", dpi=300)

# -

co_strucutre_at_INM = co_structure_ref.node_select(INM_predicted_nodes)

reindexer = gbf.get_cullin_reindexer()

reindexer

co_strucutre_at_INM.edge_values

n_pdb_missing(humap_hc_at_INM_predicted_nodes)


# +
# Plot indirect vs missing edges
def plotf(x, fmt, pkwargs):
    plt.plot(n_pdb_missing(x), n_true_pos(x, co_structure_ref), fmt, **pkwargs)

#plotf(humap_hc, 'k.', pkwargs = dict(label="hu MAP 2.0 HC"))
#plotf(humap_med, 'k^', pkwargs = dict(label="hu MAP 2.0 MC"))
plotf(saint_pred_all, 'b.', pkwargs = dict(label = "SAINT All"))
plotf(saint_pred_05, 'bo', pkwargs = dict(label = "SAINT 0.5"))
plotf(saint_pred_07, 'b^', pkwargs = dict(label = "SAINT 0.7"))
plotf(inm_all, 'g.', pkwargs = dict(label = "Av INM All"))
plotf(inm_09, 'go', pkwargs = dict(label = "Av INM 0.9"))
plotf(inm_099, 'g^', pkwargs = dict(label = "Av INM 0.99"))


plt.xlabel("N false (missing) positives")
plt.ylabel("N true (co-structure) positives")
plt.ylim(0, 160)
plt.legend()


# +
# Plot indirect vs missing edges
def plotf(x, fmt, pkwargs):
    plt.plot(n_pdb_missing(x), n_true_pos(x, direct_ref), fmt, **pkwargs)

#plotf(humap_hc, 'k.', pkwargs = dict(label="hu MAP 2.0 HC"))
#plotf(humap_med, 'k^', pkwargs = dict(label="hu MAP 2.0 MC"))
plotf(saint_pred_all, 'b.', pkwargs = dict(label = "SAINT All"))
plotf(saint_pred_05, 'bo', pkwargs = dict(label = "SAINT 0.5"))
plotf(saint_pred_07, 'b^', pkwargs = dict(label = "SAINT 0.7"))
plotf(inm_all, 'g.', pkwargs = dict(label = "Av INM All"))
plotf(inm_09, 'go', pkwargs = dict(label = "Av INM 0.9"))
plotf(inm_099, 'g^', pkwargs = dict(label = "Av INM 0.99"))


plt.xlabel("N false (missing) positives")
plt.ylabel("N true (direct) positives")
plt.ylim(0, 160)
plt.legend()


# +
# Plot indirect vs missing edges
def plotf(x, fmt, pkwargs):
    plt.plot(n_pdb_missing(x), n_true_pos(x, indirect_ref), fmt, **pkwargs)

#plotf(humap_hc, 'k.', pkwargs = dict(label="hu MAP 2.0 HC"))
#plotf(humap_med, 'k^', pkwargs = dict(label="hu MAP 2.0 MC"))
plotf(saint_pred_all, 'b.', pkwargs = dict(label = "SAINT All"))
plotf(saint_pred_05, 'bo', pkwargs = dict(label = "SAINT 0.5"))
plotf(saint_pred_07, 'b^', pkwargs = dict(label = "SAINT 0.7"))
plotf(inm_all, 'g.', pkwargs = dict(label = "Av INM All"))
plotf(inm_09, 'go', pkwargs = dict(label = "Av INM 0.9"))
plotf(inm_099, 'g^', pkwargs = dict(label = "Av INM 0.99"))


plt.xlabel("N false (missing) positives")
plt.ylabel("N true (indirect) positives")
plt.ylim(0, 160)
plt.legend()

# +
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
from numpyro.infer import MCMC, NUTS
import jax

# Define a simple model
def model():
    x = numpyro.sample("x", dist.Normal(0, 1))

# Setup the HMC sampler
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=10_000)
mcmc.run(rng_key=jax.random.PRNGKey(13), extra_fields = ("energy", "potential_energy"))
# -

# Get the energy trace
ef=  mcmc.get_extra_fields()
energy = ef['energy']
# Plot the energy
plt.figure(figsize=(10, 4))
plt.plot(energy)
plt.title('Total Energy Caterpillar Plot')
plt.xlabel('Iteration')
plt.ylabel('Total Energy')
plt.show()

plt.plot(ef['potential_energy'])

# Is SAINT as Accurate as INM?
co_structure_at_SAINT = co_structure_ref.node_select(saint_pred)
calc = tpr_ppr.PprTprCalculator(saint_pred, co_structure_ref)
calc_results = calc.crunch()

shared_edges = saint_pred.edge_identity_intersection(inm_all)



co_structure_ref.edge_identity_intersection

plt.plot(calc_results.ppr_points, calc_results.tpr_points)
plt.xlim(0, 1)
plt.ylim(0, 1)

calc_results.n_predicted_positives
