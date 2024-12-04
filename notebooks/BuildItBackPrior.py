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

# +
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import graph_tool
import pandas as pd
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpyro.distributions as dist
import numpy as np
from pathlib import Path
import math
import generate_benchmark_figures as gbf
import timeit
import networkx as nx
import _model_variations as mv
import pickle as pkl
from collections import namedtuple
import pyvis
import sampling_assessment as sa
from collections import defaultdict
import importlib
import dev_clique_and_community_src as dcc
importlib.reload(gbf)
# %matplotlib inline

def pklload(x):
    with open(x, "rb") as f:
        return pkl.load(f)

print(sa._e1)


# +
def gather_per_chain_results_dict(path, mname=""):
    results = sa.get_results(path, mname = mname)
    reindexer = gbf.get_cullin_reindexer()
    saint_prior = get_saint_prior(results)
    degree_prior = get_degree_prior_from_matrix_df(results.A_df)
    
    sort_matrix_df(saint_prior)
    sort_matrix_df(results.A_df)
    sort_matrix_df(degree_prior)

    pairwise_product = results.A_df * saint_prior
    deg_x_prod =  degree_prior * pairwise_product
    deg_x_edge_score = degree_prior * results.A_df

    assert np.alltrue(saint_prior.columns == results.A_df.columns)
    assert np.alltrue(saint_prior.index == results.A_df.index)
    assert np.alltrue(degree_prior.columns == results.A_df.columns)
    assert np.alltrue(degree_prior.index == results.A_df.index)
    assert np.alltrue(pairwise_product.columns == results.A_df.columns)
    assert np.alltrue(pairwise_product.index == results.A_df.index)
    assert np.alltrue(deg_x_prod.columns == results.A_df.columns)
    assert np.alltrue(deg_x_prod.index == results.A_df.index)

    edge_score_edgelist_df = sa.matrix_df_to_edge_list_df(results.A_df)
    pairwise_product_edgelist_df = sa.matrix_df_to_edge_list_df(pairwise_product)
    saint_prior_edgelist_df = sa.matrix_df_to_edge_list_df(saint_prior)
    deg_x_prod_edgelist_df = sa.matrix_df_to_edge_list_df(deg_x_prod)
    degree_prior_edgelist_df = sa.matrix_df_to_edge_list_df(degree_prior)
    deg_x_edge_score_edgelist_df = sa.matrix_df_to_edge_list_df(deg_x_edge_score)

    u_average_edge = u_from_edgelist_df(edge_score_edgelist_df)
    u_pairwise_prod = u_from_edgelist_df(pairwise_product_edgelist_df)
    u_saint_prior = u_from_edgelist_df(saint_prior_edgelist_df)
    u_deg_x_prod = u_from_edgelist_df(deg_x_prod_edgelist_df)
    u_deg_prior = u_from_edgelist_df(degree_prior_edgelist_df)
    u_deg_x_edge_score = u_from_edgelist_df(deg_x_edge_score_edgelist_df)

    return dict(
        results = results,
        saint_prior = saint_prior,
        degree_prior = degree_prior,
        pairwise_product = pairwise_product,
        deg_x_prod = deg_x_prod,
        deg_x_edge_score = deg_x_edge_score,
        average_edge_edgelist_df = edge_score_edgelist_df,
        pairwise_product_edgelist_df = pairwise_product_edgelist_df,
        saint_prior_edgelist_df = saint_prior_edgelist_df,
        deg_x_prod_edgelist_df = deg_x_prod_edgelist_df,
        degree_prior_edgelist_df = degree_prior_edgelist_df,
        deg_x_edge_score_edgelist_df = deg_x_edge_score_edgelist_df,
        u_average_edge = u_average_edge,
        u_pairwise_prod = u_pairwise_prod,
        u_saint_prior = u_saint_prior,
        u_deg_x_prod = u_deg_x_prod,
        u_deg_prior = u_deg_prior,
        u_deg_x_edge_score = u_deg_x_edge_score,
    )

# 1. Calculate maximal composite prey scores
def calculate_maximal_prey_saint_score_dict(results_path):
    composite_table = pd.read_csv(Path(sa._e1) / "composite_table.tsv", sep="\t")
    
    t = defaultdict(int)
    for i, r in composite_table.iterrows():
        prey_name = r['Prey']
        val = r['MSscore']
        if t[prey_name] < val:
            t[prey_name] = val
    return t

def calculate_pairwise_maximal_saint(max_saint_score_dict):
    """
    The prior probability of an edge given two SAINT scores
    p(co-purifying| A), p(co-purifying| B)
    """
    t = max_saint_score_dict
    n = len(t)
    matrix = np.zeros((n, n))
    k_list = [(name, max_score) for name, max_score in t.items()]
    names = [h[0] for h in k_list]
    for i in range(n):
        for j in range(1, i):
            a = k_list[i][1]
            b = k_list[j][1]
            matrix[i, j] = a * b
    matrix = matrix + matrix.T
    return pd.DataFrame(matrix, columns=names, index=names)

def get_saint_prior(r):
    # get the maximal prey scores
    max_prey_scores = calculate_maximal_prey_saint_score_dict(r.path)
    
    # filter by nodes in network
    temp = {}
    for key, value in max_prey_scores.items():
        if key in r.model_data['name2node_idx']:
            temp[key] = value
    max_prey_scores = temp
    return calculate_pairwise_maximal_saint(max_prey_scores)

def u_from_edgelist_df(e):
    u = gbf.UndirectedEdgeList()
    u.update_from_df(
        e, a_colname='a', b_colname='b',
        edge_value_colname='w', multi_edge_value_merge_strategy='max')
    u.reindex(reindexer, enforce_coverage=False)
    return u
    
def sort_matrix_df(mdf):
    mdf.sort_index(axis=0, inplace=True)
    mdf.sort_index(axis=1, inplace=True)
    
def get_degree_prior_from_matrix_df(matrix_df):
    N, _ = matrix_df.shape
    degree = np.sum(matrix_df.values, axis=0).reshape((N, 1)) / N
    degree = degree * degree.T
    degree_prior = 1 - degree
    return pd.DataFrame(degree_prior, columns = matrix_df.columns, index=matrix_df.index)

def plot_plotlist(plot_list):
    nrows = len(plot_list)
    fig, ax = plt.subplots(nrows, 1, figsize=(12, 10))
    tril_indices = np.tril_indices_from(m_erlp1u20k.A_df, k=-1)
    for i in range(nrows):
        hist(ax[i], np.ravel(plot_list[i][1].values[tril_indices]))
        ax[i].set_xlabel(plot_list[i][0])

    plt.tight_layout()

def hist(ax, x):
    ax.hist(x, bins=100, range=(0,1))

def plthist(x):
    plt.hist(np.array(x), bins=100)
    plt.show()
def multi_roc_plotter(predictions, references, reference_key, multi_save_suffix, model_output_dirpath = Path("BuildItBackPriorDir/",),
                      prediction_keys_to_omit = tuple()):
    pairs_to_plot_on_one_graph = [(x, reference_key) for x in predictions.keys() if x not in prediction_keys_to_omit]
    gbf.write_roc_curves_and_table(
        model_output_dirpath = model_output_dirpath,
        references = references,
        predictions = predictions,
        pairs_to_plot_on_one_graph = pairs_to_plot_on_one_graph,
        multi_save_suffix = multi_save_suffix,
    )

def multi_roc_plotter_from_results_dict(rd, references, reference_key, multi_save_suffix, model_output_dirpath = Path("BuildItBackPriorDir/",),
                      prediction_keys_to_omit = tuple()):
    predictions = dict(
        average_edge = rd['u_average_edge'],
        saint_pair_score = rd['u_saint_prior'],
        pair_prod = rd['u_pairwise_prod'],
        deg_score = rd['u_deg_prior'],
        deg_x_prod = rd['u_deg_x_prod'],
        deg_x_edge_score = rd['u_deg_x_edge_score'],
    )
    multi_roc_plotter(predictions, references, reference_key, multi_save_suffix, model_output_dirpath, prediction_keys_to_omit)


# -

importlib.reload(mv)

a = jnp.array([[0, 1, 1],
               [1, 0, 1],
               [1, 1, 0]])
b = jnp.array([0, 1, 1])
mv.calculate_metrics(a, b)

# Low prior mock uniform 20k
m_erlp1u20k = sa.get_results(sa._e1)

w_erlp1u20k = sa.get_results("../results/se_sr_low_prior_1_wt_20k/")

r_diag = sa.get_results("../results/se_sr_low_prior_1_uniform_mock_2k_diagnose/se_sr_low_prior_1_uniform_mock_2k_diagnose_rseed_0/", rseed=0)

reindexer = gbf.get_cullin_reindexer()
m_mini_model23_l_results_dict = gather_per_chain_results_dict(
    "../results/mini_model23_l/", mname="0_model23_l")

# +
#v_erlp1uv20k = sa.get_results("../results/se_sr_low_prior_1_uniform_")

plthist(mv.model23_unpack_model_data(m_mini_model23_l_results_dict["results"].model_data)[-4])

# +
m_saint_prior = get_saint_prior(m_erlp1u20k)

w_saint_prior = get_saint_prior(w_erlp1u20k)

m_degree_prior = get_degree_prior_from_matrix_df(m_erlp1u20k.A_df)

w_degree_prior = get_degree_prior_from_matrix_df(w_erlp1u20k.A_df)
# -

m_deg_inv_prior = 1 - m_degree_prior
w_deg_inv_prior = 1 - w_degree_prior

# +
sort_matrix_df(m_saint_prior)
sort_matrix_df(w_saint_prior)

sort_matrix_df(m_erlp1u20k.A_df)
sort_matrix_df(w_erlp1u20k.A_df)

sort_matrix_df(m_degree_prior)
sort_matrix_df(w_degree_prior)
# -

m_pairwise_product = m_erlp1u20k.A_df * m_saint_prior # element-wise product
w_pairwise_product = w_erlp1u20k.A_df * w_saint_prior # element-wise product

m_pairwise_average = (m_erlp1u20k.A_df + m_saint_prior) / 2
w_pairwise_average = (w_erlp1u20k.A_df + w_saint_prior) / 2

m_deg_x_av = m_degree_prior * m_pairwise_average
w_deg_x_av = w_degree_prior * w_pairwise_average

m_deg_x_prod = m_degree_prior * m_pairwise_product
w_deg_x_prod = w_degree_prior * w_pairwise_product

m_deg_inv_x_prod = m_deg_inv_prior * m_pairwise_product
w_deg_inv_x_prod = w_deg_inv_prior * w_pairwise_product

# +
m_corr = m_erlp1u20k.model_data["corr"]
m_corr.sort_index(axis=0, inplace=True)
m_corr.sort_index(axis=1, inplace=True)

w_corr = w_erlp1u20k.model_data["corr"]
w_corr.sort_index(axis=0, inplace=True)
w_corr.sort_index(axis=1, inplace=True)

# +
assert np.alltrue(m_saint_prior.columns == m_erlp1u20k.A_df.columns)
assert np.alltrue(m_saint_prior.index == m_erlp1u20k.A_df.index)
assert np.alltrue(m_saint_prior.index == m_erlp1u20k.A_df.columns)

assert np.alltrue(w_saint_prior.columns == w_erlp1u20k.A_df.columns)
assert np.alltrue(w_saint_prior.index == w_erlp1u20k.A_df.index)
assert np.alltrue(w_saint_prior.index == w_erlp1u20k.A_df.columns)

assert np.alltrue(m_pairwise_product.columns == m_erlp1u20k.A_df.columns)
assert np.alltrue(m_pairwise_average.columns == m_erlp1u20k.A_df.columns)
assert np.alltrue(m_corr.columns == m_saint_prior.columns)
# -

## Uncomment to plot various networks in matrix form
h = lambda x: sns.heatmap(x, vmin=0, vmax=1)
#h(m_corr)
#h(m_erlp1u20k.A_df)
#h(w_erlp1u20k.A_df)
#h(m_degree_prior * m_erlp1u20k.A_df)
#h(w_degree_prior * w_erlp1u20k.A_df)
#h(m_degree_prior)
#h(w_degree_prior)
#h(m_saint_prior.iloc[0:10, 0:10])
#h(w_saint_prior)
h(m_pairwise_product)
#h(w_pairwise_product)
#h(m_pairwise_average)
#h(w_pairwise_average)
#h(m_deg_x_av)
#h(w_deg_x_av)
#h(m_deg_x_prod)
#h(w_deg_x_prod)
#h(m_mini_model23_l_results_dict["results"].A_df)
#del h

# +
h(m_mini_model23_l_results_dict["results"].A_df)
#h(m_mini_model23_l_results_dict["saint_prior"])
#h(m_mini_model23_l_results_dict["degree_prior"])
#h(m_mini_model23_l_results_dict["results"].model_data["corr"])
#h(m_mini_model23_l_results_dict["pairwise_product"])
#h(m_mini_model23_l_results_dict["deg_x_prod"])
#h(m_mini_model23_l_results_dict["deg_x_edge_score"])

#h()
# -

m_mini_model23_l_results_dict["saint_prior"]

m_mini_model23_l_results_dict["results"].model_data["corr"]

m_mini_model23_l_results_dict["results"]

m_mini_model23_l_results_dict["degree_prior"]

m_mini_model23_l_results_dict.keys()


# +
m_plot_list = [
    ("m average edge score", m_erlp1u20k.A_df),
    ("m max pair saint score", m_saint_prior),
    ("m degree score", m_degree_prior),
    ("m pairwise product (edge & saint)", m_pairwise_product),
    ("m pairwise av (edge & saint)", m_pairwise_average),
    ("m deg_x_av", m_deg_x_av),
    ("m deg_x_prod", m_deg_x_prod),
    ("m deg dist", (m_degree_prior-1) * -1)
    ]

w_plot_list = [
    ("w average edge score", w_erlp1u20k.A_df),
    ("w max pair saint score", w_saint_prior),
    ("w degree score", w_degree_prior),
    ("w pairwise product (edge & saint)", w_pairwise_product),
    ("w pairwise av (edge & saint)", w_pairwise_average),
    ("w deg_x_av", w_deg_x_av),
    ("w deg_x_prod", w_deg_x_prod),
    ("w deg dist", (w_degree_prior-1) * -1)
    ]

plot_plotlist(m_plot_list)
    
# -

plot_plotlist(w_plot_list)

m_edge_score_edgelist_df = sa.matrix_df_to_edge_list_df(m_erlp1u20k.A_df)
w_edge_score_edgelist_df = sa.matrix_df_to_edge_list_df(w_erlp1u20k.A_df)

# +
m_pairwise_product_edgelist_df = sa.matrix_df_to_edge_list_df(m_pairwise_product)
w_pairwise_product_edgelist_df = sa.matrix_df_to_edge_list_df(w_pairwise_product)

m_pairwise_average_edgelist_df = sa.matrix_df_to_edge_list_df(m_pairwise_average)
w_pairwise_average_edgelist_df = sa.matrix_df_to_edge_list_df(w_pairwise_average)

m_saint_prior_edgelist_df = sa.matrix_df_to_edge_list_df(m_saint_prior)
w_saint_prior_edgelist_df = sa.matrix_df_to_edge_list_df(w_saint_prior)

m_degree_prior_edgelist_df = sa.matrix_df_to_edge_list_df(m_degree_prior)
w_degree_prior_edgelist_df = sa.matrix_df_to_edge_list_df(w_degree_prior)

m_degree_inv_prior_edgelist_df = sa.matrix_df_to_edge_list_df(m_deg_inv_prior)
w_degree_inv_prior_edgelist_df = sa.matrix_df_to_edge_list_df(w_deg_inv_prior)


# +

m_deg_inv_x_prod_edgelist_df = sa.matrix_df_to_edge_list_df(m_deg_inv_x_prod)
w_deg_inv_x_prod_edgelist_df = sa.matrix_df_to_edge_list_df(w_deg_inv_x_prod)

m_deg_x_prod_edgelist_df = sa.matrix_df_to_edge_list_df(m_deg_x_prod)
w_deg_x_prod_edgelist_df = sa.matrix_df_to_edge_list_df(w_deg_x_prod)


m_deg_x_av_edgelist_df = sa.matrix_df_to_edge_list_df(m_deg_x_av)
w_deg_x_av_edgelist_df = sa.matrix_df_to_edge_list_df(m_deg_x_av)

m_corr_edgelist_df = sa.matrix_df_to_edge_list_df(m_corr)
w_corr_edgelist_df = sa.matrix_df_to_edge_list_df(w_corr)

# +
u_m_pairwise_product = u_from_edgelist_df(m_pairwise_product_edgelist_df)
u_w_pairwise_product = u_from_edgelist_df(w_pairwise_product_edgelist_df)

u_m_pairwise_av = u_from_edgelist_df(m_pairwise_average_edgelist_df)
u_w_pairwise_av = u_from_edgelist_df(w_pairwise_average_edgelist_df)

u_m_corr = u_from_edgelist_df(m_corr_edgelist_df)
u_w_corr = u_from_edgelist_df(w_corr_edgelist_df)

u_m_degree_prior = u_from_edgelist_df(m_degree_prior_edgelist_df)
u_w_degree_prior = u_from_edgelist_df(m_degree_prior_edgelist_df)

u_m_degree_inv_prior = u_from_edgelist_df(m_degree_inv_prior_edgelist_df)
u_w_degree_inv_prior = u_from_edgelist_df(w_degree_inv_prior_edgelist_df)

u_m_degree_prior_x_pair_prod = u_from_edgelist_df(m_deg_x_prod_edgelist_df)
u_w_degree_prior_x_pair_prod = u_from_edgelist_df(w_deg_x_prod_edgelist_df)

u_m_degree_inv_prior_x_pair_prod = u_from_edgelist_df(m_deg_inv_x_prod_edgelist_df)
u_w_degree_inv_prior_x_pair_prod = u_from_edgelist_df(w_deg_inv_x_prod_edgelist_df)

u_m_degree_prior_x_pair_av = u_from_edgelist_df(m_deg_x_av_edgelist_df)
u_w_degree_prior_x_pair_av = u_from_edgelist_df(w_deg_x_av_edgelist_df)


# +

u_m_edge_score = u_from_edgelist_df(m_edge_score_edgelist_df)
u_w_edge_score = u_from_edgelist_df(w_edge_score_edgelist_df)
# -

u_saint_max = gbf.get_cullin_saint_scores_edgelist()
u_saint_max = u_saint_max.node_select(u_saint_max.node_intersection(u_w_pairwise_product))
assert u_saint_max.n_nodes == 235
assert u_saint_max.nedges == 654
u_saint_max_all = gbf.get_cullin_saint_scores_edgelist()

# The saint prior is defined for 234 nodes and 27261 possible edges
u_saint_pair_prior = u_from_edgelist_df(m_saint_prior_edgelist_df)
u_saint_pair_prior.node_intersection(gbf.get_pdb_ppi_predict_cocomplex_reference())
assert u_saint_pair_prior.n_nodes == 234

# +
m_predictions = dict(
    average_edge = u_m_edge_score,
    saint_pair_score = u_saint_pair_prior,
    pair_prod = u_m_pairwise_product,
    #pair_av = u_m_pairwise_av,
    deg_score = u_m_degree_prior,
    #deg_inv_score = u_m_degree_inv_prior,
    deg_x_prod = u_m_degree_prior_x_pair_prod,
    #deg_inv_x_prod = u_m_degree_inv_prior_x_pair_prod,
    #deg_x_av = u_m_degree_prior_x_pair_av,
    #corr = u_m_corr,
    saint_max = u_saint_max,
)

w_predictions = dict(
    average_edge = u_w_edge_score,
    saint_pair_score = u_saint_pair_prior,
    pair_prod = u_w_pairwise_product,
    #pair_av = u_w_pairwise_av,
    deg_score = u_w_degree_prior,
    #deg_inv_score = u_w_degree_inv_prior,
    deg_x_prod = u_w_degree_prior_x_pair_prod,
    #deg_inv_x_prod = u_w_degree_inv_prior_x_pair_prod,
    #deg_x_av = u_w_degree_prior_x_pair_av,
    #corr = u_w_corr,
    saint_max = u_saint_max,
)

m_predictions_test = {key : val for key, val in m_predictions.items() if key in ("saint_max", "average_edge")}
# -

references = dict(
  costructure = gbf.get_pdb_ppi_predict_cocomplex_reference(),
  direct = gbf.get_pdb_ppi_predict_direct_reference(),
  huri = gbf.get_huri_reference(),
  humap_hc = gbf.get_humap_high_reference()
)

key = "direct"
multi_roc_plotter_from_results_dict(m_mini_model23_l_results_dict, references, key, f"mini_model23_l_experi_2024_5_25_{key}", model_output_dirpath = Path("BuildItBackPriorDir/"))

references = references | {"indirect" : references['costructure'].edge_identity_difference(
    references['direct'])}

references = references | {
    "decoy149" : gbf.get_decoys_from_u(jax.random.PRNGKey(303), m_predictions['average_edge'], 149)}
references = references | {
    "decoy23" : gbf.get_decoys_from_u(jax.random.PRNGKey(404), w_predictions['average_edge'], 23)}

df = pd.read_excel("../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx")

sel = df["BFDR"] < 0.15
df.loc[sel, "Prey"].shape


# +
def get_inm_all_nodes():
    a = sa.get_results("../results/se_sr_mock_ctrl_20k/")
    b = sa.get_results("../results/se_sr_vif_ctrl_20k/")
    c = sa.get_results("../results/se_sr_wt_ctrl_20k/")

    def get_nodes(x):
        return set(x.model_data['name2node_idx'].keys())
    return get_nodes(a) & get_nodes(b) & get_nodes(c)

def get_saint_network():
    a = pd.read_excel("../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx", sheet_name=0)
    b = pd.read_excel("../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx", sheet_name=1)
    c = pd.read_excel("../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx", sheet_name=2)
    d = pd.concat([a, b, c])
    return d

def filter_saint_network(d):

    def selecct_bfdr(x):
        return x[x["BFDR"] < 0.15]

    def parse_bait(x):
        x = x[0:4]
        if x == "CBFB":
            x = "PEBB"
        return x
    def parse_prey(x):
        return x.split("_")[0]

    d = selecct_bfdr(d)
    d["Bait"] = [parse_bait(x) for x in d["Bait"]]
    d["PreyGene"] = [parse_prey(x) for x in d["PreyGene"]]
    return d

def saint_network_df2u(d):
    u = gbf.UndirectedEdgeList()
    u.update_from_df(d, a_colname="PreyGene", b_colname="Bait", edge_value_colname="BFDR", multi_edge_value_merge_strategy="max")
    reindexer = gbf.get_cullin_reindexer()
    u.reindex(reindexer, enforce_coverage=False)
    return u





# +

all_inm_nodes = get_inm_all_nodes()


# -

humap_hc_filtered = huap_hc.node_select({reindexer[x] for x in all_inm_nodes})

saint_network_all_df = get_saint_network()
filtered_saint_network_df = filter_saint_network(saint_network_all_df)
saint_network_at_inm_all = saint_network_df2u(filtered_saint_network_df)

saint_network_all 

humap_hc = gbf.get_humap_high_reference()
humap_hc_at_inm = humap_hc.node_select({reindexer[x] for x in all_inm_nodes})


# +
def tp(u_pred, u_ref):
    return len(u_pred.edge_identity_intersection(u_ref))
def fp(u_pred, u_ref):
    return u_pred.edge_identity_difference(u_ref).nedges


def tn(u_pred, u_ref, N):
    M = math.comb(N, 2)
    n_negatives = M - u_ref.nedges # number of negatives is the number of possible edges minus the number of positives
    return n_negatives - fp(u_pred, u_ref) 

def fn(u_pred, u_ref):
    """
    
    """
    true_positives = tp(u_pred, u_ref) 
    n_positives = u_ref.nedges
    return n_positives - true_positives  

def compare(u_pred, u_ref,N):
    return namedtuple("Comparison", ["tp", "fp", "tn", "fn"])(
        tp(u_pred, u_ref), fp(u_pred, u_ref), tn(u_pred, u_ref, N), fn(u_pred, u_ref))

def all_positive(comparison):
    return comparison.tp + comparison.fn

def all_negatives(comparison):
    return comparison.fp + comparison.tn

def compare_and_check(u_pred, u_ref, N):
    print("      Pred | Ref")
    print("edges", u_pred.nedges, u_ref.nedges)
    print("nodes", u_pred.n_nodes, u_ref.n_nodes)
    comparison = compare(u_pred, u_ref, N)
    assert all_positive(comparison) == u_ref.nedges
    assert all_negatives(comparison) == math.comb(N, 2) - u_ref.nedges
    return comparison

def accuracy(comparison):
    return (comparison.tp + comparison.tn) / (comparison.tp + comparison.tn + comparison.fp + comparison.fn)

def precision(comparison):
    return comparison.tp / (comparison.tp + comparison.fp)

def recall(comparison):
    return comparison.tp / (comparison.tp + comparison.fn)

def f1(comparison):
    p = precision(comparison)
    r = recall(comparison)
    return 2 * p * r / (p + r)

def compare_row(name, comparison):
    return pd.Series({
        "tp" : comparison.tp,
        "fp" : comparison.fp,
        "tn" : comparison.tn,
        "fn" : comparison.fn,
        "accuracy" : accuracy(comparison),
        "precision" : precision(comparison),
        "recall" : recall(comparison),
        "f1" : f1(comparison)
    }, name=name)

def make_comparison_dataframe(u_pred_dict, u_ref, N):
    df = pd.DataFrame([compare_row(name, compare_and_check(u_pred, u_ref, N)) for name, u_pred in u_pred_dict.items()])
    df.sort_values("f1", ascending=False)
    return df


# -

.edge_identity_difference(humap_hc)

tp(humap_hc_at_inm, direct_at_inm)

direct_at_inm = references['direct'].node_select({reindexer[x] for x in all_inm_nodes})
humap_all = gbf.get_humap_all_reference()


humap_all_at_inm = humap_all.node_select({reindexer[x] for x in all_inm_nodes})

humap_med = gbf.get_humap_medium_reference()

humap_med_at_inm = humap_med.node_select({reindexer[x] for x in all_inm_nodes})

# Do the comparison
compare_and_check(humap_hc_at_inm, direct_at_inm, N=len(all_inm_nodes))

compare_and_check(humap_med_at_inm, direct_at_inm, N=len(all_inm_nodes))

compare_and_check(saint_network_at_inm_all, direct_at_inm, N=len(all_inm_nodes))

compare_dict = {"humap_hc" : humap_hc_at_inm, "humap_md" : humap_med_at_inm, "saint" : saint_network_at_inm_all}
table_x = make_comparison_dataframe(compare_dict, direct_at_inm, N=len(all_inm_nodes))

table_x.round(2)
table_x.to_csv("comparison_table_x.csv")

print(math.comb(len(all_inm_nodes), 2))
print(len(all_inm_nodes))

pdb_direct = references["direct"]

saint_network_global = gbf.get_cullin_saint_scores_edgelist()



compare_dict_global = {"humap_hc" : humap_hc, "humap_md" : humap_med, "saint": saint_network_global}
table_global = make_comparison_dataframe(compare_dict_global, pdb_direct, N=1042)
table_global.to_csv("comparison_table_global.csv")

math.comb(1042, 2)

saint_network_global.nedges

table_global.round(2)

humap_all = gbf.get_humap_all_reference()


# +
def multi_roc_plotter(predictions, references, reference_key, multi_save_suffix, model_output_dirpath = Path("BuildItBackPriorDir/",),
                      prediction_keys_to_omit = tuple()):
    pairs_to_plot_on_one_graph = [(x, reference_key) for x in predictions.keys() if x not in prediction_keys_to_omit]
    gbf.write_roc_curves_and_table(
        model_output_dirpath = model_output_dirpath,
        references = references,
        predictions = predictions,
        pairs_to_plot_on_one_graph = pairs_to_plot_on_one_graph,
        multi_save_suffix = multi_save_suffix,
    )

def multi_roc_plotter_from_results_dict(rd, references, reference_key, multi_save_suffix, model_output_dirpath = Path("BuildItBackPriorDir/",),
                      prediction_keys_to_omit = tuple()):
    predictions = dict(
        average_edge = rd['u_average_edge'],
        saint_pair_score = rd['u_saint_prior'],
        pair_prod = rd['u_pairwise_prod'],
        deg_score = rd['u_deg_prior'],
        deg_x_prod = rd['u_deg_x_prod'],
        deg_x_edge_score = rd['u_deg_x_edge_score'],
    )
    multi_roc_plotter(predictions, references, reference_key, multi_save_suffix, model_output_dirpath, prediction_keys_to_omit)


# -

multi_roc_plotter_from_results_dict(
    m_mini_model23_l_results_dict, references, "direct", "mini_model23_l_all", 
    model_output_dirpath=Path("BuildItBackPriorDir/"))

multi_roc_plotter(predictions = m_predictions_test, references = references, reference_key= "direct", multi_save_suffix= "_m_direct_test")

multi_roc_plotter(predictions = m_predictions_test, references = references, reference_key= "costructure", multi_save_suffix= "_m_costructure_test")

multi_roc_plotter(predictions = m_predictions, references = references, reference_key= "direct", multi_save_suffix= "_m_direct")

multi_roc_plotter(predictions = m_predictions, references = references, reference_key= "costructure", multi_save_suffix= "_m_costructure")

multi_roc_plotter(predictions = w_predictions, references = references, reference_key= "direct", multi_save_suffix= "_w_direct")

multi_roc_plotter(predictions = w_predictions, references = references, reference_key= "costructure", multi_save_suffix= "_w_costructure")

multi_roc_plotter(predictions = w_predictions, references = references, reference_key= "decoy149", multi_save_suffix= "_w_decoy149")

m_deg_x_prod_edgelist_df.sort_values(by='w', ascending=False, inplace=False)
direct_ref = gbf.get_pdb_ppi_predict_direct_reference()
direct_ref.reindex({val: key for key,val in reindexer.items()}, enforce_coverage=False)
direct_ref._build_edge_dict()
m_deg_x_prod_edgelist_df["in_direct"] = [frozenset([r['a'], r['b']]) in direct_ref._edge_dict for i, r in m_deg_x_prod_edgelist_df.iterrows() ]

#m_deg_x_prod_edgelist_df.sort_values(by='w', ascending=False, inplace=False).iloc[0:200].loc[:, "in_direct"].sum()
m_deg_x_prod_edgelist_df.sort_values(by='w', ascending=False, inplace=False).iloc[0:20, :]

# #### Manually check the top 20 edges for sanity
# - SGT1, LLR1 | 

# direct_ref

gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", "costructure"),
    ("average_edge", "costructure"),
    ("pair_prod", "costructure"),
    ("pair_av", "costructure"),
    ("degree_prior", "costructure"),
    ("deg_x_prod", "costructure"),
    ("deg_x_av", "costructure"),
    ("saint_max", "costructure"),
    #("humap2_hc", "costructure"),
    #("humap2_med", "costructure"),
    #("m3", "costructure",)
    ),
    multi_save_suffix="_co_structure")

gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", "direct"),
    ("average_edge", "direct"),
    ("pair_prod", "direct"),
    ("pair_av", "direct"),
    ("degree_prior", "direct"),
    ("deg_x_prod", "direct"),
    ("deg_x_av", "direct"),
    ("saint_max", "direct",)
    #("humap2_hc", "direct"),
    #("humap2_med", "direct")
    #("m3", "direct"),
    ),
    multi_save_suffix="_direct")

ref_key = "indirect"
gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", ref_key),
    ("average_edge", ref_key),
    ("pair_prod", ref_key),
    ("pair_av", ref_key),
    ("degree_prior", ref_key),
    ("deg_x_prod", ref_key),
    ("deg_x_av", ref_key),
    ("saint_max", ref_key,)
    #("humap2_hc", ref_key),
    #("humap2_all", ref_key)
    #("m3", "direct"),
    ),
    multi_save_suffix=f"_{ref_key}")

ref_key = "huri"
gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", ref_key),
    ("average_edge", ref_key),
    ("pair_prod", ref_key),
    ("pair_av", ref_key),
    ("degree_prior", ref_key),
    ("deg_x_prod", ref_key),
    ("deg_x_av", ref_key),
    ("saint_max", ref_key,)
    #("humap2_hc", ref_key),
    #("humap2_all", ref_key)
    #("m3", "direct"),
    ),
    multi_save_suffix=f"_{ref_key}")

ref_key = "decoy149"
gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", ref_key),
    ("average_edge", ref_key),
    ("pair_prod", ref_key),
    ("pair_av", ref_key),
    ("degree_prior", ref_key),
    ("deg_x_prod", ref_key),
    ("deg_x_av", ref_key),
    ("saint_max", ref_key,)
    #("humap2_hc", ref_key),
    #("humap2_all", ref_key)
    #("m3", "direct"),
    ),
    multi_save_suffix=f"_{ref_key}")

ref_key = "decoy23"
gbf.write_roc_curves_and_table(
    model_output_dirpath = Path("BuildItBackPriorDir/"),
    references=references,
    predictions=predictions,
    pairs_to_plot_on_one_graph = (
    ("saint_prior", ref_key),
    ("average_edge", ref_key),
    ("pair_prod", ref_key),
    ("pair_av", ref_key),
    ("degree_prior", ref_key),
    ("deg_x_prod", ref_key),
    ("deg_x_av", ref_key),
    ("saint_max", ref_key,)
    #("humap2_hc", ref_key),
    #("humap2_all", ref_key)
    #("m3", "direct"),
    ),
    multi_save_suffix=f"_{ref_key}")

# +
#deg_x_av_top_edge_set
# -

corr_res = gbf.do_benchmark(u_m_corr)

plt.plot(corr_res.ppr_points, corr_res.tpr_points)
plt.show()

m_erlp1u20k.model_data["apms_corr_flat"]

kde = sp.stats.gaussian_kde(m_erlp1u20k.model_data["apms_corr_flat"])
jkde = jsp.stats.gaussian_kde(m_erlp1u20k.model_data["apms_corr_flat"])
x = np.arange(-1, 1, 0.01)

plt.plot(x, kde.pdf(x))
plt.hist(np.array(m_erlp1u20k.model_data["apms_corr_flat"]), bins=100, density=True)
plt.plot(x, jkde.pdf(x))
plt.show()

# +

plt.plot(x, jkde.logpdf(x))

# -

from scipy import interpolate
from numpyro.distributions import Distribution
class KDE_Distribution(Distribution):
    def __init__(self, kde, validate_args=None):
        self.kde = kde
        # Create an approximate CDF by integrating the KDE
        self.cdf_grid = jnp.linspace(-2, 2, 10000)  # Adjust the range and number of points as needed
        self.cdf_values = jnp.cumsum(self.kde(self.cdf_grid)) * (self.cdf_grid[1] - self.cdf_grid[0])
        # Create an interpolating function for the inverse CDF
        self.inv_cdf = interpolate.interp1d(self.cdf_values, self.cdf_grid, bounds_error=False, fill_value="extrapolate")
        super(KDE_Distribution, self).__init__(batch_shape=jnp.shape(kde), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        # Generate uniform random numbers
        u = jax.random.uniform(key, shape=sample_shape)
        # Transform them using the inverse CDF
        return self.inv_cdf(u)

    def log_prob(self, value):
        validate_sample(self, value)
        return jnp.log(self.kde(value))


kde_dist = KDE_Distribution(jkde)

plt.plot(kde_dist.cdf_grid, kde_dist.cdf_values)

plt.hist(mv.model23_unpack_model_data(m_mini_model23_l_results_dict["results"].model_data)[-4], bins=100)
plt.show()

x = np.arange(-1, 1, 0.01)
y = sp.stats.norm(0, 0.2).pdf(x)
n2 = sp.stats.norm(1 / 3.333, 0.2)
y2 = n2.pdf(x)
c2 = n2.cdf(x)
plt.vlines(0.5, 0, 2, 'r')
plt.hlines(0.75, -1, 1, 'r')
plt.plot(x, y)
plt.plot(x, y2)
plt.plot(x, c2)

saint_pair_score = mv.matrix2flat(m_mini_model23_l_results_dict["saint_prior"].values)

plthist(saint_pair_score)

# +
x = np.array([0.4, 0.6, 1.])
y = x + 1e-2

z = np.arange(-1, 1, 0.01)
for i, mu in enumerate(x):
    s = y[i]
    n = sp.stats.norm(mu-0.2, s)
    plt.plot(z, n.pdf(z), label=f"mu={mu}")
plt.legend()
# -

all_model23_l = sa.get_results("../results/mini_model23_l/", mname="0_model23_l")

n = 80
all_model23_l.edgelist_df.sort_values(by='w', ascending=False, inplace=False).iloc[n:n+20, :]

values = kde_dist.sample(jax.random.PRNGKey(0), (10_000,))



rynths = sa.get_results("../results/mini_model23_l_mock_synthetic/", mname="0_model23_l")

rynths.model_data["synthetic__"]

plt.hist(np.array(values), bins=100, density=True)
plt.show()

# +
# Let's say we want to predict a network based on the ROC type curve
# 1% of the total interaction space
# -

pp_top_node_set, pp_top_edge_set = dcc.get_top_edges_upto_threshold(pairwise_product_edgelist_df, threshold=0.9997)

deg_x_av_top_node_set, deg_x_av_top_edge_set = dcc.get_top_edges_upto_threshold(
    m_degree_prior_x_pair_av_edgelist_df, threshold=0.9)


# +
def fraction_of_all_pairs(n_nodes, nedges):
    all_pairs = math.comb(n_nodes, 2)
    return nedges / all_pairs

f(len(pp_top_node_set), len(pp_top_edge_set))
# -

len(pp_top_node_set), len(pp_top_edge_set), (len(pp_top_edge_set) / 27261) * 100

len(deg_x_av_top_node_set), len(deg_x_av_top_edge_set), (len(deg_x_av_top_edge_set) / 27261) * 100

# +
# Build a network using the top edges
# -

net_pp = dcc.pyvis_plot_network(pp_top_edge_set)

# +
#net_pp.show("pp.html")
# -

net_deg_x_av = dcc.pyvis_plot_network(deg_x_av_top_edge_set)

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

net_deg_x_av.show("deg_x_av.html")

deg_x_av_wt_top_node_set, deg_x_av_wt_top_edge_set = dcc.get_top_edges_upto_threshold(
    degree_prior_x_pair_av_wt_edgelist_df, threshold=0.85)

net_dev_x_av_wt = dcc.pyvis_plot_network(deg_x_av_wt_top_edge_set)

net_dev_x_av_wt.show("deg_x_av_wt.html")

x = degree_prior_x_pair_av_wt_edgelist_df
sel = x['a'] == 'vifprotein'
x.sort_values("w", ascending=False).loc[sel, :]

np.sum(pairwise_product_edgelist_df['w'] >= 0.9995)

pairwise_product_edgelist_df.sort_values('w', ascending=False).iloc[0:50, :]

pairwise_product_edgelist_df.sort_values('w', ascending=False).iloc[0:50]

d = pd.read_csv("../significant_cifs/BSASA_concat.csv")

for i, r in d.iterrows():
    try:
        float(r['SASA12'])
        float(r['BSASA'])
        float(r['SASA1'])
    except:
        break

# +
# Filter out KeyErrors
sel1 = []
sel2 = []
sel3 = []
sel4 = []

def keyerror_append(x, v):
    if v != "KeyError":
        x.append(False)
    else:
        x.append(True)
for i, r in d.iterrows():
    keyerror_append(sel1, r['BSASA'])
    keyerror_append(sel2, r['SASA12'])
    keyerror_append(sel3, r['SASA1'])
    keyerror_append(sel4, r['SASA2'])

sel1 = np.array(sel1)
sel2 = np.array(sel2)
sel3 = np.array(sel3)
sel4 = np.array(sel4)

sel = sel1 & sel2
sel = sel & sel3
sel = sel & sel4
# -

# 44 rows are key error
d_filt = d[~sel]


def update_by_key(d, key):
    d.loc[:, key] = np.array(d[key].values, dtype=np.float64)
    return d


d_filt = update_by_key(d_filt, "BSASA")
d_filt = update_by_key(d_filt, "SASA12")
d_filt = update_by_key(d_filt, "SASA1")
d_filt = update_by_key(d_filt, "SASA2")

d_filt.sum()

sel = [isinstance(i, str) for i in d["SASA12"]]

d.loc[0, "SASA12"]
