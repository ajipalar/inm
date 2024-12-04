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

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import graph_tool
import pandas as pd
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
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


# +
# Load in the average network
def pkl_load(path):
    with open(path, "rb") as f:
        return pkl.load(f)

def av_edge_load(path):
    return pd.read_csv(path, sep="\t")

def get_clique_list(G):
    return list(nx.find_cliques(G))

def get_n_from_iterator(iterator):
    """Memory efficient"""
    n = 0
    for i in iterator:
        n += 1
    return n

def clique_plot(ts, ns):
    plt.plot(ts, ns, 'k.')
    plt.xlabel("threshold")
    plt.ylabel("N cliques")

def get_filter_network(G, threshold):
    filtered_graph = nx.Graph()
    nodes = G.nodes()
    filtered_graph.add_nodes_from(nodes)
    filtered_edges = [(u, v, w) for u,v,w in G.edges(data='weight') if w > threshold]
    filtered_graph.add_weighted_edges_from(filtered_edges)
    return filtered_graph

def add_names_to_df(df, uid2name):
    df.loc[:, "aname"] = [uid2name[uid] for uid in df['auid']]
    df.loc[:, "bname"] = [uid2name[uid] for uid in df['buid']]
    
def get_results_df_spec_counts(dir_str):
    path = dir_str + "average_predicted_edge_scores.tsv"
    df = av_edge_load(path)
    spec_counts = pd.read_csv(
        dir_str + "spec_table.tsv",
        sep="\t",
        index_col=0)
    return df, spec_counts

def plot_edges(ts, ne):
    plt.plot(ts, ne, 'k')
    plt.ylabel("number of edges")
    plt.xlabel("threshold")
    
def get_n_maximal_cliques(G, ts):
    return [get_n_from_iterator(nx.find_cliques(get_filter_network(G, t))) for t in ts]
    
def h(x):
    return '{:,}'.format(x)


def get_nedges(G, ts):
    return [get_filter_network(G, t).number_of_edges() for t in ts]

def get_graph_from_df(df):
    G = nx.Graph()

    # Add edges with weights to the graph
    for i, r in df.iterrows():
        G.add_edge(r['a_gene'], r['b_gene'], weight=r['w'])
    return G

def CSN_selector(df):
    sel = np.array(["CSN" in r['a_gene'] for i,r in df.iterrows()])
    sel2 = np.array(["CSN" in r['b_gene'] for i,r in df.iterrows()])
    sel3 = sel & sel2
    return sel3

def dataframe_from_matrix(matrix, node_names, name2uid):
    matrix_predictor = []
    for i in range(236):
        for j in range(0, i):
            val = float(matrix[i, j])
            aname = node_names[i]
            bname = node_names[j]
    
            matrix_predictor.append((aname, bname, val))
    
    matrix_predictor_df = pd.DataFrame(
                matrix_predictor, 
                columns=['aname', 'bname', 'w'])

    auid = [name2uid[name] for name in matrix_predictor_df['aname']]
    buid = [name2uid[name] for name in matrix_predictor_df['bname']]
    matrix_predictor_df.loc[:, 'auid'] = auid
    matrix_predictor_df.loc[:, 'buid'] = buid
    return matrix_predictor_df

def add_names_to_df(x, name2uid):
    aname = [name2uid[name] for name in x['auid']]
    bname = [name2uid[name] for name in x['buid']]
    x.loc[:, 'aname'] = aname
    x.loc[:, 'bname'] = bname
    

def plot_benchmark(a, b, c):
    plt.plot(a.ppr_points, a.tpr_points, label="cocomplex")
    plt.plot(b.ppr_points, b.tpr_points, label="direct")
    plt.plot(c.ppr_points, c.tpr_points, label="indirect")
    plt.legend()
    
def get_comparisons(u, dref, iref, cref):
    direct = gbf.do_benchmark(u, dref)
    indirect = gbf.do_benchmark(u, iref)
    cocomplex = gbf.do_benchmark(u, cref)
    return direct, indirect, cocomplex

def matrix_df_to_edge_list_df(df):
    a, b = df.shape
    columns = df.columns
    an = []
    bn = []
    w = []
    for i in range(a):
        for j in range(0, i):
            an.append(columns[i])
            bn.append(columns[j])
            w.append(float(df.iloc[i, j]))
    return pd.DataFrame({'a': an, 'b': bn, 'w': w})

def mymatshow(x, ax=None):
    plt.matshow(x, cmap='plasma')
    plt.colorbar(shrink=0.8)
    
def heatshow(x, square=True, n=12, shrink=0.8):
    fig, ax = plt.subplots(figsize=(n, n))
    cbar_kws = {"shrink": shrink}
    sns.heatmap(x, square=square, cbar_kws=cbar_kws)
    
def get_top_edges_upto_threshold(edgelist_df, threshold=0.5):
    k = 0
    top_edges = {}
    top_node_set = []
    for i, r in edgelist_df.sort_values('w', ascending=False).iterrows():
        #print(r['a'], r['b'], round(r['w'], 2))
        top_node_set.append(r['a'])
        top_node_set.append(r['b'])
        top_edges[k] = r['a'], r['b']
        if r['w'] < threshold:
            break
        k += 1
    return list(set(top_node_set)), top_edges

def pyvis_plot_network(top_edges):
    G = nx.Graph()
    G.add_edges_from([(val[0], val[1]) for i, val in top_edges.items()])
    net = pyvis.network.Network(notebook = True)
    net.from_nx(G)
    return net

def cluster_map_from_clustergrid(data, clustergrid, n_clusters, criterion):
    row_clusters, col_clusters = get_cluster_assignments(clustergrid, n_clusters, criterion)
    row_order = clustergrid.dendrogram_row.reordered_ind
    col_order = clustergrid.dendrogram_col.reordered_ind
    return remap_to_cluster_maps(data, row_clusters, col_clusters, row_order, col_order)

def get_cluster_assignments(clustergrid, n_clusters, criterion="maxclust"):
    # Assuming you want to cut by a specific number of clusters, for example, 2 clusters
    row_clusters = fcluster(clustergrid.dendrogram_row.linkage, n_clusters, criterion='maxclust')
    col_clusters = fcluster(clustergrid.dendrogram_col.linkage, n_clusters, criterion='maxclust')
    return row_clusters, col_clusters

def remap_to_cluster_maps(data, row_clusters, col_clusters, row_order, col_order):
    # Create a dictionary mapping from cluster number to rows/columns
    row_cluster_map = {i + 1: [] for i in range(row_clusters.max())}
    for idx, cluster_id in enumerate(row_clusters):
        row_cluster_map[cluster_id].append(data.index[row_order[idx]])

    col_cluster_map = {i + 1: [] for i in range(col_clusters.max())}
    for idx, cluster_id in enumerate(col_clusters):
        col_cluster_map[cluster_id].append(data.columns[col_order[idx]])
    return {"row" : row_cluster_map, "col": col_cluster_map}

def min_max_scale_to_01(a):
    return (a - np.min(a)) / (np.max(a)-np.min(a))

def get_results(path):
    mcmc = pkl_load(path / "0_model23_se_sr_13.pkl")
    model_data = pkl_load(path / "0_model23_se_sr_13_model_data.pkl")

    samples = mcmc['samples']
    As = mv.Z2A(samples['z'])
    As_av = np.mean(As, axis=0)
    A = mv.flat2matrix(As_av, model_data['N'])
    u = gbf.model23_matrix2u(A, model_data)
    A_df = pd.DataFrame(A, 
    index =   [model_data['node_idx2name'][k] for k in range(model_data['N'])],
    columns = [model_data['node_idx2name'][k] for k in range(model_data['N'])])
    
    edgelist_df = matrix_df_to_edge_list_df(A_df)
    
    return Results(
      path = path,
      mcmc = mcmc,
      samples = samples,
      model_data = model_data,
      As = As,
      As_av = As_av,
      A = A,
      u = u,
      A_df = A_df,
      edgelist_df = edgelist_df,
    )
    


# +
# Globals
u20_base_path = Path("../results/se_sr_low_prior_1_uniform_all_20k/")

name2uid = gbf.get_cullin_reindexer()
uid2name = {val: key for key, val in name2uid.items()}

pdb_direct = gbf.get_pdb_ppi_predict_direct_reference()
pdb_cocomplex = gbf.get_pdb_ppi_predict_cocomplex_reference()
pdb_indirect = pdb_cocomplex.edge_identity_difference(pdb_direct)


# Paths
BasePaths = namedtuple("BasePaths",
        "wt vif u20")

base_paths = BasePaths(
    wt = Path("../results/se_sr_wt_ctrl_20k/"),
    vif = Path("../results/se_sr_vif_ctrl_20k/"),
    u20 = Path("../results/se_sr_low_prior_1_uniform_all_20k/"),
    )

wt_path = "../results/se_sr_wt_ctrl_20k/"
vif_path = "../results/se_sr_vif_ctrl_20k/"
mock_path = "../results/se_sr_mock_ctrl_20k/"
all_u_20 = "../results/se_sr_low_prior_1_uniform_all_20k/"
mock_lp1_uniform_results = "../results/se_sr_low_prior_1_uniform_mock_20k/"
Results = namedtuple("Results",
    "path mcmc model_data samples As As_av A u A_df edgelist_df")



#u20_mcmc = pkl_load(u20_base_path / "0_model23_se_sr_13.pkl")
#u20_model_data = pkl_load(
#    u20_base_path / "0_model23_se_sr_13_model_data.pkl")
#
#u20_samples = u20_mcmc['samples']
#u20_As = mv.Z2A(u20_samples['z'])
#u20_As_av = np.mean(u20_As, axis=0)
#u20_A = mv.flat2matrix(u20_As_av, 236)
#
#u_u20 = gbf.model23_matrix2u(u20_A, u20_model_data)
#
#u_u20_control = gbf.model23_results2edge_list(
#    Path("../results/se_sr_low_prior_1_uniform_all_20k/"),
#    "0_model23_se_sr_13")
## -
#
#
#wt_results = get_results(Path(wt_path))
#
#mock_results = get_results(Path(mock_path))
#vif_results = get_results(Path(vif_path))
#
#mock_lp1_uniform_results = get_results(Path("../results/se_sr_low_prior_1_uniform_mock_20k/"))
#
#wt_results.As[0, :]
#
#wt_flat2matrix = jax.jit(Partial(mv.flat2matrix, n=wt_results.model_data['N']))
#
#co_structure_reference = gbf.get_pdb_ppi_predict_cocomplex_reference()
#
## +
#cullin_reindexer = gbf.get_cullin_reindexer()
#def results_frame2u(results, frame_index, from_average = False):
#    u = gbf.UndirectedEdgeList()
#    N = results.model_data['N']
#    M = results.model_data['M']
#    a = np.zeros(M, dtype="U14")
#    b = np.zeros(M, dtype="U14")
#    w = np.zeros(M, dtype=np.float32)
#    node_idx2name = results.model_data['node_idx2name']
#    As = np.array(results.As)
#    k = 0
#    for i in range(N):
#        for j in range(i+1, N):
#            a[k] = node_idx2name[i]
#            b[k] = node_idx2name[j]
#            if isinstance(frame_index, int):
#                w[k] = As[frame_index, k]
#            elif frame_index == "all":
#                w[k] = np.mean(As[:, k])
#            else:
#                raise ValueError
#            k += 1
#    #print("done")
#    u.update_from_df(pd.DataFrame({"auid": a, "buid": b, "w": w}), 
#                     multi_edge_value_merge_strategy = "max",
#                    edge_value_colname = "w")
#    u.reindex(cullin_reindexer, enforce_coverage = False)
#    return u
#
#def assessment_of_scoring(results, uref, every = 1):
#    nframes, nedges = results.As.shape
#    aucs = []
#    for frame_index in range(nframes):
#        if frame_index % every == 0:
#            u_pred = results_frame2u(results, frame_index)
#            benchmark_results = gbf.do_benchmark(pred = u_pred, ref = uref)
#            aucs.append(benchmark_results.auc)
#            print(frame_index)
#    return aucs
#    
#    
## -
#
#def get_u_from_results_at_position(results, position_idx):
#    a = []
#    b = []
#    w = []
#    N = results.model_data['N']
#    M = results.model_data['M']
#    idmap = results.model_data['node_idx2name']
#    samples = results.samples
#    k = 0
#    for i in range(N):
#        for j in range(i+1, N):
#            a.append(idmap[i])
#            b.append(idmap[j])
#            w.append(float(results.As[position_idx, k]))
#            k += 1
#    u = gbf.UndirectedEdgeList()
#    df = pd.DataFrame({"auid": a, "buid": b, "w": np.array(w)})
#    reindexer = gbf.get_cullin_reindexer()
#    u.update_from_df(df, edge_value_colname="w", multi_edge_value_merge_strategy="max")
#    u.reindex(reindexer, enforce_coverage = False)
#    return u
#            
#
#
#mock_lp1_uniform_results.As.shape
#
## Assessment of scoring
#wt_assessment_of_scoring = assessment_of_scoring(wt_results, co_structure_reference, 
#                                                 every=1000)
#
#scores = [float(wt_results.mcmc['extra_fields']['potential_energy'][k]) for k in range(0, 20_000, 1000)]
#
#plt.plot(wt_assessment_of_scoring, scores, 'k.')
#plt.xlim(0, 1)
#plt.xlabel("Co-structure accuracy (AUC)")
#plt.ylabel("Score")
#
#wt_results.model_data.keys()
#
## +
#import numpyro
#import numpyro.distributions as dist
#import jax
#
#def model():
#    x = numpyro.sample('x', dist.Normal(0, 1))
#    y = numpyro.sample('y', dist.Normal(2, 3))
#
#model = mv.model23_se_sr
## Initialize the model: this also provides a callable that computes the potential energy
#rng_key = jax.random.PRNGKey(0)
#wt_model_kwargs = {"model_data": wt_results.model_data}
#model_init = numpyro.infer.util.initialize_model(rng_key, model, model_kwargs=wt_model_kwargs)
#
##(init_params, potential_fn, _, model_trace, _) = model_init
## -
#
#numpyro.infer.util.transform_fn(
#    model_init.model_trace,
#    params=get_params_at_frame(wt_results, 0))
#
#help(model_init.postprocess_fn)
#
#numpyro.infer.util.transform_fn(model_trace=model_init.model_trace)
#
#help(numpyro.infer.util.transform_fn)
#
#
#def get_params_at_frame(results, frame_idx):
#    samples = results.samples
#    return {"u": samples['u'][frame_idx],
#            "z": samples['z'][frame_idx, :]
#           }
#
#
#params = numpyro.infer.util.constrain_fn(
#    model, 
#    model_kwargs=wt_model_kwargs, 
#    params = get_params_at_frame(wt_results, 0))
#
#help(numpyro.infer.util.constrain_fn)
#
#get_params_at_frame(wt_results, 0)
#
#r_range = range(0, 20_000, 100)
#scores2 = [float(model_init.potential_fn(get_params_at_frame(wt_results, k))) for k in r_range]
#av_position = {"u": np.mean(wt_results.samples['u']), "z": np.mean(wt_results.samples['z'], axis=0)}
#av_score = float(model_init.potential_fn(av_position))
#
#import copy
#
#
## +
#def decoy_scores(zedge_values, model_init, model_data, ref=co_structure_reference):
#    decoy = copy.deepcopy(u_pred_temp)
#    decoy.edge_values = mv.Z2A(zedge_values)
#    decoy_position = {"u": 0., "z": zedge_values}
#    decoy_score = float(model_init.potential_fn(decoy_position))
#    decoy_auc = gbf.do_benchmark(pred=decoy, ref=ref).auc
#    return decoy_score, decoy_auc
#
#def get_auc(position, model_init, ref=co_structure_reference):
#    score = float(model_init.potential_fn(position))
#    
#    #decoy = copy.deepcopy(u_pred_temp)
#    #decoy.edge_values = mv.Z2A(position['z'])
#    auc = gbf.do_benchmark(pred=decoy, ref=ref).auc
#    return score, auc
#
#
## +
#zedge_values = np.ones(wt_results.model_data['M']) * 0.51
#decoy_all_score, decoy_all_accuracy = decoy_scores(zedge_values, wt_results.model_data)
#
#zedge_values = np.zeros(wt_results.model_data['M'])
#decoy_none_score, decoy_none_accuracy = decoy_scores(zedge_values, wt_results.model_data)
#
#zedge_values = np.array([float(i % 2 == 0) for i in range(wt_results.model_data['M'])], dtype=np.float32) * 0.51
#decoy_half_score, decoy_half_accuracy = decoy_scores(zedge_values, wt_results.model_data)
#
#zedge_values = np.array([float(i % 3 == 0) for i in range(wt_results.model_data['M'])], dtype=np.float32) * 0.51
#decoy_third_score, decoy_third_accuracy = decoy_scores(zedge_values, wt_results.model_data)
#
#zedge_values = np.array([float(i % 4 != 0) for i in range(wt_results.model_data['M'])], dtype=np.float32) * 0.51
#decoy_3_4_score, decoy_3_4_accuracy = decoy_scores(zedge_values, wt_results.model_data)
## -
#
#zedge_values[0:10]
#
#pd.DataFrame({"score": [decoy_all_score, decoy_none_score, decoy_half_score, decoy_third_score, decoy_3_4_score],
#              "auc": [decoy_all_accuracy, decoy_none_accuracy, decoy_half_accuracy, decoy_third_accuracy, decoy_3_4_accuracy]},
#             index=["1", "0", "1/2", "1/3", "3/4"]).sort_values("score")
#
#av_auc = 0.796 # Read from benchmark results
#
#wt_assesment_of_scoring = assessment_of_scoring(wt_results, co_structure_reference, 
#                                                 every=100)
#
#from functools import partial
#
#
#
#plt.plot(wt_assessment_of_scoring, scores2, 'ko', alpha=0.05)
#plt.plot(av_auc, av_score, 'rx', label="average")
##plt.plot(decoy_all_accuracy, decoy_all_score, 'b^', label="Decoy all")
##plt.plot(decoy_none_accuracy, decoy_none_score, 'g^', label="Decoy none") # off plot
##plt.plot(decoy_half_accuracy, decoy_half_score, 'y^', label="Decoy half") # off plot
##plt.plot(decoy_th)
#plt.xlabel("Co-structure accuracy (AUC)")
#plt.ylabel("Score")
#plt.xlim(0, 1)
#plt.ylim(50_000, 70_000)
#plt.savefig("co_structure_accuracy_300.png", dpi=300)
#plt.savefig("co_structure_accuracy_1200.png", dpi=1200)
#
#mock_lp1_uniform_assessment_of_scoring  = assessment_of_scoring(mock_lp1_uniform_results, co_structure_reference, 
#                                                 every=100)
#
#mock_lp1_uniform_assessment_of_scoring
#
#mock_
#
#model = mv.model23_se_sr
## Initialize the model: this also provides a callable that computes the potential energy
#rng_key = jax.random.PRNGKey(0)
#mock_lp1_uniform_model_kwargs = {"model_data": mock_lp1_uniform_results.model_data}
#mock_lp1_uniform_model_init = numpyro.infer.util.initialize_model(
#    rng_key,
#    model,
#    model_kwargs=mock_lp1_uniform_model_kwargs)
#
## +
#r_range = range(0, 20_000, 100)
#mock_lp1_uniform_scores2 = [float(
#    mock_lp1_uniform_model_init.potential_fn(
#    get_params_at_frame(mock_lp1_uniform_results, k))) for k in r_range]
#mock_lp1_uniform_av_position = {
#    "u": np.mean(mock_lp1_uniform_results.samples['u']),
#    "z": np.mean(mock_lp1_uniform_results.samples['z'],
#    axis=0)}
#
#mock_lp1_uniform_av_score = float(mock_lp1_uniform_model_init.potential_fn(mock_lp1_uniform_av_position))
## -
#
#mock_lp1_uniform_av_score
#
#mock_lp1_uniform_av_auc = 0.77 # read from benchmark_pub.tsv
#
## +
#ftemp = partial(decoy_scores,
#                model_init = mock_lp1_uniform_model_init,
#                model_data = mock_lp1_uniform_results.model_data)
#
#def ftemp(z):
#    return decoy_scores(
#        z, model_init = mock_lp1_uniform_model_init,
#        model_data = mock_lp1_uniform_results.model_data)
#
#
#results = mock_lp1_uniform_results
#zedge_values = np.ones(results.model_data['M']) * 0.51
#decoy_all_score, decoy_all_accuracy = ftemp(zedge_values)
#
#zedge_values = np.zeros(results.model_data['M'])
#decoy_none_score, decoy_none_accuracy = ftemp(zedge_values)
#
#zedge_values = np.array([float(i % 2 == 0) for i in range(results.model_data['M'])], dtype=np.float32) * 0.51
#decoy_half_score, decoy_half_accuracy = ftemp(zedge_values)
#
#zedge_values = np.array([float(i % 3 == 0) for i in range(results.model_data['M'])], dtype=np.float32) * 0.51
#decoy_third_score, decoy_third_accuracy = ftemp(zedge_values)
#
#zedge_values = np.array([float(i % 4 != 0) for i in range(results.model_data['M'])], dtype=np.float32) * 0.51
#decoy_3_4_score, decoy_3_4_accuracy = ftemp(zedge_values)
## -
#
#pd.DataFrame({"score": [decoy_all_score, decoy_none_score, decoy_half_score, decoy_third_score, decoy_3_4_score],
#              "auc": [decoy_all_accuracy, decoy_none_accuracy, decoy_half_accuracy, decoy_third_accuracy, decoy_3_4_accuracy]},
#             index=["1", "0", "1/2", "1/3", "3/4"]).sort_values("score")
#
## Best scoring model
## min score
#mock_lp1_min_score = np.min(mock_lp1_uniform_results.mcmc['extra_fields']['potential_energy'])
#mock_lp1_min_score_idx = np.where(
#    mock_lp1_uniform_results.mcmc['extra_fields']['potential_energy'] == mock_lp1_min_score)[0].item()
#
#mock_lp1_min_score_position = get_params_at_frame(mock_lp1_uniform_results,
#                    mock_lp1_min_score_idx)
#
#mock_lp1_uniform_best_model_score, mock_lp1_uniform_best_model_score_auc = get_auc(
#    position = mock_lp1_min_score_position,
#    model_init = mock_lp1_uniform_model_init)
#
#u_best_model = get_u_from_results_at_position(mock_lp1_uniform_results, mock_lp1_min_score_idx)
#
#u_best_model.edge_values
#
#u_best_model_auc = gbf.do_benchmark(u_best_model, co_structure_reference)
#
#u_best_model_auc.auc
#
#mock_lp1_uniform_best_model_score, mock_lp1_uniform_min_score_auc
#
## +
## Create a u that perfectly aligns with the benchmark and evaluate the score
#name2node_idx = {val : key for key, val in mock_lp1_uniform_results.model_data['node_idx2name'].items()}
#node_idx2name = mock_lp1_uniform_results.model_data['node_idx2name']
#N = mock_lp1_uniform_results.model_data['N']
#M = mock_lp1_uniform_results.model_data['M']
#k = 0
#edges = []
#for i in range(N):
#    for j in range(i+1, N):
#        edge_key = [node_idx2name[i], node_idx2name[j]]
#        reindexed_edge_key = [cullin_reindexer[u] for u in edge_key]
#        if frozenset(reindexed_edge_key) in co_structure_reference._edge_dict:
#            edges.append(1.)
#        else:
#            edges.append(0.)
#        k += 1
#        
#
#
#
#perfect_position1 = {"u": 0., "z" : np.array(edges) } # approximate 
#perfect_position1 = {"u": np.median(mock_lp1_uniform_results.samples['u']), "z" : np.array(edges) } 
#perfect_position2 = {"u": -0.5, "z": np.array(edges) }
#perfect_position3 = {"u": 0.5, "z": np.array(edges) }
#
#
## -
#
#pp1_score = float(mock_lp1_uniform_model_init.potential_fn(perfect_position1))
#pp2_score = float(mock_lp1_uniform_model_init.potential_fn(perfect_position2))
#pp3_score = float(mock_lp1_uniform_model_init.potential_fn(perfect_position3))
#
#pp1_score, pp2_score, pp3_score
#
#mv.Z2A(0.51)
#
#
#def A2Z(a):
#    return 0.5 + jax.scipy.special.logit(a) * 1/1000
#
#
## +
##A2Z(0.99999)
## -
#
#np.log(0.51/ (1-0.51))
#
#len(co_structure_reference._edge_dict)
#
#plt.plot(mock_lp1_uniform_assessment_of_scoring, mock_lp1_uniform_scores2, 'ko', alpha=0.05)
#plt.plot(mock_lp1_uniform_av_auc, mock_lp1_uniform_av_score, 'rx', label="average")
#plt.plot(u_best_model_auc.auc, mock_lp1_uniform_best_model_score, 'bo')
##plt.plot(decoy_all_accuracy, decoy_all_score, 'b^', label="Decoy all")
##plt.plot(decoy_none_accuracy, decoy_none_score, 'g^', label="Decoy none") # off plot
##plt.plot(decoy_half_accuracy, decoy_half_score, 'y^', label="Decoy half") # off plot
##plt.plot(decoy_th)
#plt.xlabel("Co-structure accuracy (AUC)")
#plt.ylabel("Score")
#plt.xlim(0.4, 1)
##plt.ylim(50_000, 70_000)
#plt.savefig("mock_lp1_uniform_co_structure_accuracy_300.png", dpi=300)
#plt.savefig("mock_lp1_uniform_co_structure_accuracy_1200.png", dpi=1200)
#
## Create a u that perfectly aligns with the benchmark and evaluate the score
#
#
#co_structure_pred = copy.deepcopy(co_structure_reference)
#co_structure_pred.edge_values = np.array([co_structure_pred._edge_dict[
#  frozenset([co_structure_pred.a_nodes[k], co_structure_pred.b_nodes[k]])]
# for k in range(len(co_structure_pred.a_nodes))])
#
#co_structure_pred.edge_values
#
#gbf.do_benchmark(co_structure_pred, co_structure_reference)
#
#mock_lp1_uniform_av_score
#
#plt.plot(u_best_model_auc.ppr_points, u_best_model_auc.tpr_points)
#plt.xlim(0, 1)
#plt.ylim(0, 1)
#
#plt.hist(mock_lp1_uniform_assessment_of_scoring, bins=20)
#plt.show()
#np.median(mock_lp1_uniform_assessment_of_scoring)
#
#
#
#decoy_all_score
#
#decoy_none_score, decoy_half_score, decoy_all_score
#
#with open("../results/se_sr_low_prior_1_uniform_mock_20k/0_model23_se_sr_hmc_warmup.pkl", "rb") as f:
#    hmc_warmup = pkl.load(f)
#
#
#potential_energy_scores = [float(wt_results.mcmc['extra_fields']['potential_energy'][k]) for k in range(100)]
#
#plt.plot(potential_energy_scores, scores2, 'k.')
#plt.xlabel("Potential energy from mcmc")
#plt.ylabel("Potential energy from model_init")
#
#help(numpyro)
#
## +
## Define a position at which you want to evaluate the potential energy
#position = {'x': 0.5, 'y': 2.5}
#
## Convert the position dictionary to JAX's format using the model trace
#params = numpyro.infer.util.constrain_fn(model_trace)(position)
#
## Calculate the potential energy at the given position
#energy = potential_fn(params)
#print("Potential Energy:", energy)
## -
#
#u_pred_temp = results_frame2u(wt_results, 19_000)
#
#cullin_reindexer
#
#results = gbf.do_benchmark(pred = u_pred_temp, ref = co_structure_reference)
#
#results.auc
#
#plt.plot(results.ppr_points, results.tpr_points)
#
#co_structure_reference
#
#u_pred_temp
#
#help(gbf.do_benchmark)
#
#N = 6
#k = 0
#for i in range(N):
#    for j in range(i+1, N):
#        print(i, j)
#        k += 1
#print(k, math.comb(N, 2))
#
#wt_results.model_data['M']
#
#results_frame2u(wt_results, 0)
#
#wt_results.As[]
#
#plt.plot(wt_results.mcmc['extra_fields']['potential_energy'])
#
#np.array(wt_flat2matrix(wt_results.As[0, :]))
#
#help(mv.flat2matrix)
#
#wt_results.
#
#SKIP = True
#if not SKIP:
#    #u20_mock = Results()
#    #u20_wt = Results()
#    #u20_vif = Results()
#
#    u20_all = Results(
#        mcmc = u20_mcmc,
#        As = u20_As,
#        As_av = u20_As_av,
#        A = u20_A)
#
#av_u20, av_u20_spec_counts = get_results_df_spec_counts(all_u_20)
#
## +
## Frames and counts
#av_wt, av_wt_spec_counts = get_results_df_spec_counts((wt_path))
#
#wt_mcmc = pkl_load(
#    Path(base_paths.wt) / "0_model23_se_sr_13.pkl")
#wt_model_data = pkl_load(
#    Path(base_paths.wt) / "0_model23_se_sr_13_model_data.pkl")
#
#wt_samples = wt_mcmc['samples']
#wt_As = mv.Z2A(wt_samples['z'])
#wt_As_av = np.mean(wt_As, axis=0)
#wt_A = mv.flat2matrix(wt_As_av, wt_model_data['N'])
##wt_u20, wt_u20_spec_counts = get_results(all_u_20)
## -
#
## Graphs of data frames
#Gu20 = get_graph_from_df(av_u20)
#Gwt = get_graph_from_df(av_wt)
#Gwt_02 = get_filter_network(Gwt, 0.2)
#
#plt.hist(u20_As_av, bins=100)
#plt.show()
#
#plot_benchmark(*get_comparisons(u_u20, pdb_direct, pdb_indirect, pdb_cocomplex))
#
#plot_benchmark(*get_comparisons(u_u20_control, pdb_direct, pdb_indirect, pdb_cocomplex))
#
#sel = ["CSN1", "CSN2", "CSN3", "CSN4", "ELOC", "CSN8", "RBX1", "KAT3", "ASB3", "ASB7", "SGT1"]
#heatshow(av_u20_spec_counts.loc[sel, :])
#
#print(Gu20)
#print(Gwt)
#
#print(Gwt_02)
#
#pos = nx.spring_layout(Gwt_02)
#nx.draw(Gwt_02, with_labels="True")
#
## Run clique detection
#cliques = list(nx.find_cliques(Gwt_02))
#
#ts = np.linspace(0, 1, 100)
##ns_wt = get_n_maximal_cliques(Gwt, ts)
#ne_wt = get_nedges(Gwt, ts)
#ne_u20 = get_nedges(Gu20, ts)
#
#plt.plot(ne_wt, label='wt')
#plt.plot(ne_u20, label='u20')
#plt.legend()
#
#av_wt.loc[CSN_selector(av_wt), :]
#
#clique_plot(ts, ns)
#
#clique_plot(ts, np.log10(ns))
#plt.ylabel("log10 N cliques")
#
#ts2 = np.linspace(0.95, 1, 20)
#ns2 = [get_n_from_iterator(nx.find_cliques(get_filter_network(G, t))) for t in ts2]
#
#ne2 = [get_filter_network(G, t).number_of_edges() for t in ts2]
#
#plot_edges(ts2, ne2)
#
#clique_plot(ts2, ns2)
#
#ts3 = np.linspace(0.999, 1.0, 20)
#ns3 = [get_n_from_iterator(nx.find_cliques(get_filter_network(G, t))) for t in ts3]
#
#ne3 = [get_filter_network(G, t).number_of_edges() for t in ts3]
#
#plot_edges(ts3, ne3)
#
#len(list(nx.enumerate_all_cliques(get_filter_network(G, 0.9998)))[::-1])
#
## +
## ns3 = [get_n_from_iterator(nx.enumerate_all_cliques(get_filter_network(G, t))) for t in ts3]
## -
#
#a = ['T22D1', 'APBP2', 'CTR9', 'PACS1']
#b = ['T22D1', 'HMDH', 'CUL3', 'PAF1']
#
#clique_plot(ts3, ns3)
#
#list(nx.enumerate_all_cliques(get_filter_network(G, 0.99989474)))
#
#list(nx.find_cliques(get_filter_network(G, 0.99973684)))
#
## +
#labels, degree = zip(*list(G2.degree()))
#
#plt.bar(x = np.arange(len(degree)), height=degree)
#plt.xticks(np.arange(),b labels=labels)
## -
#
## Print the edges with their weights
#print("Edges with weights:")
#for u, v, weight in G.edges.data('weight'):
#    print(f"Edge: {u} -- {v}, Weight: {weight}")
#
#ts = np.linspace(0, 1, 100)
#neGu = [get_filter_network(Gu20, t).number_of_edges() for t in ts]
#neG = [get_filter_network(G, t).number_of_edges() for t in ts]
#
#plt.plot(ts, neGu, label="low uniform")
#plt.plot(ts, neG, label="WT")
#plt.legend()
#plt.xlabel("threshold")
#plt.ylabel("Number of edges")
#plt.hlines(0.35 * (math.comb(236, 2)), 0, 1, 'r')
#plt.hlines(0.2 * (math.comb(236, 2)), 0, 1, 'k')
#
#av_u20
#
#Gu20_filter8 = get_filter_network(Gu20, 0.8)
#
#
#maximal_cliques = get_clique_list(Gu20_filter8)
#clique_len = np.array([len(k) for k in maximal_cliques])
#
#plt.hist(clique_len, bins=100)
#plt.show()
#
#Gtemp = get_filter_network(Gu20, 0.99)
#maximal_cliques_temp = get_clique_list(Gtemp)
#clique_len_temp = np.array([len(k) for k in maximal_cliques_temp])
#plt.hist(clique_len_temp, bins=100)
#plt.show()
#
#sorted(maximal_cliques_temp, key=(lambda x: len(x)), reverse=True)
#
#some_nodes = ['NRBP2',
#  'T22D2',
#  'T22D1',
#  'ZY11B',
#  'KLD10',
#  'LEO1',
#  'CTR9',
#  'LRC41',
#  'KLDC3',
#  'ZC4H2',
#  'HIF1A',
#  'UBC12',
#  'FEM1B',
#  'MED8']
#sns.heatmap(av_u20_spec_counts.loc[some_nodes, :], xticklabels=4)
#plt.savefig("some_spec_counts.png", dpi=300)
##plt.close()
#
#sns.heatmap(av_u20_spec_counts.loc[["EMD", "BAF"], :], vmax=5)
#
#sns.heatmap(wt_spec_counts.loc[some_nodes, :], xticklabels=4)
#
#sns.heatmap(av)
#
#plt.close()
#
#plt.matshow(u20_A, cmap='plasma')
#
#sns.clustermap(u20_A)
#
#test = gbf.model23_matrix2u(u20_A)
#
#weighted_degree = jnp.sum(u20_A, axis=0).reshape((236, 1))
#
#weighted_degree_matrix = (weighted_degree @ weighted_degree.T) / np.sum(weighted_degree)
#
#plt.matshow(weighted_degree_matrix, cmap='plasma')
#
## +
## Prior probability density of an interaction given the degree
#degree_prior = weighted_degree / 236
#
#one_minus_degree_prior = 1 - degree_prior
## -
#
#plt.hist(np.ravel(one_minus_degree_prior), bins=100)
#plt.show()
#
#mymatshow(one_minus_degree_prior * one_minus_degree_prior.T)
#
#u20_A_corrected = u20_A * (one_minus_degree_prior * one_minus_degree_prior.T)
#
#_N = 12
#plt.figure(figsize=(_N, _N))
#mymatshow(u20_A_corrected)
#
#u20_A_df = pd.DataFrame(u20_A,
#    columns = [u20_model_data['node_idx2name'][k] for k in range(u20_model_data['N'])],
#    index = [u20_model_data['node_idx2name'][k] for k in range(u20_model_data['N'])])
#
#
#
#heatshow(u20_A_df)
#
#heatshow(degree_prior * degree_prior.T)
#
#wt_A_df = pd.DataFrame(wt_A, 
#    index = [wt_model_data['node_idx2name'][k] for k in range(wt_model_data['N'])],
#    columns = [wt_model_data['node_idx2name'][k] for k in range(wt_model_data['N'])])
#
#fig, ax = plt.subplots(1, 3, figsize=(12, 4))
#sns.heatmap(mock_results.A_df, ax=ax[0])
#sns.heatmap(vif_results.A_df, ax=ax[1])
#sns.heatmap(wt_results.A_df, ax=ax[2])
#
#
#
#
#mock_top_nodes, mock_top_edges = get_top_edges_upto_threshold(mock_results.edgelist_df, threshold=0.99982)
#print(len(mock_top_nodes), len(mock_top_edges))
#
#vif_top_nodes, vif_top_edges = get_top_edges_upto_threshold(vif_results.edgelist_df, threshold=0.999828)
#print(len(vif_top_nodes), len(vif_top_edges))
#
#wt_top_nodes, wt_top_edges = get_top_edges_upto_threshold(wt_results.edgelist_df, threshold=0.999828)
#print(len(wt_top_nodes), len(wt_top_edges))
#
#net_vif = pyvis_plot_network(vif_top_edges)
#net_wt  = pyvis_plot_network(wt_top_edges)
#net_mock = pyvis_plot_network(mock_top_edges)
#
#net_mock.show("mock.html")
#
#net_wt.show("wt.html")
#
#net_vif.show("vif.html")
#
## Cross correlation of edges
#import scipy as sp
#
#
## +
## Cluster networks in a chain
#import hdbscan
#
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#import sklearn.datasets as data
## %matplotlib inline
#sns.set_context('poster')
#sns.set_style('white')
#sns.set_color_codes()
#plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
#
## -
#
#moons, _ = data.make_moons(n_samples=50, noise=0.05)
#blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
#test_data = np.vstack([moons, blobs])
#plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)
#
#mock_results.As.shape
#
#help(hdbscan.HDBSCAN)
#
#help(hdbscan.HDBSCAN)
#
#clusterer = hdbscan.HDBSCAN(
#    min_cluster_size=5,
#    min_samples = None,
#    metric = 'euclidean',
#    p = None,
#    alpha = 1.0,
#    algorithm = 'best',
#    leaf_size = 40,
#    allow_single_cluster = False,
#                            
#    gen_min_span_tree=True)
#clusterer.fit(np.array(mock_results.As[10_000:20_000, 500:1_000]))
#
#clusterer.single_linkage_tree_.plot()
#
#clusterer.condensed_tree_.plot()
#
#clusterer.condensed_tree_.
#
#help(clusterer.fit)
#
#clusterer.single_linkage_tree_.plot()
#
## +
## Correlation of Edges to each other
#
## 1. Select the set of top scoring edge
#
## 2. Calculate pairwise matrix between edges
#
## 3. Plot the matrix
## -
#
#help(sns.heatmap)
#
#wt_results.samples['z'].shape
#
#ztest = wt_results.samples['z']
#
#sp.signal.correlate(ztest[:, 0:2], ztest[:, 2:4]).shape
#
#ztest[:, 0:4] @ ztest[:, 0:4].T
#
#cross_correlate(ztest[:, 0:4])
#
#
#def cross_correlate(x):
#    return x.T @ x
#
#
#heatshow(u20_A_corrected)
#plt.close()
#
#(degree_prior * degree_prior.T).shape
#
#plt.plot(weighted_degree, 'k.')
#plt.xlabel("Node index")
#plt.ylabel("Degree")
#
#plt.close()
#
#fig, ax = plt.subplots(figsize=(12, 12))
#sns.heatmap(u20_A_corrected)
#
#u20_A_corrected_df = pd.DataFrame(
#    u20_A_corrected,
#    columns = [u20_model_data['node_idx2name'][k] for k in range(u20_model_data['N'])],
#    index =   [u20_model_data['node_idx2name'][k] for k in range(u20_model_data['N'])])
#
#heatshow(u20_A_corrected_df)
#
#u20_corrected_cluster_grid = sns.clustermap(u20_A_corrected_df, metric='correlation')
#
#u20_A_corrected_edgelist_df = matrix_df_to_edge_list_df(u20_A_corrected_df)
#
#u20_A_df = pd.DataFrame(u20_A,
#                        columns = [u20_model_data['node_idx2name'][k] for k in range(u20_model_data['N'])],
#                       index = [u20_model_data['node_idx2name'][k] for k in range(u20_model_data['N'])])
#
#u20_A_edgelist_df = matrix_df_to_edge_list_df(u20_A_df)
#
#wt_A_edgelist_df =  matrix_df_to_edge_list_df(wt_A_df)
#
#top_node_set, top_edges = get_top_edges_upto_threshold(u20_A_edgelist_df)
#top_node_set = list(set(top_node_set))
#print(len(top_node_set), len(top_edges))
#
#sns.clustermap(u20_A_corrected_df, metric="cosine")#.loc[top_node_set, top_node_set])
#
#sns.heatmap(av_u20_spec_counts.loc[top_node_set, :], vmax=10)
#
#fig, ax = plt.subplots(1, 2)
#ax[0].hist(u20_A_corrected_edgelist_df['w'], bins=100)
#ax[1].hist(u20_A_corrected_edgelist_df['w'], bins=100, range=(0.3, 1))
#plt.ylabel("Frequency")
#plt.tight_layout()
#plt.show()
#
## +
##net.show("example.html")
## -
#
#u20_A_top_node_set, u20_A_top_edges = get_top_edges_upto_threshold(u20_A_edgelist_df, threshold=0.9997)
#print(len(u20_A_top_node_set), len(u20_A_top_edges))
#
#u20_A_corrected_top_node_set, u20_A_corrected_top_edges = get_top_edges_upto_threshold(
#    u20_A_corrected_edgelist_df, threshold=0.5)
#print(len(u20_A_corrected_top_node_set), len(u20_A_corrected_top_edges))
#
#net_corrected = pyvis_plot_network(u20_A_corrected_top_edges)
#
## +
##net_corrected.show("example.html")
## -
#
## ## VIF Network
## - DCA11: known vif interactor. Binds to CUL4B
## - SPAG5: Mitotic spindle component. mTORC, CDK2, RAPTOR
## - CUL4B: E3 ubiquitin ligase
## - PDC6I: Multifuncitons. ESCRT. HIV-1 viral budding.
## - MAGD1: Involved in the apoptotic response after nerve growth factor (NGF) binding in neuronal cells. Inhibits cell cycle progression, and facilitates NGFR-mediated apoptosis. May act as a regulator of the function of DLX family members. May enhance ubiquitin ligase activity of RING-type zinc finger-containing E3 ubiquitin-protein ligases
##
##
## ## EZH2 Network
## - EZH2: Polycomb group (PcG) protein. Catalytic subunit of the PRC2/EED-EZH2 complex.
##     - ABCG2
## - ABCF2: ATP-binding cassette sub-family F member 2
##
## ## CBFB-Beta Network
## Hypothesis - this interaction network occurs at the nuclear envelope.
## - CBFB-RUNX1: Forms the heterodimeric complex core-binding factor (CBF) with CBFB
## - CBFB-RUNX2: Transcription factor
## - CBFB-ANXA1: Plays important roles in the innate immune response as effector of glucocorticoid-mediated responses and regulator of the inflammatory process.
## - CBFB-BAG3: Molecular co-chaperone for HSP70
## - CBFB-AKP8L: Could play a role in constitutive transport element (CTE)-mediated gene expression by association with DHX9. In case of HIV-1 infection, involved in the DHX9-promoted annealing of host tRNA(Lys3) to viral genomic RNA.
## - CBFB-BAF:Non-specific DNA-binding protein. EMD and BAF are cooperative cofactors of HIV-1 infection
## - EMD-BAF!: EMD and BAF are cooperative cofactors of HIV-1 infection. Association of EMD with the viral DNA requires the presence of BAF and viral integrase. The association of viral DNA with chromatin requires the presence of BAF and EMD
## - TOIP1: Required for nuclear membrane integrity. Induces TOR1A and TOR1B
##
## - HPBP1: 
## - CBFB-RF
##
##
##
## ## CUL5 Network
## - CUL5 and substrate receptors ASB1, ASB3, ASB7, ASB13
## - CUL5 and NEDD8
## - CUL5-RB40C: Substrate recognition component cullin
## - CUL5-PCMD2: Elongin BC-CUL5-SOCS-box protein
## - CUL5-RBX2: Probable component of the SCF (SKP1-CUL1-F-box protein) E3 ubiquitin ligase complex 
## - CUL5-EMC1: Endoplasmic reticulmn. Unknown
## - CUL5-UGGG1: ER Quality control
## - CUL5-SOCS4: SCF-like ECS (Elongin BC-CUL2/5-SOCS-box protein) E3 ubiquitin-protein ligase complex
## - CUL5-MYL4: Myosin
## - CUL5-DCNL1: Part of an E3 ubiquitin ligase complex for neddylation. Acts by binding to cullin-RBX1 complexes
## - CUL5-CAND1: Key assembly factor of SCF (SKP1-CUL1-F-box protein) E3 ubiquitin ligase complex
## - CUL5-SYNE2: Linking network between organelles and the actin cytoskeleton to maintain the subcellular spatial organization. As a component of the LINC
##
##
##
## ## NFKB Network
## - NFKB1: NF-kappa-B is a pleiotropic transcription factor present in almost all cell types.
## - BAG6: (Unkown) Client proteins that cannot be properly delivered to the endoplasmic reticulum are ubiquitinated by RNF126, an E3 ubiquitin-protein ligase associated with BAG6 and are sorted to the proteasome.
## - RD23B: Multiubiquitin chain receptor involved in modulation of proteasomal degradation. Binds to polyubiquitin chains
## - OXSR1: Effector serine/threonine-protein kinase component of the WNK-SPAK/OSR1 kinase cascade
## - Plays a central role in late thymocyte development by controlling both positive and negative T-cell selection.
## - ARGH1: ARHGEF1. 
## - PDS5A: Probable regulator of sister chromatid cohesion in mitosis which may stabilize cohesin complex association with chromatin
## - 2AAB: Interacts with RAD21 a APOB promoter.
##
##
##
##
## ## LLR1 Network
## - LLR1. Substrate recognition subunit of an ECS (Elongin BC-CUL2/5-SOCS-box protein)
## - SGT1. May play a role in ubiquitination and subsequent proteasomal degradation of target proteins.
##
##
##
## ## ARI1 Network
## - ARI1: E3 ubiquitin-protein ligase, which catalyzes ubiquitination of target proteins together with ubiquitin-conjugating enzyme E2 UBE2L3.
## - TFAP4: Transcription factor that activates both viral and cellular genes by binding to the symmetrical DNA sequence 5'-CAGCTG-3'.
##
## ## HSP76 Network
## - RNF114: E3 ubiquitin-protein ligase that promotes the ubiquitination of various substrates
## - NLRC3: Negative regulator of the innate immune response. HSV-1, TOLL, STING
## - STK39: Unkown
##
##
##
##
## RAB21: Small GTPase involved in membrane trafficking control
## - RAB21-CISY: Citrate synthase is found in nearly all cells capable of oxidative metabolism
##
##
##
##
## - RBX2-Nedd8: Part of an E3 ubiquitin ligase complex for neddylation
## - APC11 a cullin anaphase protomting complex
##
##
##
#
#wt_weighted_degree = np.sum(wt_A_df.values, axis=0).reshape((235, 1))
#
#plt.plot(wt_weighted_degree, 'k.')
#
#wt_top_node_set, wt_top_edges = get_top_edges_upto_threshold(wt_A_edgelist_df, threshold=0.9998)
#print(len(wt_top_node_set), len(wt_top_edges))
#
#wt_net = pyvis_plot_network(wt_top_edges)
#
## +
##wt_net.show("wt_example.html")
#
## +
##net_corrected.show("example.html")
## -
#
#net = pyvis_plot_network(u20_A_top_edges)
#
#net.show("net_example.html")
#
## ## Cop9
## - KAT3: No known viral or COP9 ascociation.Catalyzes the irreversible transamination of the L-tryptophan metabolite L-kynurenine to form kynurenic acid (KA)
## - ELOC: Elongation factor
## - RBX1: Ring box protein 1
##
## ## CUL5 - ASB
## - ASB: Substrate-recognition component of a SCF-like ECS (Elongin-Cullin-SOCS-box protein) E3 ubiquitin-protein ligase. 
## - SOCS2: Probable substrate recognition component of a SCF-like ECS (Elongin BC-CUL2/5-SOCS-box protein) E3 ubiquitin-protein.
## - SPSB: Substrate recognition component of a SCF-like ECS (Elongin BC-CUL2/5-SOCS-box protein)
## - PCMD2: May act as a substrate recognition component of an ECS (Elongin BC-CUL5-SOCS-box protein)
## - RB40C: Probable substrate-recognition component of a SCF-like ECS (Elongin-Cullin-SOCS-box protein)
##
## ## RUNX
## - RUNX: transcription factors
## - CBFB: Bind to CBFB
## - PML: Nuclear bodies
##
## ## SYNE2
## - SYNE2: Organlle linking
## - Cell cycle-regulated E3 ubiquitin ligase that controls progression through mitosis and the G1 phase of the cell cycl
#
## +
## Analysis of Edges
## -
#
#heatshow(mock_results.A_df)
#
#plt.hist(u20_A_edgelist_df['w'], bins=100)
#plt.show()
#
#u20_A_edgelist_df[u20_A_edgelist_df['w'] > 0.9998]
#
#plt.hist(u20_A_edgelist_df['w'], bins=100, range=(0.999, 1))
#plt.show()
#
#np.concatenate([np.linspace(0, 0.9), np.linspace(0.9, 1)])
#
#plt.hist(u20_A_edgelist_df['w'], bins=100, range=(0, 0.01))
#plt.show()
#
#x = u20_A_corrected_edgelist_df
#sel = x['a'] == 'ELOB'
#x.sort_values('w', ascending=False).loc[sel, :]
#
#
#u20
#
#cbfb_network = ["BAF", "EMD"]
#ezh2_network = ["EZH2", "ABCF2"]
#nfkb_network = ['NFKB1', "BAG6", "THMS1", "ARHG1", "PDS5A", "RD23B", "OXSR1", "2AAB", "ACAP1", "MOCS1", "KEAP1"]
#vif_network = ['vifprotein', 'DCA11', "CUL4B", "SPAG5", "PDC6I", "MAGD1", "2AAB"]
#sns.heatmap(av_u20_spec_counts.loc[cbfb_network, :], vmax=10)
#
#
#
## ## CBFB-Beta Network
## Hypothesis - this interaction network occurs at the nuclear envelope.
## - CBFB-RUNX1: Forms the heterodimeric complex core-binding factor (CBF) with CBFB
## - CBFB-RUNX2: Transcription factor
## - CBFB-ANXA1: Plays important roles in the innate immune response as effector of glucocorticoid-mediated responses and regulator of the inflammatory process.
## - CBFB-BAG3: Molecular co-chaperone for HSP70
## - CBFB-AKP8L: Could play a role in constitutive transport element (CTE)-mediated gene expression by association with DHX9. In case of HIV-1 infection, involved in the DHX9-promoted annealing of host tRNA(Lys3) to viral genomic RNA.
## - CBFB-BAF:Non-specific DNA-binding protein. EMD and BAF are cooperative cofactors of HIV-1 infection
## - EMD-BAF!: EMD and BAF are cooperative cofactors of HIV-1 infection. Association of EMD with the viral DNA requires the presence of BAF and viral integrase. The association of viral DNA with chromatin requires the presence of BAF and EMD
## - TOIP1: Required for nuclear membrane integrity. Induces TOR1A and TOR1B
##
## - HPBP1: 
## - CBFB-RF
##
##
##
## ## CUL5 Network
## - CUL5 and substrate receptors ASB1, ASB3, ASB7, ASB13
## - CUL5 and NEDD8
## - CUL5-RB40C: Substrate recognition component cullin
## - CUL5-PCMD2: Elongin BC-CUL5-SOCS-box protein
## - CUL5-RBX2: Probable component of the SCF (SKP1-CUL1-F-box protein) E3 ubiquitin ligase complex 
## - CUL5-EMC1: Endoplasmic reticulmn. Unknown
## - CUL5-UGGG1: ER Quality control
## - CUL5-SOCS4: SCF-like ECS (Elongin BC-CUL2/5-SOCS-box protein) E3 ubiquitin-protein ligase complex
## - CUL5-MYL4: Myosin
## - CUL5-DCNL1: Part of an E3 ubiquitin ligase complex for neddylation. Acts by binding to cullin-RBX1 complexes
## - CUL5-CAND1: Key assembly factor of SCF (SKP1-CUL1-F-box protein) E3 ubiquitin ligase complex
## - CUL5-SYNE2: Linking network between organelles and the actin cytoskeleton to maintain the subcellular spatial organization. As a component of the LINC
##
## ## EZH2 Network
## - EZH2: Polycomb group (PcG) protein. Catalytic subunit of the PRC2/EED-EZH2 complex.
## - ABCF2: ATP-binding cassette sub-family F member 2
##
##
## ## NFKB Network
## - NFKB1: NF-kappa-B is a pleiotropic transcription factor present in almost all cell types.
## - BAG6: (Unkown) Client proteins that cannot be properly delivered to the endoplasmic reticulum are ubiquitinated by RNF126, an E3 ubiquitin-protein ligase associated with BAG6 and are sorted to the proteasome.
## - RD23B: Multiubiquitin chain receptor involved in modulation of proteasomal degradation. Binds to polyubiquitin chains
## - OXSR1: Effector serine/threonine-protein kinase component of the WNK-SPAK/OSR1 kinase cascade
## - Plays a central role in late thymocyte development by controlling both positive and negative T-cell selection.
## - ARGH1: ARHGEF1. 
## - PDS5A: Probable regulator of sister chromatid cohesion in mitosis which may stabilize cohesin complex association with chromatin
## - 2AAB: Interacts with RAD21 a APOB promoter.
##
##
## ## VIF Network
## - Vif-DCA11
## - Vif-CUL4B: Core component of multiple cullin-RING-based E3 ubiquitin-protein ligase complexes. DCA11 known to interact with CUL4A. CUL4B?
## - DCA11: CUL4B
## - PD6I:
## - DCA11-PDC6I: Multifunctional protein involved in endocytosis, multivesicular body biogenesis. Role in HIV-1.
## - CUL4B-SPAG5.Essential component of the mitotic spindle required for normal chromosome segregation and progression into anaphase.
## - CUL4B-MAGD1: Involved in the apoptotic response after nerve growth factor (NGF) binding in neuronal cells. Inhibits cell cycle progression, and facilitates NGFR-mediated apoptosis. May act as a regulator of the function of DLX family members. May enhance ubiquitin ligase activity of RING-type zinc finger-containing E3 ubiquitin-protein ligases
##
## ## LLR1 Network
## - LLR1. Substrate recognition subunit of an ECS (Elongin BC-CUL2/5-SOCS-box protein)
## - SGT1. May play a role in ubiquitination and subsequent proteasomal degradation of target proteins.
##
##
##
## ## ARI1 Network
## - ARI1: E3 ubiquitin-protein ligase, which catalyzes ubiquitination of target proteins together with ubiquitin-conjugating enzyme E2 UBE2L3.
## - TFAP4: Transcription factor that activates both viral and cellular genes by binding to the symmetrical DNA sequence 5'-CAGCTG-3'.
##
## ## HSP76 Network
## - HSP76-RNF114: E3 ubiquitin-protein ligase that promotes the ubiquitination of various substrates
## - HSP76-NLRC3: Negative regulator of the innate immune response. HSV-1, TOLL, STING
## - HSP76-STK39: Unkown
##
##
##
##
## RAB21: Small GTPase involved in membrane trafficking control
## - RAB21-CISY: Citrate synthase is found in nearly all cells capable of oxidative metabolism
##
##
##
##
## - RBX2-Nedd8: Part of an E3 ubiquitin ligase complex for neddylation
## - APC11 a cullin anaphase protomting complex
##
##
##
#
#cluster_maps = cluster_map_from_clustergrid(
#    u20_A_corrected_df,
#    u20_corrected_cluster_grid,
#    n_clusters=200,
#    criterion="max")
#
#for cluster_id, cluster in cluster_maps['col'].items():
#    if len(cluster) > 1:
#        print(cluster_id, len(cluster))
#
#temp = cluster_maps['col'][94]
#
#for i in temp:
#    print(i)
#
#print(temp)
#sns.heatmap(u20_A_corrected_df.loc[temp, temp], cmap='plasma', vmin=0, vmax=1)
#
#top_node_set
#
#sns.heatmap(av_u20_spec_counts.loc[, :], vmax=10)
#
## +
#import seaborn as sns
#import pandas as pd
#
## Sample data
#data = pd.DataFrame({
#    "A": [1, 5, 3, 4, 2],
#    "B": [9, 4, 2, 4, 5],
#    "C": [6, 8, 7, 5, 6]
#}, index = ['D', 'E', 'F', 'G', 'H'])
#
## Create a clustermap
#clustergrid = sns.clustermap(data, method='ward', metric='euclidean', standard_scale=1)
#
#
## +
## Row and column reordering
#row_order = clustergrid.dendrogram_row.reordered_ind
#col_order = clustergrid.dendrogram_col.reordered_ind
#
#print("Row order:", row_order)
#print("Column order:", col_order)
#
## -
#
#cluster_map_from_clustergrid(clustergrid, 3, "max")
#
## +
#row_clusters, col_clusters = get_cluster_assignments(clustergrid, 3)
#print("Row clusters:", row_clusters)
#print("Column clusters:", col_clusters)
#
#
#cluster_maps = remap_to_cluster_maps(row_clusters, col_clusters, row_order, col_order)
#
#print("Row cluster map:", cluster_maps['row'])
#print("Column cluster map:", cluster_maps['col'])
#
#
## +
#fig, ax = plt.subplots(figsize=(10, 5))
#dendro_gram_results = sp.cluster.hierarchy.dendrogram(
#    cluster_grid.dendrogram_col.linkage,
#    p = 5,
#    ax=ax,
#    truncate_mode='level')
#
#ax.axhline()
#plt.show()
## -
#
#dendro_gram_results.keys()
#
#cluster_grid.dendrogram_col.linkage[234, 3]
#
#temp = [234, 202, 136, 20, 199, 53, 30, 20, 45, 155, 12, 197, 8, 97, 59, 21,
#       122, 145, 125, 133]
#[u20_model_data['node_idx2name'][k] for k in temp]
#
#[u20_model_data['node_idx2name'][k] for k in cluster_grid.data2d.iloc[:, -34:-25].columns]
#
#[u20_model_data['node_idx2name'][k] for k in cluster_grid.data2d.iloc[:, -80:-49].columns]
#
#cluster_grid.data2d.iloc[:, -80:-49].columns
#
#Z_u20 = cluster_grid.dendrogram_col.linkage
#
#clusters = sp.cluster.hierarchy.fcluster(Z_u20, 10, criterion='maxclust' )
#
#clusters
#
#help(sp.cluster.hierarchy.fcluster)
#
#sns.clustermap(u20_A)
#
#plt.hist(u20_A_corrected[np.tril_indices(236, k=-1)], bins=100, range=(0.02, 1))
#plt.show()
#
#null_map = u20_A * degree_prior * degree_prior.T
#
#degree = np.sum(u20_A, axis=0).reshape(236, 1)
#
#mymatshow(degree + degree.T)
#
#mymatshow(null_map * 10_000)
#
#mymatshow(u20_A)
#
#plt.matshow(weighted_degree + weighted_degree.T)
#
#plt.hist(weighted_degree_matrix[np.tril_indices(236, k=-1)], bins=100)
#plt.show()
#
#plt.matshow(u20_A, cmap="plasma")
#
## mean of the posterior $E[p(M|D, I)]$
##
##
##
#
## Prior probability based on seeing the degree
#prior_probability_from_degree = weighted_degree_matrix / np.sum(weighted_degree_matrix)
#
#
#plt.matshow(prior_probability_from_degree)
#
## Prior probability of seeing
#one_minus_prior_probability_from_degree = 1 - prior_probability_from_degree
#
#plt.matshow(one_minus_prior_probability_from_degree, cmap="plasma")
#plt.colorbar(shrink=0.8)
#
#u20_post = u20_A * one_minus_prior_probability_from_degree
#
#plt.matshow(u20_post, cmap='plasma', vmin=0, vmax=1)
#
#plt.matshow(u20_A, cmap='plasma')
#
#plt.hist(u20_post[np.tril_indices(236, k=-1)], bins=100)
#plt.show()
#
#plt.matshow(prior_probability_from_degree)
#
#plot_benchmark(*get_comparisons(
#    u_u20, pdb_direct, pdb_indirect, pdb_cocomplex))
#
#u_u20_A_corrected = gbf.model23_matrix2u(
#    u20_A_corrected, u20_model_data)
#plot_benchmark(*get_comparisons(
#    u_u20_A_corrected, pdb_direct, pdb_indirect, pdb_cocomplex))
#
#x = u_u20_A_corrected
#u_u20_A_corrected_df = pd.DataFrame({'auid': x.a_nodes, 'buid': x.b_nodes, 'w': x.edge_values})
#add_names_to_df(u_u20_A_corrected_df, uid2name)
#
#u_u20_A_corrected_df.sort_values('w', ascending=False).iloc[0: 50]
#
#help(sns.clustermap)
#
#u20_A_minus_weighted_degree_norm = u20_A - (weighted_degree_matrix / np.max(weighted_degree_matrix))
#
#plt.matshow(u20_A_minus_weighted_degree_norm, cmap="plasma")
#plt.colorbar(shrink=0.8)
#
#
#
#u20_scaled_transformed = min_max_scale_to_01(u20_A_minus_weighted_degree_norm)
#
#plt.matshow(u20_scaled_transformed, cmap='plasma', vmin=0, vmax=1)
#plt.colorbar(shrink=0.8)
#
#plt.hist(
#    min_max_scale_to_01(u20_A_minus_weighted_degree_norm[np.tril_indices(236, k=-1)]),
#    bins=100)
#plt.show()
#
## +
## Min max scale
#u_u20_scaled_transformed = gbf.model23_matrix2u(
#    u20_scaled_transformed, u20_model_data)
#
#plot_benchmark(*get_comparisons(
#    u_u20_scaled_transformed,
#    pdb_direct, pdb_indirect, pdb_cocomplex))
## -
#
## ?gbf.model23_matrix2u
#
#matrix_ratio = u20_A / weighted_degree_matrix
#AD_HOC_NORM_CONST = 78.5194
#matrix_ratio = matrix_ratio * AD_HOC_NORM_CONST
#
#plt.matshow(matrix_ratio, cmap='plasma')
#plt.colorbar(shrink=0.8)
#
#plt.hist(matrix_ratio[np.tril_indices_from(matrix_ratio, k=-1)], bins=100)
#plt.show()
#
#plt.hist(matrix_ratio[np.tril_indices_from(matrix_ratio, k=-1)], bins=100, range=(0.1, 1))
#plt.show()
#
#node_names = [u20_model_data['node_idx2name'][i] for i in range(len(matrix_ratio))]
#matrix_ratio_df = pd.DataFrame(matrix_ratio,
#                              columns=node_names,
#                              index=node_names)
#
#indices = np.where(matrix_ratio_df > 0.4)
#
#matrix_ratio_df.iloc[indices[0], indices[1]]
#
## +
#reindexer = gbf.get_cullin_reindexer()
#
#matrix_ratio_predictor_df = dataframe_from_matrix(matrix_ratio,
#                                                 node_names, name2uid)
## -
#
#
#
#u = gbf.UndirectedEdgeList()
#u.update_from_df(matrix_ratio_predictor_df, 
#                 a_colname='auid',
#                 b_colname='buid',
#                 edge_value_colname='w',
#                 multi_edge_value_merge_strategy='max')
#
#direct_comparison = gbf.do_benchmark(u, pdb_direct)
#indirect_comparison = gbf.do_benchmark(u, pdb_indirect)
#cocomplex_comparison = gbf.do_benchmark(u, pdb_cocomplex)
#
#plt.plot(cocomplex_comparison.ppr_points, cocomplex_comparison.tpr_points, label="cocomplex")
#plt.plot(direct_comparison.ppr_points, direct_comparison.tpr_points, label="direct")
#plt.plot(indirect_comparison.ppr_points, indirect_comparison.tpr_points, label="indirect")
#
#plt.matshow(weighted_degree_matrix, cmap='plasma')
#
#norm_weighted_degree_matrix = weighted_degree_matrix / np.max(weighted_degree_matrix)
#
#
#
#plt.matshow(u20_A)
#
#u20_m
#
#u_u20 = gbf.model23_matrix2u(u20_A, u20_model_data)
#
#hash(u_u20)
#
#u20_average_edge.edge_values
#
#u_u20.edge_values
#
#u20_control_direct_comparison = gbf.do_benchmark(u20_average_edge, pdb_direct)
#u20_control_indirect_comparison = gbf.do_benchmark(u20_average_edge, pdb_indirect)
#u20_control_cocomplex_comparison = gbf.do_benchmark(u20_average_edge, pdb_cocomplex)
#
#u20A_minus_weighted_degree = u20_A - norm_weighted_degree_matrix
#
#plt.matshow(u20A_minus_weighted_degree, cmap='plasma')
#plt.colorbar(shrink=0.8)
#
#u20_predictor_df = dataframe_from_matrix(u20_A, node_names, name2uid)
#
#u20_average_edge = gbf.model23_results2edge_list(
#    Path("../results/se_sr_low_prior_1_uniform_all_20k/"),
#    "0_model23_se_sr_13")
#
# 
#
#u20_average_edge
#
#plot_benchmark(u20_control_cocomplex_comparison,
#               u20_control_direct_comparison,
#               u20_control_indirect_comparison)
#
#u_u20 = gbf.UndirectedEdgeList()
#u_u20.update_from_df(
#    u20_predictor_df, 
#    a_colname='auid',
#    b_colname='buid',
#    edge_value_colname='w',
#    multi_edge_value_merge_strategy='max')
#
#u20_direct_comparison = gbf.do_benchmark(u_u20, pdb_direct)
#u20_indirect_comparison = gbf.do_benchmark(u_u20, pdb_indirect)
#u20_cocomplex_comparison = gbf.do_benchmark(u_u20, pdb_cocomplex)
#
#plt.plot(u20_cocomplex_comparison.ppr_points, u20_cocomplex_comparison.tpr_points, label="cocomplex")
#plt.plot(u20_direct_comparison.ppr_points, u20_direct_comparison.tpr_points, label="direct")
#plt.plot(u20_indirect_comparison.ppr_points, u20_indirect_comparison.tpr_points, label="indirect")
#
#plt.hist(u20A_minus_weighted_degree[np.tril_indices(236, k=-1)], bins=100)
#plt.xlabel("Difference")
#plt.show()
#
#u20_model_data['node_idx2name']
#
#np.sum(matrix_ratio > 0.25)
#
#plt.hist(matrix_ratio[np.tril_indices_from(matrix_ratio, k=-1)], bins=100)
#plt.show()
#
#reciprocal_weighted_degree = 1 / weighted_degree
#
#reciprocal_matmul = reciprocal_weighted_degree @ reciprocal_weighted_degree.T
#
#plt.matshow(reciprocal_matmul)
#
#normalized_weighted_degree = weighted_degree / jnp.sum(weighted_degree)
#
#reciprocal_normalized_weighted_degree = 1 / normalized_weighted_degree
#
#plt.plot(weighted_degree, 'k.')
#
#plt.matshow(weighted_degree.T)
#
#Gu20_filter4 = get_filter_network(Gu20, 0.4)
#
#print(Gu20_filter4)
#
#plot_edges(ts, ne)
#
#Av_u20_matrix
#
#av_u20
#
#Av_u20_matrix[0, 5]
#
#Av_u20_matrix = mv.flat2matrix(av_u20['w'].values, 236)
#
#sns.heatmap(Av_u20_matrix, cmap='plasma')
#
#weighted_degree = jnp.sum(Av_u20_matrix, axis=1)
#
#
#
#plt.plot(weighted_degree)
#plt.ylabel("Weighted Degree")
#plt.xlabel("Node ID")

