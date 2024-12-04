import pandas as pd
from pathlib import Path
import pickle as pkl
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from undirected_edge_list import UndirectedEdgeList
import _model_variations as mv

import logging

logger = logging.getLogger(__name__)

def pklload(path):
    with open(path, "rb") as f:
        return pkl.load(f)

def find_results_file_path(name, rseed, chain_path):
    hmc_file_path = find_hmc_warmup_samples_file_path(chain_path)
    results_path = hmc_file_path.name.split("_warmup_samples")[0] + ".pkl"
    logger.info(f"results path: {results_path}")
    results_file_path = chain_path / (results_path)
    return results_file_path

def find_hmc_warmup_samples_file_path(chain_path):
    for path in chain_path.iterdir():
        if path.is_file() and ("warmup_samples" in path.name):
            return path
    raise FileNotFoundError("hmc warmup samples file not found")

def find_model_data_file_path(chain_path):
    for path in chain_path.iterdir():
        if path.is_file() and ("model_data" in path.name):
            return path
    raise FileNotFoundError("model data file not found")

def load_cullin_composite_saint_scores():
    df = pd.read_csv("../data/processed/cullin/composite_table.tsv", sep="\t")
    return df

def load_sars2_composite_saint_scores():
    df = pd.read_csv("../data/processed/gordon_sars_cov_2/composite_table.tsv", sep="\t")
    return df

def get_cullin_saint_scores_edgelist():
    df = load_cullin_composite_saint_scores()
    c = UndirectedEdgeList()
    c.update_from_df(df, a_colname="Bait", b_colname="Prey", edge_value_colname = "MSscore",
                     multi_edge_value_merge_strategy = "max")
    # Reindex
    reindexer = get_cullin_reindexer()
    c.reindex(reindexer)
    return c

def get_sars2_reindexer():
    reindex_df = pd.read_csv("../data/processed/cullin/id_map.tsv", sep="\t", names=["PreyGene", "uid"])
    reindexer = {}
    for i, r in reindex_df.iterrows():
        prey_gene = r['PreyGene']
        prey_uid = r['uid']
        assert prey_gene not in reindexer
        reindexer[prey_gene] = prey_uid 
    return reindexer


def get_sars2_saint_scores_edgelist():
    df = load_sars2_composite_saint_scores() 
    c = UndirectedEdgeList()
    c.update_from_df(df, a_colname="Bait", b_colname="Prey", edge_value_colname = "MSscore",
                     multi_edge_value_merge_strategy = "max")
    # Reindex
    reindexer = get_sars2_reindexer()
    #c.reindex(reindexer, enforce_coverage = False)
    return c

def get_cullin_reindexer():
    reindex_df = pd.read_csv("../data/processed/cullin/id_map.tsv", sep="\t", names=["PreyGene", "uid"])
    reindexer = {}
    for i, r in reindex_df.iterrows():
        prey_gene = r['PreyGene']
        prey_uid = r['uid']
        assert prey_gene not in reindexer
        reindexer[prey_gene] = prey_uid 
    return reindexer

def get_huri_reference():
    df = pd.read_csv("../data/processed/references/HuRI_reference.tsv", sep="\t")
    u = UndirectedEdgeList()
    u.update_from_df(df)
    return u 

def get_humap_all_reference():
    df = pd.read_csv("../data/processed/references/humap2_ppis_all.tsv", sep="\t")
    u = UndirectedEdgeList()
    u.update_from_df(df, edge_value_colname="w", multi_edge_value_merge_strategy="max")
    return u

def get_humap_medium_reference():
    df = pd.read_csv("../data/processed/references/humap2_ppis_medium.tsv", sep="\t")
    u = UndirectedEdgeList()
    u.update_from_df(df)
    return u

def get_humap_high_reference() -> UndirectedEdgeList:
    df = pd.read_csv("../data/processed/references/humap2_ppis_high.tsv", sep="\t")
    u = UndirectedEdgeList()
    u.update_from_df(df)
    return u

def get_biogrid_reference():
    df = pd.read_csv("../data/processed/references/biogrid_reference.tsv", sep="\t")
    u = UndirectedEdgeList()
    u.update_from_df(df)
    return u

def get_cullin_run0_predictions():
    path = "../cullin_run0/0_model23_ll_lp_13.pkl"
    data_path = "../cullin_run0/0_model23_ll_lp_13_model_data.pkl"
    with open(path, "rb") as f:
        mcmc = pkl.load(f)
    with open(data_path, "rb") as f:
        dd = pkl.load(f)
    z = mcmc['samples']['z']
    a = jax.nn.sigmoid((z - 0.5) * 1_000)
    N = dd['N']
    Amean = np.mean(a, axis=0)
    Amean = mv.flat2matrix(Amean, N)
    a = []
    b = []
    w = []
    for i in range(N):
        for j in range(0, i):
            assert i != j
            a.append(dd['node_idx2name'][i])
            b.append(dd['node_idx2name'][j])
            w.append(Amean[i, j])

    df = pd.DataFrame({'auid': a, 'buid': b, 'w': w})
    u = UndirectedEdgeList()
    u.update_from_df(df, edge_value_colname='w', multi_edge_value_merge_strategy = "max")
    reindexer = get_cullin_reindexer()
    u.reindex(reindexer, enforce_coverage = False)
    return u 

def get_pdb_ppi_predict_direct_reference(source="tsv"):
    if source == "pkl":
        with open("./direct_benchmark.pkl", "rb") as f:
            ref = pkl.load(f)
            direct_matrix = ref.reference.matrix
            df = xarray_matrix2edge_list_df(direct_matrix)
    elif source == "tsv":
        df = pd.read_csv("../data/processed/references/pdb_ppi_prediction/direct_benchmark.tsv", names=["auid", "buid"], sep="\t")
    else:
        raise NotImplementedError
    u = UndirectedEdgeList()
    u.update_from_df(df)
    #c.update_from_df(df, a_colname="Bait", b_colname="Prey", edge_value_colname = "MSscore",
    #                 multi_edge_value_merge_strategy = "max")
    return u 


def get_pdb_ppi_predict_cocomplex_reference(source="tsv"):
    if source == "pkl":
        with open("./cocomplex_benchmark.pkl", "rb") as f:
            ref = pkl.load(f)
            xar = ref.reference.matrix
            df = xarray_matrix2edge_list_df(xar)
    elif source == "tsv":
        df = pd.read_csv("../data/processed/references/pdb_ppi_prediction/cocomplex_benchmark.tsv", names=["auid", "buid"], sep="\t")
    else:
        raise NotImplementedError
    u = UndirectedEdgeList()
    u.update_from_df(df)
    return u 

def get_intact_all_first_uid_reference():
    df = pd.read_csv("../data/processed/references/intact/intact_all_first_uid.tsv")
    u = UndirectedEdgeList()
    u.update_from_df(df)
    return u 

def get_pre_ppi_af_hc():
    df = pd.read_csv("../data/processed/preppi/preppi.human_af.interactome_LR379.txt", sep="\t")
    u = UndirectedEdgeList() 
    u.update_from_df(df, a_colname="prot1", b_colname="prot2", edge_value_colname="total_score", multi_edge_value_merge_strategy = "max")
    return u

def get_all_intersection_dataset_getter_funcs():
    return {"intact_all_first_uid" : get_intact_all_first_uid_reference,
            "biogrid_all" : get_biogrid_reference,
            "huri" : get_huri_reference,
            "humap2_high" : get_humap_high_reference,
            "humap2_medium" : get_humap_medium_reference,
            "humap2_all": get_humap_all_reference,
            "pdb_ppi_direct" : get_pdb_ppi_predict_direct_reference,
            "pdb_ppi_cocomplex" : get_pdb_ppi_predict_cocomplex_reference,
            "cullin_max_saint" : get_cullin_saint_scores_edgelist,
            "inm_cullin0" : get_cullin_run0_predictions,
            "pre_ppi_af_hc" : get_pre_ppi_af_hc,
#            "sars2_max_saint" : get_sars2_saint_scores_edgelist(),
            }


def get_all_intersection_datasets():
    return {key : val() for key, val in get_all_intersection_dataset_getter_funcs().items()}


def xarray_matrix2edge_list_df(xar):
    a = []
    b = []
    val = []
    N = xar.shape[0]
    ids = xar.preyu.values
    for i in range(N):
        for j in range(0, i):
            value = xar[i, j].item()
            if value > 0:
                a.append(ids[i])
                b.append(ids[j])
                val.append(value)
    return pd.DataFrame({'auid': a, 'buid': b, 'val': val})

def get_dataset_matrix(ds_list, intersection_method):
    """
    ds_list is a list of tuples each a ("name", u: UndirectedEdgeList)
    method: one of 'edge' or 'node'
    """
    if isinstance(ds_list, dict):
        ds_list = [x for x in ds_list.items()]
    if intersection_method == "edge":
        def calculator(a, b):
            return len(a.edge_identity_intersection(b))
    elif intersection_method == "node":
        def calculator(a, b):
            return len(a.node_intersection(b))
    else:
        raise NotImplementedError
    N = len(ds_list)
    results = np.zeros((N, N), dtype = int)
    for i in range(N):
        for j in range(0, i+1):
            a = ds_list[i][1]
            b = ds_list[j][1]
            results[i, j] = calculator(a, b) 
    names = [t[0] for t in ds_list] 
    df = pd.DataFrame(results, index=names, columns = names)
    return df

get_edge_intersection_matrix = partial(get_dataset_matrix, intersection_method = "edge")
get_node_intersection_matrix = partial(get_dataset_matrix, intersection_method = "node")

def get_filter10_pred(model_output_dirpath: Path) -> UndirectedEdgeList:
    df = pd.read_csv(model_output_dirpath / "average_predicted_edge_scores_filter10.tsv", sep="\t")
    u = UndirectedEdgeList()
    u.update_from_df(df, a_colname="auid", b_colname="buid", edge_value_colname="w", multi_edge_value_merge_strategy="max")
    return u

def get_hgscore():
    path = "./hgscore/hgsscore_output.csv"
    df = pd.read_csv(path)
    u = UndirectedEdgeList()
    reindexer = get_cullin_reindexer()
    
def get_minimal_saint_table(model_output_dirpath):
    path = model_output_dirpath / "composite_table.tsv"

def get_pdb_indirect(pdb_direct, pdb_cocomplex):
    return pdb_cocomplex.edge_identity_difference(pdb_direct)

def pklload(x):
    with open(x, "rb") as f:
        return pkl.load(f)

def model23_matrix2u(A, model_data):
    N = model_data['N']
    a = []
    b = []
    w = []
    for i in range(N):
        for j in range(0, i):
            a_name = model_data['node_idx2name'][i]
            b_name = model_data['node_idx2name'][j]
            assert a_name != b_name
            weight = float(A[i, j])
            a.append(a_name)
            b.append(b_name)
            w.append(weight)
    df = pd.DataFrame({'auid': a, 'buid': b, 'w': w})        
    u = UndirectedEdgeList()
    u.update_from_df(df, edge_value_colname="w", multi_edge_value_merge_strategy="max")
    reindexer = get_cullin_reindexer()
    u.reindex(reindexer, enforce_coverage = False)
    return u
    # Calculate the node intersection table for all nodes

    # Calculate the edge intersection table for all edges

