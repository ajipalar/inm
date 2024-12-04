"""
Implementation for Binary undirected edges read from tables

1. Edges are unique.
2. Edge weights are 1, 0 edges are not represented. 
3. Edges cannot be self edges.
4. If any pair is in the input table the edge is represented

Edge ordering
- a > b
u = EdgeList
"""

import click
import pandas as pd
from pathlib import Path
import pickle as pkl
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from undirected_edge_list import UndirectedEdgeList
import _model_variations as mv
import generate_sampling_figures as gsf
import tpr_ppr
import matplotlib.pyplot as plt

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
    c.reindex(reindexer, enforce_coverage = False)
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

#ds_list = [("huri", h), ("humap_med", humap_medium), ("humap_high", humap_high), ("biogrid_all", biogrid_reference),
#          ("pdb_ppi_direct", direct_ref), ("pdb_ppi_cocomplex", cocomplex_ref), ("intact_all", intact_all), ("cullin_saint", c),
#           ("cullin_all_by_all", c_dense)]



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

def benchmark_cullin_max_saint(ds_dict):
    acceptable_references = ["biogrid_all", "intact_all_first_uid", "pdb_ppi_direct", "pdb_ppi_cocomplex", "humap_medium", "humap2_all"]
    results = {} 
    for key, e in ds_dict.items():
        if key in acceptable_references: 
            x = tpr_ppr.PprTprCalculator(pred = ds_dict['cullin_max_saint'], ref = e)
            r = x.crunch()
            results[key] = r
    return results

def benchmark_and_save_humap_all():
    outdir = "../results/generate_benchmark_figures/"
    ds_dict = get_all_intersection_datasets()
    humap2_all = ds_dict['humap2_all']
    acceptable_references = ["biogrid_all", "pdb_ppi_direct", "pdb_ppi_cocomplex", "huri", "pre_ppi_af_hc"]
    results = {} 
    for key, e in ds_dict.items():
        if key in acceptable_references: 
            x = tpr_ppr.PprTprCalculator(pred = humap2_all, ref = e)
            r = x.crunch()
            results[key] = r
    plotter = tpr_ppr.PprTprPlotter()
    for key, r in results.items():
        name = f"humap2_all__{key}" 
        save_path = outdir + name 
        plotter.plot(save_path, name, r) 
        auc.append(r.auc)
        shuff_auc.append(r.shuff_auc)
        delta_auc.append(r.delta_auc)
        names.append(name)
    # Create a SAINT Auc Table
    df = pd.DataFrame({"auc" : auc, "shuff_auc" : shuff_auc, "detla_auc" : delta_auc}, index=names)
    df.to_csv(outdir + "humap2_all_summary_table.tsv", sep="\t")


def cullin_against_pre_ppi():
    a = get_cullin_saint_scores_edgelist()
    b = get_pre_ppi_af_hc()
    x = tpr_ppr.PprTprCalculator(pred = a, ref = b) 
    r = x.crunch()
    outdir = "../results/generate_benchmark_figures/"
    name = f"cullin_saint_max__pre_ppi_af_hc" 
    save_path = outdir + name 
    plotter = tpr_ppr.PprTprPlotter()
    plotter.plot(save_path, name, r)

def cullin_run0_benchmark():
    ds_dict = get_all_intersection_datasets()
    u = ds_dict['inm_cullin0']
    node_i = []
    edge_i = []
    for key in ds_dict.keys():
        v = ds_dict[key]
        node_i.append(len(u.node_intersection(v)))
        edge_i.append(len(u.edge_identity_intersection(v)))
    df = pd.DataFrame({'node' : node_i, 'edge' : edge_i}, index=list(ds_dict.keys()))
    df.T.to_csv("../results/generate_benchmark_figures/inm_cullin0_intersections.tsv", sep="\t")

def cullin_inm_vs_pre_ppi():
    a = get_cullin_run0_predictions()
    b = get_pre_ppi_af_hc()
    x = tpr_ppr.PprTprCalculator(pred = a, ref = b) 
    r = x.crunch()
    outdir = "../results/2024_03_19_cullin_model23_connectivity_and_prior/"
    name = f"cullin_run0_inm__pre_ppi_af_hc" 
    save_path = outdir + name 
    plotter = tpr_ppr.PprTprPlotter()
    plotter.plot(save_path, name, r)

def cullin_run0_inm_vs_many():
    def body(a, b, outdir, name):
        x = tpr_ppr.PprTprCalculator(pred = a, ref = b) 
        r = x.crunch()
        save_path = outdir + name 
        plotter = tpr_ppr.PprTprPlotter()
        plotter.plot(save_path, name, r)
    
    a = get_cullin_run0_predictions()
    b = get_pre_ppi_af_hc()
    outdir = "../results/2024_03_19_cullin_model23_connectivity_and_prior/"
    name = f"cullin_run0_inm__pre_ppi_af_hc" 
    body(a, b, outdir, name)

    b = get_pdb_ppi_predict_direct_reference()
    name = "cullin_run0_inm__pdb_direct"
    body(a, b, outdir, name)

    b = get_pdb_ppi_predict_cocomplex_reference()
    name = "cullin_run0_inm__pdb_cocomplex"
    body(a, b, outdir, name)

    b = get_biogrid_reference()
    name = "cullin_run0_inm__biogrid"
    body(a, b, outdir, name)

def inm_figs(dir_path : Path, fname : str = "0_model23_ll_lp_13"):
    """
    Generate several figures for an integrative network modeling run
    """
    # Load in the data
    fpath = dir_path / (fname + ".pkl")
    data_path = dir_path / (fname + "_model_data.pkl")

    with open(str(fpath), "rb") as f:
        d = pkl.load(f)

    with open(str(data_path), "rb") as f:
        model_data = pkl.load(f)

    # Get the average network
    samples = d['samples']
    ef = d['extra_fields']

    # Plot the average adjacency matrix  
    nsamples, M = samples['z'].shape
    N = model_data['N']

    a = jax.nn.sigmoid((samples['z']-0.5)*1_000)
    mean = np.mean(a, axis=0) 
    var = np.var(a, axis=0) 

    A = np.array(mv.flat2matrix(mean, N))
    V = np.array(mv.flat2matrix(var, N))

    av_df = adjacency2edgelist_df(A, model_data['node_idx2name'])
    av_df = av_df.sort_values('w', ascending=False)

    av_df.to_csv(str(dir_path / "av_A_edgelist.tsv"), sep="\t", index=False)

    u = UndirectedEdgeList()
    u.update_from_df(av_df, a_colname="a", b_colname="b", edge_value_colname="w", multi_edge_value_merge_strategy="max")

    # Reindex
    reindexer = get_cullin_reindexer()
    u.reindex(reindexer, enforce_coverage = False)

    ds_dict_getters = get_all_intersection_dataset_getter_funcs()

    for reference_key in ds_dict_getters.keys():
         ref = ds_dict_getters[reference_key]() 
         if len(ref.node_intersection(u)):
             x = tpr_ppr.PprTprCalculator(pred = u, ref = ref) 
             r = x.crunch()
             name = f"inm__{reference_key}"
             plotter = tpr_ppr.PprTprPlotter()
             plotter.plot(str(dir_path / name), name, r)
             print(f"saved {str(dir_path) + name}") 

    return u

def mapping_is_one2one(mapping):
    for key, val in mapping.items():
        if isinstance(val, str):
            ...
        elif len(val) > 1:
            print(key, val)
            return False
        elif len(val) == 0:
            print(key, val)
            return False
    return True        

def adjacency2edgelist_df(A, node_idx2name): 
    w = []
    a = []
    b = []
    N, _ = A.shape
    for i in range(N):
        for j in range(0, i):
            assert i != j
            w.append(A[i, j])
            a.append(node_idx2name[i])
            b.append(node_idx2name[j])
    return pd.DataFrame({"a" : a, "b" : b, "w": w})
                        
def model23_results2edge_list(modeling_output_dirpath, fbasename, discard_first_n=0):
    """
    1. Path to modeling results
    2. Get an edge score as average across all modeling runs
    """
    fname = f"{fbasename}.pkl"
    with open(modeling_output_dirpath / fname, "rb") as f:
        d = pkl.load(f)
    with open(modeling_output_dirpath / f"{fbasename}_model_data.pkl", "rb") as f:
        model_data = pkl.load(f)
    d = gsf.optional_flatten(d)
    # Plot the average adjacency matrix  
    samples = d['samples']
    nsamples, M = samples['z'].shape
    N = model_data['N']
    z = samples['z']
    z = z[discard_first_n:, :]
    a = jax.nn.sigmoid((z-0.5)*1_000)
    mean = np.mean(a, axis=0) 
    A = mv.flat2matrix(mean, N)
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
    
    

get_edge_intersection_matrix = partial(get_dataset_matrix, intersection_method = "edge")
get_node_intersection_matrix = partial(get_dataset_matrix, intersection_method = "node")

def any_id_id_mapping_strategy():
    # Build a global AnyID dictionary

    # Represent a single unique identifier

    # Map all nodes in terms of AnyId indices

    # 
    ...


def do_benchmark(pred, ref):
    x = tpr_ppr.PprTprCalculator(pred = pred, ref = ref)
    results = x.crunch()
    return results


def compare_and_save():
    x = tpr_ppr.PprTprCalculator(pred = pred, ref = ref)
    results = x.crunch()

    ...

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

def _test_cullin_standard():
    references = dict( 
        humap2_high = get_humap_high_reference(),
        humap2_medium = get_humap_medium_reference(),
        pre_ppi = get_pre_ppi_af_hc(),
        huri = get_huri_reference(),
    )
    pdb_direct = get_pdb_ppi_predict_direct_reference(),
    pdb_cocomplex = get_pdb_ppi_predict_cocomplex_reference(),
    breakpoint()

def to_dirpath(str_or_dirpath):
    if isinstance(str_or_dirpath, str):
        return Path(str_or_dirpath)
    elif isinstance(str_or_dirpath, Path):
        return str_or_dirpath
    else:
        raise ValueError



def humap_bench(bench_output = "../results/humap_bench", topN=None):
    bench_output = to_dirpath(bench_output)
    assert bench_output.is_dir()
    pdb_direct = get_pdb_ppi_predict_direct_reference()
    pdb_cocomplex = get_pdb_ppi_predict_cocomplex_reference()
    pdb_indirect = pdb_cocomplex.edge_identity_difference(pdb_direct)
    references =  {"pdb_direct" : pdb_direct, "pdb_cocomplex" : pdb_cocomplex, "pdb_indirect": pdb_indirect}
    predictions = {"humap_all" : get_humap_all_reference()}

    write_roc_curves_and_table(bench_output, predictions=predictions, references=references)

def write_roc_curves_and_table(model_output_dirpath, predictions, references,
                               pairs_to_plot_on_one_graph =( 
        #("average_edge", "humap2_high"),
        #("average_edge", "huri"),
        ("average_edge", "pdb_direct"),
        ("average_edge", "pdb_cocomplex"),
        ("cullin_max_saint", "pdb_cocomplex")
                                   ), multi_save_suffix=""):
    plotter = tpr_ppr.PprTprPlotter()
    # Summary Table
    ref_name_lst = []
    pred_name_lst = []
    auc = []   
    shuff_auc = []
    n_reference_positives = []
    n_predicted_positives = []

    fig_multi, ax_multi = plt.subplots(1, 1)
    tpr_ppr.set_roc_limits(ax_multi) 
    for pred_name, prediction in predictions.items():
        for ref_name, reference in references.items():
            comparison_name = f"{pred_name}_v_{ref_name}"
            compare_results = do_benchmark(prediction, reference)
            outpath = str(model_output_dirpath /  comparison_name) 
            plotter.plot(outpath, comparison_name, compare_results)
            pred_name_lst.append(pred_name)
            ref_name_lst.append(ref_name)
            auc.append(compare_results.auc)
            shuff_auc.append(compare_results.shuff_auc)
            n_ref_pos = compare_results.n_total_positives
            n_reference_positives.append(n_ref_pos)
            n_pred_pos = compare_results.n_predicted_positives
            n_predicted_positives.append(n_pred_pos)
            if (pred_name, ref_name) in pairs_to_plot_on_one_graph:
                ax_multi.plot(compare_results.ppr_points, compare_results.tpr_points, label=f"{round(compare_results.auc, 3)} {pred_name} N={n_pred_pos} {ref_name} N={n_ref_pos}")
    
    ax_multi.legend()
    ax_multi.set_xlabel("Positive predictive rate")
    ax_multi.set_ylabel("True positive rate")
    multi_savepath = str(model_output_dirpath / ("roc_multi" + multi_save_suffix))
    tpr_ppr.dpi_save(multi_savepath, fig_multi, 300)
    tpr_ppr.dpi_save(multi_savepath, fig_multi, 1200)
    plt.close(fig=fig_multi)


    df = pd.DataFrame({"prediction" : pred_name_lst,
                       "reference" : ref_name_lst,
                       "N reference" : n_reference_positives,
                       "auc" : auc,
                       "shuff_auc" : shuff_auc,}) 
    df.to_csv(str(model_output_dirpath / "benchmark.tsv"), sep="\t")
    df.round(decimals = {"auc" : 3, "shuff_auc" : 3})
    df.to_csv(str(model_output_dirpath / "benchmark_pub.tsv"))

    # Write the model prediction table
    reindexer = get_cullin_reindexer()
    inv_reindexer = {val : key for key, val in reindexer.items()}
    u = predictions['average_edge']
    df = pd.DataFrame({"auid" : u.a_nodes,
                       "buid" : u.b_nodes,
                       "a_gene" : [inv_reindexer[k] for k in u.a_nodes],
                       "b_gene" : [inv_reindexer[k] for k in u.b_nodes],
                       "w" : u.edge_values})
    df = df.sort_values("w", ascending=False)
    df.to_csv(str(model_output_dirpath / "average_predicted_edge_scores.tsv"), sep="\t", index=None)
    plt.close()

def get_indirect_edges():
    pdb_direct = get_pdb_ppi_predict_direct_reference()
    pdb_cocomplex = get_pdb_ppi_predict_cocomplex_reference()
    pdb_indirect = pdb_cocomplex.edge_identity_difference(pdb_direct)
    return pdb_indirect

def get_decoys_from_u(rng_key, u, N_decoys, a_colname="auid", b_colname="buid"):
    indices = jnp.arange(u.nedges)
    random_indices = jax.random.permutation(rng_key, indices)
    random_indices = random_indices[0:N_decoys]
    # build decoy
    #breakpoint()
    a_nodes = np.array(u.a_nodes)[random_indices]
    b_nodes = np.array(u.b_nodes)[random_indices]
    decoy = UndirectedEdgeList()
    df = pd.DataFrame({a_colname : a_nodes, b_colname : b_nodes})
    decoy.update_from_df(df, a_colname = a_colname, b_colname = b_colname)
    return decoy

def n_tp(aij, refij):
    """
    Return the values of the array where both are 1
    Boolean arrays
    """
    return np.sum(aij & refij)

def n_fp(aij, refij):
    return np.sum((aij == 1) & (refij == 0))

def _loop(aij_mat, refij, f):
    N, _ = aij_mat.shape
    output = np.zeros(N)
    for i in range(N):
        output[i] = f(aij_mat[i, :], refij)
    return output

n_tps = partial(_loop, f=n_tp)
n_fps = partial(_loop, f=n_fp)

def n_tps_from_samples(samples, refij):
    aij_mat = mv.Z2A(samples['z'])
    aij_mat = aij_mat > 0.5
    return n_tps(aij_mat, refij)

def n_fps_from_samples(samples, refij):
    aij_mat = mv.Z2A(samples['z'])
    aij_mat = aij_mat > 0.5
    return n_fps(aij_mat, refij)

def pklload(x):
    with open(x, "rb") as f:
        return pkl.load(f)

def concatenate_warmup_samples(model_output_dirpath, fbasename="0_model_23_se_sr_13"):
    p1 = model_output_dirpath / (fbasename + "_warmup_samples.pkl")
    p2 = model_output_dirpath / (fbasename + ".pkl")

    p1 = pklload(p1)
    p2 = pklload(p2)
    
    p2_samples = p2.mcmc['samples']
    p2_ef = p2.mcmc['extra_fields']

    found_keys = []
    for key, val in p1.items():
        # concatenate samples along first axis
        val2 = p2[key]
        result = np.concatenate(val, val2)
        found_keys.append()

def cullin_standard(model_output_dirpath, fbasename, with_humap_as_predictions_at_cullin=True):
    references = dict( 
        humap2_high = get_humap_high_reference(),
        humap2_medium = get_humap_medium_reference(),
        pre_ppi = get_pre_ppi_af_hc(),
        huri = get_huri_reference(),
    )
    pdb_direct = get_pdb_ppi_predict_direct_reference()
    pdb_cocomplex = get_pdb_ppi_predict_cocomplex_reference()
    pdb_indirect = pdb_cocomplex.edge_identity_difference(pdb_direct)
    references = references | {"pdb_direct" : pdb_direct, "pdb_cocomplex" : pdb_cocomplex, "pdb_indirect": pdb_indirect}
    
    predictions = dict(
        average_edge = model23_results2edge_list(model_output_dirpath, fbasename),
        cullin_max_saint = get_cullin_saint_scores_edgelist(),
        average_edge_after2k = model23_results2edge_list(
            model_output_dirpath,
            fbasename,
            discard_first_n = 2000))

    #    Add some decoy references    
    decoy_key_50 = jax.random.PRNGKey(303)
    decoy_key_100 = jax.random.PRNGKey(404)
    decoy_key_500 = jax.random.PRNGKey(504)
    decoy50  = get_decoys_from_u(decoy_key_50,  predictions['average_edge'], 50) 
    decoy100 = get_decoys_from_u(decoy_key_100, predictions['average_edge'], 100)
    decoy500 = get_decoys_from_u(decoy_key_500, predictions['average_edge'], 500)
    references = references | {"decoy50" : decoy50, "decoy100" : decoy100, "decoy500" : decoy500}

    #    hgscore_all = get_hgscore()
    if with_humap_as_predictions_at_cullin:
        # Get humap
        humap_all = get_humap_all_reference()
        # Reduce the map to the represented in the cullin system 
        predicted_nodes = predictions['average_edge'].get_node_list()
        humap_at_predicted_nodes = humap_all.node_select(predicted_nodes)
        del humap_all
        predictions = predictions | {"humap_at_average_edge_scores" : humap_at_predicted_nodes}

    if (model_output_dirpath / "average_predicted_edge_scores_filter10.tsv").is_file():
        temp = get_filter10_pred(model_output_dirpath)
        predictions = predictions | {"average_edge_filter10" : temp}

    write_roc_curves_and_table(model_output_dirpath, references=references, predictions=predictions)
    
    # Calculate the minimal pairwise SaintTable
    minimal_saint_table = get_minimal_saint_table()


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

def main():
    outdir = "../results/generate_benchmark_figures/"
    ds_dict = get_all_intersection_datasets()
    de = get_edge_intersection_matrix(ds_dict)
    dn = get_node_intersection_matrix(ds_dict)
    # Save the matrices
    de.to_csv(outdir + "edge_intersections.tsv", sep="\t")
    dn.to_csv(outdir + "node_intersections.tsv", sep="\t")

    # Do Saint Benchmarking with cullin
    cullin_saint_benchmark_results = benchmark_cullin_max_saint(ds_dict)
    # Save figures
    auc = []
    shuff_auc = []
    delta_auc = []
    names = []
    plotter = tpr_ppr.PprTprPlotter()
    for key, r in cullin_saint_benchmark_results.items():
        name = f"cullin_saint_max__{key}" 
        save_path = outdir + name 
        plotter.plot(save_path, name, r) 
        auc.append(r.auc)
        shuff_auc.append(r.shuff_auc)
        delta_auc.append(r.delta_auc)
        names.append(name)
    # Create a SAINT Auc Table
    df = pd.DataFrame({"auc" : auc, "shuff_auc" : shuff_auc, "detla_auc" : delta_auc}, index=names)
    df.to_csv(outdir + "cullin_saint_max_summary_table.tsv", sep="\t")

@click.command()
@click.option("--i")
@click.option("--fbasename")
def main(i, fbasename):
    _main(i, fbasename)

def _main(i, fbasename):
    logger = logging.getLogger(__name__)
    logging.info("Enter generate_benchmark_figures")
    logging.info(f"Params")
    logging.info(f"    i:{i}")
    logging.info(f"    fbasename:{fbasename}")
    i = Path(i)
    cullin_standard(i, fbasename)

if __name__ == "__main__":
    main()
