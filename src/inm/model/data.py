import logging
import math
from collections import defaultdict

import jax.numpy as jnp
import numpy as np
import pandas as pd

from inm.model.utils import Histogram, matrix2flat


def get_maximal_composite_dict_from_composite_table(
    composite_table: pd.DataFrame,
) -> dict:
    """Return a dict keyed by Prey name with maximal SAINT scores."""
    composite_dict = defaultdict(float)
    for i, row in composite_table.iterrows():
        prey_name = row["Prey"]
        score = row["MSscore"]
        val = composite_dict[prey_name]
        if score > val:
            composite_dict[prey_name] = score
    return composite_dict


def _maximal_composite_dict_to_max_saint_pair_score_table(max_saint_score_dict):
    """Convert a max SAINT score dict to a pairwise score DataFrame."""
    N = len(max_saint_score_dict)
    table = np.zeros((N, N))
    for i, (k1, v1) in enumerate(max_saint_score_dict.items()):
        for j, (k2, v2) in enumerate(max_saint_score_dict.items()):
            table[i, j] = v1 * v2
    columns = list(max_saint_score_dict.keys())
    return pd.DataFrame(table, index=columns, columns=columns)


def _align_saint_max_score_table_2_corr_table(saint_max_score_table, corr):
    """Align the SAINT max score table rows/columns to match the corr table."""
    columns = corr.columns
    return saint_max_score_table.loc[columns, columns]


def get_aligned_max_saint_score_table(
    composite_table: pd.DataFrame, corr: pd.DataFrame
) -> pd.DataFrame:
    """Build a pairwise SAINT max score table aligned to the correlation table index."""
    max_saint_dict = get_maximal_composite_dict_from_composite_table(composite_table)
    saint_max_score_table = _maximal_composite_dict_to_max_saint_pair_score_table(
        max_saint_dict
    )
    saint_max_score_table = _align_saint_max_score_table_2_corr_table(
        saint_max_score_table, corr
    )

    assert np.all(saint_max_score_table.index == corr.index)
    assert np.all(saint_max_score_table.columns == corr.columns)
    return saint_max_score_table


def get_saint_max_edgelist(
    aligned_max_saint_score_table: pd.DataFrame,
) -> "jnp.ndarray":
    """Flatten the aligned SAINT max score table to a 1-D edge list."""
    return matrix2flat(
        jnp.array(aligned_max_saint_score_table.values, dtype=jnp.float32)
    )


def get_apms_pair_score_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """Compute the AP-MS pair score matrix from the correlation matrix."""
    corr = np.clip(corr, 0, 1)
    degree_matrix = _get_degree_matrix_from_corr(corr)
    return corr * (1 - degree_matrix)


def _get_degree_matrix_from_corr(corr):
    """Compute the degree matrix from a correlation matrix."""
    N, _ = corr.shape
    mask = np.ones(corr.shape) - np.identity(N)
    degree = np.sum(corr.values * mask, axis=0).reshape((N, 1)) / N
    degree_matrix = degree * degree.T
    return degree_matrix


def auto_find_binning(dd: dict, bmin: int = 50, bmax: int = 500) -> int:
    """Find the smallest number of histogram bins that avoids infinite log-prob."""
    x = np.arange(-1, 1, 0.001)
    R0 = dd["apms_shuff_corr_flat"]
    for nbins in range(bmin, bmax):
        null_dist = Histogram(R0, bins=nbins)
        y = np.sum(null_dist.log_prob(x))
        if not np.isinf(y):
            return nbins
    raise ValueError(f"Failed to find binning in range {bmin}, {bmax}")


def model23_data_transformer(
    data_dict: dict,
    calculate_composites: bool = True,
) -> dict:
    """Transform input data to be ready for modeling and perform validation checks."""

    def do_checks(dd, r, c):
        assert len(dd["corr"].index) == len(set(dd["corr"].index))
        assert len(dd["corr"].index) > 2
        assert dd["corr"].shape == dd["shuff_corr"].shape
        assert dd["corr"].ndim == 2
        assert r == c

    def update_name2node_idx(dd):
        name2node_idx = {name: i for i, name in enumerate(dd["corr"].index)}
        node_idx2name = {i: name for name, i in name2node_idx.items()}
        dd["name2node_idx"] = name2node_idx
        dd["node_idx2name"] = node_idx2name
        return dd

    def update_maximal_nodes_edges(dd):
        dd = dd | {"N": len(dd["corr"].index)}
        dd["M"] = dd["N"] * (dd["N"] - 1) // 2
        return dd

    def update_composites(dd):
        new_composite_dict_norm_approx = {}
        new_composite_dict_p_is_1 = {}
        name2node_idx = dd["name2node_idx"]
        for cid, c in dd["composite_dict"].items():
            n = len(c["nodes"])
            p = c["t"]
            temp = []
            for name in c["nodes"]:
                temp.append(name2node_idx[name])
            n_pairs = math.comb(len(temp), 2)
            temp_composite = {
                "nodes": temp,
                "t": p,
                "SID": c["SID"],
                "N": len(temp),
                "n_pairs": n_pairs,
                "maximal_shortest_path_to_calculate": min(22, n_pairs),
            }

            if p == 1:
                new_composite_dict_p_is_1[cid] = temp_composite
            elif (p * n >= 5) and (n * (1 - p) >= 5):
                new_composite_dict_norm_approx[cid] = temp_composite
            else:
                logging.warning(
                    f"Excluding Composite {cid}. Invalid Binomial Approximation. n: {n} p {p} np : {n * p}) n(1-p) : {n * (1 - p)}"
                )
        dd["new_composite_dict_norm_approx"] = new_composite_dict_norm_approx
        dd["new_composite_dict_p_is_1"] = new_composite_dict_p_is_1
        return dd

    def update_correlations(dd, update_by_weighted_degree=False):
        assert np.all(dd["corr"].index == dd["shuff_corr"].index)
        if update_by_weighted_degree:
            dd["corr"] = get_apms_pair_score_matrix(dd["corr"])
        dd["apms_corr_flat"] = matrix2flat(
            jnp.array(dd["corr"].values, dtype=jnp.float32)
        )
        dd["apms_shuff_corr_flat"] = matrix2flat(
            jnp.array(dd["shuff_corr"].values, dtype=jnp.float32)
        )
        dd["apms_shuff_corr_all_flat"] = matrix2flat(
            jnp.array(dd["shuff_corr_all"], dtype=jnp.float32)
        )
        assert len(dd["apms_corr_flat"]) == dd["M"]
        return dd

    dd = data_dict.copy()
    r, c = dd["corr"].shape
    do_checks(dd, r, c)
    dd = update_name2node_idx(dd)
    dd = update_maximal_nodes_edges(dd)
    dd["n_composites"] = len(dd["composite_dict"])
    logging.info(f"N composites {dd['n_composites']}")
    if calculate_composites:
        dd = update_composites(dd)
    else:
        dd = dd | {
            "new_composite_dict_p_is_1": None,
            "new_composite_dict_norm_approx": None,
        }
    dd.pop("composite_dict")
    dd = update_correlations(dd)

    if dd["n_null_bins"] == "auto":
        dd["n_null_bins"] = auto_find_binning(dd)

    aligned_max_saint_score_table = get_aligned_max_saint_score_table(
        dd["composite_table"], dd["corr"]
    )
    saint_max_pair_score_edgelist = get_saint_max_edgelist(
        aligned_max_saint_score_table
    )
    dd["saint_max_pair_score_edgelist"] = saint_max_pair_score_edgelist

    return dd


def _build_composite_dict(composite_table: pd.DataFrame, ms_thresholds) -> dict:
    """Build composite_dict with structure {int_id: {'nodes': [...], 't': threshold, 'SID': sid}}."""
    sid_set = set(composite_table["SID"])
    composites = {}
    k = 0
    for threshold in ms_thresholds:
        for sid in sid_set:
            sel = (composite_table["SID"] == sid) & (
                composite_table["MSscore"] >= threshold
            )
            if np.sum(sel) > 0:
                d = composite_table[sel]
                bait_l = list(set(d["Bait"]))
                assert len(bait_l) == 1, bait_l
                prey_l = list(set(d["Prey"]))
                if bait_l[0] in prey_l:
                    prey_l.remove(bait_l[0])
                nodes = bait_l + prey_l
                composites[k] = {"nodes": nodes, "t": threshold, "SID": sid}
                k += 1
    return composites


def build_model_data_from_tsv(
    spec_table_path: str,
    composite_table_path: str,
    corr_path: str,
    shuff_corr_path: str,
    shuff_corr_all_path: str,
    cfg: dict,
) -> dict:
    """Load TSV files and assemble the initial data_dict for model23_data_transformer."""
    spec_table = pd.read_csv(spec_table_path, sep="\t", index_col=0, header=None)
    composite_table = pd.read_csv(composite_table_path, sep="\t")
    composite_dict = _build_composite_dict(composite_table, ms_thresholds=[0.0])
    corr = pd.read_csv(corr_path, sep="\t", index_col=0)
    shuff_corr = pd.read_csv(shuff_corr_path, sep="\t", index_col=0)
    shuff_corr_all = pd.read_csv(shuff_corr_all_path, sep="\t", index_col=0)
    dd = {
        "spec_table": spec_table,
        "composite_table": composite_table,
        "composite_dict": composite_dict,
        "corr": corr,
        "shuff_corr": shuff_corr,
        "shuff_corr_all": shuff_corr_all,
        "alpha": cfg["model"]["alpha"],
        "beta": cfg["model"]["beta"],
        "disconectivity_distance": cfg["model"]["disconectivity_distance"],
        "max_distance": cfg["model"]["max_distance"],
        "n_null_bins": cfg["model"]["n_null_bins"],
    }
    dd.update(cfg["model"].get("priors", {}))
    return dd
