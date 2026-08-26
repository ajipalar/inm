"""Postprocess MCMC samples: convergence test, edge probabilities, figures."""

import argparse
import logging
from pathlib import Path

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import yaml

from inm.model.utils import Z2A
from inm.postprocess.convergence import (
    ess_lower_triangle,
    ess_ranked,
    rhat_lower_triangle,
    rhat_ranked,
)
from inm.postprocess.ground_truth import build_refij_from_tsv
from inm.postprocess.samples import (
    edge_probabilities,
    merge_chains,
    per_sample_degree,
    save_average_edge_scores,
)
from inm.postprocess.sampling_diagnostics import (
    best_potential_energy_vs_sample_count,
    roc_auc_vs_sample_count,
    sample_count_checkpoints,
    split_half_edge_scores,
)

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "figure.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 15,
        "savefig.bbox": "tight",
    }
)

# Fixed color -> meaning bindings (Okabe-Ito), reused across all reference-comparison figures.
COLORS = {
    "direct": "#0072B2",
    "cocomplex": "#E69F00",
    "a": "#0072B2",
    "b": "#E69F00",
    "ab": "#009E73",
    "null": "#999999",
}


def _node_idx2name(corr_path: str) -> dict:
    """Derive node index → UniProt ID from corr.tsv row order."""
    index = pd.read_csv(corr_path, sep="\t", index_col=0).index
    return {i: name for i, name in enumerate(index)}


def main(config_path: str, samples_dir: str, output_dir: str) -> None:
    config_path = Path(config_path)
    samples_dir = Path(samples_dir)
    output_dir = Path(output_dir)
    warmup_extra_path = samples_dir / "warmup_extra_fields.npz"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dpi = cfg.get("analysis", {}).get("figure_dpi", 150)

    # --- Load samples ---
    samples_npz = np.load(samples_dir / "samples.npz")
    z = samples_npz["z"]  # (C, S, N, N)
    all_samples = {k: samples_npz[k] for k in samples_npz.files}  # all parameters
    # z's lower triangle drives the degree/edge-count restraints in
    # model23_n_degree_ (see inm.model.model23); its upper triangle is sampled
    # independently and unconstrained by them, so the lower triangle is the
    # canonical set of edges. rhat_ranked and ess_ranked treat every matrix
    # entry as an independent parameter, so without slicing here they'd also
    # report on (and, for ess_ranked's FFT-based ESS step, redundantly compute
    # over) the unconstrained upper triangle and diagonal too -- doubling
    # peak memory and OOM-killing on a 236x236 x 4-chain x 1000-sample run.
    if "z" in all_samples:
        i, j = np.tril_indices(all_samples["z"].shape[-1], k=-1)
        all_samples["z"] = all_samples["z"][..., i, j]
    potential_energy = np.load(samples_dir / "extra_fields.npz")[
        "potential_energy"
    ]  # (C, S)
    logger.info("Loaded z %s, potential_energy %s", z.shape, potential_energy.shape)

    # --- Node metadata ---
    node_idx2name = _node_idx2name(cfg["data"]["corr"])
    N = len(node_idx2name)
    assert N == z.shape[2], (
        f"corr.tsv has {N} nodes but samples.npz z has shape {z.shape}"
    )

    # --- Edge probabilities ---
    z_merged = merge_chains(z)  # (C*S, N, N)
    mean_aij = edge_probabilities(z_merged)  # (N, N)
    save_average_edge_scores(
        mean_aij, node_idx2name, output_dir / "average_edge_scores.tsv"
    )
    logger.info("Saved average edge scores (%d pairs)", N * (N - 1) // 2)

    # Lower triangle is canonical: it's the half model23_n_degree_'s
    # degree/edge-count restraints actually constrain (see inm.model.model23).
    tril_i, tril_j = np.tril_indices(N, k=-1)
    y_score = mean_aij[tril_i, tril_j]

    # --- Figure 10: ROC curves (direct + cocomplex) ---
    references = {
        "direct": cfg["data"]["direct_reference"],
        "cocomplex": cfg["data"]["cocomplex_reference"],
    }
    ref_flat = {}  # ref_name -> (M,) binary labels on the lower triangle, reused below
    fig, ax = plt.subplots()
    for ref_name, ref_path in references.items():
        refij_mat = build_refij_from_tsv(ref_path, node_idx2name)
        n_pos = int(refij_mat.sum()) // 2
        logger.info("Reference %s: %d positive edges in model graph", ref_name, n_pos)
        y_true = refij_mat[tril_i, tril_j]
        if y_true.sum() == 0:
            logger.warning(
                "Reference %s has no positive edges in model graph; skipping", ref_name
            )
            continue
        ref_flat[ref_name] = y_true
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_score)
        auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color=COLORS[ref_name], label=f"{ref_name} (AUC={auc:.3f})")
        logger.info("ROC %s AUC=%.3f", ref_name, auc)
    ax.plot([0, 1], [0, 1], color=COLORS["null"], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves: PDB interaction references")
    ax.legend()
    fig.savefig(output_dir / "10_roc_curve.png", dpi=dpi)
    plt.close(fig)

    # --- Figure 32: Precision-recall curves (direct + cocomplex) ---
    fig, ax = plt.subplots()
    for ref_name, y_true in ref_flat.items():
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_score)
        ap = sklearn.metrics.average_precision_score(y_true, y_score)
        ax.plot(
            recall, precision, color=COLORS[ref_name], label=f"{ref_name} (AP={ap:.3f})"
        )
        ax.axhline(y_true.mean(), color=COLORS[ref_name], linestyle="--", alpha=0.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves: PDB interaction references")
    ax.legend()
    fig.savefig(output_dir / "32_pr_curve.png", dpi=dpi)
    plt.close(fig)

    # --- Sampling-amount diagnostics: pool per-draw edge scores on the lower triangle ---
    sampling_cfg = cfg.get("analysis", {}).get("sampling_diagnostics", {})
    sample_counts = sample_count_checkpoints(
        z_merged.shape[0], n_points=sampling_cfg.get("n_points", 10)
    )
    pooled_edge_scores = np.array(Z2A(z_merged), dtype=np.float32)[:, tril_i, tril_j]

    # --- Figure 33: ROC AUC vs. amount of sampling, real reference vs. shuffled null ---
    fig, axes = plt.subplots(
        1, len(ref_flat), figsize=(6 * len(ref_flat), 5), squeeze=False
    )
    for ax, (ref_name, y_true) in zip(axes[0], ref_flat.items()):
        result = roc_auc_vs_sample_count(
            pooled_edge_scores,
            y_true,
            sample_counts,
            n_shuffles=sampling_cfg.get("n_shuffles", 5),
            n_bootstraps=sampling_cfg.get("n_bootstraps", 3),
        )
        ax.errorbar(
            result["sample_counts"],
            result["mean_auc"],
            yerr=result["std_auc"],
            fmt="o-",
            capsize=3,
            color=COLORS[ref_name],
            label="AUC",
        )
        ax.errorbar(
            result["sample_counts"],
            result["mean_null_auc"],
            yerr=result["std_null_auc"],
            fmt="o-",
            capsize=3,
            color=COLORS["null"],
            alpha=0.7,
            label="Shuffled null",
        )
        ax.set_xscale("log")
        ax.set_xlabel("N posterior draws pooled")
        ax.set_ylabel("ROC AUC")
        ax.set_title(ref_name)
        ax.legend()
    fig.suptitle("ROC AUC vs. amount of sampling")
    fig.tight_layout()
    fig.savefig(output_dir / "33_roc_auc_vs_sample_count.png", dpi=dpi)
    plt.close(fig)

    # --- Figure 34: split-half ROC consistency check (independent halves vs. full pool) ---
    split = split_half_edge_scores(pooled_edge_scores)
    fig, axes = plt.subplots(
        1, len(ref_flat), figsize=(6 * len(ref_flat), 5), squeeze=False
    )
    for ax, (ref_name, y_true) in zip(axes[0], ref_flat.items()):
        for half_name, pred in split.items():
            auc = sklearn.metrics.roc_auc_score(y_true, pred)
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, pred)
            ax.plot(
                fpr, tpr, color=COLORS[half_name], label=f"{half_name} (AUC={auc:.3f})"
            )
        ax.plot([0, 1], [0, 1], color=COLORS["null"], linestyle="--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(ref_name)
        ax.legend()
    fig.suptitle("Split-half ROC consistency (a, b = disjoint halves; ab = full pool)")
    fig.tight_layout()
    fig.savefig(output_dir / "34_split_half_roc.png", dpi=dpi)
    plt.close(fig)
    del pooled_edge_scores

    # --- Figure 35: best (lowest) potential energy per chain vs. amount of sampling ---
    energy_result = best_potential_energy_vs_sample_count(
        potential_energy, sample_counts
    )
    fig, ax = plt.subplots()
    ax.errorbar(
        energy_result["sample_counts"],
        energy_result["mean_best_energy"],
        yerr=energy_result["std_best_energy"],
        fmt="o-",
        capsize=3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("N samples used (per chain)")
    ax.set_ylabel("Best (minimum) potential energy")
    ax.set_title("Best potential energy per chain vs. amount of sampling")
    fig.savefig(output_dir / "35_best_energy_vs_sample_count.png", dpi=dpi)
    plt.close(fig)

    # --- Animation: representative chain's binary adjacency matrix over the sampling run ---
    animate_cfg = cfg.get("analysis", {}).get("animation", {})
    if animate_cfg.get("enabled", True):
        max_frames = animate_cfg.get("max_frames", 200)
        n_iter_chain0 = z.shape[1]
        stride = max(1, n_iter_chain0 // max_frames)
        frame_indices = np.arange(0, n_iter_chain0, stride)
        if len(frame_indices) < n_iter_chain0:
            logger.info(
                "Animation: using %d of %d frames from chain 0 (stride=%d)",
                len(frame_indices),
                n_iter_chain0,
                stride,
            )
        # Mirror from the lower triangle (see edge_probabilities) rather than
        # plotting Z2A(z) raw: the raw matrix is asymmetric since only the
        # lower triangle is constrained by the degree restraint.
        aij_frames = np.tril(np.array(Z2A(z[0, frame_indices])), k=-1)
        aij_frames = aij_frames + np.transpose(aij_frames, (0, 2, 1))
        binary_frames = (aij_frames > 0.5).astype(int)
        for frame in binary_frames:
            np.fill_diagonal(frame, 0)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.matshow(binary_frames[0], cmap="cividis", vmin=0, vmax=1)
        ax.set_title("Chain 0: sampled binary adjacency matrix (Z2A(z) > 0.5)")

        def _update(i):
            im.set_data(binary_frames[i])
            return (im,)

        ani = matplotlib.animation.FuncAnimation(
            fig,
            _update,
            frames=len(binary_frames),
            interval=animate_cfg.get("interval_ms", 60),
        )
        ani.save(
            output_dir / "36_sampling_animation.gif",
            writer=matplotlib.animation.PillowWriter(fps=animate_cfg.get("fps", 15)),
        )
        plt.close(fig)

    # --- Figure 11: Edge probability histogram ---
    fig, ax = plt.subplots()
    ax.hist(y_score, bins=50)
    ax.set_xlabel("Mean edge probability")
    ax.set_ylabel("Count")
    ax.set_title("Edge probability distribution")
    fig.savefig(output_dir / "11_edge_prob_hist.png", dpi=dpi)
    plt.close(fig)

    # --- Figure 12: Degree distribution of the averaged network (threshold=0.5) ---
    # Off by default (analysis.degree_distribution.enabled) — easily confused with
    # the per-sample posterior degree distributions in Figures 21-28.
    degree_dist_cfg = cfg.get("analysis", {}).get("degree_distribution", {})
    if degree_dist_cfg.get("enabled", False):
        binary_adj = (mean_aij > 0.5).astype(int)
        np.fill_diagonal(binary_adj, 0)
        degree = binary_adj.sum(axis=1)
        fig, ax = plt.subplots()
        max_degree = int(degree.max()) if degree.max() > 0 else 1
        ax.hist(degree, bins=np.arange(max_degree + 2) - 0.5)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Count")
        ax.set_title("Node degree distribution (threshold=0.5)")
        fig.savefig(output_dir / "12_degree_distribution.png", dpi=dpi)
        plt.close(fig)

    # --- R-hat convergence diagnostics ---
    rhat_vals = rhat_lower_triangle(z)
    logger.info(
        "R-hat z: mean=%.4f median=%.4f max=%.4f frac>1.1=%.3f",
        float(rhat_vals.mean()),
        float(np.median(rhat_vals)),
        float(rhat_vals.max()),
        float((rhat_vals > 1.1).mean()),
    )

    # --- Figure 13: R-hat histogram ---
    fig, ax = plt.subplots()
    ax.hist(rhat_vals, bins=50)
    ax.axvline(1.1, color="r", linestyle="--", label="cutoff (1.1)")
    ax.axvline(1.05, color="k", linestyle="--", label="strict (1.05)")
    ax.set_xlabel("R-hat")
    ax.set_ylabel("Edge count")
    ax.set_title("R-hat distribution across edges")
    ax.legend()
    fig.savefig(output_dir / "13_rhat_histogram.png", dpi=dpi)
    plt.close(fig)

    # --- Figure 14: R-hat rank plot (top 10 highest across all parameters) ---
    top_rhat = rhat_ranked(all_samples, top_n=10)
    labels = [label for label, _ in top_rhat]
    values = [val for _, val in top_rhat]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(labels)), values)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(1.1, color="r", linestyle="--", label="cutoff (1.1)")
    ax.axvline(1.05, color="k", linestyle="--", label="strict (1.05)")
    ax.set_xlabel("R-hat")
    ax.set_title("Top 10 highest R-hat values")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "14_rhat_rank_plot.png", dpi=dpi)
    plt.close(fig)

    # --- ESS convergence diagnostics ---
    total_draws = z.shape[0] * z.shape[1]
    ess_vals = ess_lower_triangle(z)
    logger.info(
        "ESS z: mean=%.1f median=%.1f min=%.1f frac<%d(1/10 draws)=%.3f",
        float(ess_vals.mean()),
        float(np.median(ess_vals)),
        float(ess_vals.min()),
        total_draws // 10,
        float((ess_vals < total_draws / 10).mean()),
    )

    # --- Figure 30: ESS histogram ---
    fig, ax = plt.subplots()
    ax.hist(ess_vals, bins=50)
    ax.axvline(
        total_draws / 10, color="r", linestyle="--", label="cutoff (10% of draws)"
    )
    ax.set_xlabel("Effective sample size")
    ax.set_ylabel("Edge count")
    ax.set_title(f"ESS distribution across edges (total draws={total_draws})")
    ax.legend()
    fig.savefig(output_dir / "30_ess_histogram.png", dpi=dpi)
    plt.close(fig)

    # --- Figure 31: ESS rank plot (bottom 10 lowest across all parameters) ---
    bottom_ess = ess_ranked(all_samples, bottom_n=10)
    labels = [label for label, _ in bottom_ess]
    values = [val for _, val in bottom_ess]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(labels)), values)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(
        total_draws / 10, color="r", linestyle="--", label="cutoff (10% of draws)"
    )
    ax.set_xlabel("Effective sample size")
    ax.set_title("Bottom 10 lowest ESS values")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "31_ess_rank_plot.png", dpi=dpi)
    plt.close(fig)

    # --- Figure 15: Potential energy trace per chain ---
    fig, ax = plt.subplots()
    for c in range(potential_energy.shape[0]):
        ax.plot(potential_energy[c], alpha=0.7, label=f"chain {c}")
    ax.set_xlabel("MCMC step")
    ax.set_ylabel("Potential energy")
    ax.set_title("Potential energy trace (sampling only)")
    ax.legend()
    fig.savefig(output_dir / "15_energy_trace.png", dpi=dpi)
    plt.close(fig)

    # --- Figure 16: Energy trace with warmup phase ---
    warmup_npz = np.load(warmup_extra_path)
    pe_with_warmup = warmup_npz["potential_energy"]  # (C, num_warmup + num_samples)
    num_warmup = int(warmup_npz["num_warmup"])
    fig, ax = plt.subplots()
    for c in range(pe_with_warmup.shape[0]):
        ax.plot(pe_with_warmup[c], alpha=0.7, label=f"chain {c}")
    ax.axvline(
        num_warmup,
        color="r",
        linestyle="--",
        label=f"warmup boundary (step {num_warmup})",
    )
    ax.set_xlabel("MCMC step")
    ax.set_ylabel("Potential energy")
    ax.set_title("Potential energy trace (warmup + sampling)")
    ax.legend()
    fig.savefig(output_dir / "16_energy_trace_with_warmup.png", dpi=dpi)
    plt.close(fig)

    # --- Matrix figure helpers ---
    node_names = [node_idx2name[i] for i in range(N)]
    tick_kw = dict(fontsize=max(4, 8 - N // 10))  # shrink labels for larger N

    def _heatmap(ax, mat, title, cmap, vmin, vmax):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(N))
        ax.set_xticklabels(node_names, rotation=90, **tick_kw)
        ax.set_yticks(range(N))
        ax.set_yticklabels(node_names, **tick_kw)
        return im

    # --- Figure 17: Ground truth matrices ---
    ref_items = list(references.items())
    fig, axes = plt.subplots(
        1, len(ref_items), figsize=(5 * len(ref_items), 4 + N * 0.05)
    )
    if len(ref_items) == 1:
        axes = [axes]
    for ax, (ref_name, ref_path) in zip(axes, ref_items):
        mat = build_refij_from_tsv(ref_path, node_idx2name)
        im = _heatmap(ax, mat, f"Ground truth: {ref_name}", "Blues", 0, 1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "17_ground_truth_matrices.png", dpi=dpi)
    plt.close(fig)

    # --- Figure 18: 4 example sampled binary matrices (2×2 grid) ---
    rng = np.random.default_rng(0)
    n_total = z_merged.shape[0]
    sample_indices = rng.choice(n_total, size=min(4, n_total), replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8 + N * 0.05))
    for ax, idx in zip(axes.flat, sample_indices):
        # Mirror from the lower triangle (see edge_probabilities): Z2A(z) raw
        # is asymmetric since only the lower triangle is degree-constrained.
        aij_lower = np.tril(np.array(Z2A(z_merged[idx])), k=-1)
        binary_mat = ((aij_lower + aij_lower.T) > 0.5).astype(int)
        np.fill_diagonal(binary_mat, 0)
        _heatmap(ax, binary_mat, f"Sample {int(idx)}", "Blues", 0, 1)
    for ax in axes.flat[len(sample_indices) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Example sampled binary adjacency matrices (Z2A(z) > 0.5)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(output_dir / "18_example_sampled_matrices.png", dpi=dpi)
    plt.close(fig)

    # --- Figure 19: Average edge probability matrix ---
    fig, ax = plt.subplots(figsize=(6, 5 + N * 0.05))
    im = _heatmap(ax, mean_aij, "Average edge probability", "Blues", 0, 1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "19_average_edge_probability_matrix.png", dpi=dpi)
    plt.close(fig)

    # --- Figure 20: Edge variance matrix (p*(1-p)) ---
    edge_variance = mean_aij * (1 - mean_aij)
    fig, ax = plt.subplots(figsize=(6, 5 + N * 0.05))
    im = _heatmap(ax, edge_variance, "Edge variance (p*(1-p))", "Oranges", 0, 0.25)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "20_edge_variance_matrix.png", dpi=dpi)
    plt.close(fig)

    # --- Figures 21-28: Posterior degree distributions ---
    binary_deg = per_sample_degree(z_merged, weighted=False)  # (S, N) int
    weighted_deg = per_sample_degree(z_merged, weighted=True)  # (S, N) float

    # Human-readable display names: map UniProt ID -> GeneName, stripping _HUMAN
    prey_names_df = pd.read_csv(cfg["data"]["prey_names"], sep="\t")
    uniprot_to_display = {
        u: g.removesuffix("_HUMAN").removeprefix("HUMAN_")
        for u, g in zip(prey_names_df["UniProt"], prey_names_df["GeneName"])
    }
    node_idx2display = {
        i: uniprot_to_display.get(node_idx2name[i], node_idx2name[i]) for i in range(N)
    }

    # Pick low/mid/high nodes by mean binary degree (used for both versions)
    mean_binary_deg = binary_deg.mean(axis=0)
    sorted_nodes = np.argsort(mean_binary_deg)
    pick = {
        "low": int(sorted_nodes[0]),
        "mid": int(sorted_nodes[len(sorted_nodes) // 2]),
        "high": int(sorted_nodes[-1]),
    }
    logger.info(
        "Representative nodes (mean binary degree): low=%s (%.2f), mid=%s (%.2f), high=%s (%.2f)",
        node_idx2name[pick["low"]],
        mean_binary_deg[pick["low"]],
        node_idx2name[pick["mid"]],
        mean_binary_deg[pick["mid"]],
        node_idx2name[pick["high"]],
        mean_binary_deg[pick["high"]],
    )

    # Figure 21: pooled binary degree posterior
    fig, ax = plt.subplots()
    pooled_bin = binary_deg.ravel()
    max_d = int(pooled_bin.max()) if pooled_bin.max() > 0 else 1
    ax.hist(pooled_bin, bins=np.arange(max_d + 2) - 0.5)
    ax.set_xlabel("Binary degree")
    ax.set_ylabel("Count (sample × node)")
    ax.set_title("Pooled posterior binary degree distribution")
    fig.savefig(output_dir / "21_pooled_binary_degree_posterior.png", dpi=dpi)
    plt.close(fig)

    # Figure 22: pooled weighted degree posterior
    fig, ax = plt.subplots()
    ax.hist(weighted_deg.ravel(), bins=50)
    ax.set_xlabel("Weighted degree (Z2A_soft)")
    ax.set_ylabel("Count (sample × node)")
    ax.set_title("Pooled posterior weighted degree distribution")
    fig.savefig(output_dir / "22_pooled_weighted_degree_posterior.png", dpi=dpi)
    plt.close(fig)

    # Figures 23-28: per-node posterior degree distributions
    bin_files = {"low": 23, "mid": 24, "high": 25}
    wgt_files = {"low": 26, "mid": 27, "high": 28}
    for tier, node_idx in pick.items():
        node_name = node_idx2display[node_idx]

        fig, ax = plt.subplots()
        node_bin = binary_deg[:, node_idx]
        max_d = int(node_bin.max()) if node_bin.max() > 0 else 1
        ax.hist(node_bin, bins=np.arange(max_d + 2) - 0.5)
        ax.set_xlabel("Binary degree")
        ax.set_ylabel("Sample count")
        ax.set_title(
            f"Representative {tier} degree node ({node_name}) — Posterior degree distribution (binary)"
        )
        fig.savefig(
            output_dir / f"{bin_files[tier]}_node_{tier}_binary_degree_posterior.png",
            dpi=dpi,
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(weighted_deg[:, node_idx], bins=50)
        ax.set_xlabel("Weighted degree (Z2A_soft)")
        ax.set_ylabel("Sample count")
        ax.set_title(
            f"Representative {tier} degree node ({node_name}) — Posterior degree distribution (weighted)"
        )
        fig.savefig(
            output_dir / f"{wgt_files[tier]}_node_{tier}_weighted_degree_posterior.png",
            dpi=dpi,
        )
        plt.close(fig)

    logger.info("Postprocess complete, outputs in %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Postprocess MCMC samples.")
    parser.add_argument("config_path", help="Path to config.yaml")
    parser.add_argument(
        "samples_dir", help="Directory containing samples.npz and extra_fields.npz"
    )
    parser.add_argument("output_dir", help="Directory to write postprocess outputs")
    args = parser.parse_args()
    main(args.config_path, args.samples_dir, args.output_dir)
