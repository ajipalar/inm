"""Run NUTS/MCMC sampling using the configured model."""

import argparse
import logging
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import yaml
from numpyro.infer import MCMC, NUTS

from inm.model.data import build_model_data_from_tsv, model23_data_transformer
from inm.model.model23 import MODEL_REGISTRY

logger = logging.getLogger(__name__)


def _resolve_composite_bait_nodes(
    composite_dict: dict, valid_nodes: set, prey_names_path: str
) -> dict:
    """Replace 4-char bait gene-name prefixes in composite nodes with UniProt IDs.

    Bait identifiers in composite_dict come from the 4-char truncated Bait column
    (e.g. "CUL5"), while the correlation matrix is indexed by UniProt IDs (e.g.
    "Q93034"). This maps them; unresolvable bait nodes are dropped.
    """
    prey_names = pd.read_csv(prey_names_path, sep="\t")
    gene_prefix_to_uniprot = dict(
        zip(prey_names["GeneName"].str.split("_").str[0], prey_names["UniProt"])
    )

    resolved = {}
    for cid, c in composite_dict.items():
        new_nodes = []
        for node in c["nodes"]:
            if node in valid_nodes:
                new_nodes.append(node)
            elif node in gene_prefix_to_uniprot:
                uniprot = gene_prefix_to_uniprot[node]
                if uniprot in valid_nodes and uniprot not in new_nodes:
                    new_nodes.append(uniprot)
                else:
                    logger.warning(
                        "Bait %r resolved to %r but not in model graph; dropping from composite %d",
                        node,
                        uniprot,
                        cid,
                    )
            else:
                logger.warning(
                    "Bait %r unresolvable; dropping from composite %d", node, cid
                )
        if len(new_nodes) >= 2:
            resolved[cid] = {**c, "nodes": new_nodes}
        else:
            logger.warning(
                "Composite %d has < 2 resolvable nodes after bait remapping; skipping",
                cid,
            )
    return resolved


def main(config_path: str, output_dir: str) -> None:
    config_path = Path(config_path)
    output_dir = Path(output_dir)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {model_name!r}. Valid names: {list(MODEL_REGISTRY)}"
        )
    model_fn = MODEL_REGISTRY[model_name]

    data_cfg = cfg["data"]
    data_dict = build_model_data_from_tsv(
        data_cfg["spec_table"],
        data_cfg["composite_table"],
        data_cfg["corr"],
        data_cfg["shuff_corr"],
        data_cfg["shuff_corr_all"],
        cfg,
    )

    # Remap 4-char bait gene-name prefixes in composite nodes to UniProt IDs
    valid_nodes = set(data_dict["corr"].index)
    data_dict["composite_dict"] = _resolve_composite_bait_nodes(
        data_dict["composite_dict"], valid_nodes, data_cfg["prey_names"]
    )
    logger.info(
        "Composite dict after bait remapping: %d composites",
        len(data_dict["composite_dict"]),
    )

    model_data = model23_data_transformer(data_dict)

    model_cfg = cfg["model"]
    rng_key = jax.random.PRNGKey(model_cfg.get("seed", 0))

    mcmc = MCMC(
        NUTS(model_fn),
        num_warmup=model_cfg.get("num_warmup", 500),
        num_samples=model_cfg.get("num_samples", 1000),
        num_chains=model_cfg.get("num_chains", 1),
    )
    num_warmup = model_cfg.get("num_warmup", 500)

    # Phase 1: warmup with PE collection (NumPyro 0.12.x: collect_warmup on warmup(), not run())
    mcmc.warmup(
        rng_key, model_data, extra_fields=("potential_energy",), collect_warmup=True
    )
    warmup_pe = mcmc.get_extra_fields(group_by_chain=True)[
        "potential_energy"
    ]  # (C, num_warmup)

    # Phase 2: sampling from warmed-up state (adaptation frozen)
    mcmc.run(rng_key, model_data, extra_fields=("potential_energy",))
    post_samples = mcmc.get_samples(group_by_chain=True)  # (C, num_samples, ...)
    sampling_pe = mcmc.get_extra_fields(group_by_chain=True)[
        "potential_energy"
    ]  # (C, num_samples)

    pe_full = np.concatenate(
        [warmup_pe, sampling_pe], axis=1
    )  # (C, num_warmup + num_samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_dir / "samples.npz"), **post_samples)
    np.savez_compressed(
        str(output_dir / "extra_fields.npz"), potential_energy=sampling_pe
    )
    np.savez_compressed(
        str(output_dir / "warmup_extra_fields.npz"),
        potential_energy=pe_full,
        num_warmup=np.array(num_warmup),
    )
    logger.info("Saved posterior samples to %s/samples.npz", output_dir)
    logger.info("Saved extra fields to %s/extra_fields.npz", output_dir)
    logger.info(
        "Saved warmup+sampling energy trace to %s/warmup_extra_fields.npz", output_dir
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run MCMC sampling.")
    parser.add_argument("config_path", help="Path to config.yaml")
    parser.add_argument(
        "output_dir", help="Directory to write samples.npz and extra_fields.npz"
    )
    args = parser.parse_args()
    main(args.config_path, args.output_dir)
