"""Single pipeline entry point for running NUTS/MCMC sampling."""

from pathlib import Path

import jax
import numpy as np
from numpyro.infer import MCMC, NUTS

from inm.model.data import build_model_data_from_tsv, model23_data_transformer
from inm.model.model23 import MODEL_REGISTRY


def run(cfg: dict, output_dir: "Path | str") -> None:
    """Run NUTS/MCMC sampling and save posterior samples to output_dir/samples.npz."""
    output_dir = Path(output_dir)

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
    model_data = model23_data_transformer(data_dict)

    model_cfg = cfg["model"]

    rng_key = jax.random.PRNGKey(model_cfg.get("seed", 0))

    mcmc = MCMC(
        NUTS(model_fn),
        num_warmup=model_cfg.get("num_warmup", 500),
        num_samples=model_cfg.get("num_samples", 1000),
        num_chains=model_cfg.get("num_chains", 1),
    )
    mcmc.run(rng_key, model_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_dir / "samples.npz"), **mcmc.get_samples(group_by_chain=True)
    )
