import jax
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tree_util import Partial
import numpy as np
import numpyro
import numpyro.distributions as dist
from collections import namedtuple
from functools import partial
from itertools import combinations
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_value,
    init_to_uniform,
    )
from numpyro.contrib.funsor import config_enumerate
from numpyro.util import format_shapes
import click
from pathlib import Path
import pandas as pd
import pickle as pkl
import time
import math
import sys
import xarray as xr
import logging
import json
from typing import Any, NamedTuple
import pdb

import _model_variations as mv
import generate_benchmark_figures as gbf


def pkl_load(x):
    with open(x, "rb") as f:
        return pkl.load(f)

Results = namedtuple("Results",
    "path mcmc model_data samples As As_av A u A_df edgelist_df hmc_warmup")

def guard_path(path):
    if isinstance(path, str):
        return Path(path)
    elif isinstance(path, Path):
        return path
    else:
        raise ValueError

def get_results(path, rseed=13, mname="0_model23_se_sr"):
    path = guard_path(path)

    mcmc = pkl_load(path / f"{mname}_{rseed}.pkl")
    model_data = pkl_load(path / f"{mname}_{rseed}_model_data.pkl")
    hmc_warmup = pkl_load(path / f"{mname}_hmc_warmup.pkl")

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
      hmc_warmup = hmc_warmup,
    )

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

def concatenate_samples(samples1, samples2):
    return {
        k: np.concatenate(
        [samples1[k][np.newaxis, ...], 
         samples2[k][np.newaxis, ...]], axis=0) for k in samples1}

def get_rhat_results(samples):
    return {k : numpyro.diagnostics.gelman_rubin(samples[k]) for k in samples}
             



_e1 = "../results/se_sr_low_prior_1_uniform_mock_20k/"
_e2 = "../results/se_sr_low_prior_1_uniform_mock_20k_rseed_9"

_e3 = "../results/se_sr_low_prior_2_uniform_mock_20k"
_e4 = "../results/se_sr_low_prior_2_uniform_mock_100k"

