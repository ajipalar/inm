# Builtins
import argparse
import collections
import cycler
import functools
import itertools
import json
import math
import operator
# from builtin
from collections import namedtuple
from functools import partial
from itertools import combinations
from typing import Set, FrozenSet
from types import SimpleNamespace
import pickle as pkl
# third party
from typing import NamedTuple
#import arviz as az
import biotite
import biotite.sequence.io
import flyplot as flt
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import notepad
#import optax
import pathlib
import pandas as pd
import scipy as sp
import sklearn
import seaborn as sns
import xarray as xr
# from third party
from numpyro.diagnostics import hpdi, summary
from numpyro import handlers
from numpyro.contrib.funsor import config_enumerate
from numpyro.distributions import constraints
from numpyro.infer import SVI, TraceEnum_ELBO
from numpyro.ops.indexing import Vindex
from numpyro.infer import MixedHMC, HMC, MCMC
from numpyro.diagnostics import hpdi, summary
from pathlib import Path

