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

# ## Overview
#
# ### 1. IO: Load in the Maximal BSASA Matrix Derived from PDB
#    - Load in BSASA (see 1-10.py)
#    - Calculate Maximal BSASA
#    - Plot histogram of Maximal BSASA
#
#    
# ### 2. Fit a logistic regression model to Maximal BSASA
#    - Fill missing values with 0
#    - Fit a logistic regression to PPI score
#      - Consider Bayesian or non-Bayesian logistic regression
#    - Return the matrix of PPI Scores
#    
# ### 3. Calculate weight matrix
#    - Calculate the sum of paths for successive k length paths
#    - Calculate the normalizing factor (the sum of all weights)
#
# ### 4. Calculate the profile similarities
# - Using the regression analysis at different lengths
#
# ### 5. Do a regression analysis for the profile similarities
#
# ### 6. Define representation, scoring, sampling for IMP model
# - Select some number of nodes
# - Apply composite connectivity
#
# ### 6. Calculate contact frequencies
#
#
# ### 7. Compare to weights
#    

# +
import chain_mechanism as cm
from functools import partial
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax.tree_util import Partial
from types import SimpleNamespace
import xarray as xr
import math
import seaborn as sns

# Custom modules
import _model_variations as mv

# +
# 0: Global Variables

# 0.1 plotting paramters
# Style
sty = SimpleNamespace()
sty.alpha = 0.7
sty.bins = 300

# 0.2 Flags
CALCULATE_MAX_PPI_SCORE_MATRIX = True


# +
#1 : Load in maximal BSASA
bsasa_all = pd.read_csv("../significant_cifs/BSASA_ALL_reference.csv")
bsasa_gt_50 = bsasa_all.loc[bsasa_all['bsasa_lst'] > 3, :]


#1 : 1.2
def plot1(bsasa_df, col="bsasa_lst", sty=sty):
    
    fig, ax = plt.subplots(3)
    plt.suptitle("PDB Pairwise Buried Solvent Accesible Surface Area")
    ax[0].hist(bsasa_df[col].values, range=(0, 1000), **vars(sty))
    ax[1].hist(bsasa_df[col].values, range=(0, 5000), **vars(sty))
    ax[1].set_ylabel("Frequency")
    ax[2].hist(bsasa_df[col].values, range=(0, 10000), **vars(sty))
    plt.xlabel("Square Angstrom")
    plt.tight_layout()
    plt.show()
    
plot1(bsasa_gt_50)
# -

# remove self protein interactions and nans
sel = (bsasa_all['Prey1'] != bsasa_all['Prey2']) & (pd.notna(bsasa_all['Prey1'])) & (pd.notna(bsasa_all['Prey2']))
sel = sel & (bsasa_all['bsasa_lst'] > 3)
bsasa_filter = bsasa_all[sel]

plot1(bsasa_filter)


# +
# 2: Fit a logistic regression model to maximal BSASA

# +
def logistic(x, l=1, x0=0, k=1):
    """
    x: horizontal value
    l: supermum
    x0: midpoint
    k : logistic growth rate
    """
    return l / (1 + jnp.exp(-k*(x - x0)))

ppi_score = Partial(logistic, x0=400, k=0.06, l=1)
grad_ppi_score = jax.vmap(jax.grad(ppi_score))

w = 1000
xtemp = np.arange(0, w, 0.1)
ytemp = ppi_score(xtemp)

weights = np.ones(len(bsasa_filter)) * 0.00015
plt.hist(bsasa_filter['bsasa_lst'], range=(0, 1000), weights=weights, **vars(sty),
        label="scaled frequency")
plt.plot(xtemp, ytemp, label="PPI Score", alpha=0.7)
plt.plot(xtemp, np.array(grad_ppi_score(xtemp)) * 60, label="gradient", alpha=0.3)

plt.vlines(500, 0, 1, 'r', label="500 square angstrom cut off", alpha=0.7)
plt.xlabel("Square Angstroms")
plt.legend()
plt.ylabel("PPI Score")
# -

bsasa_filter.loc[:, "ppi_score"] = ppi_score(bsasa_filter["bsasa_lst"].values)

# +
col = "ppi_score"

fig, ax = plt.subplots(3)
plt.suptitle("Implications of PPI Score")
ax[0].set_title("Low scoring")
ax[0].hist(bsasa_filter[col].values, range=(0, 0.01), bins=100)
ax[1].set_title("High scoring")
ax[1].hist(bsasa_filter[col].values, range=(0.999, 1), bins=100)
ax[2].set_title("Intermediate scoring")
ax[2].hist(bsasa_filter[col].values, range=(0.01, 0.999), bins=100)
#ax[2].hist(bsasa_filter[col].values, range=(0, 1), **vars(sty))
plt.xlabel("PPI Score")
ax[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()

# +
# Number of PPIs in Reference
# -

bsasa_filter

# +
# 2.2 Get the Weight Matrix - max ppi_score matrix

if CALCULATE_MAX_PPI_SCORE_MATRIX:
    prey_set = set(bsasa_filter['Prey1'].values).union(set(bsasa_filter['Prey2'].values))
    prey_arr = np.array(list(prey_set))
    # 1134 unique prey are supported by the reference
    n_prey = len(prey_set)
    max_ppi_score_matrix = np.zeros((n_prey, n_prey))
    max_ppi_score_matrix = xr.DataArray(ppi_score_matrix, coords={"preyu_uid": prey_arr, "preyv_uid": prey_arr})
    for i, r in bsasa_filter.iterrows():
        p1 = r['Prey1']
        p2 = r['Prey2']
        val1 = max_ppi_score_matrix.sel(preyu_uid=p1, preyv_uid=p2).item()
        ppi_score = r['ppi_score']
        if ppi_score > val:
            max_ppi_score_matrix.loc[p1, p2] = ppi_score

# +
# Place Symmetry
a = max_ppi_score_matrix.values
b = a.T
c = np.where(a > b, a, b)
max_ppi_score_matrix.values = c

plt.matshow(c)
plt.colorbar()
# -

fig, ax = plt.subplots(1)
max_ppi_score_matrix.plot(cmap="binary", ax=ax)
plt.title("PDB Derived Maximal PPI Score")
ax.set_xticklabels("")
ax.set_yticklabels("")
plt.show()

# +
tril_indices = np.tril_indices(len(max_ppi_score_matrix), k=-1)
max_ppi_score_values = max_ppi_score_matrix.values[tril_indices]
n_low_scoring = np.sum(max_ppi_score_values < 0.01)
n_high_scoring = np.sum(max_ppi_score_values >=0.999)
n_intermediate = np.sum((max_ppi_score_values >= 0.01) & (max_ppi_score_values < 0.999))
n_possible = math.comb(len(prey_set), 2)

y = [n_low_scoring, n_high_scoring, n_intermediate, n_possible]
x = [0, 1, 2, 3]

plt.bar(x, y, width=0.2, color='k', alpha=0.5)
plt.bar_label(bar)
plt.xticks(ticks=x, labels=labels)
plt.show()

# +
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
max_ppi_score_matrix.plot(cmap="binary", add_labels=True, ax=ax[0])
ax[0].set_xticklabels("")
ax[0].set_yticklabels("")
ax[0].set_title("Max PPI Score Matrix")

labels=["Low", "High", "Intermediate", "N possible pairs"]
bar = ax[1].bar(x, y, width=0.2)
#plt.legend()
ax[1].bar_label(bar)
plt.ylabel("N")
plt.xticks(ticks = x, labels=labels)
plt.show()

#ax[1].hist(max_ppi_score_matrix.values[tril_indices], bins=100)
#ax[1].set_ylabel("Frequency")
#ax[1].set_xlabel("Max PPI Score")
plt.tight_layout()
plt.show()


# -

def calculate_maximal_weight_up_to_N(A, N):
    n, m = A.shape
    if not np.alltrue(np.diagonal(A) == 0):
        diag_indices = np.diag_indices(n)
        A[diag_indices] = 0
    
    assert np.alltrue(np.diagonal(A) == 0)
    assert np.alltrue(A >= 0)
    assert np.alltrue(A == A.T)
    assert np.alltrue(A <= 1)
    
    W = np.zeros(A.shape)
    for i in range(N):
        break
        


"""
1. Could look at the shortest path to bait
2. Could look at all the paths
"""

calculate_maximal_weight_up_to_N(max_ppi_score_matrix.values, 2)

# +
n_low_scoring = np.sum(bsasa_filter['ppi_score'] <0.01)
n_high_scoring = np.sum(bsasa_filter['ppi_score'] >=0.999)
n_intermediate = np.sum((bsasa_filter['ppi_score'] >= 0.01) & (bsasa_filter['ppi_score'] < 0.999))


n_possible = math.comb(len(prey_set), 2)
y = [n_low_scoring, n_high_scoring, n_intermediate, n_possible]
x = [0, 1, 2, 3]

labels=["Low", "High", "Intermediate", "N possible pairs"]
bar = plt.bar(x, y, width=0.2)
#plt.legend()
plt.bar_label(bar)
plt.ylabel("N")
plt.xticks(ticks = x, labels=labels)
plt.show()
# -

# Shortest Paths
f = jax.jit(mv.shortest_paths_up_to_23)
A = max_ppi_score_matrix.values
n, _ = A.shape
diag_indices = np.diag_indices(n)
A[diag_indices] = 0
D = f(A)
Dvalues = D[tril_indices]

plt.hist(np.array(Dvalues), bins=100)
plt.show()

plt.hist(np.array(Dvalues), bins=100, range=(0, 15))
plt.xlabel("Shortest Path")
plt.show()



plt.matshow(max_ppi_score_matrix.values)
