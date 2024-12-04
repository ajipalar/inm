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

# +
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pickle as pkl
import xarray as xr
from numpyro.distributions.constraints import Constraint
import pandas as pd
from pathlib import Path
import math
import scipy as sp
# try and fit the data to a model 
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
import jax
from numpyro.contrib.funsor import config_enumerate, infer_discrete
from numpyro.infer import NUTS, MCMC
import jax.numpy as jnp
import arviz as az
import numpyro.distributions as dist
from jax.tree_util import Partial
import sympy as sympy
import jax.scipy as jsp

import inspect
from collections import namedtuple

import blackjax
import blackjax.smc.resampling as resampling

from numpyro.distributions.transforms import Transform
from numpyro.distributions import constraints
from jax.tree_util import tree_flatten, tree_unflatten

# +
# Notebook Globals

LOAD_APMS_CORRELATION = True
LOAD_SHUFFLED_APMS_CORRELATION = True
LOAD_HIV_CORRELATION = True
SKIP_NON_CRITICAL_CELLS = True

APMS_COLOR = 'blue'
GI_COLOR = 'orange'
# -

with open("direct_benchmark.pkl", 'rb') as f:
    direct_benchmark = pkl.load(f)
reference_matrix = direct_benchmark.reference.matrix

# +
# remap and relabel columns

# Read in the ids of both data sets

hiv_emap_path = "../data/hiv_emap/idmapping_2023_07_05.tsv"
hiv_emap_ids = pd.read_csv(hiv_emap_path, sep="\t")

from_ = set(hiv_emap_ids['From'])
to_ = set(hiv_emap_ids['To'])
#hiv_gene2uid = {row['To']: row['From'] for i, row in hiv_emap_ids.iterrows()}

print(hiv_emap_ids)
print(f"from : {len(from_)}")
print(f"to : {len(to_)}")

apms_data_path = "../table1.csv"

apms_ids = pd.read_csv(apms_data_path)
uids = set(apms_ids['UniprotId'])
print(f"apms uids: {len(apms_ids)}")

to_from = {r['To']: r['From'] for i,r in hiv_emap_ids.iterrows()}
intersection = uids.intersection(to_)
print(f"HIV APMS INTERSECTION: {len(intersection)}")
unique_genes = [to_from[uid] for uid in intersection]
print(f"Unique genes in intersection {len(unique_genes)}")
print(f"Conclusion - at least {math.comb(len(unique_genes), 2)} interactions are supported by both AP-MS and genetic interactions")
# -

# Create reference matrix in UID coordinate space
reference_preyu2uid = {r['PreyGene'].removesuffix('_HUMAN'): r['UniprotId'] for i,r in apms_ids.iterrows()}
reference_uid = [reference_preyu2uid[i] for i in reference_matrix.preyu.values]
reference_matrix = xr.DataArray(reference_matrix.values, coords={'preyu': reference_uid, 'preyv': reference_uid})

# Genes map to multiple possible uids so we need to pick the right ones
id_mapping = pd.read_csv("../data/hiv_emap/idmapping_2023_07_05.tsv", sep="\t")
uid2gene = {r['To']: r['From'] for i,r in id_mapping.iterrows()}

hiv_uids = np.array(list(uid2gene.keys()))
shared_reference = np.array(list(set(hiv_uids).intersection(set(reference_matrix.preyu.values))))

# +
hiv_reference = reference_matrix.sel(preyu=shared_reference, preyv=shared_reference)

# Temporarily drop the following
temp_drop = ["P42167", "Q14152", "Q9NY93", "Q9Y2Q9"]
temp_reference = reference_matrix.drop_sel(preyu=temp_drop, preyv=temp_drop)
# -

hiv_reference

# +
#Get hiv df in uniprot coords

hivdf = pd.read_csv("../data/hiv_emap/data_s1.csv", sep="\t", index_col='Gene')
columns = [k.removesuffix(' - ESIRNA') for k in hivdf.columns]
rows = [k.removesuffix(' - ESIRNA') for k in hivdf.index]

hivdf = pd.DataFrame(hivdf.values, columns = columns, index=rows)
print(hivdf)
# -

gene2uid = {}
for uid in uid2gene:
    if uid in temp_reference.preyu.values:
        gene = uid2gene[uid]
        if gene in gene2uid:
            print(gene, uid, )
        #assert gene not in gene2uid, (gene, uid)
        gene2uid[gene] = uid

print(len(gene2uid))

hiv_mat = xr.DataArray(hivdf.values, coords={'preyu': hivdf.columns.values, 'preyv': hivdf.columns.values})
gene2uid_keys = list(gene2uid.keys())
shared_keys = set(gene2uid_keys).intersection(hiv_mat.preyu.values)
shared_keys = list(shared_keys)
hiv_mat = hiv_mat.sel(preyu=shared_keys, preyv=shared_keys)
#temp_reference = temp_reference.sel(preyu=shared_keys, preyv=shared_keys)

print(len(hiv_mat))

hiv_uids = [gene2uid[key] for key in hiv_mat.preyu.values]
hiv_mat = xr.DataArray(hiv_mat.values, {'preyu': hiv_uids, 'preyv': hiv_uids})

shared_keys = list(set(hiv_mat.preyu.values).intersection(temp_reference.preyu.values))

hiv_ref = temp_reference.sel(preyu=shared_keys, preyv=shared_keys)

assert np.all(hiv_mat.preyu == hiv_ref.preyu)

nnodes = len(hiv_mat.preyu)

tril_indices = np.tril_indices(nnodes, k=-1)

# +
ref_array = hiv_ref.values[tril_indices]

gi_score_array = hiv_mat.values[tril_indices]

positive = gi_score_array[ref_array]

negative = gi_score_array[~ref_array]

# +
fig, ax = plt.subplots()
neg_alpha = 0.5
pos_alpha = 0.5
xlim = 3
plt.hist(negative, label=f'UNK\nN={len(negative)}', bins=100, alpha=neg_alpha, density=True)
plt.hist(positive, label=f'PPI\nN={len(positive)}', bins=100, alpha=pos_alpha, density=True)
plt.xlim(-xlim, xlim)
plt.legend()
plt.show()
print(np.mean(positive))
print(np.median(positive))
print(np.mean(negative))
print(np.median(negative))

ks_alpha = 0.1
c_of_alpha = 1.224
m = len(positive)
n = len(negative)
b = np.sqrt((n + m) / (n * m))
D = c_of_alpha * b

res = sp.stats.ks_2samp(positive, negative)
stat = res.statistic
pvalue = res.pvalue

print(f"pvalue {pvalue} D: {stat}")
# Confidence level 90%
# p-value must be less than 0.1
# Distance - 0.3

# If p < 0.1 reject the null
# If D > 0.145 reject the null

# Conclusion - cannot reject the null.
print("Cannot reject the null")
# -

# Let's make a GI Correlation matrix
if not LOAD_HIV_CORRELATION:
    hiv_corr_matrix = np.zeros(hiv_mat.shape)
    hiv_pval_matrix = np.zeros(hiv_mat.shape)
    for i in range(len(hiv_corr_matrix)):
        for j in range(i + 1, len(hiv_corr_matrix)):
            a = hiv_mat.values[i, :]
            b = hiv_mat.values[j, :]
            corr_coef, pval = sp.stats.pearsonr(a, b)
            hiv_corr_matrix[i, j] = corr_coef
            hiv_pval_matrix[i, j] = pval
else:
    with open("xr_hiv_correlation_matrix.pkl", "rb") as f:
        hiv_corr_matrix = pkl.load(f)

print(len(hiv_corr_matrix))

# Ideal distribution of correlation coefficients
nsamples = math.comb(237, 2)
rng_key = jax.random.PRNGKey(13)
ideal_corr_samples = jax.random.t(rng_key, df=235, shape=(nsamples,))
ideal_corr_samples = ideal_corr_samples / np.sqrt(235 + ideal_corr_samples**2)

hiv_corr_matrix

fig, ax = plt.subplots(1, 2)
density = False
gi_alpha = 0.8
pval_alpha = 0.8
ylabel = 'density' if density else 'frequency' 
x = hiv_corr_matrix.T.values[tril_indices]
ax[0].hist(x, bins=100, density=density, alpha=gi_alpha, label="GI pearson R", color=GI_COLOR)
ax[0].hist(np.array(ideal_corr_samples), bins=100, alpha=gi_alpha, label="T distribution")
ax[0].legend()
#plt.hist(hiv_pval_matrix.T[tril_indices], bins=100, density=density, alpha=pval_alpha)
ax[0].set_xlabel("GI Profile Correlation Coefficient")
ax[0].set_ylabel(ylabel)
#ax[1].scatter(hiv_corr_matrix.T[tril_indices], hiv_pval_matrix.T[tril_indices], color='k')
ax[1].set_xlabel('GI Profile Correlation Coefficient')
ax[1].set_ylabel('P-value')
print(np.mean(x), np.min(x), np.max(x))
plt.show()

# GI Correlation Coefficient Volcano plot
if False: # Don't run because we didn't save p-values and don't want to recompute.
    v1 = -np.log10(0.05)
    v2 = -np.log10(0.01)
    v3 = -np.log10(0.001)
    xmin = -0.6
    xmax = 0.8

    x = hiv_corr_matrix.T[tril_indices]
    y = - np.log10(hiv_pval_matrix.T[tril_indices])
    plt.plot(x, y, 'k.')
    plt.ylabel('-log10 p-val')
    plt.xlabel('GI Profile Pearson R')
    plt.hlines(v1, xmin, xmax, label='0.05  significacne')
    plt.hlines(v2, xmin, xmax, label='0.01  significance', color='g')
    plt.hlines(v3, xmin, xmax, label='0.001 significance', color='r')
    plt.legend()
    plt.show()

# +
with open('spectral_count_xarray.pkl', 'rb') as f:
    spectral_count_xarray = pkl.load(f)

stacked_spectral_counts_array = spectral_count_xarray.stack(y=['AP', 'condition', 'bait', 'rep'])

# Remove nodes where all counts are 0
empty_nodes = np.sum(stacked_spectral_counts_array != 0, axis=1) == 0
stacked_spectral_counts_array = stacked_spectral_counts_array.isel(preyu=~empty_nodes)

apms_nnodes, nconditions = stacked_spectral_counts_array.shape

if LOAD_APMS_CORRELATION:
    with open("xr_apms_correlation_matrix.pkl", "rb") as f:
        apms_correlation_matrix = pkl.load(f)
else:


    apms_correlation_matrix = np.zeros((apms_nnodes, apms_nnodes))
    apms_pval_matrix = np.zeros((apms_nnodes, apms_nnodes))

    # Drop indices if all rows are 0

    for i in range(apms_nnodes):
        for j in range(i + 1, apms_nnodes):
            a = stacked_spectral_counts_array.values[i, :]
            b = stacked_spectral_counts_array.values[j, :]
            corr_coef, pval = sp.stats.pearsonr(a, b)
            apms_correlation_matrix[i, j] = corr_coef
            apms_pval_matrix[i, j] = pval
        

# +
# Save the AP-MS data

# +
apms_tril_indices = np.tril_indices(apms_nnodes, k=-1)
# AP-MS Correlation Coefficient Volcano plot
v1 = -np.log10(0.05)
v2 = -np.log10(0.01)
v3 = -np.log10(0.001)
xmin = -0.5
xmax = 1.0

x = apms_correlation_matrix.T.values[apms_tril_indices]
# 1-tailed p-values 'gt'

n_dist = 96
null_dist = sp.stats.beta(n_dist/2 - 1, n_dist/2 - 1, loc=-1, scale=2)
y0 = 2 * null_dist.cdf(-np.abs(x))
#y0 = 2*sp.stats.t(df=94).cdf(-np.abs(x))
# -

# Exact Distribution of R
plt.plot(np.arange(-1, 1, 0.01), null_dist.pdf(np.arange(-1, 1, 0.01)), 'k.')
plt.title('Scipy Exact PDF of Pearson R')

#y0 = apms_pval_matrix.T[apms_tril_indices]
y = - np.log10(y0)
N = len(x)
plt.plot(x, y, 'b.')
plt.ylabel('-log10 p-val')
plt.xlabel('AP-MS Profile Pearson R')
plt.hlines(v1, xmin, xmax, label='0.05  significacne')
plt.hlines(v2, xmin, xmax, label='0.01  significance', color='g')
plt.hlines(v3, xmin, xmax, label='0.001 significance', color='r')
plt.title(f"AP-MS Correlation N={N}")
plt.legend()
plt.show()
del y

# Try mixing a Gaussian distribution and a beta distribution - beta defined from [0, 1]


mixing_dist = dist.Categorical(probs=jnp.ones(3) / 3.)
component_dist = dist.Normal(loc=jnp.array([-1, 1, 9]), scale=jnp.ones(3))
mixture = dist.MixtureSameFamily(mixing_dist, component_dist)
mixture_samples = mixture.sample(jax.random.PRNGKey(42), sample_shape=(1000,))


# +
# Start with a 2 component Gaussian mixture model

def two_component_gaussian_mixture_model(observed_values, n_components=2):
    mixing_dist = dist.Categorical(probs=jnp.ones(n_components) / n_components)
    mus = numpyro.sample('mu', dist.Normal(jnp.ones(2), jnp.ones(0.5)))
    #hyper_alpha = 
    #hyper_beta = 
    #sigma = numpyro.sample('sigma', dist.)
    


# -

plt.hist(np.array(mixture_samples), bins=100, color='k')
plt.show()

# +
fig, ax = plt.subplots()
#x = apms_correlation_matrix.T[apms_tril_indices]

density = True
apms_alpha = 0.5
pval_alpha = 0.8
DF = nconditions - 2
nsamples = math.comb(237, 2)
rng_key = jax.random.PRNGKey(13)
ideal_corr_samples = jax.random.t(rng_key, df=DF, shape=(nsamples,))
ideal_corr_samples = ideal_corr_samples / np.sqrt(235 + ideal_corr_samples**2)

ylabel = 'density' if density else 'frequency' 
ax.hist(x, bins=200, density=density, alpha=gi_alpha, label="AP-MS Pearson R")
ax.hist(np.array(ideal_corr_samples), bins=100, alpha=apms_alpha, label=f"Jax t distribution DF={DF}", density=density)


null_x = np.arange(-0.4, 1.0, 0.01)
null_y = null_dist.pdf(null_x)

pi_T = 0.6
#mixture_model = lambda x: (1-pi_T) * sp.stats.norm(-0.05, 0.02).pdf(x) + pi_T * sp.stats.norm(0.1, 0.2).pdf(x)
mixture_model = lambda x: (1-pi_T) * sp.stats.norm(-0.05, 0.06).pdf(x) + pi_T * sp.stats.beta(1.7, 3.5).pdf(x)
mixture_label = "Mixture of Gaussian and beta"
#mixture_model = lambda x: (1-pi_T) * sp.stats.norm(-0.05, 0.06).pdf(x) + pi_T * sp.stats.halfnorm(0.05, 0.3).pdf(x)
#mixture_model = lambda x: (1-pi_T) * dist.pdf(x) + pi_T * sp.stats.beta(1.5, 5).pdf(x)
mixture_y = mixture_model(null_x)

ax.plot(null_x, null_y * 0.6, label='Null', color='r')
ax.plot(null_x, mixture_y, color='k', label=mixture_label)

ax.legend()

#plt.hist(hiv_pval_matrix.T[tril_indices], bins=100, density=density, alpha=pval_alpha)
ax.set_xlabel("AP-MS Correlation Coefficient")
ax.set_ylabel(ylabel)

#ax[1].scatter(hiv_corr_matrix.T[tril_indices], hiv_pval_matrix.T[tril_indices], color='k')
#ax[1].set_xlabel('')
#ax[1].set_ylabel('P-value')

# The Gaussian mixture model cannot accomadate the left tail
print(np.mean(x), np.min(x), np.max(x))
# -

plt.hist(ideal_corr_samples, bins=100, density=True, label='ideal_corr_samples')
null_x = np.arange(-0.4, 1.0, 0.01)
null_y = sp.stats.norm(0, 0.1).pdf(null_x)
plt.plot(null_x, null_y * 1.5, label='Null model')
plt.xlim(-0.4, 0.4)
plt.legend()
plt.show()

plt.hist(np.ravel(hiv_corr_matrix.T[tril_indices]), bins=100, density=True)
xtemp = np.arange(-0.6, 0.8, 0.01)
n_dist = 256
pearson_dist = sp.stats.beta(n_dist/2 - 1, n_dist/2 - 1, loc=-1, scale=2)
ytemp = 8 * pearson_dist.cdf(-np.abs(xtemp))
dist2 = sp.stats.distributions.t(df=254)
ytemp2 = dist2.pdf(xtemp) * 8
plt.plot(xtemp, ytemp, label='PDF: exact 2 sample pearson r')
plt.plot(xtemp, ytemp2, label='t: DF=254')
plt.xlabel('GI Pearson R')
plt.legend()
plt.show()

plt.hist(hiv_mat.values[tril_indices], bins=100)
plt.show()

"""

"""


# +
def model(yobs):
    w = numpyro.sample('w', dist.Uniform(0, 1))
    mu = numpyro.sample('mu', dist.Normal(0, 0.1))
    sigma = numpyro.sample('sigma', dist.Exponential(0.1))
    alpha = numpyro.sample('alpha', dist.Gamma(1))
    beta = numpyro.sample('beta', dist.Gamma(1))
    r_F = numpyro.sample('r_F', dist.Normal(mu, sigma))
    r_T = numpyro.sample('r_T', dist.Beta(alpha, beta))
    numpyro.sample('obs', dist.Normal((1-w) * r_F + w * r_T, 0.001), obs=yobs)
    
kernel = NUTS(model)
mcmc = MCMC(kernel, num_samples=1000, num_warmup=500)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, x, extra_fields=('potential_energy',))
# -

samples = mcmc.get_samples()
extra_fields = mcmc.get_extra_fields()

len(samples)

xtemp = np.arange(-0.5, 1.0, 0.01)
xtemp = np.arange(0.1, 10, 0.1)
#ytemp = sp.stats.beta(2, 5).pdf(xtemp)
ytemp = sp.stats.gamma(1).pdf(xtemp)
#ytemp = sp.stats.norm(0, 0.001)
#plt.plot(xtemp, sp.stats.halfnorm(0.1, 0.3).pdf(xtemp))
plt.plot(xtemp, ytemp)

x_points = np.arange(-5, 5.1, 0.1)
for i in range(1, 100, 10):
    y_points = sp.stats.t.pdf(x_points, df=i)
    plt.plot(x_points, y_points, label=f"DF={i}")
plt.legend()
plt.show()
print("Conclusion - T distribution does not model spectral counts")

plt.hist(np.ravel(stacked_spectral_counts_array), bins=100)
plt.xlabel("Spectral Count")
plt.ylabel("Spectral Count")
plt.show()
print("Conclusion - distribution of counts is not Gaussian")

"""
- Ignacia: Maybe try MIC

-
"""

# +
# Shuffle

if LOAD_SHUFFLED_APMS_CORRELATION:
    with open("shuffled_apms_correlation_matrix.pkl", "rb") as f:
        shuffled_apms_correlation_matrix = pkl.load(f)
else:
    shuffled_apms_data = np.random.permutation(np.ravel(stacked_spectral_counts_array)).reshape(stacked_spectral_counts_array.shape) # Accounts for distribution
    shuffled_apms_correlation_matrix = np.zeros((apms_nnodes, apms_nnodes))
    shuffled_apms_pval_matrix = np.zeros((apms_nnodes, apms_nnodes))
    for i in range(apms_nnodes):
        for j in range(i + 1, apms_nnodes):
            a = shuffled_apms_data[i, :]
            b = shuffled_apms_data[j, :]
            corr_coef, pval = sp.stats.pearsonr(a, b)
            shuffled_apms_correlation_matrix[i, j] = corr_coef
            shuffled_apms_pval_matrix[i, j] = pval

    #with open("shuffled_apms_correlation_matrix.pkl", "wb") as f:
        #pkl.dump(shuffled_apms_correlation_matrix, f)
# -

print(len(shuffled_apms_correlation_matrix))


# +
def piecewise_constant_pdf(x, bin_edges, bin_heights, n_bins):
    bin_edges = jnp.array(bin_edges)
    bin_heights = jnp.array(bin_heights)
    
    bin_indices = jnp.digitize(x, bin_edges) - 1
    bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)
    return bin_heights[bin_indices]

def rejection_sampling(key, num_samples, pdf, proposal_sampler, proposal_pdf):
    def sample_one(key):
        """Sample a single value using rejection sampling."""
        def cond_fun(state):
            _, _, is_accepted = state
            return ~is_accepted

        def body_fun(state):
            key, proposal_key, next_key = jax.random.split(state[0], 3)
            x_proposal = proposal_sampler(proposal_key)
            u = jax.random.uniform(key, minval=0, maxval=1)
            is_accepted = (u < pdf(x_proposal) / proposal_pdf(x_proposal))
            return next_key, x_proposal, is_accepted

        init_state = (key, 0.0, False)
        _, x_sample, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
        return x_sample

    keys = jax.random.split(key, num_samples)
    samples = jax.vmap(sample_one)(keys)
    return samples

def proposal_sampler(key, sigma=0.05, mu=0):
    return jax.random.normal(key) * sigma + mu  # Sampling from a normal distribution as proposal

def proposal_pdf(x, sigma=0.05, mu=0):
    return jax.scipy.stats.norm.pdf(x, loc=mu, scale=sigma)
    #return jnp.exp(-x**2 / 2) / jnp.sqrt(2 * jnp.pi)  # Standard normal PDF

cauchy_scale = 0.1
def cauchy_sampler(key, loc=0, scale=cauchy_scale):
    return jax.random.cauchy(key) * scale + loc

def cauchy_pdf(x, loc=0, scale=cauchy_scale):
    return jax.scipy.stats.cauchy.pdf(x, loc=loc, scale=scale)

flattened_apms = apms_correlation_matrix.T.values[apms_tril_indices]
nbins = 500
alpha = 0.5
use_dens = True

bin_heights, bin_edges, patches = plt.hist(flattened_apms, label='Not shuffled', 
                        alpha=alpha, bins=nbins, density=use_dens)
bin_heights, bin_edges, patches = plt.hist(shuffled_apms_correlation_matrix.T[apms_tril_indices][0:2000000], 
         bins=nbins, label="Shuffled", alpha=alpha, density=use_dens)
plt.title("Spectral Counts Correlation")
plt.plot(linspace, sp.stats.norm(0.4, 0.15).pdf(linspace), label=f"PDF")
#plt.ylim((0, 5))
plt.legend()
plt.xlabel("AP-MS Pearson R")
plt.xlim(-0.4, 1.0)
plt.show()

def build_pdf_sampler_pair(bin_heights, 
            bin_edges, 
            proposal_pdf=proposal_pdf, proposal_sampler=proposal_sampler):
    pdf = Partial(piecewise_constant_pdf, n_bins=len(bin_heights),
                 bin_heights=bin_heights, bin_edges=bin_edges)
    sampler = Partial(
        rejection_sampling,
        pdf=pdf, 
        proposal_sampler=proposal_sampler,
        proposal_pdf=proposal_pdf)
    
    return pdf, sampler

shuffled_pdf, sampler = build_pdf_sampler_pair(
    bin_heights, bin_edges, proposal_pdf=cauchy_pdf,
    proposal_sampler=cauchy_sampler)
linspace = np.arange(-0.4, 1.0, 0.01)   

#shuffled_pdf = Partial(piecewise_constant_pdf, n_bins=len(bin_heights), bin_edges=bin_edges,
#                      bin_heights=bin_heights)
# -

mus = np.arange(0.25, 0.75, 0.05)
mus = mus[..., None]
sig = 0.2
linspace = np.arange(-1, 1, 0.01)
for i in range(len(mus)):
    plt.plot(linspace, sp.stats.norm.pdf(linspace, mus[i], sig))
plt.hist(y, bins=100, density=True)
plt.hist(np.ravel(apms_correlation_matrix.T[tril_indices])[0:3000000], label="observed correlations", density=True,
        bins=100)
plt.xlim(-1, 1)
plt.show()


# +
# Direct sampling
def direct_sampler_from_histogram(counts, bin_edges, key, num_samples):
    # Convert histogram counts to probabilities
    probabilities = counts / jnp.sum(counts)
    # Sample from the categorical distribution
    bin_indices = jax.random.categorical(key,
            jnp.log(probabilities), shape=(num_samples,))
    return bin_edges[bin_indices]

epsilon = 0
counts = bin_heights + epsilon
num_samples = 100000
alpha_plot = 0.5
direct_samples = direct_sampler_from_histogram(counts, bin_edges,
            jax.random.PRNGKey(0), num_samples = num_samples)

nbins=50
plt.hist(
direct_samples, bins=nbins, label='Sampled values', alpha=alpha_plot,
    density=use_dens)
plt.hist(flattened_shuffled_apms[0:num_samples],
    bins=nbins, alpha=alpha_plot, label='Observed values', density=use_dens)
    
x = np.arange(-0.2, 1.0, 0.001)
    
plt.plot(x, shuffled_pdf(x), label="Empirical PDF")
plt.xlabel('AP-MS Correlation')
plt.ylabel("Probability Density")
plt.title("Distribution of Non interacting correlations\nMethod of direct sampling")
plt.legend()
plt.show()
# -

mini = apms_correlation_matrix.values[0:5, 0:2]

np.random.shuffle(mini)

# +

mini = jax.random.permutation(jax.random.PRNGKey(13), mini, independent=True)
# -

plt.plot(np.mean(apms_correlation_matrix.values, axis=0),
        np.mean(np.array(jax.random.permutation(jax.random.PRNGKey(13),
                apms_correlation_matrix.values, axis=0)), axis=0), 'k.', alpha=0.1)


# +
# 1. Have an Empirical PDF
# 2. Have a sampler
# 3. Need to use in Numpyro Model
# 4. Implement a Custom PDF

class Histogram(dist.Distribution):
    def __init__(self, a, bins, density=True, validate_args=None):
        bin_heights, bin_edges = np.histogram(a, bins=bins, density=density)
        bin_heights = jnp.array(bin_heights)
        bin_edges = jnp.array(bin_edges)

        self.probs = bin_heights
        self.n_bins = len(bin_heights)
        self.bin_edges = bin_edges
        self.support = dist.constraints.real # Is this right?
        super(Histogram, self).__init__(batch_shape=(), 
              event_shape=(), validate_args=validate_args)
        
        def piecewise_constant_pdf(x, bin_edges, bin_heights, n_bins):
            #bin_edges = jnp.array(bin_edges)
            #bin_heights = jnp.array(bin_heights)
            bin_indices = jnp.digitize(x, bin_edges) - 1
            bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)
            return bin_heights[bin_indices]
        
        # Initialize the PDF
        pdf = jax.tree_util.Partial(piecewise_constant_pdf,
              bin_edges=self.bin_edges, bin_heights=self.probs,
              n_bins=self.n_bins)
        self.pdf = pdf
        
    
    def sample(self, key, sample_shape=()):
        bin_indices = jax.random.categorical(key, jnp.log(self.probs), shape=sample_shape)
        return self.bin_edges[bin_indices]
        
    def log_prob(self, value):
        return jnp.log(self.pdf(value))
    
def null_model(null_bin_heights, bin_edges):
    #probs = null_bin_heights / jnp.sum(null_bin_heights)
    #apms_corr_coef = numpyro.sample('r', dist.Normal(0, 1))
    apms_corr_coef = numpyro.sample('r', dist.Uniform(low=-1, high=1))
    numpyro.sample('obs', Histogram(null_bin_heights, bin_edges),
         obs=apms_corr_coef)
    

alpha = 1
beta = 10
hyper_mu_mu = 0.5
hyper_mu_sigma = 0.03
hyper_sigma_lambda = 10
@config_enumerate
def mixture_model(null_obs, n_null_bins, observed_data,
                 hyper_mu_mu, hyper_mu_sigma, hyper_sigma_lambda, alpha, beta):
    #probs = numpyro.sample('p', dist.Dirichlet(jnp.array([1., 1.])))

    pT = numpyro.sample('pT', dist.Beta(alpha, beta))
    pF = 1 - pT
    #apms_corr_coef = numpyro.sample('r', dist.Uniform(low=-1, high=1))
    mu = numpyro.sample('mu', dist.Normal(hyper_mu_mu, hyper_mu_sigma))
    sigma = numpyro.sample('sg', dist.Exponential(hyper_sigma_lambda))
    #weight = numpyro.sample('w', dist.Uniform(0, 1))
    # Mixture model
    #r_spurious = numpyro.sample()
    
    # r ~ (1-w)
    categorical = dist.Categorical(jnp.array([pF, pT]))
    # Mixing distribution
    assignment = numpyro.sample('assignment', categorical)
    
    if observed_data is not None:
        with numpyro.plate('data', len(observed_data)):
            numpyro.sample('obs', dist.MixtureGeneral(
                categorical,
                component_distributions=[
                    #dist.Normal(-0.1, 0.1),
                    Histogram(null_obs, bins=n_null_bins), # density is True
                    #dist.Normal(0.17, 0.22)
                    #dist.Normal(0.18, 0.22)
                    dist.Normal(mu, sigma)
                ]
            ), obs=observed_data)
    else:
        numpyro.sample('obs', dist.MixtureGeneral(
                categorical,
                component_distributions=[
                    #dist.Normal(-0.1, 0.1),
                    Histogram(null_obs, bins=n_null_bins),
                    #dist.Normal(0.17, 0.22)
                    #dist.Normal(0.18, 0.22)
                    dist.Normal(mu, sigma)
                ]
            ), obs=observed_data)


# Run MCMC to infer the probabilities
#nuts_kernel = NUTS(null_model)
# -

linspace = np.arange(-1, 1, 0.01)
plt.plot(linspace, sp.stats.norm.pdf(linspace, hyper_mu_mu, hyper_mu_sigma), label=f"Prior Mu: {hyper_mu_mu}, {hyper_mu_sigma}")
plt.plot(linspace, sp.stats.expon(scale=1/hyper_sigma_lambda).pdf(linspace), label=f"Prio sigma: prior mean {np.round(1/hyper_sigma_lambda, 3)}")
plt.legend()
plt.ylabel("PDF")

# +
nsamples=10_000
n_obs = 100_000
alpha = 2
beta = 6
n_null_obs = 50_000
n_null_bins = 1_000

# Important to not use density, Histogram class will take care of normalization
#bin_heights, bin_edges, patches = plt.hist(shuffled_apms_correlation_matrix.T[apms_tril_indices][0:n_hist_obs],
#        bins=n_hist_bins, density=False)
nuts_kernel = NUTS(mixture_model)
mcmc = MCMC(nuts_kernel, num_warmup=10000, num_samples=nsamples)
mcmc.run(jax.random.PRNGKey(0), 
         null_obs=shuffled_apms_correlation_matrix.T[apms_tril_indices][0:n_null_obs],
         n_null_bins=n_null_bins, observed_data=flattened_apms[0:n_obs], hyper_mu_mu=hyper_mu_mu, hyper_mu_sigma=hyper_mu_sigma,
         hyper_sigma_lambda=hyper_sigma_lambda, alpha=alpha, beta=beta)
samples = mcmc.get_samples()

predictive = Predictive(mixture_model, num_samples=5000)
y_prior_pred = predictive(rng_key, shuffled_apms_correlation_matrix.T[apms_tril_indices][0:n_null_obs],
                         n_null_bins, observed_data=None,
                         hyper_mu_mu=hyper_mu_mu, hyper_mu_sigma=hyper_mu_sigma, 
                         hyper_sigma_lambda=hyper_sigma_lambda, alpha=alpha, beta=beta)

predictive = Predictive(mixture_model, num_samples=5000, posterior_samples=samples)
y_post_pred = predictive(rng_key, shuffled_apms_correlation_matrix.T[apms_tril_indices][0:n_null_obs], n_null_bins, observed_data=None,
                         hyper_mu_mu=hyper_mu_mu, hyper_mu_sigma=hyper_mu_sigma, 
                         hyper_sigma_lambda=hyper_sigma_lambda, alpha=alpha, beta=beta)


# -

def distplot(y_prior_pred, y_post_pred, x, plot_prior=True, n_obs=5000):
    bins=100
    alpha=0.3
    if plot_prior:
        plt.hist(y_prior_pred['obs'], bins=bins, alpha=alpha, label="Prior predictive", density=True)
    plt.hist(y_post_pred['obs'], bins=bins, alpha=alpha, label="Posterior predictive", density=True)
    plt.hist(flattened_apms[0:n_obs], bins=bins, alpha=alpha, label=f"Observed values", density=True)
    plt.ylabel("Probability density")
    plt.xlabel("AP-MS pearson R")
    plt.legend()
    plt.show()
distplot(y_prior_pred, y_post_pred, x)

HyperPriors = namedtuple("HyperPriors", "mu_mu sigma_lambda mu_sigma alpha beta")
default_hyper_priors = HyperPriors(0.195, 200, 0.01, 1.5, 8)


def sample_model2(
    shuffled_apms_matrix,
    flattened_apms,
    n_obs,
    n_null_bins,
    n_null_obs,
    hyper_priors : HyperPriors,
    num_samples : int =10_000,
    num_warmup : int = 5_000,
    n_predictive_samples = 5_000,
    rng_key = jax.random.PRNGKey(0),
    numpyro_model=mixture_model):
    """
    Perform sampling and prior predictive checks to construct Bayesian data likelihood using
    mixture model 2
    """
    
    # Unpack hyper priors
    hyper_mu_mu, hyper_sigma_lambda, hyper_mu_sigma, alpha, beta = hyper_priors
    
    # Define the histogram
    apms_tril_indices = jnp.tril_indices_from(shuffled_apms_matrix, k=-1)
    #bin_heights, bin_edges, patches = plt.hist(shuffled_apms_matrix.T[apms_tril_indices][0:n_shuffled_hist_obs],
    #                        bins=n_hist_bins)
    anull = shuffled_apms_correlation_matrix.T[apms_tril_indices][0:n_null_obs]
    
    nuts_kernel = NUTS(numpyro_model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, 
             null_obs=anull,
             n_null_bins=n_null_bins,
             observed_data=flattened_apms[0:n_obs],
             hyper_mu_mu=hyper_mu_mu,
             hyper_mu_sigma=hyper_mu_sigma,
             hyper_sigma_lambda=hyper_sigma_lambda,
             alpha=alpha, 
             beta=beta)
    samples = mcmc.get_samples()

    predictive = Predictive(mixture_model, num_samples=n_predictive_samples)
    y_prior_pred = predictive(rng_key, anull, n_null_bins, observed_data=None,
                             hyper_mu_mu=hyper_mu_mu, hyper_mu_sigma=hyper_mu_sigma, 
                             hyper_sigma_lambda=hyper_sigma_lambda, alpha=alpha, beta=beta)

    predictive = Predictive(mixture_model, num_samples=n_predictive_samples, posterior_samples=samples)
    y_post_pred = predictive(rng_key, anull, n_null_bins, observed_data=None,
                             hyper_mu_mu=hyper_mu_mu, hyper_mu_sigma=hyper_mu_sigma, 
                             hyper_sigma_lambda=hyper_sigma_lambda, alpha=alpha, beta=beta)
    
    return mcmc, samples, y_prior_pred, y_post_pred


def general_sampling_helper():
    """
    model : numpyro model
    rng_key : random number generator for sampling
    do_prior_pred : bool do prior predictive check
    do_post_pred : bool do posterior predictive check
    
    num_samples : int number of mcmc samples
    num_warmup : int number of warmup chains
    
    
    """


# +
f = sample_model2

mcmc2, samples2, prior_pred, post_pred = f(
    shuffled_apms_correlation_matrix,
    flattened_apms,
    n_obs=100_000,
    n_null_bins=200,
    n_null_obs=50_000,
    hyper_priors=default_hyper_priors, num_warmup=500, num_samples=2000)
# -

#debug_log_prob_scale(bin_edges, bin_heights)
fig, ax = plt.subplots(1, 2)
example_null_hist = Histogram(shuffled_apms_correlation_matrix.T[apms_tril_indices][0:n_null_obs],
                             bins=200)
xtemp = np.arange(0, 0.4, 0.001)
ax[0].plot(xtemp, example_null_hist.pdf(xtemp), label="Histogram PDF")
a, b, c = ax[0].hist(shuffled_apms_correlation_matrix.T[apms_tril_indices][0:500_000], density=True, bins=1000)
ax[0].legend()
xtemp = np.arange(0.15, 0.25, 0.005)
ax[1].plot(xtemp, example_null_hist.log_prob(xtemp))
ax[1].plot(xtemp, dist.Normal(0.2, 0.005).log_prob(xtemp))


xtemp


# +
def debug_log_prob_scale(anull, bins):
    xtemp = np.arange(-1, 1, 0.01)
    y1 = Histogram(anull, bins).log_prob(xtemp)
    y2 = dist.Normal(0.21, 0.005).log_prob(xtemp)
    plt.plot(xtemp, y1, label="Hist log prob")
    plt.plot(xtemp, y2, label="Normal Log Prob")
    y3 = sp.stats.norm(0.21, 0.005).logpdf(xtemp)
    plt.plot(xtemp, y3, label="Scipy Normal log prob")
    plt.legend()
    
debug_log_prob_scale(shuffled_apms_correlation_matrix.T[apms_tril_indices][0:n_null_obs], n_null_bins)
# -

mcmc2.print_summary()

sig = inspect.signature(f)
Params = namedtuple("Params", tuple(sig.parameters))
params = Params(**sig.parameters)

# +
fig, axs = plt.subplots(2, 2)
def _1(ax, hp: HyperPriors, prior_pred_obs = None, post_pred_obs = None, nbins=100):
    scale = 1/hp.sigma_lambda
    ax.set_title("Exponential PDF")
    ax.set_xlabel("Sigma")
    xtemp = np.arange(0, 1, 0.01)
    ax.plot(np.arange(0, 1, 0.01), sp.stats.expon(scale=scale).pdf(xtemp),
            label=f'Rate={1/scale}: mean {scale}')
    if prior_pred_obs is not None:
        ax.hist(prior_pred_obs, bins=nbins, label="Prior pred", density=True)
    ax.legend()
    

ax = axs[0, 0]
_1(ax,default_hyper_priors, prior_pred['sg'])

def _2(ax, hp, prior_pred_obs = None, nbins=100):
    xtemp = np.arange(-0.2, 1.2, 0.01)
    y = sp.stats.norm(loc=hp.mu_mu, scale=hp.mu_sigma).pdf(xtemp)
    ax.plot(xtemp, y, label=f"mu: {(hp.mu_mu)}, sg: {(hp.mu_sigma)}")
    ax.set_title("Normal PDF")
    ax.set_xlabel("mu")
    ax.legend()
    if prior_pred_obs is not None:
        ax.hist(prior_pred_obs, bins=nbins, density=True)
    
    
    
ax = axs[0, 1]
_2(ax, default_hyper_priors, prior_pred["mu"])
ax = axs[1, 0]
def _3(ax, hp, prior_pred_obs = None):
    xtemp = np.arange(0, 1, 0.01)
    y = sp.stats.beta(hp.alpha, hp.beta).pdf(xtemp)
    ax.plot(xtemp, y)
    ax.set_xlabel("pi_T")
    ax.set_title("Beta distribution")
    if prior_pred_obs is not None:
        ax.hist(prior_pred_obs, bins=nbins, density=True)
_3(ax, default_hyper_priors, prior_pred['pT'])


plt.tight_layout()
# -

default_hyper_priors


# Plot the prior predictive distribution
# Pathological model
def plot_pred(ax, n_obs, apms_obs, prior_pred, nbins=100, alpha_plot=0.5, label1="Prior"):
    ax.hist(prior_pred['obs'], bins=nbins, density=True, label=label1 + " predictive", alpha=alpha_plot)
    ax.hist(apms_obs[0:n_obs], bins=nbins, density=True, label="Observed", alpha=alpha_plot)
    ax.legend()
fig, ax = plt.subplots(1, 2)
plot_pred(ax[0], n_obs, flattened_apms, prior_pred)
plot_pred(ax[1], n_obs, flattened_apms, post_pred, label1="Posterior")

# +
improved_hyper_priors = HyperPriors(0.30, 10, 0.1, 4, 6)

mcmc2, samples2, prior_pred, post_pred = sample_model2(
    shuffled_apms_correlation_matrix,
    flattened_apms,
    n_obs=1_000_000,
    n_null_bins=1000,
    n_null_obs=50_000,
    hyper_priors=improved_hyper_priors, num_warmup=1000, num_samples=10_000)
# -

mcmc2.print_summary()

fig, ax = plt.subplots(1, 3)
_1(ax[0], improved_hyper_priors, prior_pred['sg'])
#_1(ax[0, 1], improved_hyper_priors, samples2['sg'])
_2(ax[1], improved_hyper_priors, prior_pred['mu'])
_3(ax[2], improved_hyper_priors, prior_pred['pT'])
#ax[0].hist(samples2['sg'], label='post', density=True)
ax[0].vlines(np.mean(samples2['sg']), 0, 1, color='k')
ax[1].vlines(np.mean(samples2['mu']), 0, 4, color='k')
ax[2].vlines(np.mean(samples2['pT']), 0, 4, color='k')
#ax[1].hist(samples2['mu'], label='post', density=True)
#ax[2].hist(samples2['pT'], label='post', density=True)
plt.tight_layout()

plt.hist(samples2['mu'], bins=100)
plt.show()

fig, ax = plt.subplots(1, 2)
plot_pred(ax[0], n_obs, flattened_apms, prior_pred)
plot_pred(ax[1], n_obs, flattened_apms, post_pred, label1="Posterior")

"""
Mutual information criteria




"""

# +
MatrixModelHP = namedtuple("MatrixModelHP", "")

def matrix_model(D, hp):
    """
    D: observed data for the likelihood
    hp: model hyperparameters
    nnodes: int - number of nodes in the matrix
    """
    
    nnodes, _ = apms_similarity_matrix.shape
    n_edges = math.comb(nnodes, 2)
    apms_tril_indices = jnp.tril_indices(nnodes, k=-1)
    
    # Prior model density
    Aij = numpyro.sample("Aij", dist.Beta(hp.Aij.alpha, hp.Aij.beta, event_shape=(n_edges)))
    
    a = numpyro.sample('a', dist.Normal(0.23, 0.21), event_shape=(n_edges))
    b = numpyro.sample('b', null_dist)
    #d = Aij * a + (1-Aij) * b
    
    numpyro.sample("d", dist.Normal(Aij * a + (1-Aij) * b, 0.01), obs=obs)
    
    
    
null_dist = Histogram(shuffled_apms_correlation_matrix.T[apms_tril_indices][0:100_000], 500)
kernal = NUTS(matrix_model)
    
# -

plt.hist(null_dist.sample(rng_key, sample_shape=(1000,)), bins=100)
plt.show()

# ?Histogram

jnp.tril_indices(4, k=-1)

# +
# Sampling 3 replaces the Normal Distribution with a Beta distribution

alpha = 1
beta = 10
hyper_mu_mu = 0.5
hyper_mu_sigma = 0.03
hyper_sigma_lambda = 10
@config_enumerate
def mixture_model_beta(null_obs, n_null_bins, observed_data,
                 hyper_mu_mu, hyper_mu_sigma, hyper_sigma_lambda, alpha, beta):
    """
    Similiar to the above mixture model except a Beta mixture is used
    - Justification: Assymetry
    - Bounded between 0 and 1
    - 
    """
    #probs = numpyro.sample('p', dist.Dirichlet(jnp.array([1., 1.])))

    pT = numpyro.sample('pT', dist.Beta(alpha, beta))
    pF = 1 - pT
    #apms_corr_coef = numpyro.sample('r', dist.Uniform(low=-1, high=1))
    #mu = numpyro.sample('mu', dist.Normal(hyper_mu_mu, hyper_mu_sigma))
    #sigma = numpyro.sample('sg', dist.Exponential(hyper_sigma_lambda))
    a = numpyro.sample('a', dist.Uniform(1, 4))
    b = numpyro.sample('b', dist.Uniform(1, 4))
    
    #weight = numpyro.sample('w', dist.Uniform(0, 1))
    # Mixture model
    #r_spurious = numpyro.sample()
    
    # r ~ (1-w)
    categorical = dist.Categorical(jnp.array([pF, pT]))
    # Mixing distribution
    assignment = numpyro.sample('assignment', categorical)
    
    # Do an affine transformation to provide a real valued support
    causal_dist = dist.TransformedDistribution(dist.Beta(a, b), dist.transforms.AffineTransform(0, 1))
    #null_dist = Histogram(null_obs, n_null_bins)
    causal_dist = dist.Beta(1, 1)
    null_dist = dist.Beta(1, 2)
    
    if observed_data is not None:
        with numpyro.plate('data', len(observed_data)):
            numpyro.sample('obs', dist.MixtureGeneral(
                categorical,
                component_distributions=[
                    null_dist, # density is True
                    causal_dist
                ]
            ), obs=observed_data)
    else:
        numpyro.sample('obs', dist.MixtureGeneral(
                categorical,
                component_distributions=[
                    null_dist,
                    causal_dist
                ]
            ), obs=observed_data)


# +
# Sampling 3 replaces the Normal Distribution with a Beta distribution

alpha = 1
beta = 10
hyper_mu_mu = 0.5
hyper_mu_sigma = 0.03
hyper_sigma_lambda = 10
@config_enumerate
def mixture_model_beta(null_obs, n_null_bins, observed_data,
                 hyper_mu_mu, hyper_mu_sigma, hyper_sigma_lambda, alpha, beta):
    """
    Similiar to the above mixture model except a Beta mixture is used
    - Justification: Assymetry
    - Bounded between 0 and 1
    - 
    """
    #probs = numpyro.sample('p', dist.Dirichlet(jnp.array([1., 1.])))

    pT = numpyro.sample('pT', dist.Beta(alpha, beta))
    pF = 1 - pT
    #apms_corr_coef = numpyro.sample('r', dist.Uniform(low=-1, high=1))
    #mu = numpyro.sample('mu', dist.Normal(hyper_mu_mu, hyper_mu_sigma))
    #sigma = numpyro.sample('sg', dist.Exponential(hyper_sigma_lambda))
    a = numpyro.sample('a', dist.Uniform(1, 4))
    b = numpyro.sample('b', dist.Uniform(1, 4))
    
    #weight = numpyro.sample('w', dist.Uniform(0, 1))
    # Mixture model
    #r_spurious = numpyro.sample()
    
    # r ~ (1-w)
    categorical = dist.Categorical(jnp.array([pF, pT]))
    # Mixing distribution
    assignment = numpyro.sample('assignment', categorical)
    
    # Do an affine transformation to provide a real valued support
    causal_dist = dist.TransformedDistribution(dist.Beta(a, b), dist.transforms.AffineTransform(0, 1))
    #null_dist = Histogram(null_obs, n_null_bins)
    causal_dist = dist.Beta(1, 1)
    null_dist = dist.Beta(1, 2)
    
    
    
# -



# +
"""
a ~ Null
b ~ Beta(alpha, beta)
c = (1-pi) * a + pi * b

numpyro.sample(dist.No)

"""


# -

samples2


class A(Transform):
    ...


"""
Goal is to have a mixture model with some assymetry




"""

xtemp = np.arange(0, 10, 0.01)
plt.plot(xtemp, np.exp(dist.Exponential(1).log_prob(xtemp)))

# +
improved_hyper_priors = HyperPriors(0.30, 10, 0.1, 4, 6)

mcmc2, samples2, prior_pred, post_pred = sample_model2(
    shuffled_apms_correlation_matrix,
    flattened_apms,
    n_obs=1_000_000,
    n_null_bins=1000,
    n_null_obs=50_000,
    hyper_priors=improved_hyper_priors, num_warmup=500, num_samples=1000,
    numpyro_model=mixture_model_beta)
# -

fig, ax = plt.subplots()
plot_pred(ax, n_obs, flattened_apms, prior_pred)

mcmc2.print_summary()

hyper_mu_mu, hyper_mu_sigma, hyper_sigma_lambda, alpha, beta

plt.hist(np.array(ypP2['obs']), bins=100, alpha=0.5, density=True, label='Predictive')
plt.hist(np.array(x[0:100000]), bins=100, alpha=0.5, density=True, label='True')
plt.legend()
plt.show()

xtemp = np.arange(-0.6, 1.0, 0.01)
plt.plot(xtemp, Histogram(bin_heights, bin_edges).log_prob(xtemp))
plt.plot(xtemp, dist.Normal(0.21, 0.21).log_prob(xtemp))
plt.ylabel("score")
plt.show()

sp.stats.norm(0.21, 0.21).pdf(0.8)

# +
# mu, sigma, lambda, alpha, beta
# 0.17, 0.01, 10, 0.05, 10
# 0.21, 0.01, 200, 3, 5
# 0.19, 0.01, 200, 3, 5
# 0.205, 0.01, 200, 2, 8

distplot(ypp2, ypP2, x, plot_prior=False, n_obs=50000)
plt.vlines
# -

fig, ax = plt.subplots(1, 5)
for i, key in enumerate(y_prior_pred.keys()):
    ax[i].hist(np.array(y_prior_pred[key]))
    ax[i].set_xlabel(key)
plt.tight_layout()
plt.show()

predictive = Predictive(mixture_model, num_samples=n_predictive_samples, posterior_samples=samples)
y_pred_post = predictive(rng_key, bin_heights, bin_edges, x[0:n_obs],
                          hyper_mu_mu, hyper_mu_sigma, hyper_sigma_lambda)["obs"]

plt.hist(np.ravel(y_pred_post), bins=100)
plt.show()

plt.hist(np.ravel(y_pred_prior), bins=1000)
plt.show()


def joint_pdf(x, pT, mu, sigma, bin_heights, bin_edges):
    return (1-pT) * np.exp(Histogram(bin_heights, bin_edges).log_prob(x)) + sp.stats.norm(mu, sigma).pdf(x) * pT


plt.plot(linspace, joint_pdf(linspace, 0.57, 0.51, 0.23, bin_heights, bin_edges))

plt.plot(linspace, np.exp(Histogram(bin_heights, bin_edges).log_prob(linspace)))

mcmc.print_summary()

values = dist.MixtureGeneral(
            dist.Categorical(jnp.array([0.0, 1.0])),
            component_distributions=[
                Histogram(bin_heights, bin_edges),
                dist.Normal(0.22, 0.17)
            ]).sample(rng_key, sample_shape=(1000,))
plt.hist(np.array(values), bins=100)
plt.show()

# +
n_obs = len(x) // 100
#observed_data = np.ravel(apms_correlation_matrix.T[tril_indices])[0:n_obs]
plt.hist(x[0:n_obs], bins=100, label=f"Model observed {n_obs} data", density=True)
xtemp = np.arange(-0.4, 1.0, 0.01)
plt.plot(xtemp, np.exp(Histogram(bin_heights, bin_edges).log_prob(xtemp))*100)
plt.ylabel("Probability Density")
plt.legend()
plt.plot(xtemp, np.exp(dist.Normal(0.17, 0.22).log_prob(xtemp)))
plt.xlabel("Pearson R")
plt.show()

# Mean 0.17
# std 0.22

# +
from numpyro.infer.util import initialize_model
# The initial model Fits the above histogram, lets try using Sequential Monte Carlo to fix problem

# 1. Create a blackjax model
rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key)
init_params, potential_fn_gen, post_process_fn, model_trace = initialize_model(
    init_key,
    mixture_model,
    model_args=(bin_heights, bin_edges, x[0:n_obs]),
    dynamic_args=True,
)

logdensity_fn = lambda position: -potential_fn_gen(bin_heights, bin_edges, x[0:n_obs])(position)
initial_position = init_params.z
# 2. Create the Paralllel temporing algorithm
def inference_loop(rng_key, mcmc_kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, k):
        state, _ = mcmc_kernel(k, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def smc_inference_loop(rng_key, smc_kernel, initial_state):
    """Run the temepered SMC algorithm.

    We run the adaptive algorithm until the tempering parameter lambda reaches the value
    lambda=1.

    """

    def cond(carry):
        i, state, _k = carry
        return state.lmbda < 1

    def one_step(carry):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        state, _ = smc_kernel(subk, state)
        return i + 1, state, k

    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )

    return n_iter, final_state
# 3. Sample with hmc


inv_mass_matrix = jnp.eye(3)
n_samples = 10_000

hmc_parameters = dict(
    step_size=1e-4, inverse_mass_matrix=inv_mass_matrix, num_integration_steps=50
)

hmc = blackjax.hmc(logdensity_fn, **hmc_parameters)
hmc_state = hmc.init(initial_position)

# +
# An example of SMC using Blackjax following the notebook

from datetime import date
rng_key = jax.random.PRNGKey(int(date.today().strftime("%Y%m%d")))


# +
def V(x):
    return 5 * jnp.square(jnp.sum(x**2, axis=-1) - 1)


def prior_log_prob(x):
    d = x.shape[-1]
    return jsp.stats.multivariate_normal.logpdf(x, jnp.zeros((d,)), jnp.eye(d))


# +
linspace = jnp.linspace(-2, 2, 5000)[..., None]
lambdas = jnp.linspace(0.0, 1.0, 5)
prior_logvals = prior_log_prob(linspace)
potential_vals = V(linspace)
log_res = prior_logvals - lambdas[..., None] * potential_vals

density = jnp.exp(log_res)
normalizing_factor = jnp.sum(density, axis=1, keepdims=True) * (
    linspace[1] - linspace[0]
)
density /= normalizing_factor
# -

#

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(linspace.squeeze(), density.T)
ax.legend(list(lambdas));

density = jnp.exp(log_res)
normalizing_factor = jnp.sum(density, axis=1, keepdims=True) * (
    linspace[1] - linspace[0]
)
density /= normalizing_factor

asdlfj

rng_key, sample_key = jax.random.split(rng_key)
hmc_samples = inference_loop(sample_key, hmc.step, hmc_state, 5000)#n_samples)

# Parameters take on unrealistic values
plt.plot(hmc_samples.potential_energy)

plt.hist(hmc_samples.position['pT'][4000:5000])

# +
# Sample with NUTS
nuts_parameters = dict(step_size=1e-4, inverse_mass_matrix=inv_mass_matrix)

nuts = blackjax.nuts(logdensity_fn, **nuts_parameters)
nuts_state = nuts.init(initial_position)

rng_key, sample_key = jax.random.split(rng_key)
nuts_samples = inference_loop(sample_key, nuts.step, nuts_state, 1000)#n_samples)

# -

plt.plot(nuts_samples.potential_energy)

plt.hist(nuts_samples.position['pT'])

start = 400
end = 1000
plt.scatter(nuts_samples.position['mu'][start:end], nuts_samples.position['pT'][start:end])

plt.hist(nuts_samples.position['mu'])
plt.show()

plt.hist(nuts_samples.position['sg'])


# +
# Sample with SMC

#loglikelihood = lambda x: -V(x)

def prior_log_prob(x):
    return (jsp.stats.expon.logpdf(x['sg'], scale=0.1) + 
            jsp.stats.norm.logpdf(x['mu'], loc=0.2, scale=0.1) + 
            jsp.stats.norm.logpdf(x['pT'], loc=0.2, scale=0.1))

logdensity_fn

# +
hmc_parameters = dict(
    step_size=1e-4, inverse_mass_matrix=inv_mass_matrix, num_integration_steps=1
)

tempered = blackjax.adaptive_tempered_smc(
    prior_log_prob,
    logdensity_fn,
    blackjax.hmc.build_kernel(),
    blackjax.hmc.init,
    hmc_parameters,
    resampling.systematic,
    0.5,
    num_mcmc_steps=1,
)

rng_key, init_key, sample_key = jax.random.split(rng_key, 3)
initial_smc_state = jax.random.multivariate_normal(
    init_key, jnp.zeros([1]), jnp.eye(1), (n_samples,)
)
initial_smc_state = tempered.init(initial_smc_state)

n_iter, smc_samples = smc_inference_loop(sample_key, tempered.step, initial_smc_state)
print("Number of steps in the adaptive algorithm: ", n_iter.item())
# -



plt.plot(xtemp, jsp.stats.expon.pdf(xtemp, scale=0.1))
#plt.ylim(-5, 0)

# ?jsp.stats.expon.pdf

plt.plot(xtemp, np.exp(dist.Beta(1, 10).log_prob(xtemp)))

plt.plot(dist.Normal(0, 1).log_prob(xtemp), label='Normal(0, 1)')
plt.plot(Histogram(bin_heights, bin_edges).log_prob(xtemp) + 13, label='Hist')
plt.ylabel("log density")

plt.title("Exponential")
plt.plot(xtemp, np.exp(dist.Exponential(1).log_prob(xtemp)))
plt.show()

plt.plot(bin_heights)

nsamples=10000
nuts_kernel = NUTS(mixture_model)
mcmc = MCMC(nuts_kernel, num_warmup=10000, num_samples=nsamples)
mcmc.run(jax.random.PRNGKey(0), 
         null_bin_heights=bin_heights,
         bin_edges=bin_edges, observed_data=x[0:n_obs])
samples = mcmc.get_samples()

mcmc.print_summary()

plt.hist(samples['pT'])
plt.show()

plt.plot(xtemp, np.exp(dist.Normal(0.17, 0.22).log_prob(xtemp)))

# +
fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(samples['mu'])
axs[0, 0].set_xlabel('mu')
axs[0, 1].hist(samples['sg'])
axs[0, 1].set_xlabel('sg')
axs[1, 0].hist(samples['pT'], range=(0, 1))
axs[1, 0].set_xlabel('pT')
#axs[1, 0].set_xlim(0, 1)
plt.tight_layout()

plt.show()
# -

# I think its because the density functions aren't on the same scale


mcmc.print_summary()

inf_obj = az.from_numpyro(mcmc)

plt.hist(samples['pT'])
plt.show()

plt.hist(np.array(observed_data), bins=100, 
         label='Model observed data')
plt.show()



plt.hist(observed_data, bins=100)
plt.show()

len(observed_data)

plt.hist(samples['mu'])

probs = dist.Dirichlet(jnp.array([1. ,1.])).sample(rng_key)
samples = dist.MixtureGeneral(
            dist.Categorical(probs),
            component_distributions=[
                Histogram(bin_heights, bin_edges),
                dist.Normal(1, 0.1)
            ]
        ).sample(rng_key, sample_shape=(1000,))

plt.hist(samples, bins=100)
plt.show()

probs

Histogram(bin_heights, bin_edges).support

dist.Normal(0, 1).support

samples

probs = dist.Dirichlet(jnp.array([1., 1.])).sample(rng_key)

help(dist.Dirichlet)

plt.hist(np.array(samples['r']), bins=100, label='Generated', density=True)
ytemp = np.ravel(shuffled_apms_correlation_matrix.T[tril_indices])[0:nsamples]
plt.hist(ytemp, label="Real", bins=100, alpha=0.5, density=True)
plt.xlabel("R")
plt.ylabel("Density")
plt.title("Numpyro generated samples")
plt.legend()
plt.show()
print(len(ytemp))
del ytemp



plt.plot(bin_heights)

# +
example_hist = Histogram(bin_heights, bin_edges)
example_samples = example_hist.sample(rng_key, sample_shape=(10000,))
plt.hist(np.array(example_samples),
        bins=100, label="Normalized generated samples", density=True)
x = np.arange(-0.2, 1.0, 0.005)
y = example_hist.log_prob(x)
y2 = np.exp(y) * 500

plt.plot(x, y, label='log PDF')
plt.plot(x, y2, label='scaled PDF')
plt.xlabel("AP-MS Pearson R")
plt.legend()
plt.title("Null Histogram Model")
plt.savefig("2023-12-1_NullHistogramModel.png", dpi=300)
plt.close()


# +
def fruit_preference_model(data=None):
    # Prior probabilities for each fruit
    # For simplicity, we use a uniform prior
    probabilities = numpyro.sample('probabilities', dist.Dirichlet(jnp.ones(3)))

    # Categorical distribution for respondent choices
    with numpyro.plate('data', size=len(data)):
        choices = numpyro.sample('choices', dist.Categorical(probs=probabilities), obs=data)

# Example survey data (encoded as 0: Apple, 1: Banana, 2: Orange)
data = jnp.array([0, 2, 1, 0, 2, 2, 1, 0, 0])

# Run MCMC to infer the probabilities
nuts_kernel = NUTS(fruit_preference_model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), data=data)

# Get the posterior samples
posterior_samples = mcmc.get_samples()

# +

kernel = kernel = NUTS(null_model)
mcmc = MCMC(kernel, num_samples=1000, num_warmup=500)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, bin_heights, extra_fields=('potential_energy',))
samples = mcmc.get_samples()
# -

# ?dist.Categorical



plt.hist(np.array(sampler(jax.random.PRNGKey(0), 100000)),
         bins=100, density=True)
plt.xlim(-0.4, 1.0)
plt.show()


# +
def plot_shuffled_pdf(pdf):
    x = np.arange(-1, 1, 0.01)
    y = pdf(x)
    plt.plot(x, y, 'r.', label='Null Target PDF')
    y = proposal_pdf(x)
    plt.plot(x, y, label='Proposal')
    plt.title("PDF of Null Distribution")
    shuffled_pdf, sampler = build_pdf_sampler_pair(
        bin_heights, bin_edges, proposal_pdf=proposal_pdf,
        proposal_sampler=proposal_sampler)
    
    plt.hist(np.array(sampler(jax.random.PRNGKey(0), 200000)),
         bins=500, density=True, label='Generated values', color='k',
            alpha=0.5)
    plt.plot(x, cauchy_pdf(x, scale=0.03), label='cauchy')
    plt.plot(x, jax.scipy.stats.gamma.pdf(x, 2, 2, 0.1), label='Gamma')
    plt.xlabel("AP-MS Pearson R")
    plt.xlim(-0.4, 1.0)
    plt.legend()
    plt.show()
    
plot_shuffled_pdf(shuffled_pdf)


# -

def plot_gamma():
    x = np.arange(-0.4, 1, 0.01)
    alpha = 2
    beta = 2
    plt.plot(x,jax.scipy.stats.gamma.pdf(x, 1.0, -0.5), label='Gamma')
    plt.plot(x, jax.scipy.stats.beta.pdf(x, 2, 2, loc=-0.4))
    plt.legend()
    plt.show()
plot_gamma()



# ?jax.scipy.stats.cauchy.pdf

np.log(bin_heights)

x = np.arange(-3, 3, 0.01)
plt.plot(x, proposal_pdf(x))

# +
sigma = 0.2
mu = 0.25
N=40000
some_shuffled = shuffled_apms_correlation_matrix.T[apms_tril_indices][0:N]
some_normal = np.array(jax.random.normal(rng_key, shape=(N,))*sigma + mu)
alpha_=1.4
beta_=6
some_beta = np.array(jax.random.beta(rng_key, alpha_, beta_, shape=(N,)))

w = 0.4
k = 1.0
M = int(N * w)
O = int(N * k)
#x_ = np.concatenate((some_shuffled[0:M], some_normal[0:O]))
#x_ = np.concatenate((some_shuffled[0:M], some_beta[0:O]))
sigma2=0.03
mu2 = -0.05
norm2 = np.array(jax.random.normal(rng_key, shape=(N,))* sigma2 + mu2)
x_ = np.concatenate((norm2[0:M], some_beta[0:O]))


plt.hist(x_, bins=500, density=True, alpha=alpha, label='synth beta')
plt.hist(x, bins=1000, density=True, alpha=alpha, label='observed r')
plt.legend()
plt.show()
del x_
# -

xtemp = np.arange(0, 20, 0.01)
k=9
theta=0.5
ytemp = sp.stats.gamma(k, theta).pdf(xtemp)
plt.plot(xtemp, ytemp)

plt.hist(some_shuffled, bins=1000)
plt.show()

norm_std = 0.5
norm_mu = 0.2
normal_random_values = np.array(jax.random.normal(rng_key, shape=(10000,)))*norm_std + norm_mu

# +
joint_values = np.concatenate(np.ravel(shuffled_apms_correlation_matrix.T[apms_tril_indices]),
                              normal_random_values)

joint_values = np.shuffle(joint_values)
# -

x = apms_correlation_matrix.T[apms_tril_indices]
x_points = np.arange(np.min(x), np.max(x), 0.05)
y_points = sp.stats.t.pdf(x_points, df=2)
r_points = y_points / np.sqrt(nconditions - 2 + y_points**2)
r_points *= 5000000
plt.hist(x, bins=100)
plt.hist(shuffled_apms_correlation_matrix.T[apms_tril_indices], alpha=0.5, bins=100)
plt.plot(x_points, r_points, label='r: student t df=94')
plt.xlabel("AP-MS Pearson Correlation")
plt.show()

# +


x, a, b = sp.symbols('x a b')  # a and b represent parameters of your PDF
pdf = sp.exp(-x**2 / 2)  # Example: a Gaussian PDF

# Assuming the lower bound is -infinity
cdf = sp.integrate(pdf, (x, -sp.oo, x))
# -

x_points = np.arange(-5, 5, 0.1)
y_points = sp.stats.t.pdf(x_points, df=2)
r_points = y_points / np.sqrt(nconditions - 2 + y_points**2)
plt.plot(x_points, r_points)

sp.stats.t.pdf(x, 10)

# +
# 

nsamples=5000000
rng_key = jax.random.PRNGKey(13)
#ideal_corr_samples = jax.random.t(rng_key, df=DF, shape=(nsamples,))
#ideal_corr_samples = ideal_corr_samples / np.sqrt(235 + ideal_corr_samples**2)

x_points = np.arange(-1, 1, 0.1)
t_94 = sp.stats.norm.pdf(x=x_points)
plt.hist(x, bins=500)
t_94 = t_94 *50000
plt.plot(x_points, t_94)
plt.show()
# -

n, bins, patches = plt.hist(x, bins=1000, density=True)

np.sum(n)

np.arange(-1, 1, 0.1)

plt.hist(xr_apms_correlation_matrix[tril_indices])

# +
# Save the AP-MS and GI data as xarrays
# AP-MS not mapped to uniprot IDs

apms_ids = [reference_preyu2uid[k] for k in stacked_spectral_counts_array.preyu.values]

xr_apms_correlation_matrix = xr.DataArray(apms_correlation_matrix + apms_correlation_matrix.T,
        coords={'uid_preyu': apms_ids,
                'uid_preyv': apms_ids})


xr_hiv_correlation_matrix = xr.DataArray(hiv_corr_matrix + hiv_corr_matrix.T,
        coords={'uid_preyu': hiv_mat.preyu.values, 'uid_preyv': hiv_mat.preyu.values})


# ID Mapping

shared_uids = set(list(xr_apms_correlation_matrix.uid_preyu.values)).intersection(
list(xr_hiv_correlation_matrix.uid_preyu.values))
print(len(shared_uids))



del apms_ids

# +
with open("xr_apms_correlation_matrix.pkl", "wb") as f:
    pkl.dump(xr_apms_correlation_matrix, f)
    
with open("xr_hiv_correlation_matrix.pkl", "wb") as f:
    pkl.dump(xr_hiv_correlation_matrix, f)
# -

# Scoring
"""
p(r| w) = (1-w) * t(r) + w * N(r| 1.0, 0.1)
"""

# Correlation plot
# Crashes
WONT_CRASH = False
if WONT_CRASH:
    apms_at_emap = xr_apms_correlation_matrix.sel(uid_preyu=xr_hiv_correlation_matrix.uid_preyu,
                                                 uid_preyv=xr_hiv_correlation_matrix.uid_preyv)
    _tril_indices = np.tril_indices(apms_at_emap.shape[0], k=-1)
    _a = apms_at_emap[_tril_indices]
    _b = xr_hiv_correlation_matrix[_tril_indices]

    plt.plot(_a.values, _b.values)
    plt.xlabel('AP-MS r')
    plt.ylabel('EMAP r')
    plt.show()

t_gi = sp.stats.t(df=(hiv_mat.shape[0]-2))

# +
c = np.zeros(94)
d = np.zeros(94)

c[2] = 1
d[88] = 88
# -

hiv_tvalues = 1 - t_gi.cdf(hiv_corr_matrix.T[tril_indices])

plt.plot(hiv_pval_matrix.T[tril_indices], hiv_tvalues, 'k.')

help(t_gi.cdf)

hiv_mat.shape

# +
# 1. Take the Union of Nodes from GI and AP-MS data

# 2. Map the Nodes to Uniprot IDs

# 3. Ensure no Uniprot IDs map to the same gene?

# 4. Perform tests for linear correlation with p-value

# +
# AP-MS Linear correlation coefficient



# +
# Save matrices
# -

genes = [uid2gene[k] for k in hiv_mat.preyu.values]

# +
# We can do modeling in UID coordinate space because UIDS are shared between E-Map and AP-MS data


# -

len(set(genes).intersection(set(direct_benchmark.prediction.cosine_similarity.matrix.preyu.values)))

direct_benchmark.prediction.cosine_similarity.matrix.preyu

with open('direc')

# +
fig, ax = plt.subplots()
neg_alpha = 0.5
pos_alpha = 0.5
xlim = 3

ref_array = hiv_ref.values[tril_indices]
gi_score_array = hiv_corr_matrix.T[tril_indices]
positive = gi_score_array[ref_array]
negative = gi_score_array[~ref_array]

plt.hist(negative, label=f'UNK\nN={len(negative)}', bins=100, alpha=neg_alpha, density=True)
plt.hist(positive, label=f'PPI\nN={len(positive)}', bins=100, alpha=pos_alpha, density=True)
plt.xlim(-xlim, xlim)
plt.legend()
plt.show()
print(np.mean(positive))
print(np.median(positive))
print(np.mean(negative))
print(np.median(negative))

ks_alpha = 0.1
c_of_alpha = 1.224
m = len(positive)
n = len(negative)
b = np.sqrt((n + m) / (n * m))
D = c_of_alpha * b

res = sp.stats.ks_2samp(positive, negative)
stat = res.statistic
pvalue = res.pvalue

print(f"pvalue {pvalue} D: {stat}")
# Confidence level 90%
# p-value must be less than 0.1
# Distance - 0.3

# If p < 0.1 reject the null
# If D > 0.145 reject the null

# Conclusion - cannot reject the null.
print("Cannot reject the null")
# -

plt.matshow(hiv_mat.values)

gene2uid = {}
for uid in uid2gene:
    if uid in temp_reference.preyu.values:
        gene = uid2gene[uid]
        if gene in gene2uid:
            print(gene, uid, )
        #assert gene not in gene2uid, (gene, uid)
        gene2uid[gene] = uid

# +
hivdf = pd.read_csv("../data/hiv_emap/data_s1.csv", sep="\t", index_col='Gene')
columns = [k.removesuffix(' - ESIRNA') for k in hivdf.columns]
rows = [k.removesuffix(' - ESIRNA') for k in hivdf.index]


hivdf = pd.DataFrame(hivdf.values, columns = columns, index=rows)

# -

# Remap hiv data
columns = [gene2uid[k.removesuffix(' - ESIRNA')] for k in hivdf.columns]
rows = [[k.removesuffix(' - ESIRNA')] for k in hivdf.index]

# +
columns = np.array(columns)
rows = np.array(rows)
assert np.all(columns == rows)

hivdf = pd.DataFrame(hivdf.values, columns = columns, index=rows)
# -

# Genes map to multiple possible uids so we need to pick the right ones
id_mapping = pd.read_csv("../data/hiv_emap/idmapping_2023_07_05.tsv", sep="\t")
uid2gene = {r['To']: r['From'] for i,r in id_mapping.iterrows()}
gene2uid = {}
for uid in uid2gene:
    if uid in reference_matrix.preyu.values:
        gene = uid2gene[uid]
        assert gene not in gene2uid
        gene2uid[gene] = uid

shared = set(id_mapping['From']).intersection(set(reference_matrix.preyu.values))

# %matplotlib inline
cax = plt.matshow(mat, cmap='seismic')
plt.colorbar(cax, label='GI score')
plt.tight_layout()

mat = xr.DataArray(mat, coords={'preyu': rows, 'preyv': columns})
