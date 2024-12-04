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

import matplotlib.pyplot as plt
import _model_variations as mv
import numpyro
from numpyro.infer import (NUTS, MCMC)
import numpyro.distributions as dist
import numpy as np
import jax
import importlib
from numpyro.infer.util import log_density
importlib.reload(mv)

# +
_pascal = {0: [1],
           1: [1, 1],
           2: [1, 2, 1],
           3: [1, 3, 3, 1],
           4: [1, 4, 6, 4, 1],
           5: [1, 5, 10, 10, 5, 1],
           6: [1, 6, 15, 20, 15, 6, 1],
           7: [1, 7, 21, 35, 35, 21, 7, 1]}

SLOPE = 100

def pascal(n):
    return _pascal[n]
# -



key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model17_test)
mcmc = MCMC(nuts, num_warmup=500, num_samples=1000)

mcmc.run(key, extra_fields=('potential_energy',))

samples = mcmc.get_samples()
extra_fields = mcmc.get_extra_fields()
pe = np.array(extra_fields['potential_energy'])
div = np.array(extra_fields['diverging'])

plt.hist(pe, bins=100)
plt.show()

plt.hist(np.ravel(samples['pT']), bins=100)
plt.xlabel("$\pi_T$")
plt.show()

for w in range(1, 10):
    k = w * 0.01
    x = np.arange(0, 1, 0.01)
    y = np.exp(dist.Beta(k, k).log_prob(x))
    plt.plot(x, y, label=f"k={k}")
    plt.legend()

for w in range(1, 10):
    k = w * 0.1
    x = np.arange(0, 1, 0.01)
    d = dist.Beta(0.01, k)
    y = np.exp(d.log_prob(x))
    plt.plot(x, y, label=f"k={k} mu={round(d.mean, 3)}")
    plt.legend()

alpha = 0.01
beta = 0.4
d = dist.Beta(alpha, beta)
x = np.arange(0, 1, 0.01)
y = np.exp(d.log_prob(x))
plt.plot(x, y)
print(d.mean)

plt.plot(x, jax.nn.sigmoid((x-0.5)*20))

# +
import jax.numpy as jnp
def f(x, N):
    scale = jnp.square(N).astype(jnp.float32)
    return jax.nn.sigmoid((x-N+1.5)*10.0)

# The 


# +
x = np.arange(0, 20, 0.1)
ns = [3, 6, 9, 100, 1000]
for n in range(3, 20):
    if n % 2 == 0:
        y = f(x, n)
        plt.plot(x, y, label=f"n={n}")
        plt.vlines(n-1, 0, 1, color='b', alpha=0.5)
        
plt.xlabel("Sum of edges")
plt.legend()
# -

ns = [3, 6, 9, 100, 1000]
for n in range(3, 20):
    if n % 2 == 0:
        y = f(x, n)
        plt.plot(x, y, label=f"n={n}")
        plt.vlines(n-2, 0, 1, color='r', alpha=0.5)        
plt.xlabel("Sum of edges")
plt.legend()

# +

x = np.arange(490, 520)
for n in range(500, 510):
    if n % 2 == 0:
        y = f(x, n)
        plt.plot(x, y, label=f"n={n}")
        plt.vlines(n-1, 0, 1, color='r')
plt.xlabel("Sum of edges")
plt.legend()
# -

importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model18_test7)
mcmc = MCMC(nuts, num_warmup=500, num_samples=1000)
mcmc.run(key, extra_fields=('potential_energy',))

samples = mcmc.get_samples()
extra_fields = mcmc.get_extra_fields()
pe = np.array(extra_fields['potential_energy'])
div = np.array(extra_fields['diverging'])

results = plt.hist(np.ravel(samples['e']), bins=100)
plt.show()

mcmc.print_summary(exclude_deterministic=False)

np.sum(samples['e'] > 0.2) / len(np.ravel(samples['e']))

dist.Beta(0.01, 0.4).mean

edge_arr = np.array([0, 1, 5, 8, 7, 9])

plt.hist(samples['pT'][:, 14], bins=100, alpha=0.5)
plt.hist(np.ravel(samples['pT'][:, edge_arr]), bins=100, label='Composite1', alpha=0.5)
plt.show()

key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model18_test)
mcmc = MCMC(nuts, num_warmup=500, num_samples=5000)
mcmc.run(key, 0.2, extra_fields=('potential_energy',))

samples = mcmc.get_samples()
extra_fields = mcmc.get_extra_fields()
pe = np.array(extra_fields['potential_energy'])
div = np.array(extra_fields['diverging'])

plt.hist(pe, bins=100)
plt.show()

plt.hist(np.ravel(samples['pT']), bins=100)
plt.xlabel("$\pi_T$")
plt.show()

plt.hist(samples['pT'][:, 14], bins=100, alpha=0.5)
plt.hist(np.ravel(samples['pT'][:, edge_arr]), bins=100, label='Composite1', alpha=0.5)
plt.show()

plt.hist(samples['pT'][:, 0], bins=100)
plt.show()

plt.hist(samples['pT'][:, 22], bins=100)
plt.show()

key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model18_test2)
mcmc = MCMC(nuts, num_warmup=500, num_samples=5000)
mcmc.run(key, 0.2, extra_fields=('potential_energy',))

samples = mcmc.get_samples()
extra_fields = mcmc.get_extra_fields()
pe = np.array(extra_fields['potential_energy'])
div = np.array(extra_fields['diverging'])

plt.hist(pe, bins=100)
plt.show()

plt.hist(np.ravel(samples['pT']), bins=100)
plt.xlabel("$\pi_T$")
plt.show()

plt.hist(samples['pT'][:, 14], bins=100, alpha=0.5)
plt.hist(np.ravel(samples['pT'][:, edge_arr]), bins=100, label='Composite1', alpha=0.5)
plt.show()

edge_arr

mcmc.print_summary()

np.mean(samples['pT'][:, edge_arr], axis=0)

# Set the weight and the scale
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model18_test3)
mcmc = MCMC(nuts, num_warmup=500, num_samples=5000)
mcmc.run(key, 0.2, 10, extra_fields=('potential_energy',))


# +

def model(y):
    ap = numpyro.sample("ap", dist.Normal(-1.5, 1))
    al = numpyro.sample("al", dist.Normal(1, 0.5))
    p = expit(ap)
    lambda_ = jnp.exp(al)
    log_prob = jnp.log1p(-p) + dist.Poisson(lambda_).log_prob(y)
    numpyro.factor("y|y>0", log_prob[y > 0])
    numpyro.factor("y|y==0", jnp.logaddexp(jnp.log(p), log_prob[y == 0]))


# -

mcmc.print_summary()

# Set the weight and the scale
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model18_test4)
mcmc = MCMC(nuts, num_warmup=500, num_samples=5000)
mcmc.run(key, 0.2, 10, extra_fields=('potential_energy',))

mcmc.print_summary()

samples = mcmc.get_samples()
extra_fields = mcmc.get_extra_fields()
pe = np.array(extra_fields['potential_energy'])
div = np.array(extra_fields['diverging'])

plt.hist(np.ravel(samples['x']), bins=100)
plt.show()

# Set the weight and the scale
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model18_test5)
mcmc = MCMC(nuts, num_warmup=500, num_samples=5000)
mcmc.run(key, extra_fields=('potential_energy',))

samples = mcmc.get_samples()
extra_fields = mcmc.get_extra_fields()
pe = np.array(extra_fields['potential_energy'])
div = np.array(extra_fields['diverging'])

# Set the weight and the scale
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model18_test6)
mcmc = MCMC(nuts, num_warmup=500, num_samples=1000)
mcmc.run(key, 0.1, 2, 10, extra_fields=('potential_energy',))

mcmc.print_summary(exclude_deterministic=False)

importlib.reload(mv)

scale = 5
x = np.arange(-2, 2, 0.001) * scale
y = jax.nn.sigmoid(x * 100)
plt.plot(x, y)
plt.xlabel("")

# Set the weight and the scale
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model19_test1)
mcmc = MCMC(nuts,num_warmup=500, num_samples=1000)
mcmc.run(key, 20, 6, 15, extra_fields=('potential_energy',))

mcmc.print_summary(exclude_deterministic=False)

# Set the weight and the scale
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model19_test2)
mcmc = MCMC(nuts,num_warmup=500, num_samples=1000)
mcmc.run(key, 20, 6, 15, extra_fields=('potential_energy',))

mcmc.print_summary(exclude_deterministic=False)

samples = mcmc.get_samples()
extra_fields = mcmc.get_extra_fields()
pe = np.array(extra_fields['potential_energy'])
div = np.array(extra_fields['diverging'])

plt.hist(jax.nn.sigmoid(np.ravel(samples['e_at_cc1'])), bins=100)
plt.show()


def composite_connect(x, n_nodes, mu_edge = 0.5):
    m_edges = math.comb(n_nodes, 2)
    mu_x = mu_edge * m_edges
    y = jax.nn.sigmoid(x - n_nodes - mu_x)
    return y
    


from functools import partial
import math
f = partial(composite_connect, n_nodes=20)
x = np.arange(0, 190, 1)
y = f(x)
plt.plot(x, y)

# Set the weight and the scale
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test4)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, extra_fields=('potential_energy',))

# +
"""
model20_test1:
  A :: (5, 5)
  E :: (10,)
  [0, 1, 2]
  [1, 2, 3]
model20_test2:
  A :: (20, 20)
  E :: (10,)
  [0, 1, 2]
  [1, 2, 3, 4]
  [1, 2, 7, 9, 11]

model20_test2:
  A :: (20, 20)
  E :: (10,)
  [0, 1, 2]
  [1, 2, 3, 4]
  [1, 2, 7, 9, 11]
  [1, 2]
  
model20_test4:
  A :: (3, 3)
  E :: 3
  c1 :: [0, 1]
  

For the following composites
[0, 1, 2]
[1, 2, 3, 4]
[1, 2, 7, 9, 11]
"""


mcmc.print_summary(exclude_deterministic=False)
# -

samples = mcmc.get_samples()
fig, ax = plt.subplots(3, 1)
for i in range(3):
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k')
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$\pi_T$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()

# +
# Set the weight and the scale
eidx = 0
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test6a)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, eidx, 1., extra_fields=('potential_energy',))

samples = mcmc.get_samples()
fig, ax = plt.subplots(3, 1)
for i in range(3):
    label = "Composite" if i == eidx else None
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k', label=label)
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$\pi_T$")

plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()
# -

"""
Two composites of size 2
[0, 1]
[0, 2]

Vs 1 composite of size 3
[0, 1, 2]
"""

# +
# Set the weight and the scale
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test6a)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, 1, 1., extra_fields=('potential_energy',))
samples = mcmc.get_samples()

fig, ax = plt.subplots(3, 1)
for i in range(3):
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k')
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$\pi_T$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()

# +
# V={0, 1, 2} Cm={0, 2}
# Set the weight and the scale
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test6a)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, 1, 8., extra_fields=('potential_energy',))
samples = mcmc.get_samples()

fig, ax = plt.subplots(3, 1)
for i in range(3):
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k')
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$\pi_T$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()

# +
# V={0, 1, 2} {Cm} = {{0,2}{1,2}}
# Set the weight and the scale
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test6c)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, 8., extra_fields=('potential_energy',))
samples = mcmc.get_samples()

fig, ax = plt.subplots(3, 1)
for i in range(3):
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k')
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$\pi_T$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()

# +
solution_base_x = np.arange(1, 7)
solution_base_y = np.array(pascal(5)) / 32.
solution_base_y = 5000 * solution_base_y

plt.hist(np.sum(samples['e'], axis=1), color='k', bins=100, alpha=0.5)
plt.plot(solution_base_x, solution_base_y, 'bx', label="Uniform over N=3 Networks")
plt.xlabel("Sum of edge weights in a microstate")
plt.ylabel("Frequency")
#plt.vlines(1.99, 0, 3000, label='Satiscation N-1', color='r')
plt.title("V={0, 1, 2, 4}, Cm={0, 1, 2, 4}")
plt.legend()
plt.show()

# +
# V={0, 1, 2, 3} {Cm} = {0, 1, 2, 3}
# Set the weight and the scale
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test6c)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, 100., extra_fields=('potential_energy',))
samples = mcmc.get_samples()

fig, ax = plt.subplots(6, 1)
for i in range(6):
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k')
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$\pi_T$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()
#plt.close()

# +
solution_base_x = np.array([0, 1, 2, 3, 4, 5 ,6])
solution_base_y = np.array([1, 6, 15, 20, 15, 6, 1])
solution_base_y = solution_base_y / np.sum(solution_base_y)
solution_base_y = 5000 * solution_base_y

fig, ax = plt.subplots(1, 2, sharex=True)
results = ax[0].hist(np.sum(samples['e'], axis=1), color='k', bins=100, alpha=0.5)
ax[0].plot(solution_base_x, solution_base_y, 'bx', label="Uniform over N=4 Networks")
#ax[0].set_xlabel("Sum of edge weights in a microstate")
ax[0].set_ylabel("Frequency")
#plt.vlines(1.99, 0, 3000, label='Satiscation N-1', color='r')
ax[0].set_title("V={0, 1, 2, 3}, Cm={0, 1, 2, 3}")
#ax[0].legend()
bin_heights = results[0]
residuals = bin_heights[bin_heights > 100]
residuals = [0, 0, 0] + list(residuals)
residuals = np.array(residuals)
residuals = residuals - solution_base_y
ax[1].plot(solution_base_x, residuals, 'k^')
ax[1].set_ylabel("Residual (observed - expected)")
fig.text(0.5, -0.04, 'Sum of edge weights in a microstate', ha='center')
plt.tight_layout()

plt.show()

# +
# V={0, 1, 2, 3} {Cm} = {0, 1, 2, 3}
# Set the weight and the scale
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test6d)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, 100., extra_fields=('potential_energy',))
samples = mcmc.get_samples()

fig, ax = plt.subplots(6, 1)
for i in range(6):
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k')
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$\pi_T$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()
#plt.close()

# +
solution_base_x = np.array([0, 1, 2, 3, 4, 5 ,6])
solution_base_y = np.array([1, 6, 15, 20, 15, 6, 1])
solution_base_y = solution_base_y / np.sum(solution_base_y)
solution_base_y = 5000 * solution_base_y

fig, ax = plt.subplots(1, 2, sharex=True)
results = ax[0].hist(np.sum(samples['e'], axis=1), color='k', bins=100, alpha=0.5)
ax[0].plot(solution_base_x, solution_base_y, 'bx', label="Uniform over N=4 Networks")
#ax[0].set_xlabel("Sum of edge weights in a microstate")
ax[0].set_ylabel("Frequency")
#plt.vlines(1.99, 0, 3000, label='Satiscation N-1', color='r')
ax[0].set_title("V={0, 1, 2, 3}, C0={0, 1, 2} C1={1,2,3}")
bin_heights = results[0]
residuals = bin_heights[bin_heights > 100]
residuals = [0, 0, 0] + list(residuals)
residuals = np.array(residuals)
residuals = residuals - solution_base_y
ax[1].plot(solution_base_x, residuals, 'k^')
ax[1].set_ylabel("Residual (observed - expected)")
fig.text(0.5, -0.04, 'Sum of edge weights in a microstate', ha='center')
plt.tight_layout()

plt.show()
#ax[0].legend()

# +
# V={0, 1, 2, 3} {Cm} = {0, 1, 2, 3}
# Set the weight and the scale
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test6e_1a)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, SLOPE, 10., extra_fields=('potential_energy',))
samples = mcmc.get_samples()

fig, ax = plt.subplots(6, 1)
for i in range(6):
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k')
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$\pi_T$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()
#plt.close()
# -

plt.hist(np.sum(samples['e'], axis=1), color='k', bins=200, alpha=0.5)
y = np.array([1, 6, 15, 20, 15, 6, 1])
y = y / np.sum(y)
y = y * 5000
x = np.arange(7)
plt.plot(x, y, 'b^')
plt.title("V={0, 1, 2, 3}; Cm={0, 1, 2}")
plt.xlabel("Sum of edges in microstate i")
plt.show()
print("Losing solutions in the tails")

# +
# V={0, 1, 2, 3} {Cm} = {0, 1, 2, 3}
# Set the weight and the scale
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test6e_1b)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, extra_fields=('potential_energy',))
samples = mcmc.get_samples()

fig, ax = plt.subplots(6, 1)
for i in range(6):
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k')
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$\pi_T$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()
#plt.close()
# -

plt.hist(np.sum(samples['e'][:, [0, 1, 3]], axis=1), color='k', bins=200, alpha=0.5)
y = np.array([1, 3, 3, 1])
y = y / np.sum(y)
y = y * 5000
x = np.arange(4)
plt.plot(x, y, 'b^')
plt.title("V={0, 1, 2, 3}; Cm={0, 1, 2}")
plt.xlabel("Sum of edges in microstate i")
plt.show()

# V={0, 1, 2, 3} {Cm} = {0, 1, 2, 3}
# Set the weight and the scale
model14_data = mv.model14_data_getter()
print(model14_data.keys())


def experiment_sample_and_savefig(
    lower_bound = 0,
    upper_bound = 0.5,
    pc0 = 0,
    pc1 = 0,
    plot_and_save=True):

    data = {"N": 4,
            "lower_edge_prob_bound": lower_bound,
            "upper_edge_prob_bound": upper_bound,
            "z2edge_slope": 1000,
            "composites": [mv.BaitPreyInfo(np.array([0, 1, 2]), 3, 8, pc0),
                           mv.BaitPreyInfo(np.array([0, 1, 2, 3, 4, 5]), 4, 8, pc1)],
            "flattened_apms_similarity_scores": model14_data['flattened_apms_similarity_scores'][0:6],
            "flattened_apms_shuffled_similarity_scores": model14_data['flattened_apms_shuffled_similarity_scores'],
            "BAIT_PREY_SLOPE": 20,}

    importlib.reload(mv)
    num_samples = 10_000
    key = jax.random.PRNGKey(13)
    nuts = NUTS(mv.model22_ll_lp)
    mcmc = MCMC(nuts,num_warmup=1000, num_samples=num_samples)
    mcmc.run(key, data, extra_fields=('potential_energy',))
    samples = mcmc.get_samples()
    
    edges = jax.nn.sigmoid((samples['z']-0.5)*data['z2edge_slope'])
    if plot_and_save:
        plt.hist(np.sum(edges, axis=1), color='k', bins=200, alpha=0.5)
        y = np.array([1, 6, 15, 20, 15, 6, 1])
        y = y / np.sum(y)
        y = y * num_samples
        x = np.arange(7)
        plt.plot(x, y, 'b^', label="Pascal expectation")
        plt.title(f"V={0, 1, 2, 3}; C0={0,1,2}({pc0}) C1={0,1,2,3}({pc1}) BR {lower_bound}-{upper_bound}")
        plt.xlabel("Sum of edges in microstate i")
        plt.ylabel("Frequency")
        plt.ylim(0, 7000)
        plt.xlim(-0.5, 6.5)
        plt.legend()
        plt.savefig(f"ExampleCompositeConnectivity_V4_C0({pc0})C1{pc1}_BR({lower_bound}_{upper_bound}).png", dpi=300)
        plt.show()
        print("Losing solutions in the tails")
    return samples


# +
    
pc0 = [0, 0.5, 1.]
pc1 = [0, 0.5, 1.]

lower_bound = [0, 0.5]
upper_bound = [0.1, 0.51, 1.0]

for p0 in pc0:
    for p1 in pc1:
        for lb in lower_bound:
            for ub in upper_bound:
                if lb < ub:
                    experiment_sample_and_savefig(
                        lower_bound=lb,
                        upper_bound=ub,
                        pc0=p0,
                        pc1=p1)


    
# -

samples = experiment_sample_and_savefig(0.5, 0.51, 1, 0, plot_and_save=False)

fig, ax = plt.subplots(6, 1, sharex=True)
for i in range(6):
    ax[i].hist(np.array(samples['z'][:, i]), alpha=0.8, color='k', bins=100)
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$z$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()
#plt.close()

edges = jax.nn.sigmoid((samples['z']-0.5)*data['z2edge_slope'])
fig, ax = plt.subplots(6, 1)
for i in range(6):
    ax[i].hist(np.array(edges[:, i]), alpha=0.8, color='k')
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$e$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()
print(f"Base Edge Rate {np.sum(edges) / np.prod(edges.shape)}")
#plt.close()

plt.hist(np.array(samples['mu']), color='k', alpha=0.8)
plt.ylabel("Frequency")
plt.xlabel("Mu")
plt.show()

# Plot distance to Bait
edges = jax.nn.sigmoid((samples['z']-0.5)*20)
# Over all solutions count the distance to the prey
r = mv.BaitPreyConnectivity(np.array([0, 1, 2]), 3, maximal_shortest_path_to_calculate=8,
                           bait_prey_slope=20)
distances_of_cc0 = []
#distances_of_all
for Alist in edges:
    A = r.get_dense_matrix_from_edge_weight_lst(Alist[0:3])
    A = r.weight2binary(A)
    D = r.apsp_up_to_dmax(A)
    Distance2bait = D[0, :]
    Distance2bait = Distance2bait.at[0].set(0)
    d = Distance2bait.tolist()
    distances = distances + d

distances

Distance2bait

Alist[0:3]



# +
plt.hist(np.sum(edges, axis=1), color='k', bins=200, alpha=0.5)


y = np.array([1, 6, 15, 20, 15, 6, 1])
y = y / np.sum(y)
y = y * num_samples
x = np.arange(7)
plt.plot(x, y, 'b^', label="Pascal expectation")
plt.title(f"V={0, 1, 2, 3}; C0={0,1,2}({pc0}) C1={0,1,2,3}({pc1}) BR {lower_bound}-{upper_bound}")
plt.xlabel("Sum of edges in microstate i")
plt.ylabel("Frequency")
plt.ylim(0, 7000)
plt.xlim(-0.5, 6.5)
plt.legend()
plt.savefig(f"ExampleCompositeConnectivity_V4_C0({pc0})C1{pc1}_BR({lower_bound}_{upper_bound}).png", dpi=300)
plt.show()
print("Losing solutions in the tails")
# -

plt.hist(np.array(samples['lp_score']), bins=100)
plt.show()



plt.hist(np.array(edges.T))
plt.show()

# +
# V={0, 1, 2, 3} {Cm} = {0, 1, 2, 3}
# Set the weight and the scale
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test6e)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, 20., 10., extra_fields=('potential_energy',))
samples = mcmc.get_samples()

fig, ax = plt.subplots(6, 1)
for i in range(6):
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k')
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$\pi_T$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()
#plt.close()
# -

solution_base_x = np.array([0, 1, 2, 3, 4, 5 ,6])
solution_base_y = np.array([1, 6, 15, 20, 15, 6, 1])
solution_base_y = solution_base_y / np.sum(solution_base_y)
solution_base_y = 5000 * solution_base_y
plt.close()
fig, ax = plt.subplots(1, 2, sharex=True)
results = ax[0].hist(np.sum(samples['e'], axis=1), color='k', bins=100, alpha=0.5)
#ax[0].plot(solution_base_x, solution_base_y, 'bx', label="Uniform over N=4 Networks")
#ax[0].set_xlabel("Sum of edge weights in a microstate")
ax[0].set_ylabel("Frequency")

# +
#plt.vlines(1.99, 0, 3000, label='Satiscation N-1', color='r')
ax[0].set_title("V={0, 1, 2, 3}, Cm={0, 1, 2, 3}")
bin_heights = results[0]
residuals = bin_heights[bin_heights > 100]
residuals = [0, 0, 0] + list(residuals)
residuals = np.array(residuals)
residuals = residuals - solution_base_y
ax[1].plot(solution_base_x, residuals, 'k^')
ax[1].set_ylabel("Residual (observed - expected)")
fig.text(0.5, -0.04, 'Sum of edge weights in a microstate', ha='center')
plt.tight_layout()

plt.show()
#ax[0].legend()
# -

N=4
M = math.comb(N, 2)
x = np.arange(0, M, 0.1) - N
y = jax.nn.sigmoid((x+0.5)*10)
plt.vlines(0, 0, 1, 'r')
plt.plot(x, y)

# +
solution_base_x = np.array([0, 1, 2, 3, 4, 5 ,6])
solution_base_y = np.array([1, 6, 15, 20, 15, 6, 1])
solution_base_y = solution_base_y / np.sum(solution_base_y)
solution_base_y = 5000 * solution_base_y

fig, ax = plt.subplots(1, 2, sharex=True)
results = ax[0].hist(np.sum(samples['e'], axis=1), color='k', bins=100, alpha=0.5)
ax[0].plot(solution_base_x, solution_base_y, 'bx', label="Uniform over N=4 Networks")
#ax[0].set_xlabel("Sum of edge weights in a microstate")
ax[0].set_ylabel("Frequency")
#plt.vlines(1.99, 0, 3000, label='Satiscation N-1', color='r')
ax[0].set_title("V={0, 1, 2, 3}, Cm={0, 1, 2, 3}")
#ax[0].legend()
bin_heights = results[0]
residuals = bin_heights[bin_heights > 100]
residuals = [0, 0, 0] + list(residuals)
residuals = np.array(residuals)
residuals = residuals - solution_base_y
ax[1].plot(solution_base_x, residuals, 'k^')
ax[1].set_ylabel("Residual (observed - expected)")
fig.text(0.5, -0.04, 'Sum of edge weights in a microstate', ha='center')
plt.tight_layout()

plt.show()
# -

samples['e'].shape

# +
# V={0, 1, 2} {Cm} = {{0,2}{1,2}}
# Set the weight and the scale
importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test6b)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, 1, 8., extra_fields=('potential_energy',))
samples = mcmc.get_samples()

fig, ax = plt.subplots(3, 1)
for i in range(3):
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k')
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$\pi_T$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()

# +
solution_base_x = np.array([0, 1, 2, 3])
solution_base_y = np.array([1, 3, 3, 1])
solution_base_y = solution_base_y / np.sum(solution_base_y)
solution_base_y = 5000 * solution_base_y

fig, ax = plt.subplots()
results = ax.hist(np.sum(samples['e'], axis=1), color='k', bins=100, alpha=0.5)
ax.plot(solution_base_x, solution_base_y, 'bx', label="Uniform over N=4 Networks")
ax.set_xlabel("Sum of edge weights in a microstate")
ax.set_ylabel("Frequency")
#plt.vlines(1.99, 0, 3000, label='Satiscation N-1', color='r')
ax.set_title("V={0, 1, 2}, Cm={0, 2} Cm={1, 2}")
#ax[0].legend()
plt.tight_layout()

plt.show()
# -

data = np.array(samples['e'])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.scatter(xs=data[:, 0], 
                     ys=data[:, 1], 
                     zs=data[:, 2])
ax.set_xlabel("Edge (0, 1)")
ax.set_ylabel("Edge (0, 2)")
ax.set_zlabel("Edge (1, 2)")
plt.show()

data = np.array(jax.nn.sigmoid((data-0.5)*5000))

data

plt.hist(np.sum(data, axis=1))
plt.show()

# +
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test6c)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, 8., extra_fields=('potential_energy',))
samples = mcmc.get_samples()

fig, ax = plt.subplots(3, 1)
for i in range(4):
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k')
    if i == 1:
        ax[i].set_ylabel("Frequency")
    plt.xlabel("$\pi_T$")
plt.suptitle("Bait Prey Connectivity")
plt.tight_layout()

# +
solution_base_x = np.array([0, 1, 2, 3, 4, 5, 6])
solution_base_y = np.array([1, 6, 15, 20, 15, 6, 1])
solution_base_y = solution_base_y / np.sum(solution_base_y)
solution_base_y = 5000 * solution_base_y

fig, ax = plt.subplots()
results = ax.hist(np.sum(samples['e'], axis=1), color='k', bins=100, alpha=0.5)
ax.plot(solution_base_x, solution_base_y, 'bx', label="Uniform over N=4 Networks")
ax.set_xlabel("Sum of edge weights in a microstate")
ax.set_ylabel("Frequency")
#plt.vlines(1.99, 0, 3000, label='Satiscation N-1', color='r')
ax.set_title("V={0, 1, 2, 3}, Cm={0, 1, 2, 3}")
#ax[0].legend()
plt.tight_layout()

plt.show()
# -

importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.model20_test6d)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mcmc.run(key, 8., extra_fields=('potential_energy',))
samples = mcmc.get_samples()

n_edges = 6
fig, ax = plt.subplots(n_edges, 1)
for i in range(n_edges):
    ax[i].hist(np.array(samples['e'][:, i]), alpha=1.0, color='k')
    nodes = mv.ij_from(i, 4)
    ax[i].set_ylabel(f"{nodes}", rotation=90)
    ax[i].yaxis.set_label_position("right")
    plt.xlabel("$\pi_T$")
plt.suptitle("V={0, 1, 2, 3}; C0={0, 1, 2}, C1={1, 2, 3}")
plt.tight_layout()

# +
solution_base_x = np.array([0, 1, 2, 3, 4, 5, 6])
solution_base_y = np.array([1, 6, 15, 20, 15, 6, 1])
solution_base_y = solution_base_y / np.sum(solution_base_y)
solution_base_y = 5000 * solution_base_y

fig, ax = plt.subplots()
results = ax.hist(np.sum(samples['e'], axis=1), color='k', bins=100, alpha=0.5)
ax.plot(solution_base_x, solution_base_y, 'bx', label="Uniform over N=4 Networks")
ax.set_xlabel("Sum of edge weights in a microstate")
ax.set_ylabel("Frequency")
#plt.vlines(1.99, 0, 3000, label='Satiscation N-1', color='r')
ax.set_title("V={0, 1, 2, 3}, C0={0, 1, 2} C1={1, 2, 3}")
#ax[0].legend()
plt.tight_layout()

# -

importlib.reload(mv)
key = jax.random.PRNGKey(13)
nuts = NUTS(mv.test_dynamic_model)
mcmc = MCMC(nuts,num_warmup=1000, num_samples=5000)
mus = jnp.arange(4)
mcmc.run(key, [0, 1, 2], extra_fields=('potential_energy',))
samples = mcmc.get_samples()
