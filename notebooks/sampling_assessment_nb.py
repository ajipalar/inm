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

import sampling_assessment as sa
import importlib
import numpy as np
import matplotlib.pyplot as plt
import _model_variations as mv
import numpyro
importlib.reload(sa)
# %matplotlib inline

# +
def plot_rhat_results(rhat_results, name="AssessmentOfSampling", title=""):
    plt.hist(rhat_results['z'], bins=100)
    plt.xlabel("R hat")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.vlines(1.1, 0, 10_000, 'r', label="cut-off")
    plt.vlines(1.05, 0, 10_000, 'k', label="strict")
    plt.legend()
    plt.ylim(0, 12_000)
    plt.xlim(1, 3)
    plt.savefig(f"{name}_300.png", dpi=300)
    plt.savefig(f"{name}_1200.png", dpi=1200)
    plt.show()
    print(np.mean(rhat_results['z']), np.median(rhat_results['z']))
    
def two_chain_catterpillar(r1, r2, name="TwoChain", dpi=300):
    plt.plot(r1.mcmc['extra_fields']['potential_energy'], alpha=0.5)
    plt.plot(r2.mcmc['extra_fields']['potential_energy'], alpha=0.5)
    plt.xlabel("MCMC Step")
    plt.ylabel("Score")
    plt.savefig(f"{name}_{dpi}.png", dpi=dpi)


# -

r1  = sa.get_results(sa._e1)
r2  = sa.get_results(sa._e2, rseed=9)

r3 = sa.get_results("../results/se_sr_low_prior_1_uniform_mock_100k_rseed_1998/", rseed=1998)
r4 = sa.get_results("../results/se_sr_low_prior_1_uniform_mock_100k_rseed_1999/", rseed=1999)

r_mini = sa.get_results("../results/mini_se_sr_low_prior_1_uniform/")

r_mini_1 = sa.get_results("../results/mini_se_sr_low_prior_1_uniform_rseed_1/", rseed=1)

r_mini_warmup_1000 = sa.get_results("../results/mini_se_sr_low_prior_1_uniform_num_warmup_1000/")

r_5 = sa.get_results("../results/se_sr_low_prior_2_uniform_mock_20k")

r_6 = sa.get_results("../results/se_sr_low_prior_1_uniform_mock_20k")

r_7 = sa.get_results("../results/se_sr_low_prior_1_uniform_mock_20k_rseed_9/", rseed=9)

concatenated_samples2 = sa.concatenate_samples(r3.samples, r4.samples)

concatenated_samples = sa.concatenate_samples(r1.samples, r2.samples)

rhat2_results = sa.get_rhat_results(concatenated_samples2)

# +
# Compute Rhat for each parameter
rhat_results = sa.get_rhat_results(concatenated_samples)

# Print Rhat results
#print("Rhat diagnostics:")
#for param, rhat in rhat_results.items():
#    print(f"{param}: Rhat = {rhat}")

# -

two_chain_catterpillar(r1, r2)

two_chain_catterpillar(r3, r4, name="TwoChain_100k_thin5")

auto_correlation = numpyro.diagnostics.autocorrelation(concatenated_samples['u'][0, :])

plt.plot(numpyro.diagnostics.autocorrelation(concatenated_samples['z'][0, :, 100:102], axis=0)[:, 0])

np.corrcoef(concatenated_samples['z'][:, 0, 0:2])

# +
a = concatenated_samples['z'][0, 0, :]
b = concatenated_samples['z'][1, 0, :]


# -

import pandas as pd


# Initial configuration 1
def matrix_plot(r, name):
    cols = [val for key, val in r.model_data['node_idx2name'].items()]
    df = pd.DataFrame(
        mv.flat2matrix(mv.Z2A(r.samples['z'][0, :]), n=r.model_data['N']),
        columns = cols,
        index = cols)

    df = df.sort_index(axis = 1)
    df = df.sort_index(axis = 0)
    plt.matshow(df.values)
    plt.savefig(name, dpi=300)


matrix_plot(r1, "Chain1InitialPositionMatrixPlot")

matrix_plot(r2, "Chain2InitialPositionMatrixPlot")

matrix_plot(r3, "InitialPosition_se_sr_low_prior_1_uniform_mock_100k_rseed_1998")

matrix_plot(r4, "InitialPosition_se_sr_low_prior_1_uniform_mock_100k_rseed_1999")

matrix_plot(r_mini, "InitialPosition_mini_se_sr_low_prior_1_uniform")

matrix_plot(r_mini_warmup_1000, "InitialPosition_mini_se_sr_low_prior_1_uniform_num_warmup_1000")

plt.plot(r_mini.mcmc['extra_fields']['potential_energy'], 'o', label="10 warmup steps. Rseed=13")
plt.plot(r_mini_1.mcmc['extra_fields']['potential_energy'], 'o', label="10 warmup steps. Rseed=1")
plt.plot(r_mini_warmup_1000.mcmc['extra_fields']['potential_energy'], 'o', label="1000 warmup steps. Rseed=13")
plt.ylabel("Score")
plt.xlabel("MCMC step")
plt.legend()
plt.tight_layout()
plt.savefig("EffectOfWarmupPhase", dpi=300)

plt.plot(0, r_mini.hmc_warmup.adapt_state.step_size, 'o')
plt.plot(1, r_mini_1.hmc_warmup.adapt_state.step_size, 'o')
plt.plot(2, r_mini_warmup_1000.hmc_warmup.adapt_state.step_size, 'o')

r_mini_warmup_1000.hmc_warmup.num_steps

plt.hist(r1.mcmc['extra_fields']['accept_prob'])
plt.show()

np.mean(r1.mcmc['extra_fields']['accept_prob'])

plt.plot(r_mini.mcmc['extra_fields']['num_steps'])
plt.plot(r_mini_1.mcmc['extra_fields']['num_steps'])
plt.plot(r_mini_warmup_1000.mcmc['extra_fields']['num_steps'])



# As above but with 1000 warmup steps


matrix_plot(r_mini_1, "InitialPosition_mini_se_sr_low_prior_1_uniform_rseed_1")



matrix_plot(r_5, "InitialPosition_se_sr_low_prior_2_uniform_mock_20k")

matrix_plot(r_6, "InitialPosition_se_sr_low_prior_1_uniform_mock_20k")

matrix_plot(r_7, "InitialPosition_se_sr_low_prior_1_uniform_mock_20k_rseed_9")

two_chain_catterpillar(r_5, r_5, "demo")

two_chain_catterpillar(r_6, r_7, "TwoChainCat_low_prior_1_uniform_mock_20k_vs_rseed_9")

# Check the mini dev run
r_mini = 

# +
# Initial configuration 1
cols = [val for key, val in r2.model_data['node_idx2name'].items()]
df1 = pd.DataFrame(
    mv.flat2matrix(mv.Z2A(concatenated_samples['z'][1, 0, :]), n=r2.model_data['N']),
    columns = cols,
    index = cols)
                  
df1 = df1.sort_index(axis = 1)
df1 = df1.sort_index(axis = 0)
plt.matshow(df1.values)
plt.savefig("Chain2_initial_position", dpi=300)
# -



plt.matshow(df1.sort_index(axis=0).values)

help(df1.sort_index)

r1.model_data['node_idx2name']

r2.model_data['node_idx2name']

plt.hist(mv.Z2A(concatenated_samples['z'][0, 0, :]), bins=100)
plt.hist(mv.Z2A(concatenated_samples['z'][1, 0, :]) + 0.1, bins=100)
plt.show()

sp.stats.pearsonr(a, b)

# +
plt.title("Auto correlation of the parameter u")
plt.plot(numpyro.diagnostics.autocorrelation(concatenated_samples['u'][0, :]))
plt.plot(numpyro.diagnostics.autocorrelation(concatenated_samples['u'][1, :]))
plt.ylabel("Auto correlation")
plt.xlabel("Monte carlo step")

plt.ylim(-1.1, 1.1)
# -

plt.title("Auto correlation of the first edge")
plt.plot(numpyro.diagnostics.autocorrelation(concatenated_samples['z'][0, :, 0]))
plt.plot(numpyro.diagnostics.autocorrelation(concatenated_samples['z'][1, :, 0]))
plt.ylabel("Auto correlation")
plt.xlabel("Monte carlo step")
plt.ylim(-1, 1)

# +
plt.plot(mv.Z2A(concatenated_samples['z'][0, :, 0]) + 0.1, 'k.', alpha=0.5, label="First Edge")
plt.plot(mv.Z2A(concatenated_samples['z'][0, :, 1]), 'r.', alpha=0.5, label="Second Edge")

#plt.plot(mv.Z2A(concatenated_samples['z'][1, :, 0]) + 0.1, 'g.', alpha=0.5, label="First Edge")
#plt.plot(mv.Z2A(concatenated_samples['z'][1, :, 1]), 'y.', alpha=0.5, label="Second Edge")
# -

plt.plot(mv.Z2A(concatenated_samples['z'][0, :, 0]), mv.Z2A(concatenated_samples['z'][1, :, 0]), 'k.')
plt.plot(mv.Z2A(concatenated_samples['z'][0, :, 1]), mv.Z2A(concatenated_samples['z'][1, :, 1]), 'r.')

plt.plot(mv.Z2A(concatenated_samples['z'][0, :, 1]), 'k.')
plt.plot(mv.Z2A(concatenated_samples['z'][1, :, 1]), 'r.')

np.max(numpyro.diagnostics.autocorrelation(concatenated_samples['u'][1, :]))

concatenated_samples['u']

auto_correlation[2]

a = r3.mcmc['extra_fields']['energy']
b = r3.mcmc['extra_fields']['potential_energy']
plt.plot(a, alpha=0.5, label="energy")
plt.plot(b, label="potential (score)", alpha=0.5)
plt.plot(a - b, label="kinetic energy", alpha=0.5)
plt.legend()



plot_rhat_results(rhat2_results, name="TwoChain100k", title="TwoChain100k thin 5")

plot_rhat_results(rhat_results, name="TwoChain20k", title="TwoChain20k")



nbins = 100
plt.hist(rhat_results['z'], bins=nbins,)
plt.hist(rhat2_results['z'], bins=nbins, )
#plt.ylim(0, 1000)
plt.xlim(.9, 3.)
plt.show()

np.sum(np.isnan(rhat2_results['z']))

len(rhat_results['z'])

rhat_after_2_5k = sa.get_rhat_results({key : val[:, 2500:, ...] for key, val in concatenated_samples.items()})

rhat2_after_2_5k = sa.get_rhat_results({key : val[:, 10_000:, ...] for key, val in concatenated_samples2.items()})

plot_rhat_results(rhat_after_2_5k, name="AssessmentAfter2_5k")

plot_rhat_results(rhat2_after_2_5k, name="TwoChain100kAssessmentAfter2_5k")

plt.hist(
    r1.samples['z'][5000:10_000, np.where(rhat_results['z'] < 1.05)[0]], bins=100)
plt.show()


def plot_edges_in_groups(name="ConvergedEdges",dpi=300):
    # Converged Edges
    plt.hist(np.ravel(
        mv.Z2A(
            r1.samples['z'][0:20_000, np.where(rhat_results['z'] <= 1.05)[0]])), bins=100, label="Strict", alpha=0.5)

    plt.hist(np.ravel(
        0.05 + mv.Z2A(r1.samples['z'][0:20_000, np.where(rhat_results['z'] <= 1.1)[0]])), bins=100, label="Cut-off", alpha=0.5)
    plt.hist(np.ravel(
        0.1 + mv.Z2A(r1.samples['z'][0:20_000, np.where(rhat_results['z'] >= 2.5)[0]])), bins=100, label="R>2.5", alpha=0.5)

    plt.legend()
    plt.xlabel("A (shifted)")
    plt.savefig(f"{name}_{dpi}.png", dpi=300)
    plt.show()


# Converged Edges
plt.hist(np.ravel(
    mv.Z2A(r1.samples['z'][0:20_000, np.where(rhat_results['z'] > 2.05)[0]])), bins=100)
plt.xlabel("A")
plt.show()

plt.plot(r1.mcmc['extra_fields']['energy'])

plt.plot(r2.mcmc['extra_fields']['energy'])
