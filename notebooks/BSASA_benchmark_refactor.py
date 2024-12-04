def interact():
    import code
    code.InteractiveConsole(locals=globals()).interact()

from _BSASA_functions import *
from _BSASA_imports import *

from types import SimpleNamespace
import plot_data

# Globals 
SAVE_CHAIN_MAPPING = True
LOAD_CHAIN_MAPPING_FROM_PKL = False

# Objects for returning user info
pad = notepad.NotePad()
# init bsasa_reference

bsasa_reference = SimpleNamespace()
bsasa_reference.df = init_bsasa_ref()
pad.write(("Loading Buried Solvent Accessible Surface Area Reference",
           bsasa_reference.df.shape))

# Add pdb and uid info
bsasa_reference.pdb_set = set(bsasa_reference.df['PDBID'].values)
pad.write(("N PDB Files in BSASA REF", len(bsasa_reference.pdb_set)))
bsasa_reference.nuid = init_nuid(bsasa_reference.df)
pad.write(("N uids represented in BSASA ref", bsasa_reference.nuid)) 

# Remove nans
bsasa_reference.df  = remove_prey_nan(bsasa_reference.df)
pad.write(("Removing Nans", bsasa_reference.df.shape)) 

# Remove duplicates
bsasa_reference.df  = remove_self_interactions(bsasa_reference.df)
pad.write(("Removing self interactions", bsasa_reference.df.shape))
bsasa_reference.df  = remove_every_other(bsasa_reference.df)
pad.write(("Removing duplicates", bsasa_reference.df.shape))
bsasa_reference.prey_set = set(
        bsasa_reference.df['Prey1'].values).union(
        bsasa_reference.df['Prey2'].values)

# update the pdb set        
bsasa_reference.pdb_set = set(bsasa_reference.df['PDBID'].values)
bsasa_reference.nuid = init_nuid(bsasa_reference.df)
pad.write(("N PDB Files in BSASA REF", len(bsasa_reference.pdb_set)))
pad.write(("N uids represented in BSASA ref",
           bsasa_reference.nuid)) 

# Summarize direct interactions
bsasa_reference.direct_interaction_set = init_direct_interaction_set(
        bsasa_reference.df)
bsasa_reference.n_direct = len(bsasa_reference.direct_interaction_set)
pad.write(("N pairwise direct interactions found",
           bsasa_reference.n_direct))

# Load fastas
fasta = SimpleNamespace()
uid2seq, uid2seq_len = init_uid2seq()
fasta.uid2seq = uid2seq.copy()
fasta.uid2seq_len = uid2seq_len.copy()
fasta.nuid = len(fasta.uid2seq)
del uid2seq
del uid2seq_len
pad.write(("Reading Primary Amino Acid Sequences",
           fasta.nuid))

# +
#uid_total = 3_062
#pad.write(("N Expected Uniprot IDs", uid_total))
# Summarize fasta interaction space

fasta.n_possible_pairwise_combinations = math.comb(
        fasta.nuid, 2)
pad.write(("Size of interaction space",
           fasta.n_possible_pairwise_combinations))
bsasa_reference.n_possible_pairwise_combinations = math.comb(
        bsasa_reference.nuid, 2)
pad.write(("Maximal possible size of reference interactions",
           bsasa_reference.nuid))

# How many positive cases are there?
dataset_balance_total = bsasa_reference.n_direct / fasta.n_possible_pairwise_combinations
dataset_balance_matched = bsasa_reference.n_direct / bsasa_reference.n_possible_pairwise_combinations

pad.write(("percent dataset balance", dataset_balance_total * 100))
pad.write(("percent matched balance", dataset_balance_matched * 100))

# Get the previously calculated mappings
auth_prey_gene_and_uid = SimpleNamespace()
auth_prey_gene_and_uid.df = pd.read_csv("../table1.csv")
pad.write(("Table 1: Maps PreyGene to Uniprot ID",
           auth_prey_gene_and_uid.df.shape))
auth_prey_gene_and_uid.gene2uid = {
        key:val for key, val in auth_prey_gene_and_uid.df.values}
auth_prey_gene_and_uid.uid2gene = {
     val:key for key,val in auth_prey_gene_and_uid.gene2uid.items()}
pad.write("-- Prey in BSASA--")
pad.write(prey_in_bsasa(
    gene2uid=auth_prey_gene_and_uid.gene2uid,
    prey_set=bsasa_reference.prey_set))

# Check for expected PDBs
write_prey_in_BSASA_2pad(pad,
    auth_prey_gene_and_uid.gene2uid, bsasa_reference.prey_set)
pad.write("-- PDBs in BSASA --")
pad.write((f"4n9f", '4n9f' in bsasa_reference.pdb_set))

#summarize_col(bsasa_ref, 'bsasa_lst')
bsasa_reference.interaction_set = init_interaction_set(bsasa_reference.df)
#n_possible_interactions = math.comb(uid_total, 2)
#n_possible_found_interactions = math.comb(len(prey_set), 2)
#npdbs_per_interaction = init_npdbs_per_interaction(bsasa_ref, interaction_set)
# Filter by percent sequence identity
# Map PDB chains to Uniprot Queries

##chain_mapping = SimpleNamespace()
##chain_mapping.df  = pd.read_csv("../significant_cifs/chain_mapping_all.csv")
##pad.write(("Loading chain mapping", chain_mapping.df.shape))
##sel = chain_mapping.df['bt_aln_percent_seq_id'] >= 0.3
##chain_mapping.df = chain_mapping.df[sel]
##pad.write(("Chains over 30% sequence identity", chain_mapping.df.shape))
### How many pdb ids and uids are there?
##
##chain_mapping.pdb_set = set(chain_mapping.df['PDBID'].values)
##chain_mapping.uid_set = set(chain_mapping.df['QueryID'].values)
##pad.write(("N PDBS in chain mapping",  len(chain_mapping.pdb_set)))
##pad.write(("N UIDS in chain mapping",  len(chain_mapping.uid_set)))
##
##chain_mapping.complexes = init_complexes(
##        chain_mapping.pdb_set, chain_mapping.df)
##
##chain_mapping.cocomplexes = SimpleNamespace()
##
##chain_mapping.cocomplexes.pdb_id__uid_fset  = init_cocomplexes(chain_mapping.complexes)
##pad.write(("N cocomplexes", len(chain_mapping.cocomplexes.pdb_id__uid_fset.keys())))
##chain_mapping.cocomplexes.edge_id__cocomplex_edge = init_cocomplex_edge_id__cocomplex_edge(
##        chain_mapping.cocomplexes.pdb_id__uid_fset)
##pad.write(("N cocomplex edges",
##           len(chain_mapping.cocomplexes.edge_id__cocomplex_edge.keys())))
##
##chain_mapping.cocomplexes.uid_set = init_cocomplex_uid_set(chain_mapping.cocomplexes.pdb_id__uid_fset)
##
##
##
##
##
### Long Running Cell
##chain_mapping.cocomplexes.pairs = list(combinations(chain_mapping.cocomplexes.uid_set, 2))
##
##chain_mapping.cocomplexes.df = init_cocomplex_df(
##        chain_mapping.cocomplexes.pairs, chain_mapping.cocomplexes.pdb_id__uid_fset)
##
##pad.write(("Co-complex df", chain_mapping.cocomplexes.df.shape))

if LOAD_CHAIN_MAPPING_FROM_PKL:
    with open("chain_mapping.pkl", "rb") as f:
        chain_mapping = pkl.load(f)
else:
    chain_mapping, pad = init_chain_mapping(pad)

if SAVE_CHAIN_MAPPING:
    with open("chain_mapping.pkl", "wb") as f:
        pkl.dump(chain_mapping, f)

cullin_data = SimpleNamespace()
df1, df2, df3 = init_dfs()
cullin_data.df1 = df1
cullin_data.df2 = df2
cullin_data.df2 = df3

cullin_data.df_all = init_df_all(df1, df2, df3, auth_prey_gene_and_uid.gene2uid)
cullin_data.df_all = update_df_all_bait_and_condition(cullin_data.df_all)

cullin_data.rsel = [f"r{i}" for i in range(1, 5)]
cullin_data.csel = [f"c{i}" for i in range(1, 13)]
cullin_data.df_all = parse_spec(cullin_data.df_all)
pad.write(("DF ALL", cullin_data.df_all.shape))

cullin_data.df_new = cullin_data.df_all[['bait', 'condition', 'Prey', 'SaintScore', 'BFDR', 'BaitUID'] + cullin_data.rsel + cullin_data.csel]

pad.write(("DF NEW", cullin_data.df_new.shape))
cullin_data.df_new = set_PreyName_as_index(cullin_data.df_new)

# Update df_new based on assumed control mappings
assert cullin_data.df_new.shape[0] == cullin_data.df_all.shape[0]
df_new, control_mappings  = update_df_new_based_on_assumed_control_mappings(cullin_data.df_new)

cullin_data.df_new = df_new.copy()
cullin_data.control_mappings = control_mappings.copy()

del df_new
del control_mappings

pad.write(("Assumed ELOB control mapping", cullin_data.control_mappings["elob"]))

# Removing extra counts
cullin_data.df_new = cullin_data.df_new.drop(columns=(
    cullin_data.control_mappings["cul5"] + cullin_data.control_mappings["elob"]))
pad.write(("DF NEW - Remove Extra Counts", cullin_data.df_new.shape))

cullin_data.df_new = update_df_new_with_summary_stats(cullin_data.df_new)
pad.write(("DF New - Add summary stats", cullin_data.df_new.shape))

cullin_data.df_new = update_df_new_with_query_aa_sequence(cullin_data.df_new, chain_mapping.df)
cullin_data.df_new = update_df_new_with_aa_seq_len(fasta.uid2seq_len, cullin_data.df_new)

#df_new.loc[:, 'exp_aa_seq_len'] = np.exp(seq_lens)  # overflow
cullin_data.df_new = update_df_new_with_tryptic_sites(cullin_data.df_new, fasta.uid2seq_len,
    fasta.uid2seq)
pad.write(("DF NEW - Add sequence info", cullin_data.df_new.shape))

SAVE_DF_NEW = True

if SAVE_DF_NEW:
    with open("df_new.pkl", "wb") as f:
        pkl.dump(cullin_data.df_new, f)

# Error
# 
df_new, cocomplex_ref_pairs, direct_ref_pairs = init_interactions(
        df_all=cullin_data.df_new,
        cocomplex_df=chain_mapping.cocomplexes.df,
        bsasa_ref=bsasa_reference.df)

cullin_data.df_new = df_new.copy()
bsasa_reference.cocomplex_ref_pairs = cocomplex_ref_pairs.copy()
bsasa_reference.direct_ref_pairs = direct_ref_pairs.copy() 

PICKLE_DF_NEW = True
if PICKLE_DF_NEW:
    with open("df_new.pkl", "wb") as f:
        pkl.dump(df_new, f)

del df_new

pad.write(("DF NEW - Init interactions", cullin_data.df_new.shape))
bait = ['CBFB', 'ELOB', 'CUL5', 'LRR1']

cullin_data.prey_set = sorted(list(set(cullin_data.df_new.index)))
cullin_data.preyu = np.array(cullin_data.prey_set)
cullin_data.preyv = np.array(cullin_data.prey_set)
cullin_data.nprey = len(cullin_data.preyu)

cullin_data.csel = [f"c{i}" for i in range(1, 5)]
#Initialize tensors
tensorR, tensorC = init_tensorRandC(
    cullin_data.df_new,
    cullin_data.rsel,
    cullin_data.csel,
    cullin_data.preyu,
    bait,
    fill_tensors)

pad.write(("Tensor Replicate", tensorR.shape))
pad.write(("Tensor Control", tensorC.shape))

spectral_count_xarray = init_spectral_count_xarray(
        tensorR, tensorC)

import pickle as pkl
save_spectral_count_xarray = True
if save_spectral_count_xarray:
    with open("spectral_count_xarray.pkl", "wb") as f:
        pkl.dump(spectral_count_xarray, f)

pad.write(("Spectral Counts", spectral_count_xarray.shape))

cullin_data.preyname2uid = {
        row['PreyName']:row['Prey'] for i,row in cullin_data.df_new.iterrows()}
cullin_data.uid2preyname = {
        val:key for key,val in cullin_data.preyname2uid.items()}

chain_mapping.cocomplexes.df = update_cocomplex_df_with_PreyXNames(
        chain_mapping.cocomplexes.df, cullin_data.uid2preyname)

cocomplex_matrix = init_cocomplex_matrix(
        nprey = cullin_data.nprey,
        preyu = cullin_data.preyu,
        preyv = cullin_data.preyv,
        cocomplex_df = chain_mapping.cocomplexes.df)

chain_mapping.cocomplexes.matrix = cocomplex_matrix.copy()
#del cocomplex_matrix

bsasa_reference.df = update_bsasa_ref_with_PreyXNames(bsasa_reference.df, cullin_data.uid2preyname)
# Long running: 2 min
direct_matrix = init_direct_matrix(
        cullin_data.nprey,
        cullin_data.preyu,
        cullin_data.preyv, bsasa_reference.df)

ds = xr.Dataset({'cocomplex': cocomplex_matrix,
                 'direct': direct_matrix,
                 'CRL_E':tensorR,
                 'CRL_C':tensorC})

SAVE_DS = True
if SAVE_DS:
    with open("ds.pkl", "wb") as f:
        pkl.dump(ds, f)

# Append Cocomplex labels to DataFrame
cullin_data.df_new = update_df_new_PDB_COCOMPLEX(cullin_data.df_new, ds)
pad.write(("Add cocomplex labels to DF NEW", cullin_data.df_new.shape))

# Benchmark cos sim on cocomplex reference
cos_sim_matrix, mag_v = cosin_sim_df2cos_sim_matrix(
        spectral_counts2cosin_sim_df(spectral_count_xarray))

# Benchmark cos sim on direct reference
direct_benchmark = SimpleNamespace()
direct_benchmark.reference = SimpleNamespace()
direct_benchmark.reference.matrix = (direct_matrix > 0).copy()
direct_benchmark.prediction = SimpleNamespace()
direct_benchmark.prediction.cosine_similarity = SimpleNamespace()
direct_benchmark.prediction.cosine_similarity.matrix = cos_sim_matrix
direct_benchmark.reference.matrix = direct_benchmark.reference.matrix.sel(
        preyu=direct_benchmark.prediction.cosine_similarity.matrix.preyu,
        preyv=direct_benchmark.prediction.cosine_similarity.matrix.preyv,)

direct_benchmark.reference.n_edges = np.sum(np.tril(
    direct_benchmark.reference.matrix, k=-1))
direct_benchmark.reference.n_possible_edges = math.comb(
        len(direct_benchmark.reference.matrix), 2)
direct_benchmark.thresholds = np.arange(0, 1.01, 0.01)
pps, tps = pp_tp_from_pairwise_prediction_matrix_and_ref(
        direct_benchmark.reference.matrix,
        direct_benchmark.prediction.cosine_similarity.matrix,
        direct_benchmark.thresholds)

direct_benchmark.prediction.cosine_similarity.pps = np.array(pps) 
direct_benchmark.prediction.cosine_similarity.tps = np.array(tps) 
direct_benchmark.prediction.cosine_similarity.ppr = (
    direct_benchmark.prediction.cosine_similarity.pps /
    direct_benchmark.reference.n_possible_edges
    )
del pps
del tps

direct_benchmark.prediction.cosine_similarity.tpr = (
    direct_benchmark.prediction.cosine_similarity.tps /
    direct_benchmark.reference.n_edges
    )
direct_benchmark.prediction.cosine_similarity.auc = (
    sklearn.metrics.auc(
        x = direct_benchmark.prediction.cosine_similarity.ppr,
        y = direct_benchmark.prediction.cosine_similarity.tpr,
        )
    )

direct_benchmark.prediction.cosine_similarity.auc = np.round(
   direct_benchmark.prediction.cosine_similarity.auc, 2)

SAVE_DIRECT_BENCHMARK = True
if SAVE_DIRECT_BENCHMARK:
    with open("direct_benchmark.pkl", "wb") as f:
        pkl.dump(direct_benchmark, f)

save_direct_benchmark_fig = False
if save_direct_benchmark_fig:
    fig, ax = plt.subplots()
    ax.set_title("Direct Interaction Benchmark")
    ax.plot(direct_benchmark.prediction.cosine_similarity.ppr,
            direct_benchmark.prediction.cosine_similarity.tpr,
            label=f"cosine similarity score: AUC {direct_benchmark.prediction.cosine_similarity.auc}")
    xlabel = ("Positive predictive rate"
        f" (N={h(direct_benchmark.reference.n_possible_edges)})"
    )
    ylabel = ("True positive rate"
        f" (N={h(direct_benchmark.reference.n_edges)})"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.savefig("direct_benchmark.png", dpi=1200)
    plt.close()

# Benchmark cos sim on cocomplex reference
cocomplex_benchmark = SimpleNamespace()
cocomplex_benchmark.reference = SimpleNamespace()
cocomplex_benchmark.reference.matrix = (chain_mapping.cocomplexes.matrix > 0).copy() 

cocomplex_benchmark.prediction = SimpleNamespace()
cocomplex_benchmark.prediction.cosine_similarity = SimpleNamespace()
cocomplex_benchmark.prediction.cosine_similarity.matrix = cos_sim_matrix

cocomplex_benchmark.reference.matrix = (
        cocomplex_benchmark.reference.matrix.sel(
            preyu=cocomplex_benchmark.prediction.cosine_similarity.matrix.preyu,
            preyv=cocomplex_benchmark.prediction.cosine_similarity.matrix.preyv))

cocomplex_benchmark.reference.n_edges = np.sum(np.tril(
    cocomplex_benchmark.reference.matrix, k=-1))
cocomplex_benchmark.reference.n_possible_edges = math.comb(
        len(cocomplex_benchmark.reference.matrix), 2)
cocomplex_benchmark.thresholds = np.arange(0, 1.01, 0.01)
pps, tps = pp_tp_from_pairwise_prediction_matrix_and_ref(
        cocomplex_benchmark.reference.matrix,
        cocomplex_benchmark.prediction.cosine_similarity.matrix,
        cocomplex_benchmark.thresholds)
cocomplex_benchmark.prediction.cosine_similarity.pps = np.array(pps) 
cocomplex_benchmark.prediction.cosine_similarity.tps = np.array(tps) 
cocomplex_benchmark.prediction.cosine_similarity.ppr = (
    cocomplex_benchmark.prediction.cosine_similarity.pps /
    cocomplex_benchmark.reference.n_possible_edges
    )
cocomplex_benchmark.prediction.cosine_similarity.tpr = (
    cocomplex_benchmark.prediction.cosine_similarity.tps /
    cocomplex_benchmark.reference.n_edges
    )
cocomplex_benchmark.prediction.cosine_similarity.auc = (
    sklearn.metrics.auc(
        x = cocomplex_benchmark.prediction.cosine_similarity.ppr,
        y = cocomplex_benchmark.prediction.cosine_similarity.tpr,
        )
    )

cocomplex_benchmark.prediction.cosine_similarity.auc = np.round(
   cocomplex_benchmark.prediction.cosine_similarity.auc, 3)

SAVE_COCOMPLEX_BENCHMARK = True

if SAVE_COCOMPLEX_BENCHMARK:
    with open("cocomplex_benchmark.pkl", "wb") as f:
        pkl.dump(cocomplex_benchmark, f)

fig, ax = plt.subplots()
ax.set_title("Cocomplex Interaction Benchmark")
ax.plot(cocomplex_benchmark.prediction.cosine_similarity.ppr,
        cocomplex_benchmark.prediction.cosine_similarity.tpr,
        label=f"cosine similarity score: AUC {cocomplex_benchmark.prediction.cosine_similarity.auc}")
ax.set_xlabel("Positive predictive rate")
ax.set_ylabel("True positive rate")
ax.legend()
fig.savefig("cocomplex_benchmark.png", dpi=1200)
plt.close()

#Plot the relationship between cosine similarity and shortest paths
#A is the direct interaction reference
A1 = direct_benchmark.reference.matrix.values
A1 = np.array(A1, dtype=float)
assert sum(np.diag(A1)) == 0 # valid adjacency
assert np.all(A1 == A1.T)     # valid adjacency
assert np.min(A1) == 0       # no negative cycles 
assert np.max(A1) == 1
#The matrix power 

n = 30 # For the Cullin System the max path length for direct interactions is 22 
D = np.linalg.matrix_power(A1, n)
n -= 1
while n > 0:
    if (n % 10) == 0:
        print(f"power {n}")
    #A2 = A1 @ A1
    #A3 = A1 @ A2
    #A4 = A1 @ A3
    #A5 = A1 @ A4
    An = np.linalg.matrix_power(A1, n)
    D = np.where(An > 0, n, D)
    n -= 1

# Plot the histogram of path lengths
fig, ax = plt.subplots(1, 1)
_x = D[np.tril_indices_from(D, k=-1)]
_nz = np.sum(_x == 0)
_x = _x[_x != 0]

plt.hist(_x, bins=100)
plt.text(10, 15000, f"Disconnected: {h(_nz)}")
plt.title("Cullin Benchmark Direct Paths")
plt.xlabel("Shortest path length")
plt.ylabel(f"Frequency (N={len(_x)})")
plt.legend()
plt.savefig("direct_path_length_hist.png", dpi=1200)
plt.close()
# 

D = xr.DataArray(D, dims=["preyu", "preyv"],
                 coords=cos_sim_matrix.coords,
                 )
bait_prey_D = D.sel(preyu=["ELOB", "CUL5", "PEBB"])
fig, ax = plt.subplots(1, 1)
ax.hist(np.ravel(bait_prey_D.values), bins=100)
plt.xlabel("Shortest path length")
plt.ylabel("Frequency")
plt.title(f"Bait Prey path lengths\nN={np.prod(bait_prey_D.values.shape)}")
plt.savefig("bait_prey_path_lengths.png", dpi=300)

# Now plot Saint Scores vs path-length
#saint_score_array = bait_prey_D.copy()
#saint_score_array.loc[:, :] = 0.0
#saint_score_array = spectral_count_xarray.sel(preyu=saint_score_array.preyu,
interact()
# Bait prey difference
bait_prey_spectral_count_diff = bait_prey_D.copy()
sc_diff = spectral_count_xarray.sel(AP=True) - spectral_count_xarray(AP=False)
sc_diff = sc_diff.mean("rep", "condition")



for bait in set(cullin_data.df_new['bait']):
    sel = cullin_data.df_new['bait'] == bait
    subdf = cullin_data.df_new.loc[sel, 'SaintScore']
    if bait == "CBFB":
        bait = "PEBB"
    #saint_score_array.loc[subdf.inde

    






# Make a histogram pair plot of d and cos sim
# Select the non-zero paths
# This looks a lot like two independant marginal distributions

def hist_plot2d_features(D, savename, feature_name):
    """
    """
    xbins = 21
    ybins = 100
    idxs = np.tril(D, k=-1) != 0
    _x = D[idxs]
    _y = cos_sim_matrix.values[idxs]
    fig, ax = plt.subplots(2, 2)
    mapp = ax[1, 0].hist2d(_x, _y, bins=[xbins, ybins], vmax=300)
    #plt.colorbar(mapp)
    ax[0, 0].sharex(ax[1, 0])
    ax[0, 0].hist(_x, bins=xbins)
    ax[0, 0].set_ylabel("Frequency")
    ax[1, 1].sharey(ax[1, 0])
    ax[1, 1].hist(_y, orientation='horizontal', bins=ybins)
    ax[1, 0].set_xlabel("Shortest path length")
    ax[1, 0].set_ylabel(feature_name)
    plt.savefig(savename, dpi=300)
    plt.close()

#Plot the relationship between SaintScore and Path length


# Feature Average SC Pair
_X = spectral_count_xarray.sel(AP=True) - spectral_count_xarray.sel(AP=False)
_X = _X.sel(bait=["ELOB", "CUL5", "CBFB"], preyu=cos_sim_matrix.preyu)
_X = _X.mean(dim=["condition", "rep"])
_D = cos_sim_matrix.copy()



sns.set_theme(style="ticks")

# Load the planets dataset and initialize the figure
planets = sns.load_dataset("planets")
g = sns.JointGrid(data=planets, x="year", y="distance", marginal_ticks=True)

# Set a log scaling on the y axis
g.ax_joint.set(yscale="log")

# Create an inset legend for the histogram colorbar
cax = g.figure.add_axes([.15, .55, .02, .2])

# Add the joint and marginal histogram plots
g.plot_joint(
            sns.histplot, discrete=(True, False),
                cmap="light:#03012d", pmax=.8, cbar=True, cbar_ax=cax
                )
g.plot_marginals(sns.histplot, element="step", color="#03012d")




#benchmark_summary = init_benchmark_summary(
#        chain_mapping.cocomplexes.df,
#        bsasa_reference.df,
#        cullin_data.df_all, unknown, direct_interaction,
#        cocomplex_interactions)
#pad.write(("Benchmark Summary", benchmark_summary.shape))
#
#edge_list = list(combinations(preyu, 2))
#prey_pairs_df = init_prey_pairs_df(jax.random.PRNGKey(13), prey_set, bsasa_ref)

base = "../../benchmark/data/cullin/PRIDE/PXD009012/"
evidence = pd.concat(
    [pd.read_csv(i, sep="\t") for i in [
        base + "MaxQuant_results_CBFB_HIV_APMS/RH022_evidence.txt"]])#,
        #base + "MaxQuant_results_CUL5_HIV_APMS/evidence.txt",
        #base + "MaxQuant_results_ELOB_HIV_APMS/evidence.txt"]])
proteinGroups = pd.read_csv(base + "MaxQuant_results_CBFB_HIV_APMS/proteinGroups.txt", sep="\t")
proteinGroups[proteinGroups['Protein IDs'] == 'vifprotein'].iloc[:, [0, 2, 3, 8, 9, 
                                        10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 25, 33]]
sel = evidence['Raw file'] == 'FU20151020-04'
evidence[evidence['Proteins']=='vifprotein'].loc[sel, 
            ['Peptide ID', 'MS/MS Count', 'Number of scans', 'Raw file']]


# Modeling Begin
pad.write("Modeling Begin")

init_params = namedtuple("p", "x y")(x=df_new['rAv'].values, y=df_new['rVar'].values)

numpyro.render_model(model, *init_params)  

nuts_kernal = numpyro.infer.NUTS(model)
mcmc = numpyro.infer.MCMC(nuts_kernal, num_warmup=500, num_samples=1000)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, x=df_new['rAv'].values, y=df_new['rVar'].values, extra_fields=('potential_energy',))
mcmc.print_summary()

df_test = df_new
nuts_kernal = numpyro.infer.NUTS(m1)
mcmc = numpyro.infer.MCMC(nuts_kernal, num_warmup=1000, num_samples=1000)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, df_test, extra_fields=('potential_energy',))

samples = mcmc.get_samples()

dsel = ds.sel(bait=['CBFB', 'ELOB', 'CUL5'])
ctrl_data = np.ravel(dsel['CRL_C'].values)
e_data = np.ravel(dsel['CRL_E'].values)
nuts_kernal = numpyro.infer.NUTS(model4)
mcmc = numpyro.infer.MCMC(nuts_kernal, num_warmup=1000, num_samples=10000, thinning=2)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, e_data=e_data, ctrl_data=ctrl_data, extra_fields=('potential_energy',))
mcmc.print_summary()
posterior_samples = mcmc.get_samples()
posterior_predictive = numpyro.infer.Predictive(model4, posterior_samples)(jax.random.PRNGKey(1))
prior = numpyro.infer.Predictive(model4, num_samples=1000)(jax.random.PRNGKey(2))
m4_data = az.from_numpyro(mcmc, prior=prior, posterior_predictive=posterior_predictive)

n = 0
y_c = np.array(df_new.iloc[n, :][csel].values, dtype=int)
y_e = np.array(df_new.iloc[n, :][rsel].values, dtype=int)
model_kwargs={'y_c': y_c, 'y_e': y_e}
numpyro.render_model(model5, model_args=(None, None, y_c, y_e, True, ()),
                     render_distributions=True, render_params=True)
search = do_mcmc(model5, 1, model_kwargs, num_warmup=1000, num_samples=5000)
search.print_summary()
posterior_samples = search.get_samples()
prior = numpyro.infer.Predictive(model5, num_samples=500)
posterior_predictive = numpyro.infer.Predictive(model5, posterior_samples)

start=10
end=20#len(df_new)
l = end - start
y_c = df_new.iloc[start:end, :][csel].values
y_e = df_new.iloc[start:end, :][rsel].values

kd = {'hyper_a': (np.mean(y_e, axis=1).reshape((l, 1)) + 10) * 1.5,
      'hyper_b': (np.mean(y_c, axis=1).reshape((l, 1)) + 10) * 1.5}
m5f = model52_f(df_new, start=start, end=end, numpyro_model=model6,
               numpyro_model_kwargs=kd)
kernel = m5f.init_kernel(rescale_model=False)
sample_init = m5f.init_sampling(jax.random.PRNGKey(13), num_warmup=1000, num_samples=1000)
search = m5f.sample(kernel, sample_init)
sample_init = m5f.init_sampling(PRNGKey(12), num_warmup=500, num_samples=500)
pp = m5f.sample_pp(kernel, sample_init)
sample_init = m5f.init_sampling(PRNGKey(11), num_warmup = 500, num_samples = 500)
Pp = m5f.sample_Pp(kernel, search.get_samples(), sample_init)
inf_data10_20 = m5f.init_InferenceData(search, pp, Pp, kernel.meta, rescale=False, append_sample_stats=True)

#model = model8
nuts_kernal = numpyro.infer.NUTS(model)
mcmc = numpyro.infer.MCMC(nuts_kernal, num_warmup=1000, num_samples=1000, thinning=1)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, ds_sel, extra_fields=('potential_energy',))
samples = mcmc.get_samples(group_by_chain=False)
summary_dict = summary(mcmc.get_samples(group_by_chain=True))
posterior_predictive = numpyro.infer.Predictive(model, samples)(jax.random.PRNGKey(1),
                                                                ds_sel)
prior_predictive = numpyro.infer.Predictive(model, 
                                            num_samples=1000)(PRNGKey(2), ds_sel)
i8data = az.from_numpyro(mcmc, 
        coords={'bait': ['CBFB', 'CUL5', 'ELOB'], 'condition': ds_sel.condition.values,
                'rrep': ds_sel.rrep.values,
                'preyu': ds_sel.preyu.values[0:prey_max]},
        dims={'alpha': ['bait', 'condition', 'rrep'],
              'beta': ['preyu', 'bait', 'condition', 'rrep'],
              'epsilon': ['preyu', 'bait', 'condition', 'rrep']})

# Zero inflated Poisson
data = model_data_from_ds(ds)
start=0
end=1000
data = data.sel(preyu=data.preyu[start:end], bait=['CBFB', 'CUL5', 'ELOB'])
numpyro.render_model(model, model_args=(data, True), render_distributions=True, render_params=True)
num_samples = 1000
mcmc = run_mcmc(PRNGKey(0), 
                model_zero_inflated_poisson, 
                data,
                num_samples=num_samples,
                num_warmup=500)
posterior_predictive = numpyro.infer.Predictive(model_zero_inflated_poisson,
    posterior_samples=mcmc.get_samples())(PRNGKey(1), data, observed=False)
prior_predictive = numpyro.infer.Predictive(model_zero_inflated_poisson, num_samples=num_samples)(
    PRNGKey(2), data, observed=False)
dims = {"lam": ["preyu", "bait", "condition", "test"],
        "pi": ["preyu", "bait", "condition", "test"],
        "sc": ["rep", "preyu", "bait", "condition", "test"]}
coords = {key: val for key, val in data.coords.items()} | {'draw': np.arange(0, num_samples)}
izdata = az.from_numpyro(mcmc, 
                        coords=coords, 
                        dims=dims,
                        pred_dims = {'sc': ['draw', 'preyu', 'bait', 'condition', 'test']},
                        posterior_predictive=posterior_predictive,
                        prior=prior_predictive)


# Begin


# End




"""
Some functions to benchmark AP-MS scoring functions
"""
tp_over_pp_score_from_df = partial(score_from_df, score_fun=tp_over_pp_score)
tp_from_df = partial(score_from_df, score_fun=tp)
ref_set = ref_df2ref_set(bsasa_ref, 'Prey1Name', 'Prey2Name')
tmp = df_new
f1 = partial(tp_from_df, a='bait', b='PreyName')
f2 = partial(tp_over_pp_score_from_df, a='bait', b='PreyName')
f3 = lambda sub_df, ref_set: len(sub_df)
t = np.arange(0, 1, 0.01)
col = 'SaintScore'
#tp_scores = scores_from_df(tmp, t, col, ref_set, f1)
#tp_over_pp_scores = scores_from_df(tmp, t, col, ref_set, f2)
scores = scores_from_df_fast(tmp, t, col, ref_set, [f1, f3])
df_new.loc[:, "PostCertainty"] = posterior_certainties.values
df_new.loc[:, "PriorCertainty"] = prior_certainties.values
# N direct interactions
n_possible_interactions = math.comb(len(set(df_new['PreyName'].values)),2)
n_direct = len(set(bsasa_ref_pairs))
thresholds = np.arange(0, 1, 0.05)
saint_predictions = []
posterior_predictions = []
prior_predictions = []
for threshold in thresholds:
    saint_subdf = df_new.loc[df_new['SaintScore'] >= threshold]
    prior_subdf = df_new.loc[df_new['PriorCertainty'] >= threshold]
    post_subdf  = df_new.loc[df_new['PostCertainty'] >= threshold]
    saint_predictions.append(df2pp_tp(saint_subdf, threshold))
    prior_predictions.append(df2pp_tp(prior_subdf, threshold))
    posterior_predictions.append(df2pp_tp(post_subdf, threshold))
    print(threshold)
saint_predictions_arr = np.array(saint_predictions)
prior_predictions_arr = np.array(prior_predictions)
postr_predictions_arr = np.array(posterior_predictions)
#saint_predictions[:, 0] = sp.special.comb(saint_predictions[:, 0], 2)
fig, ax = plt.subplots(1, 1)
step = 1000
xend = n_possible_interactions
xstart=0
every = 1e4
x = np.arange(0, xend, every)
y = sp.stats.hypergeom.median(M=n_possible_interactions,
                           n=n_direct, N=x)
Null = sp.stats.hypergeom(M=n_possible_interactions, n=n_direct, N=x)
#ylower = Null.ppf(0.1)
#yupper = Null.ppf(0.9)
#yerr = np.array([ylower, yupper])
yerr = Null.var()
y = Null.median()
ax.errorbar(x, y, yerr=yerr, alpha=0.2, label='Hypergeometric Null')
ax.set_xlabel("Predicted prey pairs")
ax.set_ylabel("Recovered PDB direct Interactions")
#xy_plot(ax, x, y, '.', alpha=0.4, xlabel="$n$ predicited positives", ylabel="Recovered PDBs")
ax.plot(saint_predictions_arr[:, 0], saint_predictions_arr[:, 1], label='Saint All Prey')
ax.plot(prior_predictions_arr[:, 0], prior_predictions_arr[:, 1], label='Prior certainty')
ax.plot(postr_predictions_arr[:, 0], postr_predictions_arr[:, 1], label="Posterior certainty")
ax.legend()
ax.set_xlim((xstart, xend))
#ax.set_ylim((0, 500))
plt.savefig("null_benchmark.png", dpi=notebook_dpi)
    

with mpl.rc_context(sali_style):
    fig, ax = plt.subplots(1, 1)
    fig_data = flt.FigData(savename="tmp.png", dpi=NOTEBOOK_DPI)
    # Define the plotting data
    hist_data = flt.HistData(x = vals, bins = 100, range=(0, int(1e4)),
        label="BSASA", edgecolor="white", linewidth=0.5)
    set_data = flt.SetData(xlabel = r"$\mathrm{\AA}^2$",
        title = "Chain Buried Solvent Accesible Surface Area",ylabel = "Count")
    legend_data = flt.LegendData(
        x=500, ymin=0, ymax=1, color=red,
        linestyle="dashed", label=r"500 $\mathrm{\AA}$ cutoff")
    ax_data = flt.AxData(
        set_data = set_data,
        hist_data=hist_data,
        axvline_data=axvline_data,
        legend_data=legend_data)
    # Call the plotting function
    flt.vline_hist_with_extra_text_legend(ax, ax_data)
    flt.savefig(fig, fig_data)

vals = bsasa_ref['bsasa_lst'].values
plot_histogram_with_vline(vals)
# +


# Plot Number of Prey
plt.close()
fig, ax = plt.subplots(1, 1)
g = df_new.groupby(["condition", "bait"])
#ax = g.apply(lambda x: len(x[x['rMax']>0])).unstack(level=0).plot(kind="bar",
#    title="tmp") 
ax = g.apply(lambda x: len(x[x['cMax']>0])).unstack(level=0).plot(kind="bar",
    title="tmp", ax=ax, alpha=0.5) 
plt.ylabel("N unique prey detected")
plt.tight_layout()
save_plt()
plt.close()

tmp_scatter(df_new[df_new["bait"] == "CBFB"], "rAv", "cAv", title="CBFB Average")
tmp_scatter(df_new[df_new["bait"] == "CUL5"], "rAv", "cAv", title="CUL5 Average")
tmp_scatter(df_new[df_new["bait"] == "ELOB"], "rAv", "cAv", title="ELOB Average")
tmp_scatter(df_new[df_new["bait"] == "LRR1"], "rAv", "cAv", title="LRR1 Average")


tmp_scatter(df_new[df_new["bait"] == "CBFB"], "rVar", "cVar", title="CBFB Variance")
tmp_scatter(df_new[df_new["bait"] == "CUL5"], "rVar", "cVar", title="CUL5 Variance")
tmp_scatter(df_new[df_new["bait"] == "ELOB"], "rVar", "cVar", title="ELOB Variance")
tmp_scatter(df_new[df_new["bait"] == "LRR1"], "rVar", "cVar", title="LRR1 Variance")

tmp_scatter(df_new[df_new["bait"] == "CBFB"], "rMax", "cMax", title="CBFB Max")
tmp_scatter(df_new[df_new["bait"] == "CUL5"], "rMax", "cMax", title="CUL5 Max")
tmp_scatter(df_new[df_new["bait"] == "ELOB"], "rMax", "cMax", title="ELOB Max")
tmp_scatter(df_new[df_new["bait"] == "LRR1"], "rMax", "cMax", title="LRR1 Max")

tmp_scatter(df_new, "Av_diff", "SaintScore", title="Saint score", plot_xy=False)
vals = cocomplex_df['NPDBS'].values
title = "N PDBS per Cocomplex prey pair"
savename = title.replace(" ", "") + ".png"
plt.title(title)
plt.hist(vals, bins=(len(set(vals)) // 1), range=(0, 600))
plt.xlabel("N PDBS")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(savename, dpi=NOTEBOOK_DPI)
plt.close()
# -

#npairs = math.comb(3062, 2)
#nobs = math.comb(1400, 2)

# +


# +
vals = df_all['SaintScore'].values
results, true_result = permutation_test(13, vals, vals[np.where(direct_interaction)], np.mean, 1000000)
ax = plot_sampling(results, true_result, 0, 250000, 
                   test_stat="T(x): Mean Saint Score of 9 bait-prey pairs", 
                   title="Permutation test",
                   tx=0.05,
                   ty=200_000,
                   ts=f"N samples w/ replacement {h(len(results))}\nSize {sum(direct_interaction)}",
                   nbins=30,
                   histkwargs={'label': 'Null'},
                   savefig=True)
plt.close()
print("H0: Direct Bait-Prey interactions have no relation to SaintScore")

# +
vals = df_all['SaintScore'].values

results, true_result = permutation_test(13, vals, vals[np.where(cocomplex_interactions)], np.mean, 1000000)
ax = plot_sampling(results, true_result, 0, 130000, 
                   test_stat="T(x): Mean Saint Score of 19 bait-prey pairs", 
                   title="Cocomplex Permutation test",
                   tx=0.05,
                   ty=120000,
                   ts=f"N samples w/ replacement {h(len(results))}\nSize {sum(cocomplex_interactions)}",
                   nbins=30,
                   histkwargs={'label': 'Random pair'}, savefig=True)
plt.close()

print("H0: Cocomplex bait-prey interactions have no relation to Saint Score")

# +
# N unknown


title="Direct Interaction Benchmark Summary"
cols = ['Co-complex', 'Direct', 'Bait-prey']
sns.categorical.barplot(benchmark_summary.T.iloc[:, 0:4])
plt.ylabel('N Interactions')
plt.title(title)
plt.tight_layout()
save_plt()
#plt.savefig(, dpi=NOTEBOOK_DPI)
plt.close()
# -

#benchmark_summary

sns.categorical.barplot(benchmark_summary.T.iloc[:, 4:6])
plt.ylabel('N Interactions')
plt.title("Bait Prey Benchmark Summary")
plt.tight_layout()
save_plt()
plt.close()

# +
# Load in the xarrays and scores



# Bait, condition, preyu, r, c

# Create a tensor filled with zeros
    
# -

#df_new[df_new['condition']=='wt']

saint_scores = df_all["SaintScore"].values
kws = {'stat':'probability'}
sns.histplot(saint_scores, label=f"All N: {len(saint_scores)}", bins=10, **kws)
sns.histplot(saint_scores[np.where(direct_interaction)], 
             label=f"Direct {sum(direct_interaction)}", bins=10, alpha=0.5, **kws)
sns.histplot(saint_scores[np.where(cocomplex_interactions)], 
             label=f"Cocomplex {sum(cocomplex_interactions)}", bins=10, alpha=0.5, **kws)
plt.legend()
plt.xlabel("Saint Score")
plt.title("Saint Scores")
save_plt()
#savename = "SaintScoreHist.png")
#plt.savefig(savename, dpi=NOTEBOOK_DPI)
#plt.show()
plt.close()

sns.histplot(saint_scores, label=f"All N: {len(saint_scores)}", bins=10, **kws)
#plt.show()
plt.title("tmp")
save_plt()
plt.close()

#saint_scores[np.where(direct_interaction)]

#for (i,j) in direct_interaction_labels:
#    print(uid2gene[i], uid2gene[j])
#
#for (i,j) in cocomplex_interaction_labels:
#    print(uid2gene[i], uid2gene[j])

#df_all['Spec']

#sns.histplot(unknown)

#np.sum(unknown == 0)

vals = np.array([len(v) for k,v in cocomplexes.items()])
plt.hist(vals, bins=(len(set(vals)) // 1))
title = "N Protein types per cocomplex"
plt.title(title)
plt.ylabel("Count")
plt.tight_layout()
save_plt()
#plt.savefig(savename, dpi=NOTEBOOK_DPI)
#plt.show()
plt.close()

# +
vals = np.array([val for key,val in npdbs_per_interaction.items()])
title = "N PDBS per Interaction"
plt.title(title)
plt.hist(vals, bins=20)
save_plt()
#savename = title.replace(" ", "") + ".png"
#plt.savefig(savename, dpi=NOTEBOOK_DPI)
#plt.show()
plt.close()


# -


# +
# Direct Interactions benchmark

# +
# -

# DataSet
print(f"N direct {h(np.sum(np.tril(ds['direct'] > 0, k=-1)))}")
print(f"N cocomplex {h(np.sum(np.tril(ds['cocomplex'] > 0, k=-1)))}")

# +

x, y = xy_from(df_new, 'SaintScore', np.arange(1, 0, -0.05), comp=operator.ge, pos_col='PDB_COCOMPLEX')
pos_col = 'PDB_COCOMPLEX'
npairs = len(df_new)
npdb_pos = sum(df_new[pos_col].values)

yplot = y / npdb_pos
xplot = x / npairs
plt.plot(xplot, yplot, 'k.', label='Saint Score')
plt.ylabel(f"Fraction PDB Cocomplex Positives (N={h(npdb_pos)})")
plt.xlabel(f"Fraction Total Positives (N={h(npairs)})")
plt.title("tmp")
save_plt()
plt.close()

#y2 = 1.0
#x2 = npdb_pos / npairs
#plt.vlines([0.0035, 0.015], 0, 1, label="Estimated fraction of true PPIs")


#plt.plot(x2, y2, 'r+', label='PDB Benchmark')
##xmul = 10
#plt.plot(x2 * xmul, y2, 'rx', label=f'{xmul}x PDB')
#plt.savefig('f1.png', dpi=300)
#plt.savefig("FractionPlot.png", dpi=NOTEBOOK_DPI)
#plt.legend()
#plt.show()
# -

# +

# -


xt = np.arange(-10, 10)
yt = np

plt.title("Experimental Counts")
x = np.ravel(df_new.loc[:, rsel].values)
plt.hist(x, bins=100)
s = f"Mean {np.mean(x)}\nMedian {np.median(x)}\nVar {np.var(x)}\nMin {np.min(x)}\nMax {np.max(x)}"
plt.text(100, 30000, s)
#plt.show()
save_plt()
plt.close()


# +

x1 = df_new['rAv'].values
y1 = df_new['rVar'].values
x2 = df_new['cAv'].values
y2 = df_new['cVar'].values

r1, p1 = sp.stats.pearsonr(x1, y1)
r2, p2 = sp.stats.pearsonr(x2, y2)

r1 = np.round(r1, 2)
r2 = np.round(r2, 2)

plt.plot(x1, y1, 'k.', label='experiment')
plt.plot(x2, y2, 'rx', label='control')
s = f"R Experiemnt {r1}\nR control {r2}"
plt.text(60, 1400, s)
plt.xlabel('Mean')
plt.ylabel('Variance')
plt.legend()
plt.title("Mean Variance Correlation")
save_plt()
#plt.show()
plt.close()

sns.regplot(x=x1, y=y1, scatter_kws={'alpha': 0.05}) 
plt.title("AP Mean Variance Regression")
save_plt()
plt.close()


# -

ctrl_counts = (df_new.loc[(df_new['condition']=='wt') | (df_new['condition']=='mock'), csel])
x = np.ravel(ctrl_counts.values)
plt.title("Control Counts")
plt.hist(x, bins=100)
svals = tuple(h(np.round(f(x), 2)) for f in (np.mean, np.median, np.var, np.min, np.max)) 
s = "Mean %s\nMedian %s\nVar %s\nMin %s\nMax %s" % svals
plt.text(100, 30000, s)
#plt.show()
save_plt()
plt.close()

# +
xall = np.concatenate([np.ravel(ctrl_counts.values), np.ravel(df_new.loc[:, rsel].values)])
bounds = (0, 50)
xaxis = np.arange(0, bounds[1], 1)
r = 1/5
yvalue = r * np.exp(-r * xaxis) * 5e5
plt.hist(xall, bins=50, range=bounds)
plt.plot(xaxis, yvalue, 'r', label='Model')
plt.title("All Counts")
svals = tuple(h(np.round(f(xall), 2)) for f in (np.mean, np.median, np.var, np.min, np.max)) 
s = "Mean %s\nMedian %s\nVar %s\nMin %s\nMax %s" % svals
plt.xlabel("Spectral Count")
plt.legend()
plt.text(100, 30000, s)
save_plt()
plt.tight_layout()
plt.close()

#plt.show()

# -

sel = df_new['aa_seq_len'] <= 5000
sns.histplot(df_new[sel], x='n_first_tryptic_cleavage_sites', y='aa_seq_len')
plt.title("Number of tryptic cleavage sites and seq len")
plt.tight_layout()
save_plt()
plt.close()

#sns.regplot(df_new[sel], x='rAv', y='n_first_tryptic_cleavage_sites')

sel = df_new['aa_seq_len'] <= 200
sns.histplot(df_new[sel], x='cAv', y='n_first_tryptic_cleavage_sites',
             cbar=True, kde=True)
plt.title("Seq and cAv lt 200")
plt.tight_layout()
save_plt()
plt.close()

sns.regplot(df_new[sel], x='rAv', y='aa_seq_len')
plt.title("Seq and rAv lt 200")
plt.tight_layout()
save_plt()
plt.close()


plt.plot(np.arange(len(df_new)), df_new['rAv'] - df_new['cAv'], 'k.')
plt.title("Average Difference")
save_plt()
plt.close()

#plt.plot(np.arange(len(df_new)), df_new['rAv'], 'k.')

#plt.plot(np.arange(len(df_new)), df_new['cAv'], 'k.')

# +
nbins=30
aa_max_len=2_000
xlabel='n_possible_first_tryptic_peptides'
#xlabel='aa_seq_len'
sns.histplot(df_new[df_new['aa_seq_len'] <=aa_max_len], x=xlabel, y='rAv',
            bins=(nbins, nbins), cbar=True, cmap='hot')
plt.title("tmp")
save_plt()
plt.close()

print("The sequence length appears to set the upper bound on Spectral Count")
print("The protein spectral count is the sum of peptide spectral count")
print(f"Proteins are well sampled in the {np.min(df_new[xlabel].values), np.mean(df_new[xlabel].values)} {xlabel} range")
print(f"At least two possibilities")
print(f"1. Longer sequences are cleaved into more peptides")
print("2. Longer sequences fragment into more fragment ions")
print(f"3. Both")
print(f"The result of this effect is that longer sequences are more detectable")
print(f"Therefore longer sequences may be have lower abundance at lower abundance ")
# -

sns.histplot(df_new[df_new['aa_seq_len'] <=aa_max_len], x=xlabel, y='cAv',
            bins=(nbins, nbins), cbar=True, cmap='hot')
plt.title("tmp")
save_plt()
plt.close()

sns.histplot(df_new[df_new['aa_seq_len'] <=aa_max_len], x=xlabel, y='rMin',
            bins=(nbins, nbins), cbar=True, cmap='hot')
plt.title("tmp")
save_plt()
plt.close()





# There is one sequence that is very long
col = 'aa_seq_len'
sel = df_new[col] < 5000
sns.regplot(df_new.loc[sel, :], x=col, y='rAv')
print("It appears like the sequence length places an upper bound on the spectral counts")

col = 'aa_seq_len'
xcol = 'rAv'
sel = df_new[col] <= 500
plt.plot((df_new.loc[sel, col].values), np.max(df_new.loc[sel, rsel].values, axis=1), 'k.')
plt.xlabel(col)
plt.ylabel('Max spectral counts')
#plt.ylabel(col)

col = 'exp_aa_seq_len'
sel = df_new[col] < 1.4e217
sns.regplot(df_new[sel], x=col, y='rAv')

# +
"""
95 % Frequentist confidence interval
"""

x = np.arange(0, 215)
b = 0
m = 1.565
y = m * x + b
sns.regplot(x="rAv", y="rVar", data=df_new)
plt.plot(x, y)
plt.title("Experimental Data")

# +
x = np.arange(0, 215)
b = 0
m = 2.8
y = m * x + b

ax = sns.regplot(x='cAv', y="cVar", data=df_new)
plt.plot(x, y)
plt.title("Control Data")

# ## Probabalistic Filter
#
# The model is
# $$ p(M | D, I) \propto p(D | M, I)p(M | I) $$
#
# $$ D = Y_E, Y_C $$
# $$ I = I_1, I_2 $$
#
#


# +
# First create a function that reads in a vector of

#jax.vmap(jax.scipy.stats.poisson.pmf)

x1 = samples['k'].T[:, 0]
y1 = df_new.loc[:, rsel].values

# +

Yexp = df_new.loc[:, rsel].values
odds_kappa, odds_lambda = posterior_odds(samples, Yexp)

# -

odds_kappa[np.where(odds_kappa==0)] = 1e-7

plt.hist(odds_kappa, bins=100, label="kappa")
plt.xlabel("Posterior odds")
plt.title("tmp")
save_plt()
plt.close()
#plt.show()

plt.hist(odds_lambda, bins=100)
plt.title("tmp")
save_plt()
plt.close()

log_odds_ratio = np.log10(odds_lambda) - np.log10(odds_kappa)
# Capped at -4 and 4
log_odds_ratio[np.where(log_odds_ratio >= 4)] = 4
log_odds_ratio[np.where(log_odds_ratio <= -4)] = -4
print(np.min(log_odds_ratio), np.max(log_odds_ratio))

capped_odds_ratio = 10**log_odds_ratio

plt.hist(capped_odds_ratio, bins=100, range=(0,10))
plt.xlabel("Odds Ratio")
#plt.show()
plt.title("tmp")
save_plt()
plt.close()

plt.hist(log_odds_ratio, bins=100, range=(-5, 5))
plt.xlabel("Log Odds Ratio M1 vs M2")
#plt.show()
plt.title("tmp")
save_plt()
plt.close()

# +
# Impact of selecting a threshold

thresholds = np.arange(-5, 5, 0.1)
remaining = []
for i in thresholds:
    remaining.append(thresh_sel(i, log_odds_ratio))

fig, ax = plt.subplots(1, 1)
remaining = np.array(remaining)
ax.plot(thresholds, remaining)
ax.set(xlabel='Log10 odds threshold', ylabel='Data Remaining',title="tmp")
save_plt()
plt.close()


# +
    
bait_box_plot(ds, var="CRL_E")

# -

bait_box_plot(ds, var="CRL_C")

bait_box_plot(ds, preysel=False)#, boxkwargs={'ylim': 50})


# +

corr_mean_var_action = partial(corr_action, col1='rAv', col2='rVar')


thresholds = np.arange(0, 211.75, 0.1)

corr = slice_up_df_metric(df_new, thresholds, 'rAv', operator.le, corr_mean_var_action)
rcorr, pval = zip(*corr)
n_remaining = slice_up_df_metric(df_new, thresholds, 'rAv', operator.le, n_remaining_action)


# +


# -

wf = partial(compare_window_f, b=30)
corr_list = slice_up_df_metric(df_new, thresholds, 'rAv', wf, corr_mean_var_action)
n_remaining_win = slice_up_df_metric(df_new, thresholds, 'rAv', wf, n_remaining_action)
rcorr_win, pval_win = zip(*corr_list)

#plt.plot(n_remaining, rcorr, 'k.')
#plt.xlabel("N datapoints remaining")
#plt.ylabel("Mean variance correlation")
#plt.title("tmp")
#save_plt()
#plt.close()
#
#simple_scatter(x=rcorr, y=pval, xname='Pearson R', yname='P-val')
#plt.title("tmp")
#save_plt()
#plt.close()

simple_scatter(x=thresholds, y=rcorr, xname='SC threshold', yname='Pearson R')
plt.close()

# +
# Low abundance
sel = df_new['rAv'] <= 200
sel2 = df_new['bait'] != 'LRR1'
sel = sel & sel2
x = df_new.loc[sel, 'rAv'].values
y = df_new.loc[sel, 'rVar'].values

simple_scatter(x, y, xname='rAv', yname='rVar')
plt.close()
# -

# +
fig, ax = plt.subplots(1, 2, sharex=True)
aa_max_len = 5_000
sns.histplot(df_new[df_new['aa_seq_len'] <= aa_max_len], 
             x='n_possible_first_tryptic_peptides', 
             y='cMax',
             cmap='hot', ax=ax[0], cbar=True)
ax[0].set_title("Control Likelihood")
sns.histplot(df_new[df_new['aa_seq_len'] <= aa_max_len], 
             x='n_possible_first_tryptic_peptides', 
             y='rMax',
             cmap='hot', ax=ax[1], cbar=True)
ax[1].set_title("AP Likelihood")
plt.tight_layout()
#plt.title("Control Likelihood")
save_plt()
plt.close()

print("This plot can be used to formulate a data likelihood")
print("The likelihood of the Max spectral count | n_first_tryptic_peptides")
print("We are interested in protein abundance")
print("We need p(Max SC | abundance, n_first_tryptic_peptides)")
print("Specifically we are interested in the abundance at a specific time")
print("Not the abundance at the detector")
print("We are interested in the abundance of the peptide in the mixture")
print("If we assume an unkown abundance distribution (can place a prior)")
print("That is indpendent of N possible first tryptic peptides")
print("")

# +
#sns.regplot(df_new, x='rAv', y='rMax')
#m=1.15
#y = df_new['rAv'].values * m
#x=df_new['rAv'].values
#
#plt.plot(x, y, 'r')
## -
#
#sns.regplot(df_new, x='rAv', y='rMin')
#
#sns.regplot(df_new, x='cAv', y='cMax')
#
#sns.regplot(df_new, x='cAv', y='cMin')
#
#sns.regplot(df_new, x='rMin', y='rMax')
#
#
#plt.plot(df_new['rAv'], df_new['rMax'].values - df_new['rMin'], 'k.')
#
#sns.regplot(df_new, x='cMin', y='cMax')
#
#
#
#sel = df_new['rAv'] <= 12
#sns.regplot(df_new[sel], x='rAv', y='rMax')
#
#sel1 = df_new['cAv'] <=10
#sel2 = df_new['bait'] != 'LRR1'
#sel = sel1 & sel2
#sns.regplot(df_new[sel], x='cAv', y='cVar')
#
#df_new.loc[sel, csel + ['cAv']].sort_values('cAv', ascending=False).iloc[150:200, :]
#
#plt.plot(np.arange(0, 30), jax.scipy.stats.poisson.pmf(np.arange(0, 30), 15))
#
## +
#sel2 = df_new['cAv'] <= 5
#sns.regplot(df_new[sel2], x='cAv', y='cMax')
#x=df_new.loc[sel2, 'cAv']
#m=4
#y = m * x
#
#plt.plot(x, y, 'r')
## -
#
#y2 = np.max(df_new.loc[sel, rsel].values, axis=1)
#simple_scatter(x, y2, xname='rAv', yname='rMax')
#m=1
#y = m * x
#plt.plot(x, y, 'r')
#
#y2
#
#df_new
#
#simple_scatter(x=thresholds, y=n_remaining, xname='SC threshold', yname='N remaining')
#
#plt.hist(pval_win, bins=50)
#plt.show()
#
#simple_scatter(x=thresholds, y=rcorr_win, xname='SC threshold', yname='Win R')
#
#simple_scatter(x=thresholds, y=n_remaining_win, xname='SC threshold', yname='N remaining')
#
#
#
#sp.stats.pearsonr
#
#operator.lt(df_new['rAv'].values, 2)
#
#bait_box_plot(ds, preysel=False, var='CRL_C')

# +
# Check the double counting of the condition
#x = ds['CRL_C']
#k = 'CUL5'
#k2 = 'ELOB'
#x2 = x.sel(bait=k, condition='wt')
#
#x2[np.any((x.sel(bait=k, condition='vif').values != x.sel(bait=k2, condition='wt').values), axis=1)]
## -

di = numpyro.render_model(model_test,render_distributions=True, render_params=True,
                     filename="ModelTest.png")
#plt.title("Model Test")
#save_plt()
#plt.close()
# +

print(f"N free parameters {h(5 * 3 + 5 * 3 * 3062)}")  
# -

x = np.arange(0, 1, 0.01)
a=2
b=6
y = jax.scipy.stats.beta.pdf(x, a=a ,b=b)
#y = jax.scipy.stats.expon.pdf(x* 250, 100)
plt.plot(x * 300, y, 'k.')
plt.text(0.8 * 250, 2.0, f"a={a}\nb={b}")
plt.title("Beta")
save_plt()
plt.close()

# +
#plt.hist(np.ravel(np.std(df_new[rsel].values, axis=1)), bins=50, density=True)

x = np.arange(0, 30, 0.1)
for rate in np.arange(0, 2.2, 0.2):
    plot_exp_dens(x, rate)
#y = np.exp(dist.Exponential(exp_rate).log_prob(x))
plt.ylabel("Probability density / rate")
#plt.text(25, 0.4, f"rate={exp_rate}")
plt.legend()
#plt.plot(x, y, 'k')
plt.title("Exponential")
plt.tight_layout()
save_plt()
plt.close()
# -

# ?dist.Normal

#x = np.arange(0, 300)
#plt.plot(x, np.exp(dist.Exponential(1/50).log_prob(x)))

sum(df_new['Av_diff'] < -10)

sns.scatterplot(df_new, x='Av_diff', y='SaintScore', alpha=0.1)
plt.title("tmp")
save_plt()
plt.close()

plt.hist(np.ravel(ds.sel(bait=["CBFB", "ELOB", "CUL5"])['CRL_E'].values), bins=100)
plt.title("tmp")
save_plt()
plt.close()

plt.hist(np.ravel(ds.sel(bait=["CBFB", "ELOB", "CUL5"])['CRL_C'].values), bins=100)
plt.title("tmp")
save_plt()
plt.close()


posterior_predictive['y_c'].shape

az.plot_trace(m4_data['sample_stats'], var_names=['lp'])

prior.keys()

prior['y_c'].shape

plt.hist(prior['y_c'], bins=100, label='prior control', alpha=0.5)
plt.hist(prior['y_e'], bins=100, label='prior experiment', alpha=0.5)
plt.hist(posterior_predictive['y_c'], bins=100, label='posterior control')
plt.hist(posterior_predictive['y_e'], bins=100, label='posterior experiment', alpha=0.5)
plt.show()

plt.hist(posterior_samples['a'], bins=50, label='Experiment')
plt.hist(posterior_samples['b'], bins=50, label='Control')
#plt.hist(prior['a'], bins=50)
plt.title(f"Posterior N={len(posterior_samples['a'])}")
plt.xlabel("Exponential rate")
plt.legend()
plt.show()

# +
# Rate 

x = np.arange(0, 300)
y = np.exp(dist.Exponential(1/80).log_prob(x))
plt.scatter(x, y)
# -

x = np.arange(0, 400)
y = np.exp(dist.HalfNormal(200).log_prob(x))
sns.scatterplot(x=x, y=y)
plt.show()

nuts_kernal = numpyro.infer.NUTS(model4)
mcmc = numpyro.infer.MCMC(nuts_kernal, num_warmup=1000, num_samples=10000, thinning=2)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, e_data=e_data, ctrl_data=ctrl_data, extra_fields=('potential_energy',))
model = model5

# +
hists = [posterior_samples['a'], posterior_samples['b'], posterior_samples['a'] - posterior_samples['b']]
labels = ['Experiment', 'Control', 'Difference']
alphas = [0.8, 0.8, 0.3]

multi_hist(hists, bins=50, labels=labels, alphas=alphas, xlabel='Poisson rate')
# -

posterior_samples['a'].shape

# +

start = 0
end = 100

# -


# +
# Variants - number of chains
# Keep the model as a variable


# +

# +
tmp = df_new.iloc[:, :]
keys = jax.random.PRNGKey(13)
step = 10
crops = range(0, len(tmp), 1000)
start = 0

for stop in crops:
    if stop == 0:
        continue
    k1, rng_key = jax.random.split(rng_key)
    
    i_now = action_f(k1, tmp, start, stop)
    
    
    if start == 0:
        idata = i_now
    else:
        idata = concat_inf(idata, i_now)
    start = stop
        
stop = len(tmp)
k1, rng_key = jax.random.split(rng_key)
i_now = action_f(k1, tmp, start, stop)
idata = concat_inf(idata, i_now)


# -

# +
#ufunc = partial(un_center_and_scale_predictor, param_scale = kernel.meta['y_e'], safe=False)
#inf_data.posterior['a'] = xr.apply_ufunc(ufunc, inf_data.posterior.a)

# +
#az.plot_trace(inf_data, var_names=['a'], plot_kwargs={'xlim': (0, 10)})

# +
#yobs = inf_data.observed_data

# +
#models = run_and_merge_models(df_new, 0, 20, 10, jax.random.PRNGKey(13))
# -

"""
Let's say you have roughly 20,000 independant parameters.
How many should fall outside the prior and posterior predictive checks

"""


# +
# Analysis

summary_stats(idata)
plot_lp(idata)

# +
axes = az.plot_trace(idata.sample_stats['lp'])
ax = axes[:, 0].item()
ax.hist(np.ravel(inf_data.sample_stats['lp'].values), bins=100, color='C1', alpha=0.5)
ax.grid()
ax.set_ylabel("Frequency")

plt.show()

# +
fig, axs = plt.subplots(nrows=1, ncols=2)
alphas = [0.1, 0.1]
for i, key in enumerate(['a', 'b']):
    axs[i].plot(idata.posterior.stats.sel(stat=f"{key}_rhat").values,
                idata.posterior.stats.sel(stat=f"{key}_neff").values, 'k.', alpha=alphas[i])
    axs[i].set_xlabel(f"{key} Rhat")
    axs[i].set_ylabel(f"{key} Neff")
    axs[i].set_ylim((500, 6000))
    axs[i].set_xlim((0.998, 1.02))
fig.tight_layout()


#inf_data.posterior.stat

# +



# Shared dimensions: irow
# Shared variable names, y_c 

# Passing Frame



# -

sat = satisfaction(idata)

prior_sat = to_satisfaction_frame(sat.pp)
post_sat = to_satisfaction_frame(sat.Pp)

# +
# Prior and posterior predictive check example
# Checks of simulation

irow_sel = 9_000 #8432

fig, axs = plt.subplots(nrows = 1, ncols=2)
dsel = idata.prior_predictive.sel(irow=irow_sel)
dobs = idata.observed_data.sel(irow=irow_sel)

xrange = (0, 80)
x1 = np.ravel(dsel['y_e'].var(dim='rrep').values)

hdi_x1 = hpdi(x1)

axs[0].fill_betweenx([0, 200], hdi_x1[0], hdi_x1[1], alpha=0.3, interpolate=True, label='90% HDI')
axs[0].hist(x1, bins=20, color=grey, range=xrange)

axs[0].set_title("Prior Predictive")
#axs[0].vlines(dobs['y_e'].values, 0, 200, 'r', label='observed')
axs[0].vlines(dobs['y_e'].var(dim='rrep').values, 0, 150, color=red, label='Var(observed)')

xlabel = "$D_{i}^2$"
axs[1].set_title("Posterior Predictive")
axs[0].set_xlabel(xlabel)
axs[1].set_xlabel(xlabel)


dsel = idata.posterior_predictive.sel(irow=irow_sel)
x1 = np.ravel(dsel['y_e'].var(dim='rrep').values)
hdi_x1 = hpdi(x1)
axs[1].fill_betweenx([0, 250], hdi_x1[0], hdi_x1[1], alpha=0.3, interpolate=True, label='90% HDI')
axs[1].hist(x1, bins=20, color='k', alpha=0.5, range=(0, 20),
           label='Simulated variance')

#axs[1].vlines(dobs['y_e'].values, 0, 700, 'r', label='Observed data')

axs[1].vlines(dobs['y_e'].var(dim='rrep').values, 0, 200, color=red, label='observed variance')
axs[1].legend()
axs[0].set_ylabel("counts")

plt.suptitle("Predicted variance of $i^{th}$ bait-prey pair")
plt.savefig("ExamplePC.png", dpi=300)
plt.show()

# +
alpha=0.9


#fig, axs = plt.subplots(nrows=1, nc)
(prior_sat.sum() / len(prior_sat)).plot(kind='bar', label='prior predictive', color=red)
(post_sat.sum() / len(post_sat)).plot(kind='bar', label='posterior predictive', color=blue, alpha=0.9)
plt.ylabel("Proportion satisfied")
nsamples = 1000
nobs = len(tmp)
plt.title(f"Data satisfaction of predicted checks\nN MCMC samples {nsamples}\nN Obs {nobs}")
plt.grid()
plt.legend(loc=(0, 0.4))
plt.savefig(f"O{nobs}N{nsamples}.png", dpi=300)


# +

fig, ax = plt.subplots(1, 1)
phase_space_of_protein_interactions(ax)

# +
# Let's look at the data in the control that so are not satisfied

# +
#not_satisfied = idata.sel(irow=~post_sat['all_c'].values)
#satisfied = idata.sel(irow=post_sat['all_c'].values)
# -

cvar = idata['posterior_predictive'].sel(chain=0).var(dim=['crep', 'rrep'])
av_var = cvar.mean('draw')
av_var_std = cvar.std('draw')


observed_var = idata.observed_data.var(dim=['rrep', 'crep'])
plt.plot(observed_var['y_c'], av_var['y_c'], 'k.', alpha=0.1, label='control')
#plt.plot(df_new['cVar'].values, df_new['cVar'].values, 'b.', alpha=0.1, label='Equality')
plt.plot(observed_var['y_e'], av_var['y_e'], 'r.', alpha=0.1, label='AP')
x = np.arange(0, 200)
y = x
plt.plot(x, y, 'b-', label='equality')
plt.xlabel("Observed variance")
plt.ylabel("Average simulated variance")
plt.title("Posterior predictive variances")
#plt.xlim((0, 600))
plt.legend()
plt.savefig("PosteriorPredictiveVariances.png", dpi=notebook_dpi)


# +
cav = idata['posterior_predictive'].sel(chain=0).mean(dim=['crep', 'rrep'])
pav = idata['prior_predictive'].sel(chain=0).mean(dim=['crep', 'rrep'])
av_var = cav.mean('draw')
av_var_std = cav.std('draw')
observed_mean = idata.observed_data.mean(dim=['rrep', 'crep'])

plt.plot(observed_mean['y_c'], av_var['y_c'], 'k.', alpha=0.1, label='control')
#plt.plot(df_new['cVar'].values, df_new['cVar'].values, 'b.', alpha=0.1, label='Equality')
plt.plot(observed_mean['y_e'], av_var['y_e'], 'r.', alpha=0.1, label='AP')
plt.plot(observed_mean['y_e'], pav.mean('draw')['y_e'].values, 'b.', alpha=0.1, label='Prior')
x = np.arange(0, 200)
y = x
plt.plot(x, y, 'b-', label='equality')
plt.xlabel("Observed average")
plt.ylabel("Average simulated mean")
plt.title("Predictive Means")
plt.savefig("PredictiveMeans.png", dpi=notebook_dpi)
#plt.xlim((0, 600))
plt.legend()
# -

idata.observed_data

df_new['nprVar']

df_new.sort_values('nprVar', ascending=False).loc[:, ['bait'] + rsel + ['nprVar'] + c2]

df_new.loc[~post_sat['var_c'].values, rsel + csel + ['bait']]

# +
# Plot posterior shrinkage


with mpl.rc_context(rc_dict):
    fig, axs = plt.subplots(1, 2) # The original figure must share the rc_dict
    ylim = (0, 300)
    tmp = idata#.sel(irow=np.arange(20))
    a = np.ravel(tmp.posterior.sel(chain=0)['a'])
    b = np.ravel(tmp.posterior.sel(chain=0)['b'])
    xlabels = ["$$ \\alpha $$", "$$ \\beta $$"]
    #xlabels = ['a', 'b']
    boxen_pair(axs[1], [a, b], "Posterior", [0, 1], xlabels, rc_dict, ylim)
    a = np.ravel(tmp.prior.sel(chain=0)['a'])
    b = np.ravel(tmp.prior.sel(chain=0)['b'])
    boxen_pair(axs[0], [a, b], "Prior", [0, 1], xlabels, rc_dict, ylim, ylabel="Poisson rate")
    plt.savefig("PosteriorShrinkage.png", dpi=300)
    plt.close()


# -

with mpl.rc_context(mpl.rcParams):
    fig, ax = plt.subplots(1, 1)
    x1 = (idata.posterior['a'] - idata.posterior['b']).sel(chain=0, irow=3)
    x2 = (idata.prior['a'] - idata.prior['b']).sel(chain=0, irow=3)
    xs = [x2, x1]
    labels = ['Prior', 'Posterior']
    colors = [None, None]
    alphas = [0.7, 0.7]
    bins = [30] * len(xs)
    hists(ax, xs, labels, colors, alphas, bins)
    ax.legend()
    ax.set_xlabel("Poisson Rate Difference")
    ax.set_ylabel("Counts")
    plt.savefig("RateDifference", dpi=NOTEBOOK_DPI)


# +
# Posterior Certainty - cumulative probability mass above 0

# -

posterior_certainties = certainty_score((idata.posterior['a'] - idata.posterior['b']).sel(chain=0, col=0))
prior_certainties = certainty_score((idata.prior['a'] - idata.prior['b']).sel(chain=0, col=0))

# +
fig, ax = plt.subplots(1, 1)
hists(ax, [prior_certainties, posterior_certainties, df_new['SaintScore'].values], 
      labels=['Prior certainty score', 'Posterior certainty score', 'Saint score'],
      colors = [None, None, 'k'],
      alphas = [0.6, 0.9, 0.9],
      bins = [500, 500, 500], 
      hist_kwargs = None)#{'cumulative': True})

ax.set_ylim(0, 2500)
ax.set_xlabel("Score")
ax.vlines(0.6, 0, 2000, red, linestyles='dashed', label='typical Saint threshold')
ax.legend()
ax.set_ylabel("Count")
#plt.savefig("ScoreDistFar.png", dpi=notebook_dpi)
plt.savefig("ScoreDistNear.png", dpi=notebook_dpi)


# -

# ?xy_from

# +
fig, ax = plt.subplots(1, 2)
#ax[0].plot(prior_certainties, df_new['SaintScore'].values, 'b.', alpha=0.1)

xy_plot(ax[0], prior_certainties, df_new['SaintScore'].values, 'b.',
       xlabel="Certainty score", ylabel="Saint score", alpha=0.05, title="Prior")

xy_plot(ax[1], posterior_certainties, df_new['SaintScore'].values, 'C1.',
       xlabel="Certainty score", alpha=0.05, title="Posterior")

plt.savefig("SaintPostComparison.png", dpi=notebook_dpi)

# +
tmp = df_new.sort_values('cVar', ascending=False)

tmp.plot(x='rVar', y='cVar', style='k.', alpha=0.1)

# +
tmpr = df_new.sort_values('rVar', ascending=False)

#tmp.plot(x='rVar', y='cVar', style='k.', alpha=0.1)
tmpr.iloc[0:10][['bait', 'condition'] + rsel + ['rVar']]
# -

"""
mock have the same controls
vif + wt have sample controls
Each control is processed in parallel with 
"""

sel = df_new['bait'] == 'ELOB'
plt.hist(np.ravel(df_new.loc[sel, rsel].values - df_new.loc[sel, ['c1', 'c2', 'c3', 'c4']].values) - 20,
        bins=200, alpha=0.2)
plt.hist(np.ravel(df_new.loc[sel, rsel].values - df_new.loc[sel, ['c5', 'c6', 'c7', 'c8']].values),
        bins=200, alpha=0.2)
plt.hist(np.ravel(df_new.loc[sel, rsel].values - df_new.loc[sel, ['c9', 'c10', 'c11', 'c12']].values) + 20,
        bins=200, alpha=0.2)
plt.xlim(-50, 50)
plt.show()

sel = df_new['bait'] == 'CUL5'
plt.hist(np.ravel(df_new.loc[sel, rsel].values - df_new.loc[sel, ['c1', 'c2', 'c3', 'c4']].values) - 20,
        bins=400, alpha=0.2)
plt.hist(np.ravel(df_new.loc[sel, rsel].values - df_new.loc[sel, ['c5', 'c6', 'c7', 'c8']].values),
        bins=400, alpha=0.2)
plt.hist(np.ravel(df_new.loc[sel, rsel].values - df_new.loc[sel, ['c9', 'c10', 'c11', 'c12']].values) + 20,
        bins=400, alpha=0.2)
plt.xlim(-50, 50)
plt.show()

sel = df_new['bait'] == 'CBFB'
plt.hist(np.ravel(df_new.loc[sel, rsel].values - df_new.loc[sel, ['c1', 'c2', 'c3', 'c4']].values) - 20,
        bins=200, alpha=0.2)
plt.hist(np.ravel(df_new.loc[sel, rsel].values - df_new.loc[sel, ['c5', 'c6', 'c7', 'c8']].values),
        bins=200, alpha=0.2)
plt.hist(np.ravel(df_new.loc[sel, rsel].values - df_new.loc[sel, ['c9', 'c10', 'c11', 'c12']].values) + 20,
        bins=200, alpha=0.2)
plt.xlim(-50, 50)
plt.show()

# +
#sel1 = df_new['r1'] == 0
#$sel2 = df_new['c1'] == 0

sel3 = df_new['bait'] == 'CBFB'

sel4 = sel3 #sel1 & sel2
sel5 = sel4 & sel3
sel6 = np.sum(df_new[csel], axis=1) != 0
sel7 = sel5 & sel6


c1 = csel[0:4]
c2 = csel[4:8]
c3 = csel[8:12]

df_new.loc[sel7, ['rVar'] + rsel + c1].sort_values('rVar', ascending=False).iloc[0:50]

# +
# Batch effects
# Some replicates simply drop 



df_new.sort_values('cVar', ascending=False).iloc[0:50].loc[:, c1 + ['condition'] + c2 + ['condition'] + c3 + ['bait'] + rsel]
# -


"""
Compare 
"""
a = df_new[rsel].values
b = df_new[c1].values
(a @ b.T).shape

# +

# Compare control counts
tmp = ds.sel(bait=['CBFB', 'CUL5', 'ELOB'])
for bait_test in ['CBFB', 'CUL5', 'ELOB']:
    for p1, p2 in combinations(['wt', 'vif', 'mock'], 2):
        
        csel_map = {'CBFB': [0, 1, 2, 3],
                    'CUL5': [4, 5, 6, 7],
                    'ELOB': [8, 9, 10, 11]}
        
        columns = csel_map[bait_test]

        compare = tmp['CRL_C'].sel(
            condition=p1, bait=bait_test).values[:, columns] == tmp['CRL_C'].sel(
            condition=p2, bait=bait_test).values[:, columns]

        assert np.alltrue(~np.isnan(compare))
        # How many rows are False?
        # 96% of the controls between 
        print(f"{bait_test} {p1} {p2} : {np.sum(np.alltrue(compare, axis=1)) / len(compare[:, 0])}")
        #break

# WT & VIF Share most control
# -

ds_sel = ds.sel(bait=['CBFB', 'CUL5', 'ELOB'])#, preyu=['PEBB', 'CUL5', 'ELOB', 'vifprotein'])

rng_key = PRNGKey(13)
prey_max=100
model = partial(model8, prey_max=prey_max)

numpyro.render_model(model8, model_args=(ds_sel, None), render_distributions=True, render_params=True)


pickle.dump(model, open("model_test.p", "wb"))
m = pickle.load(open("model_test.p", "rb"))

# -

az.plot_trace(i8data.sample_stats['lp'])

i8data.sel(bait='CBFB', preyu='PEBB', condition='wt')

bait = 'CUL5'
preyu='PEBB'
condition = 'wt'
az.plot_trace(i8data.posterior.sel(bait=bait, condition=condition, preyu='CUL5'), 
              var_names=['epsilon', 'alpha', 'beta'])

az.plot_trace((i8data.posterior['alpha'] * i8data.posterior['beta']).sel(condition=condition, preyu='CUL5',
            bait=bait))

az.plot_trace(i8data.posterior['beta'].sel(bait='CBFB', preyu='vifprotein', condition='wt'))

ds_sel.sel(preyu=ds_sel.preyu[0:prey_max])

x = np.arange(0, 3, 0.1)
plt.plot(x, np.exp(dist.HalfNormal(scale=1).log_prob(x)))

df_new.sort_values('rMax', ascending=False).loc[:,['bait']+ rsel + csel].iloc[0:50]

# +
"""
Example of Mixed Discrete and continuous HMC

"""
def Nonef():
    return None

def identity(x):
    return x

probs = jnp.array([0.15, 0.3, 0.3, 0.25])
locs = jnp.array([-2, 0, 2, 4])
tmp = numpyro.render_model(mixed_model, model_args=(probs, locs), render_distributions=True, render_params=True)

len(data.test)

def cax_behavoir(carry, method): 
    method(carry.ax, *carry.args, **carry.kwargs) 
    return carry

def cplot(c):
    c.ax.plot(c.x, c.y, *c.plot_args, **c.plot_kwargs)
    return c

def cxlabel(c):
    c.ax.set_xlabel(c.xlabel, *c.xlabel_args, **c.xlabel_kwargs)
    return c


axplot(xf = lambda : np.arange(0, 1, 0.01),
       yf = lambda x: np.exp(dist.Beta(1.1, 4.0).log_prob(x)),
       tx=0.8, ty=2.0, text="1.1, 4.0", title="Beta1-1_4-0", savename="title",
       plot_f=lambda x, y, *args, ax, **kwargs: ax.plot(x, y, *args, **kwargs))

# +

numpyro.render_model(gpt_model, model_args=(jnp.ones((4, 10, 3, 3, 2)), False), 
                     render_distributions=True, render_params=True)

# +
# mixture example
mixture_data = jnp.array([0.0, 1.0, 10.0, 11.0, 12.0])

numpyro.render_model(mixture_example, model_args=(mixture_data,), render_distributions=True, render_params=True)
# -

x = np.arange(0, 10, 0.1)
y = np.exp(dist.Dirichlet(jnp.ones(1) * 0.5).log_prob(x))
plt.plot(x, y, 'k.')

dist.LogNormal(0., 2.).support

y = np.exp(dist.LogNormal(0.0, 2.0).log_prob(x))
plt.plot(x, y)

mcmc = run_mcmc(rng_key, mixture_example, mixture_data, num_samples=500, num_warmup=500)

mixture_samples = mcmc.get_samples()

X, Y = mixture_samples['locs'].T

# +
plt.figure(figsize=(8, 8), dpi=100).set_facecolor("white")
h, xs, ys, image = plt.hist2d(X, Y, bins=[20, 20])

plt.contour(
    jnp.log(h + 3).T,
    extent=[xs.min(), xs.max(), ys.min(), ys.max()],
    colors="white",
    alpha=0.8,
)
plt.title("Posterior density as estimated by collapsed NUTS")
plt.xlabel("loc of component 0")
plt.ylabel("loc of component 1")
plt.tight_layout()
plt.show()
# -

# ?dist.Mixture

dist.Beta(jnp.ones(2), jnp.ones(2)).to_event(1).event_shape


az.plot_trace(izdata.sample_stats, var_names=['lp'])
plt.tight_layout()

var_pp, var_obs = predictive_check(np.var, izdata.prior_predictive['sc'].sel(chain=0), 
                                   izdata.observed_data['sc'])
var_Pp, _ = predictive_check(np.var, izdata.posterior_predictive['sc'].sel(chain=0))

# +
var_pp = jax.jit(lambda x: jnp.var(x, axis=1))(izdata.prior_predictive['sc'].sel(chain=0).values)
var_Pp = jax.jit(lambda x: jnp.var(x, axis=1))(izdata.posterior_predictive['sc'].sel(chain=0).values)
var_obs = jax.jit(lambda x: jnp.var(x, axis=0))(izdata.observed_data['sc'].values)

var_pp = jax.jit(lambda x: jnp.mean(x, axis=0))(var_pp)
var_Pp = jax.jit(lambda x: jnp.mean(x, axis=0))(var_Pp)
#var_obs = jax.jit(lambda x: jnp.mean(x, axis=0))

# +
fig, ax = plt.subplots(1, 1)
cumulative = False
density=True
nbins=100
xlim = (0, 50)
ylim= None
hrange = (0, 200)

if density:
    ylabel= "Probability density"
    
def _helper(ax, x, color, alpha, label):
    ax.hist(x, bins=nbins, density=density, cumulative=cumulative, label=label, alpha=alpha, color=color,
           range=hrange)

_helper(ax, np.ravel(izdata.prior_predictive['sc'].values), label='Prior Pred', color='k', alpha=0.8)
_helper(ax, np.ravel(izdata.posterior_predictive['sc'].values), alpha=0.3, label='Post Pred', color='r')
_helper(ax, np.ravel(izdata.observed_data['sc'].values), label="Obs", color='b', alpha=0.1)
    
ax.set_xlabel("Spectral count")
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_ylabel(ylabel)
ax.legend()

plt.show()
plt.savefig("ZeroIPoissBeta11_4.png")

def obs_scatter(o, pp, Pp, iz=True, xlabel="Observed sample variance", ylabel="Simluated mean sample variance"):
    if iz:
        o = np.ravel(obs.values)
        pp = np.ravel(pp.mean('draw').values)
        Pp = np.ravel(Pp.mean('draw').values)

    fig, ax = plt.subplots(1, 1)
    ax.plot(o, pp, 
            'b.', alpha=0.1, label='Prior')

    ax.plot(o, Pp,
           'r.', alpha=0.1, label='Posterior')

    ax.plot(o, o, label='y=x', color='grey', alpha=0.3)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


fig, ax = obs_scatter(np.ravel(var_obs), np.ravel(var_pp), np.ravel(var_Pp), iz=False)
fig.savefig("ZeroIPoissBeta1-1_4_1000_prey.png", dpi=notebook_dpi)

fig, ax = obs_scatter(np.ravel(var_obs), np.ravel(var_pp), np.ravel(var_Pp), iz=False)
ax.set_ylim((0, 400))
fig.savefig("ZeroIPoissBeta23_2_1000_preyZoom.png", dpi=notebook_dpi)

# +
@jax.jit
def simulated_min_max_gap(x):
    return jnp.max(x, axis=1) - jnp.min(x, axis=1)

min_max_Pp = simulated_min_max_gap(izdata.posterior_predictive['sc'].sel(chain=0).values)
min_max_pp = simulated_min_max_gap(izdata.prior_predictive['sc'].sel(chain=0).values)

@jax.jit
def observed_min_max_gap(x):
    return jnp.max(x, axis=0) - jnp.min(x, axis=0)

min_max_obs = observed_min_max_gap(izdata.observed_data['sc'].values)

# +
fig, ax = obs_scatter(np.ravel(min_max_obs), np.ravel(min_max_pp.mean(axis=0)), 
                      np.ravel(min_max_Pp.mean(axis=0)), iz=False,
                     xlabel="Observed max min gap", ylabel="Simulated mean max min gap")

fig.savefig("ZeroIPoissBeta1-1_4_1000_prey_Max_min_gap.png", dpi=notebook_dpi)
# -

min_max_Pp.mean(axis=0)

min_max_obs.shape

var_pp.mean('draw').to_numpy().flatten()

var_obs

# +
fig, ax = plt.subplots(1, 1)

a = var_pp.isel(bait=0, condition=0, preyu=0, test=0).values
b = var_Pp.isel(bait=0, condition=0, preyu=0, test=0).values
ax.hist(np.ravel(a), bins=100, alpha=0.1)
ax.hist(np.ravel(b), bins=100)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
plt.show()
# -

plt.hist(a, cumulative=True)
plt.hist(b, color='C1', cumulative=True)
plt.xlim(0, 100)

b

plt.hist(np.ones(100))
plt.hist(np.ones(100), alpha=0.5)

plt.hist

izdata.observed_data['sc'].sel(bait='CUL5',
                                     condition='wt',
                                     test=True, preyu=preyu[1])

preyu = izdata.posterior_predictive.preyu
izdata.posterior_predictive['sc'].sel(chain=0, bait='CUL5',
                                     condition='wt',
                                     test=True, preyu=preyu[1])

# +
fig, ax = plt.subplots(1, 1)

ax.hist(np.ravel(izdata.sel(bait='CUL5', 
                            preyu=izdata.posterior.preyu[0], 
                            condition='wt', test=True).posterior_predictive['sc']))
# -



idata.preyu

# +
fig, ax = plt.subplots(1, 1)

predictive_hist(ax, izdata.posterior_predictive['sc'].sel(bait='CUL5', 
                                                  preyu=izdata.posterior_predictive.preyu[0],
                                                  condition='wt', test=True),
               observed=izdata.observed_data['sc'].sel)
# -

posterior_predictive

order = ('preyu', 'bait', 'condition', 'test')
plt.errorbar(np.ravel(var_obs.transpose(*order)),
             np.ravel(var_pp.mean('draw').transpose(*order)),
             yerr=np.ravel(var_pp.std('draw').transpose(*order)), fmt='b.', alpha=0.1)
#plt.plot(np.ravel(var_obs.transpose(*order)),
#         np.ravel(var_Pp.mean('draw').transpose(*order)), 'r.', alpha=0.1)
plt.show()



izdata.posterior_predictive['sc'].sel(chain=0)

izdata.observed_data['sc']

sd = get_summary_stats(mcmc)
plt.plot(np.ravel(sd['n_eff']), np.ravel(sd['r_hat']), 'b.', alpha=0.5)
plt.xlabel('N effictive samples')
plt.ylabel("R hat")
plt.show()
print("")

sd['n_eff']

sd['lam'].keys()

az.plot_trace(izdata.posterior.sel(bait='CUL5'))
plt.tight_layout()

izdata.posterior.condition

sd = summary(mcmc.get_samples())

xr.Dataset(summary(mcmc.get_samples()))

sd['lam']['r_hat'].shape

data.preyu

# ?dist.HalfNormal

data.sel(bait='LRR1', condition='wt')

set(df_new[df_new['bait']=='LRR1']['condition'])

data.sel(preyu='CUL2', bait='LRR1', condition='mock')

df_new.loc['GDE',['bait', 'condition'] + rsel + csel]

df_new[df_new['bait']=='LRR1'].sort_values('rVar', ascending=False)[rsel + csel].iloc[0:50]

df_new['bait'] 

data = model_data_from_ds(ds)

for bait in data.bait.values:
    print(bait)

data.loc[:, 'CBFB', :, :, True]

data.loc[:, 'ELOB', :, :, :]

data.transpose('preyu', 'rep', 'test', 'condition', 'bait')

# +
plt.hist(np.ravel(ds['CRL_C'].values), bins=100, alpha=0.5, range=(0, 50))
plt.hist(np.ravel(ds['CRL_E'].values), bins=100, alpha=0.9, range=(0, 50))

plt.show()
# -
numpyro.render_model(zero_inflated_spectral_counts,
                    model_args=(obs_e, obs_c),
                    render_params=True,
                    render_distributions=True)

obs_e = df_new.loc[:, rsel].values[0:100]
obs_c = df_new.loc[:, c1].values[0:100]
k = numpyro.infer.NUTS(zero_inflated_spectral_counts)
mcmc = MCMC(k, num_warmup=500, num_samples=1000)
mcmc.run(PRNGKey(1), obs_e, obs_c)

mcmc.print_summary()

samples = mcmc.get_samples()

mcmc.print_summary()

dist.ZeroInflatedPoisson(0.1, 5).sample(key)

numpyro.render_model(zero_inflated_model, model_args=obs)

x = np.arange(0, 1, 0.01)
y = np.exp(dist.Beta(2.0, 2.3).log_prob(x))
plt.plot(x, y)

kernel = MixedHMC(HMC(mixed_model, trajectory_length=1.2), num_discrete_updates=20)
mcmc = MCMC(kernel, num_warmup=10000, num_samples=10000)
mcmc.run(jax.random.PRNGKey(0), probs, locs)

samples_m = mcmc.get_samples()
mcmc.print_summary()

numpyro.infer.NUTS

samples_m

nuts_kernal = numpyro.infer.NUTS(model8)
mcmc = numpyro.infer.MCMC(nuts_kernal, num_warmup=1000, num_samples=1000, thinning=1)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, purification=cul5_e, control=cul5_c, extra_fields=('potential_energy',))

samples = mcmc.get_samples(group_by_chain=True)

summary_dict = summary(samples)

posterior_predictive = numpyro.infer.Predictive(model8, samples)(jax.random.PRNGKey(1), cul5_e, cul5_c)

posterior_predictive['y_c'].shape

m8data = az.from_numpyro(mcmc)

az.plot_trace(m8data.sample_stats['lp'])

# Background on Mass Spectrometry data analysis
# - RawFile: Each RawFile corresponds to an individual experimental / technical replicate / run
# - Condition: CBFB_mock, CBFB_wt, CUL5_vif are examples of conditions. A condition can be a mix of bait and 
# infection. 
# - The Experiemntal Design tables from https://ftp.pride.ebi.ac.uk/pride/data/archive/2019/04/PXD009012/ displayed in the cell below.
# - Based on these table
# - The MaxQuant evidence.txt -> spectral counts
# - Based on the artMS github page https://github.com/biodavidjm/artMS/blob/9fca962bc5c36926d9eeeda1a32a8210ce665f7a/R/evidenceToSaintExpressFormat.R#L6 artMS uses the MS/MS count column to obtain the spectral counts
# - One could use MS/MS Count, or MS1 Intensity
# - artMS filters for potential contaminants.
# - For
#
#
# ## Key conclusions for modeling
#
plt.hist(evidence['Length'].values, bins=45)
plt.show()

plt.hist(evidence['MS/MS Count'].values, bins=40)
plt.xlabel("MS/MS Count")
plt.show()
print(f"Min Max {np.min(evidence['MS/MS Count'].values), np.max(evidence['MS/MS Count'].values)}")

plt.plot(evidence['Length'].values, evidence['MS/MS Count'].values, 'b.', alpha=0.01)
plt.show()

"""
What is a protein's spectral count? The sum of peptide spectral counts.
What is a peptide spectral count? The number of MS/MS spectra that map to the peptide.
0 to 17, often 1 MS/MS spectra match to a peptide.
"""

"""
Evidence.txt feature
- Sequence - coverage
- Length
- Missed cleavages
- RawFile - match the condition / experimental design
"""

plt.hist(evidence['Intensity'].values, bins=100, range=(0, 1e8))
plt.show()

len(set(df_new['Prey'].values))

"""
Raw             Condition   Replicate
FU20151013-21	CBFB_HIVvif	1
FU20151013-26	CBFB_HIVvif	2
FU20151030-02	CBFB_HIVvif	3
FU20151030-04	CBFB_HIVvif	4
FU20151013-28	CBFB_HIVvif_MG132	1
FU20151013-30	CBFB_HIVvif_MG132	2
FU20151030-06	CBFB_HIVvif_MG132	3
FU20151030-08	CBFB_HIVvif_MG132	4
FU20151020-06	CBFB_HIVwt	1
FU20151013-38	CBFB_HIVwt	2
FU20151030-11	CBFB_HIVwt	3
FU20151030-13	CBFB_HIVwt	4
FU20151013-42	CBFB_HIVwt_MG132	1
FU20151020-04	CBFB_HIVwt_MG132	2
FU20151030-15	CBFB_HIVwt_MG132	3
FU20151030-17	CBFB_HIVwt_MG132	4
FU20151013-11	CBFB_mock	1
FU20151013-13	CBFB_mock	2
FU20151030-24	CBFB_mock	3
FU20151030-26	CBFB_mock	4
FU20151013-15	CBFB_mock_MG132	1
FU20151013-17	CBFB_mock_MG132	2
FU20151030-28	CBFB_mock_MG132	3
FU20151030-30	CBFB_mock_MG132	4
FU20151013-03	JPmock	1
FU20151013-05	JPmock_MG132	1
FU20151013-07	JPwt	1
FU20151013-09	JPwt_MG132	1
FU20151005-05	control_mock	1
FU20151005-09	control_HIVwt	1
FU20160111-03	control_mock	2
FU20160111-07	control_HIVwt	2
FU20151005-07	control_mock_MG132	1
FU20151005-11	control_HIVwt_MG132	1
FU20160111-05	control_mock_MG132	2
FU20160111-09	control_HIVwt_MG132	2
FU20151005-26	CUL5_HIVvif	1
FU20151005-28	CUL5_HIVvif	2
FU20160111-19	CUL5_HIVvif	3
FU20160111-21	CUL5_HIVvif	4
FU20151005-30	CUL5_HIVvif_MG132	1
FU20151005-32	CUL5_HIVvif_MG132	2
FU20160111-23	CUL5_HIVvif_MG132	3
FU20160111-25	CUL5_HIVvif_MG132	4
FU20151005-42	CUL5_HIVwt	1
FU20151005-44	CUL5_HIVwt	2
FU20160111-27	CUL5_HIVwt	3
FU20160111-29	CUL5_HIVwt	4
FU20151005-46	CUL5_HIVwt_MG132	1
FU20151005-48	CUL5_HIVwt_MG132	2
FU20160111-31	CUL5_HIVwt_MG132	3
FU20160111-33	CUL5_HIVwt_MG132	4
FU20151005-13	CUL5_mock	1
FU20151005-15	CUL5_mock	2
FU20160111-11	CUL5_mock	3
FU20160111-13	CUL5_mock	4
FU20151005-21	CUL5_mock_MG132	1
FU20151005-23	CUL5_mock_MG132	2
FU20160111-15	CUL5_mock_MG132	3
FU20160111-17	CUL5_mock_MG132	4
FU20170905-33	control_mock_MG132	1
FU20170905-37	control_mock_MG132	2
FU20170905-67	control_mock_MG132	3
FU20170905-71	control_mock_MG132	4
FU20170905-35	control_wt_MG132	1
FU20170905-39	control_wt_MG132	2
FU20170905-69	control_wt_MG132	3
FU20170905-73	control_wt_MG132	4
FU20170905-09	ELOB_HIVvif_MG132	1
FU20170905-22	ELOB_HIVvif_MG132	2
FU20170905-43	ELOB_HIVvif_MG132	3
FU20170905-56	ELOB_HIVvif_MG132	4
FU20170905-11	ELOB_HIVwt_MG132	1
FU20170905-24	ELOB_HIVwt_MG132	2
FU20170905-45	ELOB_HIVwt_MG132	3
FU20170905-58	ELOB_HIVwt_MG132	4
FU20170905-07	ELOB_mock_MG132	1
FU20170905-20	ELOB_mock_MG132	2
FU20170905-41	ELOB_mock_MG132	3
FU20170905-54	ELOB_mock_MG132	4
"""

df_new[df_new['n_possible_first_tryptic_peptides']<50]
plt.hist(df_new['aa_seq_len'], bins=100, range=(0, 2000))
plt.show()
plt.hist(np.ravel(df_new[rsel].values), range=(0, 50), bins=100)
plt.show()
np.min(df_new['n_possible_first_tryptic_peptides'])

# +
# (prey, cell_type, infection, replicates)
# prey : 0...
# bait : CBFB, CUL5, ELOB, LRR1, Parent
# condition: wt, dvif, mock

# Do bait share the same controls accross conditions?
# wt & mock no
# wt & vif yes
# 

# -

dist.Poisson(jnp.ones((3000, 4, 3, 4))).batch_shape
df_new[df_new['bait'] == 'ELOB'].loc['vifprotein',['bait', 'condition'] + csel]
key = PRNGKey(13)
dist.MultivariateNormal(jnp.array([1, 10, 4]), jnp.eye(3)).log_prob(jnp.array([1, 2, 3]).reshape((1, 3)))
df_new[df_new['PreyName']=='LLR1'].loc[:, csel]

# +
from_scores = True
log_scale = True
yl1 = 'TP'
yl2 = 'PP'
if from_scores:
    tp_scores = scores[:, 0]
    tp_over_pp_scores = np.divide(scores[:, 0], scores[:, 1] ** 2, where=scores[:, 1] != 0)
    pp_ = scores[:, 1]
    unk = scores[:, 1] - scores[:, 0]
if log_scale:
    tp_scores = np.log10(tp_scores)
    pp_ = np.log10(pp_)
    yl1 = 'log10 ' + yl1
    yl2 = 'log10 ' + yl2
    
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].plot(t, tp_scores, label=yl1)
axs[0].plot(t, pp_, label=yl2)
axs[0].set_xlabel(col)
axs[0].legend()
axs[1].plot(t, tp_over_pp_scores, label='TP / PP^2', color='C2')
axs[1].set_xlabel(col)
axs[1].legend()
fig.tight_layout()
# -

plt.plot(scores[:, 0], unk, 'k.', alpha=0.1)
plt.xlabel('TP')
plt.ylabel('Unknown (PP - TP)')
plt.xlim(0, 100)
plt.ylim

tp(set2frozen_pairs(two_col_union(df_new, 'bait', 'PreyName')), ref_set)


elob_av = df_new.loc[df_new['bait']=='ELOB', rsel + csel].mean()
cul5_av = df_new.loc[df_new['bait'] == 'CUL5', rsel + csel].mean()
cbfb_av = df_new.loc[df_new['bait'] == 'CBFB', rsel + csel].mean()
lrr1_av = df_new.loc[df_new['bait'] == 'LRR1', rsel + csel].mean()

d = cbfb_av
plt.plot(d[rsel], d[c1], 'k.')
plt.plot(d[rsel], d[c2], 'r.')
plt.plot(d[rsel], d[c3], 'b.')

plt.plot(elob_av[rsel], elob_av[c1], 'k.')
plt.plot(elob_av[rsel], elob_av[c2], 'r.')
plt.plot(elob_av[rsel], elob_av[c3], 'b.')

tmp[['bait', 'condition'] + rsel + ['cVar'] + csel].loc['MYH9'].sort_values('bait')

tmp[csel].iloc[0: 50]

# +
# Batch effects 
# -

df_new.sort_values('rVar', ascending=False).iloc[0:50, :]

# +
# Plot the observed variance to satisfaction
# -

pp_s, Pp_s, o_s = inf_get_summary_stats(inf_data)

pc = satisfaction(inf_data)
ppc = pd.DataFrame(index=inf_data.posterior.coords['irow'].values, columns=['min', 'max', 'var', 'mean'])
sat = satisfaction(inf_data)
ds_hpdi(inf_data.posterior_predictive.sel(chain=0).mean(dim=['crep', 'rrep']), 
        var_names=['y_c', 'y_e'], dim_name='draw')
inf_data.posterior_predictive.sel(chain=0).mean(dim=['crep', 'rrep']).map(f)
f(inf_data.posterior_predictive.sel(chain=0).mean(dim=['crep', 'rrep']).y_c, dim_name='draw')
inf_data.posterior_predictive.sel(chain=0).mean(dim=['rrep', 'crep']).reduce(hpdi, dim=['draw'])
inf_data.posterior_predictive.sel(chain=0).mean(dim=['rrep', 'crep']).coords
pps, Pps = satisfaction(inf_data)
pps[0].mean(dim=['draw'])
pp_stats, Pp_stats, obs_stats = satisfaction(inf_data)
pp_stats.mean - obs_stats.mean
inf_data.observed_data.mean(dim=['crep', 'rrep'])
plot_prior(m_data, start=0, end=10)
predictive_check(m_data.sel(irow=0))

q = np.array([0.1, 0.5, 0.9])
q_func = partial(np.quantile, q=q, axis=0)
q_func(m_data.prior_predictive.sel(chain=0)['y_c'].reduce(np.max, dim='crep')).T

test_stats = {'min': np.min, 'max': np.max, 'var': np.var, 'med': np.median}
data_sets = {'pp': m_data.prior_predictive, 'Pp': m_data.posterior_predictive, 'o': m_data.observed_data}
summary_stats = {}

for Tname, T in test_stats.items():
    for dname, data_set in data_sets.items():
        if dname != 'o':
            s = data_set.sel(chain=0).reduce(T, dim=['crep', 'rrep'])
        else:
            s = data_set.reduce(T, dim=['crep', 'rrep'])
        for key in ['y_e', 'y_c']:
            skey = dname + "_" + Tname + "_" + key
            if dname != 'o':
                summary_stats[skey] = q_func(s[key])
            else:
                summary_stats[skey] = s[key].values
        
plot_from_summary_stats(summary_stats, 'var', 'prior', 'y_c')

plt.errorbar(np.arange(len(median)), median)

min_o = m_data.observed_data.reduce(np.min, dim=['crep', 'rrep'])
min_pp = m_data.prior_predictive.reduce(np.min, dim=['crep', 'rrep']).sel(chain=0).min(dim='draw')

x = np.arange(1000)
plt.plot(x, (min_o - min_pp).sortby('y_c')['y_c'].values, '.', label='Prior Predictive Control')
plt.plot(x,)
#plt.plot(x, (min_o - min_pp).sortby('y_c')['y_e'].values, '.', label='Preior Predictive AP')
plt.ylim(-4, 4)
plt.ylabel("Min Obs - min D")
plt.grid()

min_o = m_data.observed_data.reduce(np.min, dim=['crep', 'rrep'])
#max_o = m_data.observed_data.reduce(np.max, dim=['crep', 'rrep'])

min_pp = m_data.prior_predictive.reduce(np.min, dim=['crep', 'rrep']).sel(chain=0).min(dim='draw')
min_Pp = m_data.posterior_predictive.reduce(np.min, dim=['crep', 'rrep']).sel(chain=0).min(dim='draw')

a = (min_o - min_pp).sortby('y_c')['y_c'].values
b = (min_o - min_pp).sortby('y_c')['y_e'].values
c = (min_o - min_Pp)['y_c'].values
d = (min_o - min_Pp)['y_e'].values

ppc_boxen([a, b, c, d])

def in_hpdi(x, arr, prob, hpdi_fun=hpdi):
    """
    HPDI: Narrowest Interval with probability mass of prob
    Is it in the narrowest interval with 0.9 probability mass?
    Is it in the narrowest interval 
    1. Apply Test statistic
    2. Compute HPDI at row
    3. Check 
    """
    i = hpdi_fun(arr, prob)
    if i[0] < x < i[1]:
        return True
    else:
        return False

def in_hpdi(prob, m_data, dataset, ysel, repsel, chainsel, t):
    T_of_pp = m_data[dataset][ysel].sel(chain=chainsel).reduce(t, dim=repsel)
    hpdi_of_T = hpdi(T_of_pp, prob=prob, axis=0).T
    T_obs = m_data.observed_data[ysel].reduce(t, dim=repsel)
    a = hpdi_of_T[:, 0] <= T_obs
    b = T_obs < hpdi_of_T[:, 1]
    in_interval = a & b
    return in_interval


def hpdi_check(interval_probs, m_data, dataset, ysel, repsel, chainsel, t):
    """
    Return the Highest Posterior Density Interval with the smallest probability mass
    """
    shape = (m_data[dataset].dims['irow'], len(interval_probs))
    results = np.ones(shape) * 2
    for i in range(shape[1]):
        in_intervals = in_hpdi(prob=interval_probs[i], m_data=m_data, dataset=dataset,
                              ysel=ysel, repsel=repsel, chainsel=chainsel,
                              t=t)
        
        results[in_intervals, i] = interval_probs[i]
    results = np.min(results, axis=1)
    results[results==2] = -0.1
    return results



# +
thresholds = np.arange(0.999, 0.01, -0.005)
results = hpdi_check(thresholds, m_data, 'prior_predictive', 'y_c', 'crep', 0, np.max)

def hpdi_cdf(results, title, hist_kwargs={}):
    results[results==[-0.1]]=1.2
    plt.title(title)
    plt.hist(results, cumulative=True, bins=100, density=True, **hist_kwargs)
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid(which='both')
    plt.xlim(0, np.max(results[results <= 1]))
    plt.xlabel("Minimal HPDI")
    plt.ylabel("Probability Mass")
    
hpdi_cdf(results, title="Prior Predictive T(x): Max")
# -

results = hpdi_check(thresholds, m_data, 'prior_predictive', 'y_c', 'crep', 0, np.min)

hpdi_cdf(results, title='Prior Predictive T(x): Min')

# +
results = hpdi_check(thresholds, m_data, 'prior_predictive', 'y_c', 'crep', 0, np.var)

r2 = hpdi_check(thresholds, m_data, 'prior_predictive', 'y_e', 'rrep', 0, np.var)
# -

hpdi_cdf(results, title='Prior Predictive T(x): Var', hist_kwargs={'label': 'control'})
plt.hist(r2, label='AP', cumulative=True, density=True, alpha=0.5, bins=100)
plt.xlim(0, 0.8)
plt.legend()
plt.show()

results = hpdi_check(thresholds, m_data, 'posterior_predictive', 'y_c', 'crep', 0, np.var)

r2 =      hpdi_check(thresholds, m_data, 'posterior_predictive', 'y_e', 'rrep', 0, np.var)

hpdi_cdf(results, title='Prior Predictive T(x): Var', hist_kwargs={'label': 'control'})
plt.hist(r2, label='AP', cumulative=True, density=True, alpha=0.5, bins=100)
plt.xlim(0, 0.8)
plt.legend()
plt.show()

"""
Conclusions


"""

results = hpdi_check(thresholds, m_data, 'prior_predictive', 'y_c', 'crep', 0, np.mean)

results[results==[-0.1]]=1.2
plt.title(f"T: Mean")
plt.hist(results, cumulative=True, bins=100, density=True)
plt.grid()
plt.xlim(0, np.max(results[results <= 1]))
plt.xlabel("Minimal HPDI")
plt.ylabel("Probability Mass")

plt.plot(np.arange(1000), sorted(results, reverse=False), 'k.')

# ?np.sort

a = in_hpdi(0.9999, m_data, dataset='prior_predictive', ysel='y_c', repsel='crep', chainsel=0, t=np.max)

m_data['posterior'].dims['irow']



in_interval

m_data.observed_data.reduce(lambda x, axis: hpdi(x, axis=axis), dim=['crep', 'rrep'])

hpdi(m_data.observed_data['y_c'].values, axis=1)



# What decile does the data fall in?
hpdi()

# +
max_o = m_data.observed_data.reduce(np.max, dim=['crep', 'rrep'])
#max_o = m_data.observed_data.reduce(np.max, dim=['crep', 'rrep'])

max_pp = m_data.prior_predictive.reduce(np.max, dim=['crep', 'rrep']).sel(chain=0).max(dim='draw')
max_Pp = m_data.posterior_predictive.reduce(np.max, dim=['crep', 'rrep']).sel(chain=0).max(dim='draw')

a = (max_pp - max_o).sortby('y_c')['y_c'].values
b = (max_pp - max_o).sortby('y_c')['y_e'].values
c = (max_Pp - max_o)['y_c'].values
d = (max_Pp - max_o)['y_e'].values
# -

ppc_boxen([a, b], ['Control', 'AP'], title="Max: T(y~) - T(y)")

ppc_boxen([c, d], ['Control', 'AP'])

# +
# Magnitude of the deviation outside the 
# -

ticks = np.hstack([np.arange(i) for i in [4, 4]]).reshape((2, 4))

"$y \sim $"

# ?min_o.sortby

m_data.reduce(np.min, dim=['crep', 'rrep'])

plt.plot(np.arange(20000), np.arange(20000), 'k.')

# +

plt.plot(np.arange(len(y)), sorted(y, reverse=True), 'k.')
plt.ylabel('Out of distribution score')
# -


# ?np.min



# +
#1 Does the observed value fall within the 1st and 9th decile of the prior?
#2 Does the observed value fall within the 1st and 9th decile of the posterior pred?
#3 How much did the model learn?

# What are the most uncertain distributions?


summary_stats['pp_min_y_e'].T - summary_stats['o_min_y_e'].reshape((1000, 1))
# -

summary_stats['pp_min_y_e'].T

summary_stats['o_min_y_e'].reshape((1000, 1))

q_func(s[key])

m_data.prior_predictive.sel(chain=0).reduce(np.min, dim=['crep', 'rrep'])

q_func(m_data.prior_predictive.sel(chain=0)['y_c'].reduce(np.min, dim='crep')).T

m_data.observed_data.reduce(np.min, dim=['crep', 'rrep'])

#m_data.posterior_predictive.sel(chain=0)['y_c'].reduce(np.mean, dim='crep').reduce(q_func, dim='draw')
# -

np.quantile(np.arange(10), [0.3, 0.6])

# ?np.quantile

m_data.posterior_predictive

hpdi(m_data.posterior['a'].values, axis=1)

predictive_check(m_data.sel(irow=0))

m_data.posterior_predictive['y_c'].mean('crep')

xr.apply_ufunc(np.mean, m_data.posterior)

# ?xr.apply_ufunc

m_data.prior_predictive

m_data.prior_predictive['y_c'].shape

neff_rhat = {'a_r_hat': summary_dict['a']['r_hat']}

samples = search.get_samples()

summary_dict = summary(samples, group_by_chain=False)

summary_dict['a'].keys()


def mcmc_rank_plot(summary_dict, stat='r_hat'):
    rhats = []
    for key in summary_dict:
        r = list(np.ravel(summary_dict[key][stat]))
        rhats = rhats + r
    
    rhats = sorted(rhats, reverse=True)
    plt.scatter(np.arange(len(rhats)), rhats)
    plt.ylabel(stat)
    plt.xlabel('Param Rank')
    


mcmc_rank_plot(summary_dict, stat='n_eff')

mcmc_rank_plot(summary_dict)

hpdi(posterior_samples['a'] - posterior_samples['b'])

plt.hist(posterior_samples['a'], bins=100, label='Experimental rate')
plt.hist(posterior_samples['b'], bins=100, label='Control rate')
plt.hist(posterior_samples['a'] - posterior_samples['b'], bins=100, label='Rate difference')
plt.legend()
plt.show()

# +

y= np.exp(dist.HalfCauchy(200).log_prob(x))
sns.scatterplot(x=x, y=y)
plt.show()
# -

posterior_predictive['y_c'][-5]

m4_data = az.from_numpyro(mcmc, prior=prior, posterior_predictive=posterior_predictive)



df_new.sort_values('Av_diff')

df_new.loc[:, rsel].mean()

ds.sel(bait=["CUL5", "ELOB", "CBFB", "LRR1"])['CRL_E'].mean(['rrep', 'preyu'])

nuts_kernal = numpyro.infer.NUTS(model2)
mcmc = numpyro.infer.MCMC(nuts_kernal, num_warmup=1000, num_samples=50000, thinning=10)
rng_key = jax.random.PRNGKey(13)
mcmc.run(rng_key, ds=ds, extra_fields=('potential_energy',))

posterior_samples = mcmc.get_samples()

posterior_predictive = numpyro.infer.Predictive(model2, posterior_samples)(jax.random.PRNGKey(1), ds)

prior = numpyro.infer.Predictive(model2, num_samples=500)(jax.random.PRNGKey(2), ds)

numpyro_data = az.from_numpyro(mcmc, prior=prior, posterior_predictive=posterior_predictive,
                              coords={"protein": ds.preyu.values, "cell": np.arange(5), "infection": np.arange(3)},
                              dims={"a": ["protein", "cell", "infection"]})

az.plot_trace(numpyro_data['sample_stats'], var_names=['lp'])

az.plot_trace(numpyro_data['posterior'], var_names=['Nc'])

post = numpyro_data['posterior']

az.plot_trace(post.sel(mu_dim_0=np.arange(0, 2)), var_names=['s'])

numpyro_data['posterior']


    

az.plot_forest(numpyro_data, var_names="N")

az.plot_trace(numpyro_data['posterior'], var_names=['N'])



a = jnp.ones((5,3)) * 2
b = jnp.arange(3062 * 5 * 3).reshape((3062, 5, 3))

(a * b)[:, :, 0]

samples = mcmc.get_samples()

ds.sel(preyu='ELOC')['CRL_C']

(ds.sel(bait=['ELOB', 'CBFB', 'CUL5'])['CRL_E'].sum('rrep') / 4 - ds['CRL_C'].sum('crep') / 12)

df_new.loc[:, 'log_odds_ratio'] = log_odds_ratio

df_new.loc[:, 'odds_ratio'] = odds_ratio

sns.scatterplot(df_new, x='log_odds_ratio', y='SaintScore')
plt.hlines(0.6, -3, 4, 'k')
plt.vlines(0, 0, 1, 'r')

# +
a = np.ravel(df_new.loc[:, [f"c{i}" for i in range(1, 5)]])
b = np.ravel(df_new.loc[:, [f"c{i}" for i in range(5, 9)]])
c = np.ravel(df_new.loc[:, [f"c{i}" for i in range(9, 13)]])
sns.boxplot([a, b, c])

sp.stats.ks_2samp(a, b)
# -

sp.stats.ks_2samp(a, a)

sp.stats.ks_2samp(b, c)

tmp = df_new['log_odds_ratio'].values.copy()
np.random.shuffle(tmp)
plt.plot(tmp, df_new['SaintScore'].values, 'k.')



 # ?sns.pairplot

n=600
df_new.sort_values('cAv', ascending=False).loc[:,
    ['rAv', 'cAv', 'rVar', 'cVar','bait', 'condition', 'SaintScore'] + rsel + csel ].iloc[n:50 + n]

query = 'PEBB_HUMAN'

df2[df2['PreyGene']==query]

df1[df1['PreyGene']==query]

df3[df3['PreyGene']==query]

tmp = np.ravel(ds.sel(bait='LRR1')['CRL_E'].values)
print(sum(tmp != 0))
plt.hist(tmp, bins=100)
plt.show()


# +
def m1(ds, preyrange=slice(0, None), 
       bait = ['CBFB', 'ELOB', 'CUL5'],
       conditions = ['wt', 'vif', 'mock']):
    
    
    prey_sel = ds.preyu[preyrange]
    
    d = ds.sel(bait=bait, condition=conditions, preyu=prey_sel, preyv=prey_sel)
    
    
    
# -

ds.sel(condition='mock', bait='CBFB')['CRL_E']

m1(ds)

ds.sel(bait=["CBFB", 'ELOB', 'CUL5'])

ds.sel(bait=['CBFB', 'ELOB', 'CUL5'])

ds.preyu[slice(0, None)]

ds.sel(bait=['LRR1', 'CBFB', 'CUL5', 'ELOB'], condition='mock')

df1[df1['Bait']=='CUL5wt_MG132'].sort_values('SaintScore', ascending=False).iloc[0:20, :]

bait_box_plot(ds, 'CRL_C')

ds.sel


def m2(ds):
    cbfb_preyname = 'Q13951'
    


ds.sel(condition='CBFB', preyu='CBFB')

df_new[df_new['bait'] == 'CBFB'].sort_values('SaintScore', ascending=False).iloc[0:20, :]

# +
# What values should we exclude from the analysis?

thresh_sel(0, log_odds_ratio)
# -

np.sum(odds_lambda == 0)

log_odds_ratio = np.log(odds_lambda) - np.log(odds_kappa)

plt.hist(np.array(odds_ratio), bins=100)
plt.show()

np.isnan(samples['k']).sum()

np.isnan(samples['l']).sum()

# +
# Vectorize loops
# -

a = np.array([[1, 2, 3],
              [5, 5, 5]])
b = np.array([10, 20])



jax.vmap(np.sum)(a, b.T)

jax.vmap(np.sum)(np.sum(a), b)



p1, p2 = posterior_odds(samples, df_new)

jax.scipy.stats.poisson.pmf(df_new[rsel].iloc[:, 0].values, 1)

jax.scipy.stats.poisson.pmf()

plt.hist(np.ravel(samples['l']), bins=100)
plt.show()

plt.hist(np.ravel(samples['k']), bins=100)
plt.show()





# +
def model(Yexp=None, Yctrl=None):
    
    numpyro.sample("C", dist.Poisson(kappa_), C=Yctrl)
    numpyro.sample("E", dist.Poisson(lambda_), E=Yexp)
    
    
    
# -

key = jax.random.PRNGKey(13)
ppc = jax.random.exponential(key)

# ?jax.random.exponential

df_new_all_json = df_new2json(df_new)
with open("../sm1/df_new_all.json", 'w') as f:
    json.dump(df_new_all_json, f)

# + language="bash"
# ls ../sm1
# -

# ?json.dump

df_new_all_json



sp.stats.beta(0.1, 0.9).pdf(xaxis / 401)

(df_new['condition'] == 'wt') | (df_new['condition'] == 'mock')

chain_mapping[chain_mapping['PDBID'] == '4n9f']

prey2seq

df_newjson = df_new2json(df_new)

df_newjson['Ncsel']



hist_bins, bin_edges, patches = plt.hist(df_new['Av_diff'].values, bins=100, range=(-5, 5))
plt.vlines([-1, 1], 0, 2000, 'r')
plt.title
plt.show()

diff_sel = ~(np.abs(df_new['Av_diff'].values) <= 1)
sum(diff_sel)
df_new2json(df_new[diff_sel])

df_new[df_new['condition'] == 'vif'].loc['RUNX1', ['condition', 'bait'] + csel]

set(df_new['condition'])

df_new[df_new['Prey'] == 'P69723'].loc[:, ['bait', 'condition'] + csel]

# +
xaxis = np.arange(0, 10, 0.5)

nremaining = [sum(~(np.abs(df_new['Av_diff'].values) <= i)) for i in xaxis]
thresh = 1

plt.title("Filtering thresholds")
plt.plot(xaxis, nremaining, 'k.')
plt.vlines(1, 0, 21000, 'r')
plt.ylabel('N datapoints remaining')
plt.xlabel("Spectral count threshold")
plt.show()
# -

diff_sel = ~(np.abs(df_new['Av_diff'].values) <= 1)
sum(diff_sel)




# +
x, y = xy_from(prey_pairs_df, 'rand', np.arange(0, 0.05, 0.005))

npairs = len(prey_pairs_df)
npdb_pos = sum(prey_pairs_df['pdb_pos'].values)

yplot = y / npdb_pos
xplot = x / npairs
plt.plot(xplot, yplot, 'k.', label='Random Classifier')
plt.ylabel(f"Fraction PDB Positives (N={h(npdb_pos)})")
plt.xlabel(f"Fraction Total Positives (N={h(npairs)})")

y2 = 1.0
x2 = npdb_pos / npairs
plt.vlines([0.0035, 0.015], 0, 1, label="Estimated fraction of true PPIs")


plt.plot(x2, y2, 'r+', label='PDB Benchmark')
xmul = 10
plt.plot(x2 * xmul, y2, 'rx', label=f'{xmul}x PDB')
plt.savefig('f1.png', dpi=300)
plt.legend()
plt.show()
