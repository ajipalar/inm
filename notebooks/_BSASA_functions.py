from _BSASA_imports import *

#Begin functions
def action_f(rng_key, df, start, stop):
    print(start, stop)
    return m5crop2idata(rng_key, df, start, stop)

def bait_box_plot(ds, var='CRL_E', preysel=True, boxkwargs={}):
    vals = []
    z = [('CBFB', 'PEBB'), ('ELOB', 'ELOB'), ('CUL5', 'CUL5')]#, ('LRR1', 'LLR1')]
    for i in z:
        if preysel:
            arr = np.ravel(ds.sel(bait=i[0], preyu=i[1])[var].values)
        else:
            arr = np.ravel(ds.sel(bait=i[0])[var].values)
        vals.append(arr)
    z.append(('LRR1', 'LLR1'))
    arr = np.ravel(ds.sel(bait='LRR1', preyu='LLR1', condition='mock')[var].values)
    vals.append(arr)
    assert len(vals) == 4, len(vals)
    if var == 'CRL_E' and preysel==False:
        arr = np.ravel(ds.sel(bait='ELOB')['CRL_C'].values) # Bait can be any because control is similiar
        vals.append(arr)
        z.append('Parent')
    sns.boxplot(vals, **boxkwargs)#, labels=['A'] * 4)
    plt.title("Amount of Bait in own purification across 3 conditions")
    plt.xticks(np.arange(len(vals)), [i[0] for i in z])
    plt.ylabel("Spectral count")
    plt.tight_layout()
    save_plt()
    plt.close()

def boxen_pair(ax, vals, title, xticks, xticklabels, rc_context, ylim, ylabel=None):
    sns.boxenplot(vals, ax=ax)
    with mpl.rc_context(rc_context):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xticks, xticklabels)
        ax.set_ylim(ylim)
        return ax

def colprint(name, value):
    return f"N {name} {value}"

def center_and_scale_predictor(y, shift=2):
    assert y.ndim == 2
    col_shape = (len(y), 1)
    mean, var = np.mean(y, axis=1).reshape(col_shape), np.var(y, axis=1).reshape(col_shape)
    var[var==0] = 1
    y = ((y - mean) / var) + shift
    return namedtuple('ParamScale', 'y mean var shift col_shape')(
        y, mean, var, shift, col_shape)
    
def check_satisfaction(sym_s, obs_s, prob=0.9, var_names=['y_c', 'y_e'], dim='draw'):
    """Does the distribution satisfy the observed data?"""
    a = sym_s.min <= obs_s.min
    b = sym_s.max >= obs_s.max
    hdi_mean = ds_hdi(sym_s.mean, prob = prob, var_names = var_names, dim = dim)
    hdi_var  = ds_hdi(sym_s.var,  prob = prob, var_names = var_names, dim = dim)
    s1 = hdi_mean.sel(hdi='min') <= obs_s.mean
    s2 = hdi_mean.sel(hdi='max') >= obs_s.mean
    mean_sat = s1 & s2
    s1 = hdi_var.sel(hdi='min') <= obs_s.var
    s2 = hdi_var.sel(hdi='max') >= obs_s.var
    var_sat = s1 & s2
    return namedtuple('satisfaction',
        'min max mean var')(a, b, mean_sat, var_sat)

def certainty_score(ds):
    """
    The cumulative probabilty of a variable above 
    """
    return (ds > 0).sum(dim="draw") / idata.posterior.dims['draw']

def corr_action(df, col1, col2):
    return sp.stats.pearsonr(df[col1].values, df[col2])

def concat_inf(i1, i2, dim='irow', sample_stats = True):
    assert i1 is not None
    assert i2 is not None
    inference_data = [i1, i2]
    dim_concat = lambda key: xr.concat([i[key] for i in inference_data], dim=dim)
    post = dim_concat('posterior')
    Pp   = dim_concat('posterior_predictive')
    ll   = dim_concat('log_likelihood')
    pp   = dim_concat('prior_predictive')
    prior= dim_concat('prior')
    obs  = dim_concat('observed_data')
    if sample_stats:
        ss   = dim_concat('sample_stats')
        return az.InferenceData(posterior=post, posterior_predictive=Pp, log_likelihood=ll,
                           sample_stats=ss, prior_predictive=pp, prior=prior, observed_data=obs)
    else:
        raise ValueError('Sample stats not implemented')

def compare_window_f(a, N, b):
    return (N - b < a) & (a < N + b)

def df_new2json(df):
    csel = [f"c{i}" for i in range(1, 13)]
    rsel = [f"r{i}" for i in range(1, 5)]
    Nrows = len(df)
    Nrsel = len(rsel)
    Ncsel = len(csel)
    RMatrix = df.loc[:, rsel].values
    CMatrix = df.loc[:, csel].values
    RMatrix = [[int(i) for i in row] for row in RMatrix]
    CMatrix = [[int(i) for i in row] for row in CMatrix]
    return {'Nrsel': Nrsel, 'Ncsel': Ncsel, 'Nrows': Nrows, 'Yexp': RMatrix,
           'Yctrl': CMatrix}

def do_mcmc(model, rng_key, model_kwargs,Kernal=None,Search=None,
        num_warmup=500, num_samples=1000, thinning=1,extra_fields=('potential_energy',)):

    Kernal = numpryo.infer.NUTS if Kernal == None else Kernal
    Search = numpryo.infer.MCMC if Search == None else Search

    search_kwargs = {'num_warmup': num_warmup, 'num_samples': num_samples,
        'thinning': thinning}
    run_kwargs = {'extra_fields': extra_fields}
    search_kwargs = {} if search_kwargs is None else search_kwargs
    run_kwargs = {} if run_kwargs is None else run_kwargs
    kernel = Kernel(model)
    search = Search(kernel, **search_kwargs)
    search.run(rng_key, **model_kwargs, **run_kwargs)
    return search

def _get_summary_stats(y_sim, dims=('rrep', 'crep')):
    return namedtuple('T', "mean min max var")(
            y_sim.mean(dim=dims),
            y_sim.min(dim=dims),
            y_sim.max(dim=dims),
            y_sim.var(dim=dims))

def ds_get_summary_stats(ds, dims=('rrep', 'crep')):
    return _get_summary_stats(ds, dims=dims)

def ds_hdi(x, var_names, dim, prob=0.9):
    """
    Dataset High Density Interval

    x: xr.Dataset
    var_names: 
    dim: int
    prob: float
    """
    axes = [x[name].get_axis_num(dim) for name in var_names]
    assert len(np.unique(axes)) == 1, axes
    axis_num = axes[0]
    f = partial(da_hdi, dim=dim, prob=prob)
    return x.map(f)
    
def da_hdi(x, dim, prob):
    """
    x: xr.DataArray
    """
    axis_num = x.get_axis_num(dim)
    y = hpdi(x.values, axis=axis_num, prob=prob)
    hdi = np.array(['min', 'max'])
    coords = {'hdi': hdi, 'irow': x.coords['irow']}
    return xr.DataArray(data=y, coords=coords, dims=['hdi', 'irow'])

def exp_pdf(x, r):
    return r * np.exp(-x * r)

def fill_tensors(df, tensorR, tensorC, condition_name, bait_mapping, prey_mapping,
        condition_mapping,r_cols, c_cols, prey_col_name, bait_col_name="Bait"):
    # Fill the tensor with the 'C' value where 'A' and 'B' intersect
    for _, row in df.iterrows():
        bait_name = row[bait_col_name].split("_")[0][0:4] #bait are 4 char
        bait_index = bait_mapping[bait_name]
        prey_index = prey_mapping[row[prey_col_name]]
        condition_index = condition_mapping[condition_name]
        tensorR[condition_index][bait_index][prey_index] = [row[c] for c in r_cols]
        tensorC[condition_index][bait_index][prey_index] = [row[c] for c in c_cols]
    return tensorR, tensorC

def format_loffset(x: str, offset=4):
    return " " * offset + x


def htup(x) -> tuple:
    return tuple('{:,}'.format(i) for i in x)

def h(x):
    if isinstance(x, tuple):
        return htup(x)
    else:
        return '{:,}'.format(x)

def hists(ax, xs, labels, colors, alphas, bins, hist_kwargs = None):
    if hist_kwargs is None:
        hist_kwargs  = {}
    for i in range(len(xs)):
        ax.hist(xs[i], color=colors[i], alpha=alphas[i], bins=bins[i],
               label=labels[i], **hist_kwargs)
    return ax

def inf_get_summary_stats(inf_data, chain_num=0):
    return (ds_get_summary_stats(inf_data.prior_predictive.sel(chain=chain_num)),
            ds_get_summary_stats(inf_data.posterior_predictive.sel(chain=chain_num)),
            ds_get_summary_stats(inf_data.observed_data))

def info_it(i):
    return len(i)

def info_unique_it(i):
    return len(set(i))


def init_bsasa_ref(path="../significant_cifs/BSASA_reference.csv"):
    """Initialize the BSASA reference"""
    bsasa_ref = pd.read_csv(path)
    bsasa_ref.loc[:, 'hasna'] = (
        pd.isna(bsasa_ref['Prey1'].values) |
        pd.isna(bsasa_ref['Prey2'].values))
    return bsasa_ref

def init_bait2uid(conditions, gene2uid, baits=None):
    if baits == None:
        baits = {"CBFB": "PEBB_HUMAN", "ELOB": "ELOB_HUMAN", "CUL5": "CUL5_HUMAN", "LRR1": "LLR1_HUMAN"}
    bait2uid = {}
    for bait, genename in baits.items():
        for condition in conditions:
            key = bait + condition
            uid = gene2uid[genename]
            bait2uid[key] = uid
    return bait2uid

def init_Bait2bait_Bait2condition(df_all):
    df_all.index = [i.split("_")[0] for i in df_all["PreyGene"]]
    conditions = ['wt', 'vif', 'mock']
    bait = ['CBFB', 'ELOB', 'CUL5', 'LRR1']
    Bait2bait = {}
    Bait2condition = {}
    for key in bait:
        for val in conditions:
            Bait2bait[key + val + '_MG132'] = key
            Bait2condition[key + val + '_MG132'] = val
    return Bait2bait, Bait2condition        

def init_dfs(xlsx_path="../1-s2.0-S1931312819302537-mmc2.xlsx"):
    xlsx_file = xlsx_path
    df1 = pd.read_excel(xlsx_file, 0)
    df2 = pd.read_excel(xlsx_file, 1)
    df3 = pd.read_excel(xlsx_file, 2)
    df1 = df1[df1['PreyGene'] != 'IGHG1_MOUSE']
    df2 = df2[df2['PreyGene'] != 'IGHG1_MOUSE']
    df3 = df3[df3['PreyGene'] != 'IGHG1_MOUSE']
    assert "IGHG1_MOUSE" not in df1.values
    assert "IGHG1_MOUSE" not in df2.values
    assert "IGHG1_MOUSE" not in df3.values
    return df1, df2, df3

def init_direct_matrix(nprey, preyu, preyv, bsasa_ref):
    direct_matrix = xr.DataArray(np.zeros((nprey, nprey)), 
        coords={"preyu": preyu, "preyv": preyv}, dims=["preyu", "preyv"])
    for i, r in bsasa_ref.iterrows():
        p1 = r['Prey1Name']
        p2 = r['Prey2Name']
        val = direct_matrix.loc[p1, p2].values
        direct_matrix.loc[p1, p2] = val + 1
        direct_matrix.loc[p2, p1] = val + 1
    return direct_matrix

def init_cocomplex_matrix(nprey, preyu, preyv, cocomplex_df):
    cocomplex_matrix = xr.DataArray(np.zeros((nprey, nprey)), 
        coords={"preyu": preyu, "preyv": preyv}, dims=["preyu", "preyv"])
    for i, r in cocomplex_df.iterrows():
        p1 = r['Prey1']
        p2 = r['Prey2']
        val = r['NPDBS']
        assert p1 in preyu, p1
        assert p2 in preyu, p2
        cocomplex_matrix.loc[p1, p2] = val
        cocomplex_matrix.loc[p2, p1] = val
    return cocomplex_matrix

def update_df_all_bait_and_condition(df_all):
    Bait2bait, Bait2condition = init_Bait2bait_Bait2condition(df_all)
    df_all.loc[:, 'bait'] = [Bait2bait[i] for i in df_all['Bait'].values]
    df_all.loc[:, 'condition'] = [Bait2condition[i] for i in df_all['Bait'].values]
    return df_all

def update_df_new_based_on_assumed_control_mappings(
        df_new):
    c_cbfb = [f"c{i}" for i in (1, 2, 3, 4)] 
    c_cul5 = [f"c{i}" for i in (5, 6, 7, 8)] 
    c_elob = [f"c{i}" for i in (9, 10, 11, 12)] 
    csel= c_cbfb
    s = df_new["bait"] == "CBFB"
    df_new.loc[s, csel] = df_new.loc[s, c_cbfb].values
    s = df_new["bait"] == "CUL5"
    df_new.loc[s, csel] = df_new.loc[s, c_cul5].values 
    s = df_new["bait"] == "ELOB"
    df_new.loc[s, csel] = df_new.loc[s, c_elob].values
    control_mappings = {"cbfb": c_cbfb, "cul5": c_cul5, "elob": c_elob}
    return df_new, control_mappings

def update_df_new_with_aa_seq_len(uid2seq_len, df_new): 
    seq_lens = np.array([uid2seq_len[prey] for prey in df_new['Prey'].values])
    df_new.loc[:, 'aa_seq_len'] = seq_lens
    return df_new

def update_df_new_with_tryptic_sites(df_new, uid2seq_len, uid2seq):
    n_first_sites = [n_first_tryptic_cleavages(uid2seq[prey]) for prey in df_new['Prey']]
    df_new.loc[:, 'n_first_tryptic_cleavage_sites'] = np.array(n_first_sites)
    df_new.loc[:, 'n_possible_first_tryptic_peptides'] = df_new.loc[
            :,'n_first_tryptic_cleavage_sites'] + 1
    return df_new

def update_df_new_with_query_aa_sequence(df_new, chain_mapping):
    prey2seq = {r['QueryID']: r['Q'] for i, r in chain_mapping.iterrows()}
    df_new.loc[:, 'Q'] = np.array([(prey2seq[prey]
        if prey in prey2seq else np.nan) for prey in df_new['Prey'].values])
    return df_new

def update_df_new_with_summary_stats(df_new, rsel=None, csel=None):
    if rsel == None:
        rsel = [f"r{i}" for i in range(1, 5)]
    if csel == None:
        csel = [f"c{i}" for i in range(1, 5)]
    df_new.loc[:, 'rAv'] = df_new.loc[:, rsel].mean(axis=1)
    df_new.loc[:, 'cAv'] = df_new.loc[:, csel].mean(axis=1)
    df_new.loc[:, 'rVar'] = df_new.loc[:, rsel].var(axis=1).values
    df_new.loc[:, 'cVar'] = df_new.loc[:, csel].var(axis=1).values
    df_new.loc[:, 'Av_diff'] = df_new['rAv'].values - df_new['cAv'].values
    df_new.loc[:, 'rMax'] = np.max(df_new.loc[:, rsel].values, axis=1)
    df_new.loc[:, 'rMin'] = np.min(df_new.loc[:, rsel].values, axis=1)
    df_new.loc[:, 'cMax'] = np.max(df_new.loc[:, csel].values, axis=1)
    df_new.loc[:, 'cMin'] = np.min(df_new.loc[:, csel].values, axis=1)
    return df_new

def init_df_all(df1, df2, df3, gene2uid, conditions=None):
    viral_remapping = {
    "vifprotein"          :   "P69723",
    "polpolyprotein"      :   "Q2A7R5",
    "nefprotein"     :        "P18801",
    "tatprotein"         :    "P0C1K3",
    "gagpolyprotein"     :    "P12493",
    "revprotein"          :   "P69718",
    "envpolyprotein"      :   "O12164"}
    
    if conditions == None:
        conditions = ["wt_MG132", "vif_MG132", "mock_MG132"]
    #pad.write(("N conditions", len(conditions)))
    bait2uid = init_bait2uid(conditions, gene2uid)

    df1.loc[:, "BaitUID"] = [bait2uid[i] for i in df1["Bait"].values]
    df2.loc[:, "BaitUID"] = [bait2uid[i] for i in df2["Bait"].values]
    df3.loc[:, "BaitUID"] = [bait2uid[i] for i in df3["Bait"].values]
    # Map viral proteins to uniprot IDS 
    df_all = pd.concat([df1, df2, df3])
    df_all.loc[:, "Prey"] = [viral_remapping[i]
        if i in viral_remapping else i for i in df_all['Prey'].values]
    return df_all

def set_PreyName_as_index(df_new):
    df_new.loc[:, 'PreyName'] = df_new.index
    return df_new

def init_complexes(chain_pdb_set, chain_mapping) -> dict:
    """Takes all uniprot ids at a given pdb id and puts them
    in a dict keyed by pdbid
    (a, b -> {pdb_id : frozenset(uid_str...)})
    """
    complexes = {}
    for pdb_id in chain_pdb_set:
        sel = chain_mapping['PDBID'] == pdb_id
        chain_mapping_at_pdb = chain_mapping[sel]
        uids = frozenset(chain_mapping_at_pdb['QueryID'].values)
        complexes[pdb_id] = uids
    return complexes

def init_cocomplexes(complexes):
    """Filters out self complexes"""
    cocomplexes = {}
    for pdb_id, fset in complexes.items():
        if len(fset) > 1:
            cocomplexes[pdb_id] = fset
    return cocomplexes

def init_cocomplex_edge_id__cocomplex_edge(cocomplexes):
    """
    Build a dictionary of cocomplex edge ids
    """
    d = {}
    eid = 0
    for pdb_id, fset in cocomplexes.items():
        assert len(fset) > 1
        combos = list(combinations(fset, 2))
        combos = [frozenset(x) for x in combos]
        for pair in combos:
            if pair not in d:
                d[pair] = eid
                eid += 1
    return {val: key for key, val in d.items()}
        
def init_cocomplex_uid_set(cocomplexes):
    cocomplex_uid_set = []
    for pdb_id, fset in cocomplexes.items():
        for uid in fset:
            cocomplex_uid_set.append(uid)
    cocomplex_uid_set = set(cocomplex_uid_set)
    return cocomplex_uid_set

def init_cocomplex_df(cocomplex_pairs, cocomplexes):
    cocomplex_df = {pair: "" for pair in cocomplex_pairs}
    for pair in cocomplex_pairs:
        for pdb_id, fset in cocomplexes.items():
            p1, p2 = pair
            if (p1 in fset) and (p2 in fset):
                val = cocomplex_df[pair]
                if len(val) == 0:
                    val = pdb_id
                else:
                    val = val + f";{pdb_id}"
                cocomplex_df[pair] = val
    cocomplex_list = [(key[0], key[1], val) for key, val in cocomplex_df.items()]
    cocomplex_df = pd.DataFrame(cocomplex_list, columns=['Prey1', 'Prey2', 'PDBIDS'])
    sel = cocomplex_df["PDBIDS"] != ""
    cocomplex_df = cocomplex_df[sel]
    cocomplex_df.loc[:, 'NPDBS'] = [
        len(i.split(";")) for i in cocomplex_df['PDBIDS'].values]
    return cocomplex_df

def init_direct_interaction_set(bsasa_ref) -> set:
    """Initialize the direct interactions"""
    sel = ~bsasa_ref['hasna'].values
    return {frozenset(
        (r['Prey1'], r['Prey2'])) : "" for i, r in bsasa_ref.loc[sel, :].iterrows()}

def init_spectral_count_xarray(tensorR, tensorC):
    coords = {"AP": [True, False]} | dict(tensorR.coords)  
    sc = xr.DataArray(coords = coords, dims = ["AP", "condition", "bait", "preyu", "rep"]) 
    sc.isel(AP=0)[:] = tensorR
    sc.isel(AP=1)[:] = tensorC
    return sc

def update_cocomplex_df_with_PreyXNames(cocomplex_df, uid2preyname):
    cocomplex_df.loc[:, 'Prey1Name'] = cocomplex_df.loc[:, "Prey1"].map(
        lambda x: uid2preyname[x])
    
    cocomplex_df.loc[:, 'Prey2Name'] = cocomplex_df.loc[:, "Prey2"].map(
        lambda x: uid2preyname[x])
    return cocomplex_df

def update_bsasa_ref_with_PreyXNames(bsasa_ref, uid2preyname):
    bsasa_ref.loc[:, 'Prey1Name'] = bsasa_ref.loc[:, "Prey1"].map(
        lambda x: uid2preyname[x])
    bsasa_ref.loc[:, 'Prey2Name'] = bsasa_ref.loc[:, "Prey2"].map(
        lambda x: uid2preyname[x])
    return bsasa_ref

def init_benchmark_summary(cocomplex_df, bsasa_ref, df_all, unknown,
    direct_interaction, cocomplex_interactions):
    benchmark_summary = pd.DataFrame([len(cocomplex_df), len(bsasa_ref), len(df_all),
    sum(unknown), sum(direct_interaction), sum(cocomplex_interactions),math.comb(3062, 2)],
    columns=["N"], index=['PDB Co-complex', 'PDB Direct', 'Bait-prey',
                          'Bait-prey absent in PDB',
                          'Bait-prey Direct',
                          'Bait-prey cocomplex',
                          'Possible Interactions'])
    return benchmark_summary
def init_interactions(df_all, cocomplex_df, bsasa_ref):
    #direct_interaction_saint = []
    direct_interaction = []
    direct_interaction_labels = []
    cocomplex_interactions = []
    #cocomplex_interaction_saint = []
    cocomplex_interaction_labels = []
    # Make the sets dictionaries for faster lookup
    cocomplex_pairs = {frozenset((row['Prey1'], row['Prey2'])) : 0 for i, row in cocomplex_df.iterrows()}
    bsasa_ref_pairs = {frozenset((row['Prey1'], row['Prey2'])) : 0 for i, row in bsasa_ref.iterrows()}
    for i, row in df_all.iterrows():
        bait_uid = row["BaitUID"]
        prey_uid = row["Prey"]
        pair = frozenset((bait_uid, prey_uid))
        if pair in cocomplex_pairs:
            cocomplex_interactions.append(1)
            cocomplex_interaction_labels.append(pair)
        else:
            cocomplex_interactions.append(0)
        if pair in bsasa_ref_pairs:
            direct_interaction.append(1)
            direct_interaction_labels.append(pair)
        else:
            direct_interaction.append(0)
    direct_interaction = np.array(direct_interaction)
    cocomplex_interactions = np.array(cocomplex_interactions)
    df_all.loc[:, 'bsasa_direct_interaction'] = direct_interaction
    df_all.loc[:, 'cocomplex_interaction'] = cocomplex_interactions
    return df_all, cocomplex_pairs, bsasa_ref_pairs 

def init_tensorRandC(df_new, rsel, csel, preyu, bait, fill_tensors, 
                     conditions = None):
    # expeRiment tensor
    if conditions == None:
        conditions = ['wt', 'vif', 'mock']
    tensorR = [[[[0 for _ in rsel]
                for _ in range(len(preyu))]
                for _ in range(len(bait))]
                for _ in range(len(conditions))]
    # Control tensor
    tensorC = [[[[0 for _ in csel]
                 for _ in range(len(preyu))]
                 for _ in range(len(bait))]
                 for _ in range(len(conditions))]
    #Mappings
    bait2idx = {b:i for i, b in enumerate(bait)}
    condition2idx = {c:i for i, c in enumerate(conditions)}
    #r2idx = {r:i for i, r in enumerate(rsel)}
    #c2idx = {c:i for i, c in enumerate(csel)}
    prey2idx = {p:i for i,p in enumerate(preyu)}
    fill_tensors_applied = partial(fill_tensors, bait_mapping=bait2idx,
        prey_mapping=prey2idx, condition_mapping=condition2idx,
        r_cols=rsel, c_cols=csel, prey_col_name='PreyName', bait_col_name="bait")
    s1 = df_new['condition'] == 'wt'
    s2 = df_new['condition'] == 'vif'
    s3 = df_new['condition'] == 'mock'
    for key, sel in {"wt": s1, "vif": s2, "mock": s3}.items():
        tensorR, tensorC = fill_tensors_applied(df=df_new[sel],
            tensorR=tensorR, tensorC=tensorC,condition_name=key)
    tensorC = xr.DataArray(np.array(tensorC, dtype=int),
        dims=['condition', 'bait', 'preyu', 'rep'],
        coords={'condition': conditions,
            'bait': bait,
            'preyu': preyu,
            'rep': np.arange(0, len(csel))})
    tensorR = xr.DataArray(np.array(tensorR, dtype=int),
        dims=['condition', 'bait', 'preyu', 'rep'],
        coords={'condition': conditions,
            'bait': bait,
            'preyu': preyu,
            'rep': np.arange(0, len(rsel))})
    return tensorR, tensorC

def init_prey_pairs_df(rng_key, prey_set, bsasa_ref):
    prey_pairs = list(combinations(prey_set, 2))
    direct_interaction_set = {}
    for i, r in bsasa_ref.iterrows():
        p1 = r['Prey1Name']
        p2 = r['Prey2Name']
        direct_interaction_set[frozenset((p1, p2))] = ""
    pdb_positive = []
    prey1 = []
    prey2 = []
    for pair in prey_pairs:
        fs = frozenset(pair)
        p1, p2 = pair
        prey1.append(p1)
        prey2.append(p2)
        if fs in direct_interaction_set:
            pdb_positive.append(True)
        else:
            pdb_positive.append(False)
    prey_pairs_df = pd.DataFrame(
        {'Prey1Name': prey1, 'Prey2Name': prey2, 'pdb_pos': pdb_positive})
    nrows, ncols = prey_pairs_df.shape
    prey_pairs_df.loc[:, 'rand'] = np.array(jax.random.uniform(rng_key, shape=(nrows, )))
    return prey_pairs_df

def init_interaction_set(bsasa_ref):
    return set(
        [frozenset((r['Prey1'],r['Prey2']))
         for i, r in bsasa_ref.iterrows()])

def init_npdbs_per_interaction(bsasa_ref, interaction_set):
    """Initialize the number of PDB IDs per interaction"""
    npdbs_per_interaction = {}
    for key in interaction_set:
        prey1, prey2 = key
        sel1 = bsasa_ref['Prey1'] == prey1
        sel2 = bsasa_ref['Prey1'] == prey2
        bsasa_sel = bsasa_ref[sel1 | sel2]
        sel1 = bsasa_sel['Prey2'] == prey1
        sel2 = bsasa_sel['Prey2'] == prey2
        bsasa_sel = bsasa_sel[sel1 | sel2]
        pdb_set_sel = set(bsasa_sel['PDBID'])
        npdbs_per_interaction[key] = len(pdb_set_sel)
    return npdbs_per_interaction

def init_uid2seq(seq_path="../input_sequences"):
    """Initialize the UniprotID to Sequence mapping"""
    uid2seq = {}
    tmp = [i for i in Path(seq_path).iterdir() if ".fasta" in str(i)]
    #assert len(tmp) == 3062, len(tmp)
    for i in tmp:
        uid = i.name.removesuffix(".fasta")
        seq = biotite.sequence.io.load_sequence(str(i))
        seq = str(seq)
        uid2seq[uid] = seq
    uid2seq_len = {uid: len(seq) for uid, seq in uid2seq.items()}
    return uid2seq, uid2seq_len

def init_nuid(bsasa_ref) -> int:
    return len(set(
      bsasa_ref['Prey1'].values).
      union(bsasa_ref['Prey2'].values))

def isOdd(x):
    return x % 2 != 0

def isHashable(x):
    try:
        hash(x)
        return True
    except TypeError:
        return False

def known_positives(ref_set):
    return len(ref_set)

def model(x=None, y=None):
    b = 0 #numpyro.sample('b', dist.Normal(0, 0.1))
    m = numpyro.sample('m', dist.Normal(0, 1))
    numpyro.sample('Y', dist.Normal(m * x + b), obs=y)


def m1(df):
    nrows = len(df)
    hyper_prior = np.ones(nrows) * 1/5
    Yexp = df.loc[:, rsel].values
    Yctrl = df.loc[:, csel].values
    nexp = len(rsel)
    nctrl = len(csel)
    kappa_ = numpyro.sample('k', dist.Exponential(rate=hyper_prior))
    lambda_ = numpyro.sample('l', dist.Exponential(rate=hyper_prior))
    for i in range(0, nctrl):
        numpyro.sample(f'Ycrtl_{i}', dist.Poisson(kappa_), obs=Yctrl[:, i])
    for i in range(0, nexp):
        numpyro.sample(f'Yexp_a{i}', dist.Poisson(lambda_), obs=Yexp[:, i])
        numpyro.sample(f'Yexp_b{i}', dist.Poisson(kappa_), obs=Yexp[:, i])

def mixed_model(probs, locs):
    """Example mixture model"""
    c = numpyro.sample("c", dist.Categorical(probs))
    numpyro.sample("x", dist.Normal(locs[c], 0.5))

def mixture_example(data):
    K = 2 # Two mixture components
    weights = numpyro.sample("weights", dist.Dirichlet(0.5 * jnp.ones(K)))
    scale = numpyro.sample("scale", dist.LogNormal(0.0, 2.0))
    with numpyro.plate("components", K):
        locs = numpyro.sample("locs", dist.Normal(0.0, 10))
    with numpyro.plate("data", len(data)):
        # Local variables
        assignment = numpyro.sample("assignment", dist.Categorical(weights))
        numpyro.sample("obs", dist.Normal(locs[assignment], scale), obs=data)

def model2(ds, N=3062):
    # wt, vif, mock
    # 
    # [condition, bait, prey, rrep]
    ELOB_wt = ds.sel(condition='wt', bait='ELOB')['CRL_E'].values
    CUL5_wt = ds.sel(condition='wt', bait='CUL5')['CRL_E'].values
    CBFB_wt = ds.sel(condition='wt', bait='CBFB')['CRL_E'].values
    # 
    ELOB_vif = ds.sel(condition='vif', bait='ELOB')['CRL_E'].values
    CUL5_vif = ds.sel(condition='vif', bait='CUL5')['CRL_E'].values
    CBFB_vif = ds.sel(condition='vif', bait='CBFB')['CRL_E'].values
    # 
    ELOB_mock = ds.sel(condition='mock', bait='ELOB')['CRL_E'].values
    CUL5_mock = ds.sel(condition='mock', bait='CUL5')['CRL_E'].values
    CBFB_mock = ds.sel(condition='mock', bait='CBFB')['CRL_E'].values
    # 
    LRR1_mock = ds.sel(condition='mock', bait='LRR1')['CRL_E'].values
    # 
    ctrl_ELOB_wt = ds.sel(condition='wt', bait='ELOB')['CRL_C'].values
    ctrl_CUL5_wt = ds.sel(condition='wt', bait='CUL5')['CRL_C'].values
    ctrl_CBFB_wt = ds.sel(condition='wt', bait='CBFB')['CRL_C'].values
    # 
    ctrl_ELOB_vif = ds.sel(condition='vif', bait='ELOB')['CRL_C'].values
    ctrl_CUL5_vif = ds.sel(condition='vif', bait='CUL5')['CRL_C'].values
    ctrl_CBFB_vif = ds.sel(condition='vif', bait='CBFB')['CRL_C'].values
    # 
    ctrl_ELOB_mock = ds.sel(condition='mock', bait='ELOB')['CRL_C'].values
    ctrl_CUL5_mock = ds.sel(condition='mock', bait='CUL5')['CRL_C'].values
    ctrl_CBFB_mock = ds.sel(condition='mock', bait='CBFB')['CRL_C'].values
    # 
    ctrl_LRR1_mock = ds.sel(condition='mock', bait='LRR1')['CRL_C'].values
   # max_val = ds['CRL_E'].max('rrep')
   # mu_Nc = np.ones((5, 3))
   # mu_alpha = np.ones((N, 5, 3))
    #N = numpyro.sample('N', dist.Normal(np.zeros(3), 5))
    #mu = numpyro.sample('mu', dist.Normal(max_val.sel(bait='ELOB').values.T, 50), sample_shape=(3062, 3))
    #numpyro.sample('sc', dist.Normal(N * mu), obs=max_val.sel(bait='ELOB').values.T)
    #N1 = numpyro.sample('N1', dist.Normal(0, 1))
    #N2 = numpyro.sample('N2', dist.Normal(0, 1))
    #mu_elob = numpyro.sample('mu_elob', dist.Normal(np.mean(ELOB_wt, axis=1), np.var(ELOB_wt, axis=1)))
    #mu_cul5 = numpyro.sample('mu_cul5', dist.Normal(np.mean(CUL5_wt, axis=1), np.var(ELOB_wt, axis=1)))
    #numpyro.sample('ELOB_wt', dist.Normal(mu_elob * N1, 5), obs=ELOB_wt)
    #numpyro.sample('CUL5_wt', dist.Normal(mu_cul5 * N2, 5), obs=CUL5_wt)
    #cell_abundance = numpyro.sample(dist.Normal(jnp.ones((3, 5))), 1)
    assert ELOB_wt.shape == (3062, 4)
    mu_hyper_prior = np.ones((3062, 1)) / 50
    sig_hyper_prior = np.ones((3062, 1)) / 2
    mu = numpyro.sample('mu', dist.Exponential(mu_hyper_prior))
    sigma = numpyro.sample('s', dist.Exponential(sig_hyper_prior))
    Ncells = numpyro.sample('Nc', dist.Normal(np.ones((1, 4)), 0.5))
    Ncells_rep = jnp.repeat(Ncells, 3062, axis=0)
    numpyro.sample('sc', dist.Normal(mu * Ncells_rep, sigma), obs=ELOB_wt)
    #Ncells = cell_abundance * 1e7 
    #gamma_i = numpyro.sample('gamma', dist.Beta(0.5, 0.5), sample_shape=(3062,))
    #mu_ctrl = numpyro.sample('mu0', dist.Uniform(0, 250), sample_shape=(3062,))
    #mu_wt = numpyro.sample('mu_wt', dist.Uniform(0, 250), sample_shape=(3062,))
    #numpyro.sample('ELOB_wt', dist.Normal(mu_wt, 10), obs=ELOB_wt)
    #numpyro.sample('ctrl_ELOB_wt', dist.Normal(mu_ctrl * gamma_i, 10), obs=ctrl_ELOB_wt)

def model3(ds, N=3062):
    
    # wt, vif, mock
    # 
    # [condition, bait, prey, rrep]
    
    ELOB_wt = ds.sel(condition='wt', bait='ELOB')['CRL_E'].values
    CUL5_wt = ds.sel(condition='wt', bait='CUL5')['CRL_E'].values
    CBFB_wt = ds.sel(condition='wt', bait='CBFB')['CRL_E'].values
    
    ELOB_vif = ds.sel(condition='vif', bait='ELOB')['CRL_E'].values
    CUL5_vif = ds.sel(condition='vif', bait='CUL5')['CRL_E'].values
    CBFB_vif = ds.sel(condition='vif', bait='CBFB')['CRL_E'].values
    
    ELOB_mock = ds.sel(condition='mock', bait='ELOB')['CRL_E'].values
    CUL5_mock = ds.sel(condition='mock', bait='CUL5')['CRL_E'].values
    CBFB_mock = ds.sel(condition='mock', bait='CBFB')['CRL_E'].values
    
    LRR1_mock = ds.sel(condition='mock', bait='LRR1')['CRL_E'].values
    
    ctrl_ELOB_wt = ds.sel(condition='wt', bait='ELOB')['CRL_C'].values
    ctrl_CUL5_wt = ds.sel(condition='wt', bait='CUL5')['CRL_C'].values
    ctrl_CBFB_wt = ds.sel(condition='wt', bait='CBFB')['CRL_C'].values
    
    ctrl_ELOB_vif = ds.sel(condition='vif', bait='ELOB')['CRL_C'].values
    ctrl_CUL5_vif = ds.sel(condition='vif', bait='CUL5')['CRL_C'].values
    ctrl_CBFB_vif = ds.sel(condition='vif', bait='CBFB')['CRL_C'].values
    
    ctrl_ELOB_mock = ds.sel(condition='mock', bait='ELOB')['CRL_C'].values
    ctrl_CUL5_mock = ds.sel(condition='mock', bait='CUL5')['CRL_C'].values
    ctrl_CBFB_mock = ds.sel(condition='mock', bait='CBFB')['CRL_C'].values
    
    ctrl_LRR1_mock = ds.sel(condition='mock', bait='LRR1')['CRL_C'].values

   # max_val = ds['CRL_E'].max('rrep')
    
   # mu_Nc = np.ones((5, 3))
   # mu_alpha = np.ones((N, 5, 3))
    

    
    #N = numpyro.sample('N', dist.Normal(np.zeros(3), 5))
    #mu = numpyro.sample('mu', dist.Normal(max_val.sel(bait='ELOB').values.T, 50), sample_shape=(3062, 3))
    #numpyro.sample('sc', dist.Normal(N * mu), obs=max_val.sel(bait='ELOB').values.T)
    
    
    
    
    #N1 = numpyro.sample('N1', dist.Normal(0, 1))
    #N2 = numpyro.sample('N2', dist.Normal(0, 1))
    
    #mu_elob = numpyro.sample('mu_elob', dist.Normal(np.mean(ELOB_wt, axis=1), np.var(ELOB_wt, axis=1)))
    #mu_cul5 = numpyro.sample('mu_cul5', dist.Normal(np.mean(CUL5_wt, axis=1), np.var(ELOB_wt, axis=1)))
    
    #numpyro.sample('ELOB_wt', dist.Normal(mu_elob * N1, 5), obs=ELOB_wt)
    #numpyro.sample('CUL5_wt', dist.Normal(mu_cul5 * N2, 5), obs=CUL5_wt)
    
    
    #cell_abundance = numpyro.sample(dist.Normal(jnp.ones((3, 5))), 1)
    
    assert ELOB_wt.shape == (3062, 4)
    
    mu_hyper_prior = np.ones((3062, 1)) / 50
    sig_hyper_prior = np.ones((3062, 1)) / 2
    
    
    mu = numpyro.sample('mu', dist.Exponential(mu_hyper_prior))
    sigma = numpyro.sample('s', dist.Exponential(sig_hyper_prior))
    
    Ncells = numpyro.sample('Nc', dist.Normal(np.ones((1, 4)), 0.5))
    
    Ncells_rep = jnp.repeat(Ncells, 3062, axis=0)
    
    
    numpyro.sample('sc', dist.Normal(mu * Ncells_rep, sigma), obs=ELOB_wt)
    
    #Ncells = cell_abundance * 1e7 
    
    #gamma_i = numpyro.sample('gamma', dist.Beta(0.5, 0.5), sample_shape=(3062,))
    #mu_ctrl = numpyro.sample('mu0', dist.Uniform(0, 250), sample_shape=(3062,))
    #mu_wt = numpyro.sample('mu_wt', dist.Uniform(0, 250), sample_shape=(3062,))
    
    #numpyro.sample('ELOB_wt', dist.Normal(mu_wt, 10), obs=ELOB_wt)
    #numpyro.sample('ctrl_ELOB_wt', dist.Normal(mu_ctrl * gamma_i, 10), obs=ctrl_ELOB_wt)
    
def model4(e_data=None, ctrl_data=None, a=None, b=None):
    # Prior around 1/5
    if a is None:
        a = numpyro.sample('a', dist.Uniform(0.0001, 1))
    if b is None:
        b = numpyro.sample('b', dist.Uniform(0.0001, 1))
    numpyro.sample('y_e', dist.Exponential(a), obs=e_data)
    numpyro.sample('y_c', dist.Exponential(b), obs=ctrl_data)

def model5(a=None, b=None, y_c=None, y_e=None, poisson_sparse=True, batch_shape=()):
    """
    a:  experimental rate
    b:  control rate
    y_e: experimental obs
    y_c: control obs
    """
    #max_val = np.max([np.max(y_c), np.max(y_e)])
    # Assum the true rates are within 10 + 2 * max observed value
    hyper = np.ones(batch_shape) * 200 #* max_val
    if a is None:
        a = numpyro.sample('a', dist.HalfNormal(hyper))
    if b is None:
        b = numpyro.sample('b', dist.HalfNormal(hyper))
    # predictive checks
    nrows, ncols = batch_shape
    if y_c is None:
        b = jnp.ones((nrows, 12)) * b
    if y_e is None:
        a = jnp.ones((nrows, 4)) * a
    numpyro.sample('y_c', dist.Poisson(b, is_sparse=poisson_sparse), obs=y_c)
    numpyro.sample('y_e', dist.Poisson(a, is_sparse=poisson_sparse), obs=y_e)

def model52_f(df_new, start, end, numpyro_model = model5,numpyro_model_kwargs = None):
    if numpyro_model_kwargs is None:
        numpyro_model_kwargs = {}
    def init_kernel(rescale_model=False):
        n = slice(start, end)
        dsel = df_new.iloc[n, :]
        l = len(dsel)
        y_c = np.array(dsel[csel].values, dtype=int)
        y_e = np.array(dsel[rsel].values, dtype=int)
        #col_shape = (l, 1)
        # Scale and shift predictors
        #mean_c, var_c = np.mean(y_c, axis=1).reshape(col_shape), np.var(y_c, axis=1).reshape(col_shape)
        #mean_e, var_e = np.mean(y_e, axis=1).reshape(col_shape), np.var(y_e, axis=1).reshape(col_shape)
        #var_c[var_c==0]=1
        #var_e[var_e==0]=1
        #y_c = ((y_c - mean_c) / var_c) + 2.
        #y_e = ((y_e - mean_e) / var_e) + 2.
        if rescale_model:
            y_c_param_scale = center_and_scale_predictor(y_c)
            y_e_param_scale = center_and_scale_predictor(y_e)
            model_kwargs={'y_c': y_c_param_scale.y, 'y_e': y_e_param_scale.y}
        else:
            model_kwargs={'y_c': y_c, 'y_e': y_e}
            y_c_param_scale = None
            y_e_param_scale = None
        model_meta={'y_c': y_c_param_scale, 'y_e': y_e_param_scale}
        model = partial(
            numpyro_model, 
            batch_shape=(l, 1),
            **numpyro_model_kwargs) # partial apply for prior and post pred checks
        return namedtuple('Init', 'model kwargs meta')(model, model_kwargs, model_meta)
    def init_sampling(rng_key, num_warmup, num_samples):
        return namedtuple('Init', 'rng_key num_warmup num_samples')(rng_key, num_warmup, num_samples)
    def sample(model_init, sample_init):
        search = do_mcmc(model_init.model, 
                         sample_init.rng_key, 
                         model_init.kwargs,
                         num_warmup = sample_init.num_warmup,
                         num_samples = sample_init.num_samples)
        #samples = search.get_samples()
        #summary_dict = summary(samples, group_by_chain=False)
        return search
    #k1, k2, k3 = jax.random.split(rng_key, 3)
    #search = do_mcmc(model, k1, model_kwargs, num_warmup=1000, num_samples=5000)
    #samples = search.get_samples()
    #summary_dict = summary(samples, group_by_chain=False)
    def sample_pp(model_init, sample_init):
        """
        Sample from the prior predictive distribution
        """
        return numpyro.infer.Predictive(
            model = model_init.model, 
            num_samples = sample_init.num_samples
        )(sample_init.rng_key)
    def sample_Pp(model_init, samples, sample_init):
        return numpyro.infer.Predictive(
            model = model_init.model, 
            posterior_samples = samples
        )(rng_key = sample_init.rng_key)
    def _pre_init_InferenceData(
        search,
        pp,
        Pp,
        model_meta
    ):
        coords = {'irow': np.arange(start, end), 'col': np.array([0]),
                  'rrep': np.arange(4), 'crep': np.arange(12)}
        dims = {'a': ['chain', 'draw', 'irow', 'col'],
                'b': ['chain', 'draw', 'irow', 'col'],
                'y_e': ['irow', 'rrep'], 
                'y_c':['irow', 'crep']}
        pred_dims = {'y_c': ['chain', 'draw', 'irow', 'col'],
                     'y_e': ['chain', 'draw', 'irow', 'col']}
        inf_data = az.from_numpyro(search, 
            prior=pp, posterior_predictive=Pp, coords=coords, dims=dims, pred_dims=pred_dims)
        assert inf_data is not None
        return inf_data
    def rescale_model(model_meta, inf_data):
        # Scale Observed
        yobs = inf_data['observed_data']
        inf_data.observed_data['y_c'].values = un_center_and_scale_predictor(
            yobs.y_c, model_meta['y_c'])
        inf_data.observed_data['y_e'].values = un_center_and_scale_predictor(
            yobs.y_e, model_meta['y_e'])
        # Scale Posterior
        ufunc = parital(un_center_and_scale_predictor, param_scale=model_meta['y_e'])
        inf_data.posterior.a = xr.apply_ufunc(
            ufunc, inf_data.posterior.a)
        inf_data.posterior.a.values = un_center_and_scale_predictor(
            inf_data.posterior.a)
        return inf_data
    def _append_posterior_statistics(
        inf_data, 
        samples, 
        group_by_chain=False
    ):
        summary_dict = summary(samples, group_by_chain=group_by_chain)
        a_rhat = summary_dict['a']['r_hat'][:, 0]
        a_neff = summary_dict['a']['n_eff'][:, 0]
        b_rhat = summary_dict['b']['r_hat'][:, 0]
        b_neff = summary_dict['b']['n_eff'][:, 0]
        tmp_df = pd.DataFrame({'a_rhat': a_rhat, 'a_neff': a_neff, 'b_rhat':b_rhat, 'b_neff': b_neff})
        inf_data['posterior']['stats'] = xr.DataArray(tmp_df.values, 
                                                    coords={'irow': np.arange(start, end),
                                                    'stat': np.array(tmp_df.columns)})
        return inf_data
    def init_InferenceData(
        search,
        pp,
        Pp,
        model_meta,
        rescale=True,
        append_sample_stats=True,
        group_by_chain=False
    ):
        inf_data = _pre_init_InferenceData(search, pp, Pp, model_meta)
        if rescale:
            inf_data = rescale_model(model_meta = model_meta, inf_data = inf_data)
        if append_sample_stats:
            inf_data = _append_posterior_statistics(inf_data, search.get_samples(), group_by_chain=group_by_chain)
        return inf_data
    return namedtuple(
        "M5F", "init_kernel init_sampling sample sample_pp sample_Pp rescale_model init_InferenceData")(
            init_kernel = init_kernel,
            init_sampling = init_sampling,
            sample = sample,
            sample_pp = sample_pp,
            sample_Pp = sample_Pp,
            rescale_model = rescale_model,
            init_InferenceData = init_InferenceData
        )

def m5crop2idata(rng_key, df, start, end, rsel = None, csel = None, rescale_model=False,
                num_warmup=1000, num_samples=1000, num_pred_samples=500, num_pred_warmup=None,
                 append_sample_stats=True):
    if rsel is None:
        rsel = [f"r{i}" for i in range(1,5)]
    if csel is None:
        csel = [f"c{i}" for i in range(1, 13)]
    if num_pred_warmup is None:
        num_pred_warmup = num_warmup
    l = end - start
    y_c = df.iloc[start:end, :][csel].values
    y_e = df.iloc[start:end, :][rsel].values
    kd = {'hyper_a': (np.mean(y_e, axis=1).reshape((l, 1)) + 10) * 1.5,
          'hyper_b': (np.mean(y_c, axis=1).reshape((l, 1)) + 10) * 1.5}
    m5f = model52_f(df_new, start=start, end=end, numpyro_model=model6,
               numpyro_model_kwargs=kd)
    kernel = m5f.init_kernel(rescale_model=rescale_model)
    k1, k2, k3 = jax.random.split(rng_key, 3)
    # Sample model
    sample_init = m5f.init_sampling(k1, num_warmup=num_warmup, num_samples=num_samples)
    search = m5f.sample(kernel, sample_init)
    # Prior predictive check
    sample_init = m5f.init_sampling(k2, num_warmup=num_pred_warmup, num_samples=num_pred_samples)
    pp = m5f.sample_pp(kernel, sample_init)
    # posterior predictive check
    sample_init = m5f.init_sampling(k3, num_warmup = num_pred_warmup,
        num_samples = num_pred_samples)
    Pp = m5f.sample_Pp(kernel, search.get_samples(), sample_init)
    idata = m5f.init_InferenceData(search, pp, Pp, kernel.meta, rescale=rescale_model,
        append_sample_stats=append_sample_stats)
    return idata

def model6(hyper_a, hyper_b, a=None, b=None, y_c=None, y_e=None, poisson_sparse=True, batch_shape=()):
    """
    Similiar to model 5 however we place stronger priors on Poisson
    rates in the hopes of speeding up HMC
    """
    if a is None:
        a = numpyro.sample('a', dist.HalfNormal(hyper_a))
    if b is None:
        b = numpyro.sample('b', dist.HalfNormal(hyper_b))
    # predictive checks
    nrows, ncols = batch_shape
    if y_c is None:
        b = jnp.ones((nrows, 12)) * b
    if y_e is None:
        a = jnp.ones((nrows, 4)) * a
    numpyro.sample('y_c', dist.Poisson(b, is_sparse=poisson_sparse), obs=y_c)
    numpyro.sample('y_e', dist.Poisson(a, is_sparse=poisson_sparse), obs=y_e)

def model7(cul5_e, elob_e, cbfb_e, cul5_ctrl, elob_ctrl, cbfb_ctrl,lrr1_e, purification_shape):
    nrows, ncols = purification_shape
    # alphas - replicate specific abundance factors
    alpha_hyper = np.ones((4, 4)) * 5
    beta_hyper = np.ones(purification_shape) * 200
    # Batch the alphas over the number of pulldowns
    alpha = numpyro.sample('a', dist.Normal(alpha_hyper, 1))
    cul5_beta = numpyro.sample('cb', dist.HalfNormal(beta_hyper))
    elob_beta = numpryo.sample('eb', dist.HalfNormal(beta_hyper))
    cbfb_beta = numpyro.sample('bb', dist.HalfNormal(beta_hyper))
    lrr1_beta = numpryo.sample('lb', dist.HalfNormal(beta_hyper))
    numypro.sample('y_cul_e', dist.Poisson(cul5_beta * alpha[0, :]), obs=cul_e)

def model8(ds, prey_max=None):
    """
    """
    if prey_max is None:
        prey_max = len(ds['preyu'].values)
    n_bait = len(ds['bait'])
    n_infections = len(ds['condition'])
    n_replicates = len(ds['rrep'])
    alpha_batch_shape = (n_bait, n_infections, n_replicates)
    n_prey = len(ds['preyu'])
    obs = ds['CRL_E'].transpose('preyu', 'bait', 'condition', 'rrep')
    obs_c = ds['CRL_C'].transpose('preyu', 'bait', 'condition', 'crep')
    obs_bait_counts = np.zeros(alpha_batch_shape)
    n_control_replicates = 4 # Global domain knowledge
    control_counts = np.zeros((n_prey, n_bait, n_infections, n_control_replicates))
    # CBFB 1:4, CUL5 5:8, ELOB 9:12
    csel_map = {'CBFB': [0, 1, 2, 3],
            'CUL5': [4, 5, 6, 7],
            'ELOB': [8, 9, 10, 11]}
    for i, bait in enumerate(ds.bait):
        if bait == 'CBFB':
            prey_sel = 'PEBB'
        else:
            prey_sel = bait
        obs_bait_counts[i, :, :] = obs.sel(bait=bait, preyu=prey_sel).values
        columns = csel_map[bait.item()]
        control_counts[:, i, :, :] = obs_c.sel(bait=bait, crep=columns).values
    obs_bait_counts = jnp.array(obs_bait_counts, dtype=int)
    obs_c = jnp.array(control_counts[0:prey_max, :, :, :], dtype=int)
    if prey_max:
        obs = obs.sel(preyu=obs.preyu[0:prey_max])
    # epsilon: control counts
    # Sample Bait alphas: bait rate
    max_obs = jnp.max(obs_bait_counts)
    #alpha_hyper_prior = jnp.ones(obs_bait_counts.shape) * max_obs
    mean_ = obs_bait_counts.mean(axis=2)
    alpha_hyper_prior = jnp.zeros(alpha_batch_shape)
    assert mean_.shape == (n_bait, n_infections), mean_.shape
    for i in range(n_replicates):
        alpha_hyper_prior = alpha_hyper_prior.at[:, :, i].set(mean_)
    alpha_hyper_prior = jnp.array(alpha_hyper_prior)
    alpha = numpyro.sample('alpha', dist.Normal(alpha_hyper_prior, 20)) # rate
    numpyro.sample('bait_obs', dist.Poisson(alpha), obs=obs_bait_counts)
    # Sample betas: prey coefficient
    n_prey = len(obs.preyu)
    beta_batch_shape = (n_prey, n_bait, n_infections, n_replicates)
    beta_hyper_prior = jnp.ones(beta_batch_shape) # HalfNormal Scale
    beta = numpyro.sample('beta', dist.HalfNormal(scale=beta_hyper_prior)) # Unitless
    numpyro.sample('obs_exp', dist.Poisson(beta * alpha), obs=obs.values)
    max_control_obs = jnp.max(obs_c)
    epsilon_hyper_prior = jnp.ones(beta_batch_shape) * max_control_obs # Prior over rates
    epsilon = numpyro.sample('epsilon', dist.HalfNormal(epsilon_hyper_prior)) # Same prior as alpha
    numpyro.sample('obs_ctrl', dist.Poisson(epsilon), obs=obs_c) # Independant of alpha * beta

def model_data_from_ds(ds):
    preyu = ds.preyu # Prey Dimension
    bait = ds.bait   # Bait dimension
    condition = ds.condition # infection dimension
    rep = np.arange(0, 4) # replicate dimension
    test = [True, False]  # control vs treatment dimension
    n_prey = len(preyu)
    n_bait = len(bait)
    n_cond = len(condition)
    n_rep = len(rep)
    n_test = len(test)
    coords = {'preyu': preyu, 'bait': bait, 'condition': condition,
             'rep': rep, 'test': test}
    dims = ['preyu', 'bait', 'condition', 'rep', 'test']
    shape = (n_prey, n_bait, n_cond, n_rep, n_test)
    data = np.zeros(shape, dtype=np.int64)
    data = xr.DataArray(data=data, coords=coords, dims=dims)
    # Assign the data
    ds = ds.transpose('preyu', 'bait', 'condition', 'rrep', 'crep', 'preyv')
    data = data.transpose('preyu', 'bait', 'condition', 'rep', 'test')
    bait2_controls = {'CBFB': [0, 1, 2, 3],
                      'CUL5': [4, 5, 6, 7],
                      'ELOB': [8, 9, 10, 11],
                      'LRR1': [8, 9, 10, 11]} # Assume that ELOB and LRR1 share control based on GDE counts
    # Don't plot these controls twice
    for bait in data.bait.values:
        for test in [True, False]:
            key = 'CRL_E' if test else 'CRL_C'
            rep_cols = [0, 1, 2, 3] if test else bait2_controls[bait]
            data.loc[:, bait, :, :, test] = ds[key].loc[:, bait, :, :].values[:, :, rep_cols]
    assert np.sum(data.sel(bait='LRR1', condition=['wt', 'vif'])) == 0
    data.loc[:, 'LRR1', 'wt', :, :] = -100   # These experiments weren't performed 
    data.loc[:, 'LRR1', 'vif', :, :] = -100  # 
    return data

def model_zero_inflated_poisson(model_data, observed=True):
    data = model_data.transpose('rep', 'preyu', 'bait', 'condition', 'test')
    n_prey = len(data.preyu)
    n_bait = len(data.bait)
    n_infect = len(data.condition)
    n_rep = len(data.rep)
    n_test = len(data.test)
    batch_shape =         (n_prey, n_bait, n_infect, n_test)
    sample_shape = (n_rep, n_prey, n_bait, n_infect, n_test)
    lam_hyper = jnp.ones(batch_shape) * 100
    beta_alpha_hyper = jnp.ones(batch_shape) * 1.1
    beta_beta_hyper =  jnp.ones(batch_shape) * 4.0
    if observed:
        data = data.values
        assert data.shape == sample_shape
    else:
        data = None
    #with numpyro.plate("rep_dim", n_rep):
    pi = numpyro.sample('pi', dist.Beta(beta_alpha_hyper, beta_beta_hyper))
    lambda_ = numpyro.sample('lam', dist.HalfNormal(lam_hyper))
    numpyro.sample('sc', dist.ZeroInflatedPoisson(gate=pi, rate=lambda_), 
                   sample_shape = (n_rep,), obs=data)

def model_test():
    numpyro.sample("a", dist.Normal(0, 1))

def multi_hist(hists, bins, labels, alphas, xlabel):
    for i in range(len(hists)):
        plt.hist(hists[i], bins=bins, label=labels[i], alpha=alphas[i])
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()

def n_tryptic_cleavages(aa_seq):
    assert aa_seq.isupper()
    aa_seq = np.array(list(aa_seq))
    Ksites = aa_seq == 'K'
    Rsites = aa_seq == 'R'
    Allsites = Ksites | Rsites
    n_sites = np.sum(Allsites)
    if (aa_seq[-1] == 'K') or (aa_seq[-1] == 'R'):
        n_sites -= 1
    return n_sites

def n_first_tryptic_cleavages(aa_seq):
    return 2 * n_tryptic_cleavages(aa_seq)

def n_first_typtic_cleavage_peptides(aa_seq):
    return 1 + n_first_tryptic_cleavages(aa_seq)

def n_remaining_action(df):
    return len(df)

def plot_exp_dens(x, rate):
    y = np.exp(dist.Exponential(rate).log_prob(x)) / rate
    plt.plot(x, y, label=f"rate={rate}")

def plot_lp(m_data):
    az.plot_trace(m_data.sample_stats['lp'])

def plot_prior(m_data, start=0, end=10):
    az.plot_trace(m_data.prior.sel(irow=np.arange(start, end)))
    
def predictive_check(m_data, T='max', map_axis_pair=('y_e', 'rrep')):
    var, dim = map_axis_pair
    Tobs = (m_data.observed_data[var]).max(dim).values
    Tprior = (m_data.prior_predictive[var]).max(dim).values
    Tpost = (m_data.posterior_predictive[var]).max(dim).values
    prior_n, *_ = plt.hist(np.ravel(Tprior), bins=10, label='Prior predictive', alpha=0.8)
    post_n, *_ = plt.hist(np.ravel(Tpost), bins=10, label='Posterior predictive', alpha=0.8)
    ymax = max(np.mean(prior_n), np.mean(post_n))
    plt.vlines(Tobs.item(), 0, ymax, 'k', label='observed')
    plt.xlabel(f"T(y) : {T} spectral count")
    plt.legend()
# Function for plotting an accuracy curve
def pos_ntotal(prey_pairs_df, col, threshold, comp=None, pos_col='pdb_pos'):
    comp = operator.le if comp == None else comp
    sel = prey_pairs_df[col] <= threshold
    sub_df = prey_pairs_df.loc[sel, :]
    npos = np.sum(sub_df[pos_col].values)
    ntotal = len(sub_df)
    return npos, ntotal

def preysel(df, prey_name, gene2uid):
    uid = gene2uid[prey_name]
    sel1 = df['Prey1'] == uid
    sel2 = df['Prey2'] == uid
    return df[sel1 | sel2]

def parse_spec(df, spec_colname='Spec', ctrl_colname='ctrlCounts',n_spec=4, n_ctrl=12):
    """Parse the Spec and CtrlCounts columns"""
    rsel = [f"r{i}" for i in range(1, n_spec + 1)]
    csel = [f"c{i}" for i in range(1, n_ctrl + 1)]
    # Treatment counts 
    specs = np.array(
      [list(map(int, i.split("|")))
       for i in df[spec_colname].values])
    # Control counts
    ctrls = np.array(
      [list(map(int, i.split("|")))
       for i in df[ctrl_colname].values])
    # Checks 
    N = len(df)
    assert specs.shape == (N, n_spec)
    assert ctrls.shape == (N, n_ctrl)
    # Populate 
    for i, rcol in enumerate(rsel):
        df.loc[:, rcol] = specs[:, i]
    for i, ccol in enumerate(csel):
        df.loc[:, ccol] = ctrls[:, i]
    return df

def phase_space_of_protein_interactions(ax):
    N = 4000
    Imax = math.comb(N, 2)
    #x = np.arange(1, N, 1)
    #y = sp.special.comb(x, 2)
    x = np.arange(1, N)
    y0 = sp.special.comb(x, 2)
    yl = 0.004 * y0
    yu = 0.014 * y0
    #ax.plot(x, y, 'k')
    ax.plot(x, np.log10(y0))
    ax.plot(x, np.log10(yl))
    ax.plot(x, np.log10(yu))
    #ax.plot(x, yl)
    ax.set_xlabel("N unique proteins")
    ax.set_ylabel("log10 N pairs")
    return ax

def get_user_info_bsasa_ref(bsasa_ref):
    nrows, ncols = bsasa_ref.shape
    n_pdb_ids = len(set(bsasa_ref['PDBID'].values))
    n_uids = len(set(bsasa_ref["Prey1"].values).union(set(bsasa_ref["Prey2"].values)))
    return {"shape" : (nrows, ncols), "N PDB IDs" : n_pdb_ids, "N UIDs" : n_uids}

def prey_in_bsasa(gene2uid, prey_set):
    return [ 
      ("VIF",   gene2uid['vifprotein'] in prey_set),
      ("ELOB",  gene2uid['ELOB_HUMAN'] in prey_set),
      ("ELOC",  gene2uid['ELOC_HUMAN'] in prey_set),
      ("LRR1",  gene2uid['LLR1_HUMAN'] in prey_set),
      ("CBFB",  gene2uid['PEBB_HUMAN'] in prey_set),
      ("CUL5",  gene2uid['CUL5_HUMAN'] in prey_set),
      ("NEDD8", gene2uid['NEDD8_HUMAN'] in prey_set)
      ] 

def print_(uid_total, interaction_total, direct_interaction_set,
        nuid, n_possible_mapped_interactions):
    a = h(len(direct_interaction_set))
    b = h(np.round(100 *len(direct_interaction_set) / interaction_total, 2))
    c = np.round((len(direct_interaction_set) / n_possible_mapped_interactions
        ) * 100, 2)

    d = h(int(n_possible_mapped_interactions * 0.0035))
    print((
        f"Of the {h(uid_total)} total prey "
        f"{h(interaction_total)} interactions are possible"
        f"Of these, {a} were found in the PDB\n"
        f"representing {b}% of interactions"
        f"Of the {h(nuid)} mapped prey "
        f"{h(n_possible_mapped_interactions)} interactions are possible"
        f"{a} ({c}%)"
        f" were found in the PDB"
        "It is estimated that 0.35%-1.5% of possible protein interactions "
        "are positive"
        f"This corresponds to {d} to "
        f"{h(int(n_possible_mapped_interactions * 0.015))} "
        "possible interactions"
        f"The remaining {uid_total - nuid} prey were not found in the PDB"
        f"This corresponds to {h(math.comb(uid_total - nuid, 2))} "
        "possible interactions"))

def predicted_positives(ref_set, pred_set):
    return len(pred_set)

def permutation_test(rseed, vals, true, T, n_samples):
    t_true = T(true)
    N = len(true)
    key = jax.random.PRNGKey(rseed)
    sampling = jax.random.choice(key, a=vals, shape=(N, n_samples))
    results = T(sampling, axis=0)
    true_result = T(true)
    return np.array(results), np.array(true_result)
    
def plot_sampling(results, true_result, v0, v1, test_stat="",
        backend='seaborn', title="", vkwargs={'color':'r', 'label': 'True'},
        histkwargs={'label':'Null'}, tx=0, ty=5, ts=f"N samples {''}\nSize {''}",
        nbins=None, savefig=False, show=False, ax=None):
    nsamples = len(results)
    if not nbins:
        nbins = min(nsamples // 10, 100)
    if backend == 'seaborn':
        ax = sns.histplot(results, bins=nbins, **histkwargs, ax=ax)
    elif backend == 'mpl':
        ax = plt.hist(results, bins=nbins, **histkwargs)
    ts = ts + f"\np-value {pval(results, true_result)}"
    plt.text(tx, ty, ts)
    plt.vlines(true_result, v0, v1, **vkwargs)
    plt.xlabel(test_stat)
    plt.legend()
    plt.title(title)
    if savefig:
        assert title != ""
        savename = title.replace(" ", "") + ".png"
        plt.savefig(savename, dpi=NOTEBOOK_DPI) 
    if show:
        plt.show()
    return ax
    
def pp(i, f, g = lambda x: x, printf=print):
    """
    Pretty print
    i :: an iterable   object
    f :: (i -> int)    info getter
    g :: (str -> str)  formater
    printf A function for printing
    ---
    h :: (int -> str)
    """
    printf(g(h(f(i))))

def pval(results, true_results):
    N = len(results)
    return np.sum(results >= true_result) / N

def posterior_odds(samples, Yexp):
    def f(x, Yexp):
        """x as a vector, return average probability"""
        return jax.vmap(jax.scipy.stats.poisson.pmf)(Yexp, x).sum(axis=1) / 4
    def f2(X, Yexp):
        nrows, nsamples = X.shape
        return jax.vmap(f, in_axes=[1, None])(X, Yexp).sum(axis=0) / nsamples
    K = samples['k'].T
    L = samples['l'].T
    return np.array(f2(K, Yexp)), np.array(f2(L, Yexp))

def remove_self_interactions(bsasa_ref):
    sel = bsasa_ref["Prey1"].values != bsasa_ref["Prey2"].values
    bsasa_ref = bsasa_ref[sel]
    return bsasa_ref

def remove_self_interactions(bsasa_ref):
    sel = bsasa_ref["Prey1"].values != bsasa_ref["Prey2"].values
    bsasa_ref = bsasa_ref[sel]
    return bsasa_ref

def remove_prey_nan(bsasa_ref):
    """Remove nans from DataFrame for Prey1 and Prey2 columns"""
    notna = pd.notna(bsasa_ref["Prey1"].values)
    bsasa_ref = bsasa_ref[notna]
    notna = pd.notna(bsasa_ref["Prey2"].values)
    bsasa_ref = bsasa_ref[notna]
    return bsasa_ref

def remove_every_other(df):
    assert len(df) % 2 == 0, f"Number of entries is not even"
    return df.iloc[list(map(isOdd, range(0, len(df)))), :]

def ref_df2ref_set(df, a, b):
    return set([frozenset(i) for i in df.loc[:, [a, b]].values])

def run_and_merge_models(df_new, from_, to, step, rng_key):
    posterior = []
    prior = []
    observed = []
    prior_p = []
    post_p = []
    log_like = []
    sample_stats = []
    intervals = list(range(from_, to, step))
    for i in range(len(intervals)):
        start = intervals[i]
        end = start + step
        print(start ,end)
        rng_key, k1 = jax.random.split(rng_key)
        m_data = model52m_data(df_new, start, end, k1)
        posterior.append(m_data.posterior)
        prior.append(m_data.prior)
        observed.append(m_data.observed_data)
        prior_p.append(m_data.prior_predictive)
        log_like.append(m_data.log_likelihood)
        sample_stats.append(m_data.sample_stats)
    posterior = xr.merge(posterior)
    prior = xr.merge(prior)
    observed = xr.merge(observed)
    prior_p = xr.merge(prior_p)
    post_p = xr.merge(post_p)
    log_like = xr.merge(log_like)
    sample_stats = xr.merge(sample_stats)
    return av.InferenceData({'posterior': posterior, 'prior': prior, 'posterior_predictive': post_p,
                            'prior_predictive': prior_p, 'log_likelihood': log_like,
                            'observed_data': observed})

def set2frozen_pairs(s):
    """
    s[a] -> s[f[a1, a2], ...]
    The set of unordered pairs 
    """
    return set([frozenset(i) for i in combinations(s, 2)])

def slice_up_df_metric(df, thresholds, col, compare_f, action_f):
    results = []
    for t in thresholds:
        sel = compare_f(df[col].values, t)
        results.append(action_f(df[sel]))
    return results

def satisfaction(inf_data, chain_num=0, dims=('rrep', 'crep')):
    """Define Data Satisfaction in a simple way
    - a dataset satifies the model if
      - the obs mean is within the sample mean
      - the obs variance is within the sample variance
      - the sim min value <= obs min value
      - the sim max value >= obs max value"""
    y_pp_sim = inf_data.prior_predictive.sel(chain=chain_num)
    y_Pp_sim = inf_data.posterior_predictive.sel(chain=chain_num)
    pp_s, Pp_s, o_s = inf_get_summary_stats(inf_data, chain_num=chain_num)
    return namedtuple("pc", "pp Pp obs")(check_satisfaction(pp_s, o_s),
        check_satisfaction(Pp_s, o_s), o_s)

def save_ax(ax):
    title = ax.title.get_text()
    savename = title.replace(" ", "") + ".png"
    plt.savefig(savename, ax=ax, dpi=NOTEBOOK_DPI)

def save_plt(savename=None):
    if savename:
        plt.savefig(savename, NOTEBOOK_DPI)
    else:
        ax = plt.gca()
        title = ax.title.get_text()
        savename = title.replace(" ", "") + ".png"
        plt.savefig(savename, dpi=NOTEBOOK_DPI)

def score_from_df(pred_df, ref_set, a, b, score_fun):
    """
    pred_df: prediction dataframe
    ref_set: reference set
    a: column
    b: column
    score_fun :: (ref_set -> pred_set -> int)
    """
    pred_set = set2frozen_pairs(two_col_union(pred_df, a, b))
    return score_fun(ref_set, pred_set)

def summary_stats(m_data):
    data = m_data.posterior.stats.sel(stat=['a_rhat', 'b_rhat', 'a_neff', 'b_neff'])
    max_ = data.max('irow').values
    min_ = data.min('irow').values
    std = data.std(dim='irow').values
    med = data.median('irow').values
    return pd.DataFrame([max_, min_, med, std], index=['max', 'min', 'med', 'std'], 
        columns=data.stat.values)

def summarize_col(df, col):
    """Print summary stats for a DataFrame column"""
    vals = df[col].values
    def f(x, g):
        return h(np.round(g(x), 2))
    d = {"min": np.min, "max": np.max,
         "mean": np.mean, "var": np.var}
    for key, ufunc in d.items():
        yield colprint(f"{col} {key}", f(vals, ufunc))

def simple_scatter(x, y, xname, yname, title=None):
    plt.plot(x, y, 'k.')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)

def scores_from_df(pred_df, thresholds, sel_col_name: str, ref_set,score_from_df_fun):
    """
    pred_df : DataFrame
    thresholds : iterable[...]
    sel_col_name: column to apply thresholds
    ref_set: set
    score_from_df_fun :: (pred_df -> ref_set -> int)
    """
    return [score_from_df_fun(df, ref_set) for df in [pred_df.loc[pred_df[sel_col_name]>=t] for t in thresholds]]

def scores_from_df_fast(pred_df, thresholds, sel_col_name, ref_set, score_funs):
    """
    1. Select the subframe
    2. Compute tp
    3. Compute pp
    4. return
    """
    nrows = len(thresholds)
    ncols = len(score_funs)
    scores = np.zeros((nrows, ncols))
    for i, sub_df in enumerate((pred_df.loc[pred_df[sel_col_name] >= t] for t in thresholds)):
        scores[i, :] = [score_fun(sub_df, ref_set) for score_fun in score_funs]
    return scores

def thresh_sel(t, x):
    """Return the number of remaining entries"""
    return len(x[x >= t])

def to_satisfaction_frame(p_c):
    """sat tup: A tuple from check_satsifaction"""
    df = pd.DataFrame(index=p_c.mean.coords['irow'], data={
        'mean_c': p_c.mean.y_c, 'mean_e': p_c.mean.y_e,
        'var_e':  p_c.var.y_e, 'var_c':  p_c.var.y_c})
    df.loc[:, 'all_e'] = np.alltrue(df.loc[:, ['mean_e', 'var_e']], axis=1)
    df.loc[:, 'all_c'] = np.alltrue(df.loc[:, ['mean_c', 'var_c']], axis=1)
    return df.loc[:, ['mean_c', 'var_c', 'all_c', 'mean_e', 'var_e', 'all_e']]

def tp(ref_set, pred_set):
    return len(ref_set.intersection(pred_set))

def tp_over_pp_score(ref_set, pred_set):
    tp_ = tp(ref_set, pred_set)
    pp_ = len(pred_set)
    if pp_ != 0:
        return tp_ / pp_
    else:
        return 0

def tp_from_df(df, ref_set):
    prey_set = set(df['bait'].values).intersection(df)
    
def df2pp_tp(sub_df, threshold):
    prey_set = set(sub_df['bait']).union(sub_df['PreyName'])
    prey_pairs = set2frozen_pairs(prey_set)
    n_predicted_positives = len(prey_pairs)
    n_tp = tp(ref_set, prey_pairs)
    return n_predicted_positives, n_tp

def two_col_union(df, a, b):
    """
    The union of unique values from two columns
    """
    bait = set(df[a].values)
    prey = set(df[b])
    return bait | prey

def tmp_scatter(df, x, y, title="tmp", plot_xy=True):
    df.plot(x=x, y=y, title=title, kind="scatter", alpha=0.1)
    if plot_xy:
        x = df[x].values
        plt.plot(x, x, label="y=x", color='r', alpha=0.1)
    plt.tight_layout()
    plt.legend()
    save_plt()
    plt.close()

def un_center_and_scale_predictor(x, param_scale, min_value=1e-8, safe=True):
    if safe:
        assert x.shape[0] == param_scale.col_shape[0], (x.shape, param_scale.col_shape)
    x = np.array(x)
    var = param_scale.var
    var[var==0] = 1
    x = (x - param_scale.shift) * var + param_scale.mean
    x[np.where(x <= min_value)] = 0.
    return x

def update_df_new_PDB_COCOMPLEX(df_new, ds):
    cocomplex_labels = []
    for i, r in df_new.iterrows():
        bait = r['bait']
        prey = r['PreyName']
        if bait == 'LRR1':
            bait = 'LLR1'
        elif bait == 'CBFB':
            bait = 'PEBB'
        val = ds['cocomplex'].sel(preyu=bait, preyv=prey).item()
        cocomplex_labels.append(val)
    # Assign
    df_new.loc[:, 'PDB_COCOMPLEX'] = np.array(cocomplex_labels, dtype=bool)
    return df_new

def xy_from(prey_pairs_df, col, thresholds, comp, pos_col):
    npos = []
    ntot = []
    for t in thresholds:
        p, nt = npos_ntotal(prey_pairs_df, col, t, comp=comp, pos_col=pos_col)
        npos.append(p)
        ntot.append(nt)
    return np.array(ntot), np.array(npos)

def is_iterable(o):
    try:
        iter(o)
        return True
    except TypeError:
        return False
    else:
        raise ValueError

def run_mcmc(rng_key, model, model_data, num_samples, num_warmup, 
             extra_fields=('potential_energy','diverging'), thinning=1):
    kernel = numpyro.infer.NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, thinning=thinning)
    mcmc.run(rng_key, model_data, extra_fields=extra_fields)
    return mcmc

def posterior_predictive_dist(rng_key, model, model_data, mcmc):
    samples = mcmc.get_samples()
    return numpyro.infer.Predictive(model, samples)(rng_key, model_data)

def prior_predictive_dist(rng_key, model, model_data, num_samples=1000):
    return numpyro.infer.Predictive(model, num_samples=num_samples)(rng_key, model_data)

def get_summary_stats(mcmc):
    sd = summary(mcmc.get_samples())
    return {'n_eff': sd[key]['n_eff'] for key in sd.keys()} | {'r_hat': sd[key]['r_hat'] for key in sd}
    

def predictive_check(T, predictive, obs=None, vectorize=True):
    """
    T: mean, var, min, max, max_min_diff
    """
    Tpred = xr.apply_ufunc(T, predictive, input_core_dims=[["rep"]], vectorize=vectorize)
    if obs is not None:
        Tobs = xr.apply_ufunc(T, obs, input_core_dims=[["rep"]], vectorize=vectorize)
    else:
        Tobs = None
    return Tpred, Tobs
    
    
def av_pred_hist(ax, idata, kind='post', y_key='sc'):
    obs = idata.observed_data
    if kind == 'post':
        pred = idata.posterior_predictive
    elif kind == 'prior':
        pred = idata.prior_predictive
    ax.hist(pred[y_key])
# +
# Prior and posterior predictive check example
# Checks of simulation

def predictive_hist(ax, samples, observed, bins=100, vmin=0, vmax=1):
    ax.hist(samples, bins=bins)
    ax.vlines(observed, vmin, vmax, 'r')
    return ax

def gpt_model(data, observed=True):
    batch_shape =     (10, 3, 3, 2)
    sample_shape = (4, 10, 3, 3, 2)
    assert data.shape == sample_shape
    if not observed:
        data = None
    with numpyro.plate('rep', 4):
        mu = numpyro.sample('mu', dist.Beta(jnp.ones(batch_shape), jnp.ones(batch_shape)))
        sigma = numpyro.sample('sigma', dist.HalfNormal(jnp.ones(batch_shape)))
        numpyro.sample('y', dist.ZeroInflatedPoisson(gate=mu, rate=sigma), obs=data)

def zero_inflated_model(obs):
    pi = numpyro.sample('pi', dist.Beta(2.3, 2))
    alpha = numpyro.sample('alpha', dist.HalfNormal(50))
    with numpyro.plate("obs", obs.shape[0]):
        numpyro.sample('z', dist.ZeroInflatedPoisson(pi, rate=alpha), obs=obs)

def zero_inflated_spectral_counts(obs_e, obs_c):
    """
    obs n prey x n replicates
    pi or gate parameter : Proportion of 
    """
    n_prey, n_rep  = obs_e.shape
    assert n_rep == 4
    assert obs_e.shape == obs_c.shape
    data = np.zeros((n_rep, n_prey, 2))
    data[:,:, 0] = obs_c.T
    data[:,:, 1] = obs_e.T
    data = jnp.array(data, dtype=int)
    # observed shape: (prey, replicate, experiment)
    pi = numpyro.sample("pi", dist.Beta(jnp.ones((n_prey, 2)) * 2.0, jnp.ones((n_prey, 2)) * 2.3))
    #pi_e = numpyro.sample("pi_e", dist.Beta(jnp.ones(n_prey) * 2.0, jnp.ones(n_prey) * 2.3))
    #epsilon = numpyro.sample("epsilon", dist.HalfNormal(jnp.ones(n_prey) * 50))
    lambda_hyper = np.ones((n_prey, 2)) * 100
    lambda_hyper[:, 0] = 50
    lambda_hyper = jnp.array(lambda_hyper)
    lambda_ = numpyro.sample("lam", dist.HalfNormal(lambda_hyper))
    #with numpyro.plate("data", n_rep):
    numpyro.sample("sc", dist.ZeroInflatedPoisson(gate=pi, rate=lambda_), obs=data)

def out_of_distribution_score(upper, lower, observed):
    length = upper - lower
    a = np.min([lower, observed], axis=0) - lower
    b = np.max([upper, observed], axis=0) - upper
    return np.abs(a + b)# / length

def ood_from_summary_stats(summary_stats, T = 'min', pred='post', y_i='y_c'):
    if pred == 'post':
        p = 'Pp_'
    elif pred == 'prior':
        p = 'pp_'
    else:
        raise ValueError
    key = p + T + "_" + y_i
    okey = 'o_' + T + "_" + y_i
    x = summary_stats[key].T
    lower, median, upper = x[:, 0], x[:, 1], x[:, 2]
    observed = summary_stats[okey]
    y = out_of_distribution_score(upper, lower, observed)
    return y

def plot_from_summary_stats(summary_stats, T, pred, y_i):
    y = ood_from_summary_stats(summary_stats, T, pred, y_i)
    x = np.arange(len(y))
    plt.plot(x, y, 'k.')
    plt.ylabel('Out distr')

def ppc_boxen(boxes, 
    labels=['ppc control', 'ppc AP', 'PpC control', 'PpC AP'], font_dict = {'size': 16},
    title=None):
    y_tilde = '\u1EF9'
    if title == None:
        title = "T(y) - T(%s)" %(y_tilde)
    sns.boxenplot(boxes)
    plt.xticks(np.arange(len(boxes)),labels , **font_dict)
    plt.title(title, **font_dict)
    plt.ylabel("Spectral Count Difference", **font_dict)
    #plt.text(2.5, 50, "T: Min", **font_dict)
    plt.grid()

def write_prey_in_BSASA_2pad(pad, gene2uid, prey_set):
    for i in sorted([f"CSN{i}_HUMAN" for i in range(1, 10)] + ["CSN7A_HUMAN", "CSN7B_HUMAN"]):
        if i in gene2uid:
            pad.write((f"{i.removesuffix('_HUMAN')}", gene2uid[i] in prey_set))

def spectral_counts2cosin_sim_df(spectral_counts_xarray):
    data = {}
    for bait in spectral_counts_xarray.bait:
        for condition in spectral_count_xarray.condition:
            for ap in spectral_count_xarray.AP:
                for rep in spectral_count_xarray.rep:
                    val = spectral_count_xarray.sel(condition=condition,
                        bait=bait, rep=rep, AP=ap)
                    ap_key = "ap" if ap else "ctrl"
                    key = bait.item() + "_" + condition.item() + "_" + ap_key + "_" + str(rep.item())
                    data[key] = val.values
    df = pd.DataFrame(data)
    df.index = spectral_count_xarray.preyu
    # Drop zeros
    sel = np.sum(df.values, axis=1) !=0
    df = df[sel]
    return df

def cosin_sim_df2cos_sim_matrix(cosin_sim_df):
    df = cosin_sim_df
    matrix = df.values @ df.values.T
    mag_v = np.sqrt(np.sum(np.square(df.values), axis=1))   
    mag_v = mag_v.reshape(mag_v.shape + (1,))
    matrix = matrix / mag_v
    matrix = matrix / mag_v.T
    matrix = np.clip(matrix, -1, 1)
    index = df.index
    matrix = xr.DataArray(matrix, coords={"preyu": index, "preyv": index}, dims=["preyu", "preyv"])
    return matrix, mag_v 


def pairwise_matrix2edge_list(pairwise_matrix):
    n = len(pairwise_matrix)
    N = math.comb(n, 2)
    prey1 = np.zeros(N, dtype=pairwise_matrix.preyu.dtype)
    prey2 = np.zeros(N, dtype=pairwise_matrix.preyu.dtype)
    score = np.zeros(N, dtype=pairwise_matrix.dtype)
    prey_pairs = combinations(pairwise_matrix.preyu, 2)
    k=0
    for p1, p2 in prey_pairs:
        prey1[k] = p1.item()
        prey2[k] = p2.item()
        score[k] = pairwise_matrix.sel(preyu=p1, preyv=p2).item()
        k+=1
    data = {"prey1": prey1, "prey2": prey2, "score": score}
    return pd.DataFrame(data)

def pp_from_pairwise_prediction_matrix(matrix, k=-1):
    """
    Compute the number of positive predictions (pp) from a pairwise symetric prediction matrix
    """
    L = np.tril(matrix, k=k)
    pp = np.sum(L)
    return pp

def tp_from_pairwise_prediction_matrix_and_ref(ref_matrix, pred_matrix, k=-1):
    LR = np.tril(ref_matrix, k=k)
    LP = np.tril(pred_matrix, k=k)
    result = LR * LP # Element-wise multiplication
    return np.sum(result)

def pp_tp_from_pairwise_prediction_matrix_and_ref(
        ref_matrix, pred_matrix, thresholds,compare_func = op.ge, k=-1):
    tps = []
    pps = []
    for thresh in thresholds:
       m = compare_func(pred_matrix, thresh) 
       pps.append(pp_from_pairwise_prediction_matrix(m, k=k))
       tps.append(tp_from_pairwise_prediction_matrix_and_ref(ref_matrix, m, k=k))
    return pps, tps 

def get_tpr(tps, n):
    return np.array(tps) / n

def get_ppr(pps, M):
    return np.array(pps) / M

def auc(x, y, rule="simpson"):
    return sp.









