"""
Inputs

Reference
- PDB Complex Matrix 
- PDB Direct Interaction Reference
- PDB Distance Reference
- Biogrid Edge List

Prediction
- GI Score Matrix
- Saint Score Edge list
- HGSCore (edge list or matrix?)
- INM Matrix Trajectory


Identifiers
- Uniprot

- HGSCore matrix
- SAINT score list
- INM Xarray trajectory

Data Structure Comparisons

Trajectory to direct-matrix
Trajectory to cocoplex-matrix
Trajectory to biogrid dataframe

Trajectory to matrix comparison
matrix to matrix

matrix vs matrix comparison
matrix vs edge_list comparison -> matrix vs matrix


"""

import sys
sys.path = ["../notebooks"] + sys.path

from _BSASA_functions import *

import click
import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr
from pathlib import Path
from functools import partial

# IO
def _pkl_load(path, site="rb"):
    with open(path, site) as f:
        obj = pkl.load(f)
    return obj

df_new = read_cullin_data_from_pickle()

id_maps = SimpleNamespace()
id_maps.preyname2uniprot_id = {
id_maps.viral_name2uniprot = {
    "vifprotein"          :   "P69723",
    "polpolyprotein"      :   "Q2A7R5",
    "nefprotein"     :        "P18801",
    "tatprotein"         :    "P0C1K3",
    "gagpolyprotein"     :    "P12493",
    "revprotein"          :   "P69718",
    "envpolyprotein"      :   "O12164"}

id_maps.uniprot2preyname = {r['Prey']: i for i,r in df_new.iterrows()}
id_maps.preyname2uniprot = {val: key for key, val in id_maps.uniprot2preyname.items()}

id_maps.uniprot2viral_name = {val: key for key, val in id_maps.viral_name2uniprot.items()}

def _load_biogrid(path="../data/BIOGRID-ALL-4.4.223.tab3.txt"):
    csv = pd.read_csv(path, sep="\t")
    return csv

def _load_gi_data():
    ...

def _idmap_gi_data():
    ...

def read_gi_data():
    ...

def _idmap_biogrid():
    """
    Map from Uniprot IDs to Biogrid IDs 
    """
    ...

def reduce_biogrid(biogrid, ommitted_organisms = None):
    """
    Reduce the size of biogrid by organism name
    """
    if ommitted_organisms == None:
        ommitted_organisms = [
        'Anas platyrhynchos',
        'Canis familiaris',
        'Arabidopsis thaliana (Columbia)',
        'Anopheles gambiae (PEST)',
        'Caenorhabditis elegans'
        'Oryctolagus cuniculus',
        'Solanum lycopersicum',
        'Sorghum bicolor',
        'Plasmodium falciparum (3D7)',
        'Rattus norvegicus',
        'Gallus gallus',
        'Schizosaccharomyces pombe (972h)',
        'Strongylocentrotus purpuratus',
        'Bos taurus',
        'Ustilago maydis (521)',
        'Emericella nidulans (FGSC A4)',
        'Escherichia coli (K12/MG1655)',
        'Escherichia coli (K12/BW2952)',
        'Chlamydomonas reinhardtii',
        'Oryctolagus cuniculus',
        'Caenorhabditis elegans',
        'Meleagris gallopavo',
        'Danio rerio',
        'Candida albicans (SC5314)',
        'Solanum tuberosum', 'Bacillus subtilis (168)',
        'Mycobacterium tuberculosis (H37Rv)', 'Streptococcus pneumoniae (ATCCBAA255)', 'Nicotiana tomentosiformis', 'Ovis aries',
        'Mus musculus', 'Pan troglodytes',
        'Zea mays', 'Saccharomyces cerevisiae (S288c)',
        'Oryza sativa (Japonica)',
        'Xenopus laevis', 'Ricinus communis',
        'Drosophila melanogaster', 'Macaca mulatta',
        'Selaginella moellendorffii', 'Sus scrofa',
        'Felis Catus', 'Leishmania major (Friedlin)',
        'Cavia porcellus']
    sel = np.ones(len(biogrid), dtype=bool)
    for i, name in enumerate(biogrid['Organism Name Interactor B']):
        if name in ommitted_organisms:
            sel[i] = 0 
    for i, name in enumerate(biogrid['Organism Name Interactor A']):
        if name in ommitted_organisms:
            sel[i] = 0
    biogrid = biogrid[sel]
    return biogrid

def biogrid2matrix(biogrid, binary=True):
    ref_matrix = read_direct_benchmark().reference.matrix
    ref_matrix.values = np.zeros(ref_matrix.shape, dtype=int)
    assert ref_matrix.ndim == 2, ref_matrix.ndim
    for i, r in biogrid.iterrows():
        preyu = r['preyu']
        preyv = r['preyv']
        val = ref_matrix.sel(preyu=preyu, preyv=preyv).item()
        val = val + 1
        ref_matrix.loc[preyu, preyv] = val
    ref_matrix.values = ref_matrix.values + ref_matrix.values.T
    ref_matrix.values[np.diag_indices(len(ref_matrix))] = np.zeros(len(ref_matrix))
    if binary:
        ref_matrix.values = np.where(ref_matrix > 0, 1, 0)
    return ref_matrix



def read_biogrid_benchmark():
    # Drop columns by organism
    biogrid = _load_biogrid()
    biogrid = reduce_biogrid(biogrid)
    biogrid.drop(columns=['#BioGRID Interaction ID',
         'Entrez Gene Interactor A', 'Entrez Gene Interactor B'], inplace=True)
    biogrid_ids = pd.read_csv("uniprot2biogrid.tsv", sep="\t")  
    id_maps.biogrid2uniprot = {r['To']: r['From'] for i, r in biogrid_ids.iterrows()}
    reference = [int(i) for i in biogrid_ids['To'].values]
    sel = np.ones(len(biogrid), dtype=bool)
    for i, biogrid_id in enumerate(biogrid['BioGRID ID Interactor A'].values):
        query = int(biogrid_id)
        if query not in reference: 
            sel[i] = 0
    for i, biogrid_id in enumerate(biogrid['BioGRID ID Interactor B'].values):
        query = int(biogrid_id)
        if query not in reference: 
            sel[i] = 0
    biogrid = biogrid[sel]
    # Remove self interactions
    #Add preyu and preyv columns
    sel = np.ones(len(biogrid), dtype=bool)
    preyu = []
    preyv = []
    for i, (label, r) in enumerate(biogrid.iterrows()):
        preya = r['BioGRID ID Interactor A']
        preyb = r['BioGRID ID Interactor B']
        if preya == preyb:
            sel[i] = 0
        uida = id_maps.biogrid2uniprot[preya]
        uidb = id_maps.biogrid2uniprot[preyb]
        preynamea = id_maps.uniprot2preyname[uida]
        preynameb = id_maps.uniprot2preyname[uidb]
        preyu.append(preynamea)
        preyv.append(preynameb)
    biogrid['preyu'] = preyu 
    biogrid['preyv'] = preyv 
    biogrid = biogrid[sel]
    return biogrid

def plot_biogrid_reference_stats(biogrid):
    fig, axs = plt.subplots(2, 2)
    #axs[0, 0].bar([0, 1], biogrid['Experimental System Type'].values)
    plt.savefig("biogrid_reference.png", dpi=1200)
     
def read_hgscore(path="hgscore/hgscore_output.csv"):
    hgscores = pd.read_csv(path)
    return hgscores

def read_pdb_chain_mapping_from_pickle(
    path="../notebooks/chain_mapping.pkl"):
    with open(path, "rb") as f:
        chain_mapping = pkl.load(f)
    return chain_mapping

def save_direct_id_maps():
    direct = read_direct_benchmark()
    ids_df = direct.reference.matrix.preyu.to_dataframe()
    # Map ids to uniprot ids
    d = id_maps.viral_name2uniprot
    l_ = [id_maps.preyname2uniprot[key] for key in ids_df['preyu'].values]
    l = [d[key] if key in d else key for key in l_]
    pd.DataFrame({"preyu": l}).to_csv("direct_ids.csv",
        index=False, header=False)
    
read_cocomplex_benchmark = partial(_pkl_load, path="../notebooks/cocomplex_benchmark.pkl")

read_direct_benchmark = partial(_pkl_load, path="../notebooks/direct_benchmark.pkl")

def read_cullin_data_from_pickle(
    path="../notebooks/df_new.pkl"):
    df_new = _pkl_load(path)
    return df_new
    
# Read in cocomplex and binary protein interactions

def set_wd(path):
    global wd
    wd = path

wd = Path(".")

# Comparison functions
def traj_vs_matrix(query_traj, ref_xr, compare_func, axis_name="draw"):
    results = []
    for label, matrix in query_traj.groupby(axis_name):
        results.append(matrix_vs_matrix(matrix, ref_xr, compar_func))
    return results 

def matrix_vs_matrix(query_xr, ref_xr, compare_func):
    assert np.all(query_xr.preyu == ref_xr.preyu)
    assert np.all(query_xr.preyv == ref_xr.preyv)
    return compare_func(query_xr, ref_xr)

def matrix_vs_df(query_xr, df, colA, colB, compare_func):
    ...

def read_references(convert_biogrid_df2matrix=True, biogrid_binary=True):
    """
    Read in references with a common idmapping
    """
    biogrid = read_biogrid_benchmark()
    direct = read_direct_benchmark()
    cocomplex = read_cocomplex_benchmark()
    reference = SimpleNamespace()
    reference.direct_matrix = direct.reference.matrix
    reference.cocomplex_matrix = cocomplex.reference.matrix
    if convert_biogrid_df2matrix:
        biogrid = biogrid2matrix(biogrid, binary=biogrid_binary)
    reference.biogrid = biogrid
    return reference

@click.command()
def main():
    ...

if __name__ == "__main__":
    main()
