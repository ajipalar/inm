import pandas as pd
import pickle as pkl
import numpy as np

# Load in all datasets
ndatasets = 5 
jaeger_uids = pd.read_csv("../../jaeger/jaeger_all_uids.tsv", sep="\t") 
huttenhain_uids = pd.read_csv("../../../table1.csv")
jhonson_uids = pd.read_csv("../../jhonson_ptm/jhonson_all_uids.tsv", sep="\t")
hiat_uids = pd.read_csv("../../hiat_2022/idmapping_2024_02_05.tsv", sep="\t")

with open("../../../notebooks/direct_benchmark.pkl", "rb") as f:
    pdb = pkl.load(f)
pdb = pdb.reference.matrix
pdb = np.sum(pdb, axis=0)
pdb = pdb[pdb != 0]
pdb_prey_uid = pdb.preyv
# Remap prey ids to uniprot ids
_tf = {r["PreyGene"] : r["UniprotId"] for i, r in huttenhain_uids.iterrows()}
tf = {(key if "_" not in key else key.split("_")[0]):val for key, val in _tf.items()}

pdb_prey_uid = np.array([tf[i] for i in pdb_prey_uid.values])
pdb_prey_uid = pd.Series(pdb_prey_uid)

#hiat_2022_uids = pd.read_csv("../../hiat_2022/")

ds = {"jaeger": jaeger_uids['jaeger_all_uid'],
      "huttenhain": huttenhain_uids['UniprotId'],
      "jhonson_uids": jhonson_uids['jhonson_all_uids'],
      "hiat_uids" : hiat_uids['To'],
      "pdb_prey_uid": pdb_prey_uid,
      }

dmat = np.zeros((ndatasets, ndatasets))

for i, k1 in enumerate(ds.keys()):
    for j, k2 in enumerate(ds.keys()):
        d1 = ds[k1].values
        d2 = ds[k2].values
        d1 = set(d1)
        d2 = set(d2)
        inter = len(d1.intersection(d2))
        dmat[i, j] = inter

names = ["Jaeger 2011", "Huttenhain 2019", "Jhonson 2022", "Hiat 2022", "PDB PPI"] 

df = pd.DataFrame(dmat, columns=names, index=names, dtype=int)
df.to_csv("dataset_intersection.tsv", sep="\t")

