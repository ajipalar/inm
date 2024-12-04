"""
Map significant hits to cluster members
"""
import pandas as pd
import numpy as np

clu_path = "pdb70_clu.tsv"
#clu_path = "~/databases/pdb70/pdb70_clu.tsv"

pdb70_clu = pd.read_csv(clu_path, sep="\t", names=["clu", "chain"])
sig_clu = pd.read_csv("hhblits_out/SignificantPreyPDB70PairAlign.csv")

pdb70clu2chain_list = {}
for i, r in pdb70_clu.iterrows():
    clu = r['clu']
    chain = r['chain']

    if clu not in pdb70clu2chain_list:
        pdb70clu2chain_list[clu] = [chain]
    else:
        pdb70clu2chain_list[clu].append(chain)

chain2pdb70clu = {}
for clu, chains in pdb70clu2chain_list.items():
    for chain in chains:
        assert chain not in chain2pdb70clu, chain
        chain2pdb70clu[chain] = clu

query_ids = []
pdb70_clu_ids = []
chains = []
probabs = []
evalues = []
scores = []
ali = []
iden = []
sim = []
sum_p = []
tneff = []
q = []
t = []

# QueryID, PDB70_CLU, Chain, Probab, Evalue, Score, Aligned_cols, Identities, Similarity, Sum_probs, Template_Neff, Q, T 

for i, r in sig_clu.iterrows():

    pdb70id = r["PDB70ID"]
    clu = chain2pdb70clu[pdb70id] 

    chain_lst = pdb70clu2chain_list[clu]
    for chain in chain_lst:
        query_ids.append(r["QueryUID"])
        pdb70_clu_ids.append(clu)
        chains.append(chain)
        probabs.append(r["Probab"])
        evalues.append(r["Evalue"])
        scores.append(r["Score"])
        ali.append(r["Aligned_cols"])
        iden.append(r["Aligned_cols"])
        sim.append(r["Similarity"])
        sum_p.append(r["Sum_probs"])
        tneff.append(r["Template_Neff"])
        q.append(r["Q"])
        t.append(r["T"])

df = pd.DataFrame({"QueryUID": query_ids, "PDB70_CLU": pdb70_clu_ids,
                   "PDB70_Chain": chains, "Probabs": probabs,
                   "Evalue": evalues, "Score": scores,
                   "Aligned_cols": ali, "Identities": iden,
                   "Similarity": sim, "Sum_probs": sum_p,
                   "Template_Neff": tneff, "Q": q, "T": t})
df.to_csv("hhblits_out/SigPDB70Chain.csv", index=False)


sig_pdbs = list(set([i.split("_")[0] for i in df['PDB70_Chain'].values]))
df = pd.DataFrame({"sig_pdbs": sig_pdbs})
df.to_csv("hhblits_out/SignificantPDB70_PDBIDs.csv", index=False, header=False)
