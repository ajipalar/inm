import pandas as pd
import xarray as xr

d = pd.read_csv("BIOGRID-ALL-4.4.223.tab3.txt", sep="\t")
allowed_organisms = ["Homo sapiens", "Human Immunodeficiency Virus 1", "Severe acute respiratory syndrome coronavirus 2"]   

sel = d['Experimental System Type'] == 'physical'
columns = ["BioGRID ID Interactor A", "BioGRID ID Interactor B",
           "Organism Name Interactor A", "Organism Name Interactor B"]

d = d.loc[sel, columns]

s1 = d['Organism Name Interactor A'] == allowed_organisms[0]
s2 = d['Organism Name Interactor A'] == allowed_organisms[1]
s3 = d['Organism Name Interactor A'] == allowed_organisms[2]

s = s1 | s2
s = s | s3

d = d[s]

s1 = d['Organism Name Interactor B'] == allowed_organisms[0]
s2 = d['Organism Name Interactor B'] == allowed_organisms[1]
s3 = d['Organism Name Interactor B'] == allowed_organisms[2]

s = s1 | s2
s = s | s3

d = d[s]

a_ids = set(d['BioGRID ID Interactor A'])
b_ids = set(d['BioGRID ID Interactor B']) 

# Filter down to unique non self interactions

dd = {}
for i, r in d.iterrows():
    a = r['BioGRID ID Interactor A']
    b = r['BioGRID ID Interactor B']
    if a == b:
        continue
    if isinstance(a, int) and isinstance(b, int):
        pair = frozenset((a, b))
        if pair not in dd:
            dd[pair] = None

aids = []
bids = []
for pair in dd:
    a, b = pair
    aids.append(a)
    bids.append(b)

dd = pd.DataFrame({'a': aids, 'b': bids})    

unique_biogrid_ids = set(dd['a']).union(set(dd['b']))
uids = pd.DataFrame({'BioGRID ID': list(unique_biogrid_ids)})

uids.to_csv("unique_biogrid_ids.tsv", sep="\t", index=False, header=False)

id_map = pd.read_csv("idmapping_2024_02_24.tsv", sep="\t")

b2uid = {r['From'] : r['Entry'] for i, r in id_map.iterrows()} 
uid2b = {r['Entry'] : r['From'] for i, r in id_map.iterrows()}

auid = []
buid = []

for i, r in dd.iterrows():
    a = r['a']
    b = r['b']
    au = b2uid[a] if a in b2uid else "NOTMAPPED"
    bu = b2uid[b] if b in b2uid else "NOTMAPPED"
    auid.append(au)
    buid.append(bu)

dd['auid'] = auid
dd['buid'] = buid

s1 = dd['auid'] == "NOTMAPPED"
s2 = dd['buid'] == "NOTMAPPED"
s3 = s1 | s2
s3 = ~s3

dd.to_csv("biogrid_reference.tsv", sep="\t", index=False)

