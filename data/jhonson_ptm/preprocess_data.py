import pandas as pd
NSHEETS = 3
d = {}
prefix = "NIHMS1798678-supplement-2" 

with open(f"{prefix}.xlsx", "rb") as f: 
    for sheet in range(NSHEETS):
        d[sheet] = pd.read_excel(f, sheet_name=sheet)

for sheet in range(NSHEETS):
    d[sheet].to_csv(prefix + f"_sheet_{sheet}.tsv")

# Get all the uniprot ids across all experiments and write to a file
d0_uids = set(d[0].loc[:, "Protein.Accession"].values)
d1_uids = set(d[1].loc[:, "Protein.Accession"].values)
d2_uids = set(d[2].loc[:, "Protein.Accession"].values)

all_uids = d0_uids.union(d1_uids).union(d2_uids)
all_uids = list(all_uids)
uid_dict = {}
for uid in all_uids:
    if ";" in uid:
        uid_lst = uid.split(";")
        for u in uid_lst:
            uid_dict[u] = None
    else:
        uid_dict[uid] = None
all_uids = list(uid_dict.keys())        

dall = pd.DataFrame({"jhonson_all_uids": list(all_uids)}) 
dall.to_csv("jhonson_all_uids.tsv", sep="\t", index=False)

