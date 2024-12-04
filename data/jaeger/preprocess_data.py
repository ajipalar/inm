import pandas as pd
d = {}
prefix = "41586_2012_BFnature10719_MOESM289_ESM" 
with open(f"{prefix}.xls", "rb") as f: 
    for sheet in range(4):
        d[sheet] = pd.read_excel(f, sheet_name=sheet)


for sheet in range(4):
    d[sheet].to_csv(prefix + f"_sheet_{sheet}.tsv", sep="\t", index=False)

# Get all the uniprot ids across all experiments and write to a file
d0_uids = set(d[0].loc[:, "Prey"].values)
d1_uids = set(d[1].loc[:, "#"].values[2:])
d2_uids = set(d[2]["Prey"].values)
d3_uids = set(d[3]["#"].values[2:])

all_uids = d0_uids.union(d1_uids).union(d2_uids).union(d3_uids)

dall = pd.DataFrame({"jaeger_all_uid": list(all_uids)}) 
dall.to_csv("jaeger_all_uids.tsv", sep="\t", index=False)

