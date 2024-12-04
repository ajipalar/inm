import pandas as pd
prefix = "41467_2022_29346_MOESM3_ESM" 

with open(f"{prefix}.xlsx", "rb") as f: 
     d = pd.read_excel(f, sheet_name=0)

d.to_csv(prefix + f"_sheet_{0}.tsv")

dall = d['Gene Name'] 
dall = pd.DataFrame({"Gene Name": dall.unique()})
dall.to_csv("hiat_all_genes_targeted.tsv", sep="\t", index=False)

