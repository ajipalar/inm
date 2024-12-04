import pandas as pd
from pathlib import Path
import math

# Read in the ids of both data sets

hiv_emap_path = "../data/hiv_emap/idmapping_2023_07_05.tsv"
hiv_emap_ids = pd.read_csv(hiv_emap_path, sep="\t")

from_ = set(hiv_emap_ids['From'])
to_ = set(hiv_emap_ids['To'])
to_from = {row['To']: row['From'] for i, row in hiv_emap_ids.iterrows()}

print(hiv_emap_ids)
print(f"from : {len(from_)}")
print(f"to : {len(to_)}")

apms_data_path = "../table1.csv"

apms_ids = pd.read_csv(apms_data_path)
uids = set(apms_ids['UniprotId'])
print(f"apms uids: {len(apms_ids)}")

intersection = uids.intersection(to_)
print(f"HIV APMS INTERSECTION: {len(intersection)}")
unique_genes = [to_from[uid] for uid in intersection]
print(f"Unique genes in intersection {len(unique_genes)}")
print(f"Conclusion - at least {math.comb(len(unique_genes), 2)} interactions are supported by both AP-MS and genetic interactions")
