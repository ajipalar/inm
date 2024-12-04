"""
HuRI ID Mapping Pipeline

1. Map ENSEMBLE ID to UniProt Entry
2. Select reviewed entries
"""

import pandas as pd
from collections import defaultdict
import sys
sys.path.append("../../notebooks/")
import generate_benchmark_figures as gbf


d = pd.read_csv("HuRI.tsv", sep="\t", names=['a', 'b'])
#a = set(d['a'].values)
#b = set(d['b'].values)
#a_b = a.union(b)

# Mapped IDs

id_map = pd.read_csv("idmapping_2024_03_05.tsv", sep="\t")
reviewed = id_map['Reviewed'] == 'reviewed'
reviewed_id_map = id_map[reviewed]
#not_reviewed_id_map = id_map[~reviewed]
id_map = reviewed_id_map

#review_uid_set = set(reviewed_id_map['Entry'])
#not_reviewed_uid_set = set(not_reviewed_id_map['Entry'])

# Is id_map 1 to 1?

#entry2ensemble_list = defaultdict(list)
#for i, r in id_map.iterrows():
#    entry2ensemble_list[r['Entry']].append(r['From'])

ensemble2entry_list = defaultdict(list)
for i, r in id_map.iterrows():
    ensemble2entry_list[r['From']].append(r['Entry'])

# 54 Uniprot Entries are supported by more than 1 ensemble ID
# One of these (ENSG00000181577) seems to be a long non coding RNA. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4405709/
# Count an edge if either identifier pair counts it

# Hemoglobin subunits alpha 1 & 2 are encoded by 

#a_b_idmap = set(id_map['From']).intersection(a_b) # 8214 ids
#not_mapped = a_b - set(id_map['From']) # 58 ids were not mapped

# Get the non self pairs 
#ensemble2acc = {r['From'] : r['Entry'] for i,r in id_map.iterrows()}

d_edgelist = gbf.UndirectedEdgeList()
d_edgelist.update_from_df(d, a_colname="a", b_colname="b")

#d_edgelist.reindex(ensemble2acc, enforce_coverage = False)

d_edgelist.reindex(ensemble2entry_list, enforce_coverage = False, all_vs_all = True)

# Write the output
a_en = []
b_en = []
for i, a in enumerate(d_edgelist.a_nodes):
    b = d_edgelist.b_nodes[i]
    
d_edgelist.to_csv("../processed/references/HuRI_reference.tsv",
                  a_colname = "auid",
                  b_colname = "buid",
                  index = False,
                  sep = "\t",
                  header = True)


def dict_list_print(d):
    n = 0
    for key, l in d.items():
        if len(l) > 1:
            n += 1
            print(key, l)
    return n
