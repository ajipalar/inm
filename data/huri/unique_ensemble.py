import pandas as pd
d = pd.read_csv("HuRI.tsv", sep="\t", names=['a', 'b'])
a = set(d['a'].values)
b = set(d['b'].values)
a_b = a.union(b)
d2 = pd.DataFrame({"EnsembleGeneID": list(a_b)})
d2.to_csv("HuRI_ensemble.tsv", index=False, sep="\t", header=False)

