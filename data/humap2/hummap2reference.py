import pandas as pd
df = pd.read_csv("humap2_ppis_ACC_20200821.pairsWprob", sep="\t", names=['auid', 'buid', 'w'])

d_medium = df[df['w'] > 0.49]
d_high = df[df['w'] > 0.94]

d_medium.to_csv("../processed/references/humap2_ppis_medium.tsv", sep="\t", index=False)
d_high.to_csv("../processed/references/humap2_ppis_high.tsv", sep="\t", index=False)

df.to_csv("../processed/references/humap2_ppis_all.tsv", sep="\t", index=False)





