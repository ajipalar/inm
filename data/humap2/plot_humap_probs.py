import pandas as pd
import matplotlib.pyplot as plt
d = pd.read_csv("humap2_ppis_ACC_20200821.pairsWprob", sep="\t", names=['a', 'b', 'p'])

data = d['p'].values

plt_range = (0.0001, 1)
plt.hist(data, bins=100, range=plt_range)
plt.xlabel("Prob")
plt.ylabel("Freq")
plt.savefig("humap_probs_300.png", dpi=300)
plt.savefig("humap_probs_1200.png", dpi=1200)
