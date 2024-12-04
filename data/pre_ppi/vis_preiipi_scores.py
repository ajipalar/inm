import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("preppi.human_af.interactome.txt", sep="\t")

x = df['total_score'].values
plt.hist(x, bins=100, range=(0, 10*np.median(x)))
plt.savefig("PrePPI_scores.png", dpi=300)
print(f"Min {min(x)}")
print(f"Max {max(x)}")
print(f"Med {np.median(x)}")
print(f"Var {np.var(x)}")
print(f"Mean {np.mean(x)}") 
print(f"Std {np.std(x)}")


