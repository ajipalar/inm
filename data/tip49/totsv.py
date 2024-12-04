import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_excel("NIHMS252278-supplement-2.xls", header=2)
d.to_csv("tip49_sheet1.tsv", sep="\t", index=False)


d.hist('SAINT')
plt.savefig("saint_hist.png")
plt.close()

d.hist()
plt.savefig("tip49_all.png")

