"""
PreProcess PrePPI scores for LR > 379
"""
import pandas as pd

df = pd.read_csv("preppi.human_af.interactome.txt", sep="\t")

sel = df['total_score'] > 379
df_proc = df[sel]

df_proc.to_csv("../processed/preppi/preppi.human_af.interactome_LR379.txt", sep="\t")
