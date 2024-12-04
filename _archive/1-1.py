import pandas as pd
import sys
sys.path = sys.path + ["../../../pyext/src/"]
import cullin_benchmark_test as cb
from pathlib import Path

def get_parsed_df(path, sheet_num=0):
    x = cb.CullinBenchMark(path, sheet_num=sheet_num)
    x.parse_spec_counts()
    return x.data

path = Path(".")
xlsx_file = "1-s2.0-S1931312819302537-mmc2.xlsx"
df1 = pd.read_excel(xlsx_file, 0)
df2 = pd.read_excel(xlsx_file, 1)
df3 = pd.read_excel(xlsx_file, 2)

# Drop the MOUSE Values

df1 = df1[df1['PreyGene'] != 'IGHG1_MOUSE']
df2 = df2[df2['PreyGene'] != 'IGHG1_MOUSE']
df3 = df3[df3['PreyGene'] != 'IGHG1_MOUSE']

assert "IGHG1_MOUSE" not in df1.values
assert "IGHG1_MOUSE" not in df2.values
assert "IGHG1_MOUSE" not in df3.values

viral_remapping = {      
"vifprotein"          :   "P69723",
"polpolyprotein"      :   "Q2A7R5",
"nefprotein"     :        "P18801",
"tatprotein"         :    "P0C1K3",
"gagpolyprotein"     :    "P12493",
"revprotein"          :   "P69718",
"envpolyprotein"      :   "O12164"}

cols = ["PreyGene", "Prey"]
prey_df = pd.concat([df1.loc[:, cols], df2.loc[:, cols], df3.loc[:, cols]])
prey_gene = []
prey = []
for i, r in prey_df.iterrows():
    pg, p = r['PreyGene'], r['Prey']
    if pg in viral_remapping:
        p = viral_remapping[pg]
    if pg not in prey_gene:
        assert p not in prey
        prey_gene.append(pg)
        prey.append(p)

prey_df = pd.DataFrame({"PreyGene": prey_gene, "UniprotId": prey})
for key in viral_remapping:
    assert key not in prey_df['UniprotId']
        
prey_df.to_csv("table1.csv", index=False)
