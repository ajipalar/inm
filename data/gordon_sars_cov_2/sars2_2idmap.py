import pandas as pd

id_map = {}

"""
bait2PreyGene = {
        "SARS-CoV2 E"  
"SARS-CoV2 M"
"SARS-CoV2 N"
"SARS-CoV2 nsp1"
"SARS-CoV2 nsp10"
"SARS-CoV2 nsp11"
"SARS-CoV2 nsp12"
"SARS-CoV2 nsp13"
"SARS-CoV2 nsp14"
"SARS-CoV2 nsp15"
"SARS-CoV2 nsp2"
"SARS-CoV2 nsp4"
"SARS-CoV2 nsp5_C145A"
"SARS-CoV2 nsp5"
"SARS-CoV2 nsp6"
"SARS-CoV2 nsp7"
"SARS-CoV2 nsp8"
"SARS-CoV2 nsp9"
"SARS-CoV2 orf10"
"SARS-CoV2 orf3a"
"SARS-CoV2 orf3b"
"SARS-CoV2 orf6"
"SARS-CoV2 orf7a"
"SARS-CoV2 orf8"
"SARS-CoV2 orf9b"
"SARS-CoV2 orf9c"
"SARS-CoV2 Spike"
"""



for sheet in range(3):
    df = pd.read_excel("./1-s2.0-S1931312819302537-mmc2.xlsx", sheet_name=sheet)
    for i, r in df.iterrows():
        bait = r['Bait'][0:4]
        bait_gene = bait2PreyGene[bait]
        prey_gene = r['PreyGene'].removesuffix("_HUMAN")
        prey_uid = r['Prey']
        if prey_gene not in id_map:
            id_map[prey_gene] = prey_uid
        else:
            assert id_map[prey_gene] == prey_uid, (prey_gene, prey_uid)

assert "PEBB" in id_map
assert "LLR1" in id_map
assert "ELOB" in id_map
assert "CUL5" in id_map

prey_gene_lst = []
prey_lst = []
for prey_gene, prey in id_map.items():
    prey_gene_lst.append(prey_gene)
    prey_lst.append(prey)

df = pd.DataFrame(data = {"PreyGene": prey_gene_lst, "Prey" : prey_lst})
df.to_csv("./../processed/cullin/id_map.tsv", sep="\t", index=False, header=False)

