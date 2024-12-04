import pandas as pd
d = pd.read_csv("BIOGRID-ALL-4.4.223.tab3.txt", sep="\t")

allowed_organisms = ["Homo sapiens", "Human Immunodeficiency Virus 1", "Severe acute respiratory syndrome coronavirus 2"]   

s0 = None
for name in allowed_organisms:
    sel1 = d['Organism Name Interactor A'] != name
    sel2 = d['Organism Name Interactor B'] != name
    sel3 = sel1 & sel2
    if s0 is not None:
        s0 = sel3 & s0
    else:
        s0 = sel3
s0 = ~s0

sel = d['Experimental System Type'] == 'physical'
s0 = s0 & sel

