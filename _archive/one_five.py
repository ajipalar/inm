from pathlib import Path
import pandas as pd
import json

def pp_aln(aln):
    if len(aln[0]) > 81:
        print(aln[0][0:80])
    else:
        print(aln[0])
    print(aln[1])
    print(f"  {aln[2]}")
    print(f"  {aln[3]}")

def parse_hhr_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    alignments = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("No ") and not line.startswith("No Hit"):
            header = lines[i + 1].strip() 
            meta = lines[i + 2].strip() 
            i += 3

            query = "" 
            template = "" 
            
            for j in range(i + 1, len(lines)):
        #        print(lines[j]) 
                if lines[j].startswith("No "):
                    break
                
                elif lines[j].startswith("Q Consensus") or lines[j].startswith("T Consensus"):
                    ...
                elif lines[j].strip() == "":
                    ...
                else:
                    parts = lines[j].strip().split()

                    if parts[0] == "Q":
                        query += parts[3]
                    elif (parts[0] == "T") and (parts[1] != "ss_dssp") and (parts[1] != "ss_pred"):
                        template += parts[3]
            alignments.append((header, meta, query, template))
    return alignments


#test_hhr = parse_hhr_file("hhblits_out/A0FGR8.hhr")

def parse_aln(aln):
    header, meta, query, template = aln

    meta_parts = meta.split()
    meta_dict = {meta_part.split("=")[0]: meta_part.split("=")[1] for meta_part in meta_parts}
    expected_keys = {'Probab', 'E-value', 'Score', 'Aligned_cols', 'Identities', 'Similarity', 'Sum_probs', 'Template_Neff'}
    assert set(meta_dict.keys()) == expected_keys

    return meta_dict

hhrs = [i for i in Path("hhblits_out").iterdir() if i.suffix == ".hhr"]
hhrs = [i for i in hhrs if i.stat().st_size != 0]
assert len(hhrs) > 1
if __name__ == "__main__":
    QueryUID = []
    PDB70ID = []
    Probab = []
    Evalue = []
    Score = []
    Aligned_cols = []
    Identities = []
    Similarity = []
    Sum_probs = []
    Template_Neff = []
    Q = []
    T = [] 
    for hhr_file in hhrs:

        alns = parse_hhr_file(str(hhr_file))
        assert len(alns) > 0, hhr_file
        for aln in alns: 
            QueryUID.append(hhr_file.name.removesuffix(".hhr"))
            assert len(aln) == 4
            header, meta, query, template = aln
            assert len(query) == len(template)

            meta_parts = meta.split()
            meta_dict = {meta_part.split("=")[0]: meta_part.split("=")[1] for meta_part in meta_parts}
            expected_keys = {'Probab', 'E-value', 'Score', 'Aligned_cols', 'Identities', 'Similarity', 'Sum_probs', 'Template_Neff'}
            assert set(meta_dict.keys()) == expected_keys

            header_parts = header.split()
            PDB70ID.append(header_parts[0].strip(">"))
            Probab.append(meta_dict['Probab'])
            Evalue.append(meta_dict['E-value'])
            Score.append(meta_dict['Score'])
            Aligned_cols.append(meta_dict['Aligned_cols'])
            Identities.append(meta_dict['Identities'])
            Similarity.append(meta_dict['Similarity'])
            Sum_probs.append(meta_dict['Sum_probs'])
            Template_Neff.append(meta_dict['Template_Neff'])
            Q.append(query)
            T.append(template)

    df = pd.DataFrame({"QueryUID" : QueryUID,
            "PDB70ID": PDB70ID,
            "Probab": Probab,
            "Evalue": Evalue,
            "Score": Score,
            "Aligned_cols": Aligned_cols,
            "Identities": Identities,
            "Similarity": Similarity,
            "Sum_probs": Sum_probs,
            "Template_Neff": Template_Neff,
            "Q": Q,
            "T": T})
    df.to_csv("hhblits_out/PreyPDB70PairAlign.csv", index=False)
