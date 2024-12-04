from pathlib import Path
import json

def pp_aln(aln):
    print(aln[0][0:80])
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
                print(lines[j]) 
                if lines[j].strip() == "":
                    break
                
                elif lines[j].startswith("Q Consensus") or lines[j].startswith("T Consensus"):
                    ...

                else:
                    parts = lines[j].strip().split()

                    if parts[0] == "Q":
                        query += parts[3]
                    elif (parts[0] == "T") and (parts[1] != "ss_dssp") and (parts[1] != "ss_pred"):
                        template += parts[3]
            alignments.append((header, meta, query, template))
    return alignments

hhrs = [i for i in Path("hhblits_out").iterdir() if i.suffix == ".hhr"]

if __name__ == "__main__":
    hhr_alns = {}
    for hhr_file in hhrs:
        hhr_alns[hhr_file.name] = parse_hhr_file(str(hhr_file))
    
    json.dump(hhr_alns, open("hhblits_out/hhr_alns.json", "w"))
