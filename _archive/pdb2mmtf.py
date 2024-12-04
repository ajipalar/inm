"""
Given a pdbfile, convert it to an mmtf file
"""
from pathlib import Path
import click
import biotite.structure.io.mmtf as mmtf
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


@click.command()
@click.option("-i", required=True, help="pdb file path")
@click.option("-o", required=True, help="output dir")
@click.option("-from", "from_", default="pdb")
@click.option("-to", default="mmtf", help="mmtf, or biommtf")
def main(i, o, from_, to):
    if from_ == "pdb": 
        Reader = pdb.PDBFile
        read_suffix = ".pdb"
        read_mod = pdb
    elif from_ == "cif":
        Reader = pdbx.PDBxFile
        read_suffix = ".cif"
        read_mod = pdbx
    else:
        raise ValueError(f"{from_} is not pdb or cif")

    if (to == "mmtf") or (to == "biommtf"):
        write_mod = mmtf
        write_suffix = "mmtf"
        Writer = mmtf.MMTFFile
    else:
        raise ValueError(f"{to} is not mmtf or biommtf")


    pdb_file_path = i
    output_dir = o
    pdb_id = Path(pdb_file_path).name.removesuffix(read_suffix)
    if to == "mmtf":
        output_path = f"{output_dir}/{pdb_id}.{write_suffix}"
    elif to == "biommtf":
        output_path = f"{output_dir}/{pdb_id}.bio.{write_suffix}"

    paths_list = [str(i) for i in Path(o).iterdir()]
    if output_path not in paths_list:
        eprint(f"Converting {pdb_id} to {output_path}") 
        db_file = Reader.read(pdb_file_path)
        
        if to == "mmtf":
            stack = read_mod.get_structure(db_file)
        elif to == "biommtf":
            stack = read_mod.get_assembly(db_file, model=1)
    
        write_file = Writer()
        write_mod.set_structure(write_file, stack)
        write_file.write(output_path)

    else:
        eprint(f"Skipping {output_path}")

if __name__ == "__main__":
    main()
