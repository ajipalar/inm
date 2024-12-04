"""
Given a ciffile, convert it to an pdb file
"""
from pathlib import Path
import click
import biotite.structure.io.mmtf as mmtf
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx

@click.command()
@click.option("-i", required=True, help="pdb file path")
@click.option("-o", required=True, help="output dir")
def main(i, o):
    cif_file_path = i
    output_dir = o

    db_file = pdbx.PDBxFile.read(cif_file_path)
    struct = db_file.get_structure()

    pdb_id = Path(cif_file_path).name.removesuffix(".")
    output_path = f"{output_dir}/{pdb_id}.mmtf"
    
    pdb_file = pdb.PDBFile()

    pdb.set_structure(pdb_file, struct)
    pdb_file.write(output_path)

if __name__ == "__main__":
    main()
