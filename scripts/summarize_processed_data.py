import pandas as pd
from pathlib import Path
import click

@click.command()
@click.option("--path")
@click.option("--no-baits", is_flag=True, default=True)
def main(path, no_baits):
    path = Path(path)
    sc = pd.read_csv(str(path / "spec_table.tsv"), sep="\t")
    c = pd.read_csv(str(path / "composite_table.tsv"), sep="\t")
    print(f"INFO::summarize_processed_data::{path}/spec_table.tsv: {sc.shape}") 
    print(f"INFO::summarize_processed_data::{path}/composite_table.tsv: {c.shape}") 
    if no_baits: 
        bait_set = set(c['Bait'])
        prey_set = set(c['Prey'])
        unknown_baits = bait_set - prey_set
        print(f"INFO::sumarize_processed_data::N Bait {len(bait_set)}")
        print(f"INFO::sumarize_processed_data::N Unmapped Bait {len(unknown_baits)}") 
        print(f"INFO::sumarize_processed_data::N Unmapped Bait {(unknown_baits)}") 

if __name__ == "__main__":
    main()
