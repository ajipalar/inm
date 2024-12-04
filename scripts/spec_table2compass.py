"""
Converts a spec_table.tsv to compass format
Experiment.id\tReplicate\tBait\tPrey\tcounts

Experiment id is a BAIT
"""
import click
from pathlib import Path
import pandas as pd

@click.command()
@click.option("--i")
@click.option("--o")
@click.option("--debug", is_flag=True, default=False)
def main(i: Path, o: Path, debug: bool):
    _main(i, o, debug)

def _main(i: Path, o: Path, debug: bool):
    if isinstance(i, str):
        i = Path(i)

    if isinstance(o, str):
        o = Path(o)

    assert i.is_file()
    assert o.parent.is_dir()

    with open(i, "r"):
        df = pd.read_csv(i, sep="\t", index_col=0)
    outdf = parse_df(df, debug)
    outdf.to_csv(o / "spec_table_compass.tsv", sep="\t", index=False)



def parse_df(df, debug):
    experiment_ids = {}

    columns = df.columns
    
    # columns of ouput df
    experiment_id_lst = []
    replicate_lst = []
    Bait_lst = []
    Prey_lst = []
    counts_lst = []

    for i, row in df.iterrows(): 
        if debug and len(experiment_id_lst) > 0:
            break
        for column in columns:
            split_col = column.split("_")
            bait = split_col[0][0:4]
            assert bait.isupper()

            experiment_id = split_col[0] + split_col[1][0]
            prey = row.name
            replicate = int(split_col[1][1])
            count = int(row[column])

            experiment_id_lst.append(experiment_id)
            replicate_lst.append(replicate)
            Bait_lst.append(bait)
            Prey_lst.append(prey)
            counts_lst.append(count)

    outdf = pd.DataFrame({
        "Experiment.id" : experiment_id_lst,
        "Replicate" : replicate_lst,
        "Bait" : Bait_lst,
        "Prey" : Prey_lst,
        "counts" : counts_lst,})
    return outdf

    
if __name__ == "__main__":
    main()
