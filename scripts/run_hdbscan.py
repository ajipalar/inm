import click
from pathlib import Path

@click.commnad()
@click.option("--i", help="input directory of modeling results")
def main(i):
    _main()

def _main(i):
    i = Path(i)

if __name__ == "__main__":
    main()
