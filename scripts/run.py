import click
import sys
from pathlib import Path
import _run

@click.command()
def main():
    _run.main()


if __name__ == "__main__":
    main()
