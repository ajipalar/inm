import pandas as pd
import datetime
import numpy as np
import logging
import argparse
from pathlib import Path
import os
import sys

def read_input_file(fpath):
    if not Path(fpath).is_file():
        logging.error(f"input {fpath} is not a file")
        exit
    # Given a path
    df = pd.read_csv(fpath, sep="\t")
    # write to a pandas df
    return df

def check_input_df(df):
    try:
        _check_input_df(df)
    except:
        logging.error("Input error")
        exit

def _check_input_df(df):
    # Check columns
    cols = df.columns
    if len(cols) != 4:
        logging.error(f"Expected four tab seperated columns")
        exit
    c1, c2, c3, c4 = cols
    # Ensure no negative probailities or probaliies above 1
    bait_set = set(df[c2].values)
    if len(bait_set) < 1:
        logging.error("Expected at least one bait")
        exit
    # Check the bait
    for bait in bait_set:
        if not isinstance(bait, str):
            logging.error(f"bait : {str(bait)} is not a str")
            exit
    # Check the prey
    if len(prey_set) < 2:
        logging.error("Expected more than 2 prey") 
        exit

    for prey in prey_set:
        if not isinstance(bait, str):
            logging.error(f"bait : {str(bait)} is not a str")
            exit
    # Check the probabilities 
    min_prob = np.min(df[c4])
    max_prob = np.max(df[c4])
    if min_prob < 0:
        logging.error(f"min MS score < 0: {min_prob}")  
        exit
    if max_prob > 1:
        logging.error(f"max MS score > 1: {max_prob}")
        exit
    purification_set = set(df[c1].values)

def validate_args(args):
    ...


def main():
    parser = argparse.ArgumentParser(description="Prepare AP-MS inputs for INM")
    parser.add_argument("--i", type=str, help="input file path")
    parser.add_argument("--o", type=str, help="output directory")
    parser.add_argument("--t", type=float, nargs="+", help="list of thresholds")

    # Check the output director exists  
    args = parser.parse_args()
    assert os.path.isdir(args.o)

    # Set up a new logfile
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y_%m_%d_%H_%M")
    log_fname = Path(args.o) / (formatted_date + "__prepare_modeling_inputs.log") 
    logger = logging.getLogger(__name__)

    # Validate args
    validate_args(args)
    
    # Read an input file
    df = read_input_file(args.i)

    # Check the file for a correct format
    check_input_df(df)

    # Define composites  
    df = define_composites(df, args.t)


    # Write to an output directory that exists

if __name__ == "__main__":
    main()

