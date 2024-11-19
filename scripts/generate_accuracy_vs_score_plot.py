import click
import sys
from pathlib import Path
import pickle as pkl
import json
import numpy as np
from numpyro.infer.util import log_density

import matplotlib.pyplot as plt

sys.path.append("../notebooks")

import data_io
import generate_sampling_figures as gsf
import _model_variations as mv

@click.command()
@click.option("--dpath", help="path to the output directory")
@click.option("--mode", help="set to 'cullin' or other preprocessing mode")
def main(dpath, mode):
    _main(dpath, mode)
    
def _main(dpath, mode, _fbasename = "merged_results"):
    dpath = Path(dpath)
    assert dpath.is_dir()
    # Read in the model data

    with open(str(dpath / "merged_results_model_data.pkl"), "rb") as f:
        model_data = pkl.load(f)

    # Read in the models w/ warmup

    x = gsf.postprocess_samples(dpath, fbasename = _fbasename, merge = True)

    # Read in the scores

    scores_from_sampled_models = x["extra_fields"]["potential_energy"]

    # Add supplementary models and scores to the plot

    # get the native network and score

    native_network_flat = get_native_network(x, mode=mode, model_data = model_data)

    native_network_matrix = mv.flat2matrix(native_network_flat, n = model_data["N"])

    native_network_score = log_density(mv.model23_n, model_args = (model_data,), model_kwargs = {}, params = {"z" : native_network_matrix}) # Not the true value of Z
    native_accuracy = get_accuracy(native_network_flat)

    # plot the figure  

    fname = "dummy"
    plot_figure(dpath, fname, scores_from_sampled_models = scores_from_sampled_models)

    # write the figure meta data

def get_accuracy(network_model):
    ...

def get_native_network(x, mode, model_data):

    pdb_ppi_direct = data_io.get_pdb_ppi_predict_direct_reference()
    native_network = gsf.align_reference_to_model(model_data, pdb_ppi_direct, mode=mode)
    return native_network

    

def write_figure_meta_data():
    ...

def plot_figure(dpath, fname, scores_from_sampled_models):

    # placeholder
    N = len(scores_from_sampled_models)
    accuracy = np.ones(N) * 0.5

    prefix = str(dpath / fname) 

    plt.plot(np.array(scores_from_sampled_models), accuracy)
    plt.xlabel("score")
    plt.ylabel("accuracy")
    plt.savefig(prefix + "_300.png", dpi=300)
    plt.savefig(prefix + "_1200.png", dpi=1200)

if __name__ == "__main__":
    main()
