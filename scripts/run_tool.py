import json
import pdb
import pickle as pkl
import logging
import sys

import click
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
import jaxlib
import pandas as pd
from pathlib import Path

import run_configurations as rc
import _run

sys.path.append("../notebooks")
from sampling_assessment import (
        get_results
)
import data_io


from data_io import (
    pklload,
    find_hmc_warmup_samples_file_path,
    find_results_file_path,
    find_model_data_file_path,
)


@click.command()
@click.argument("name") 
@click.option("--figures", default = False, is_flag = True, help="only run figures")
@click.option("--rseed", default = None)
@click.option("--merge", is_flag = True, default = False, help="merge multiple chains") 
@click.option("--jax2numpy-path", default = None, type=str, help="path to a pickle file for conversion")
def main(name, figures, rseed, merge, jax2numpy_path):
    run_configuration =rc.__dict__[name]._asdict() 
    if jax2numpy_path is not None:
        jax2numpy(jax2numpy_path)
        exit()
    if not merge:
        if rseed is not None:
            rseed = int(rseed)
            # update the run configuration dynamically
            run_configuration = update_rc_dict_rseed(run_configuration, rseed)
        model_output_dirpath = Path(run_configuration['model_output_dirpath'])
        if not figures: 
            _run.main(**run_configuration)
            jsonpath = model_output_dirpath  / "run_config.json"
            with open(str(jsonpath), "w") as f:
                json.dump(run_configuration, f)
        else:
            _run.figures(**run_configuration)
    else:
        if not figures:
             do_merge(name)
        else:
            _run.figures(**run_configuration, merge = merge)

def pkldump(x, path):
    with open(path, "wb") as f:
        pkl.dump(x, f)

class MergeObject:
    def __init__(self, results = None, model_data = None, hmc_warmup_samples = None):
        self.results = results
        self.model_data = model_data
        self.hmc_warmup_samples = hmc_warmup_samples
    def write(self, name):
        assert self.results is not None
        assert self.model_data is not None
        assert self.hmc_warmup_samples is not None

        out = Path(f"../results/{name}_merged")
        assert out.is_dir(), out

        pkldump(self.results, out / "merged_results.pkl")
        pkldump(self.model_data, out / "merged_results_model_data.pkl")
        pkldump(self.hmc_warmup_samples, out / "merged_results_warmup_samples.pkl")

def do_merge(name):
    """ Merge multiple input chains into a single output directory """
    out = Path(f"../results/{name}_merged")
    if not out.is_dir():
        out.mkdir()
    # set up logging
    
    print(logging.getLogger().hasHandlers())
    print(logging.getLogger())

    logging.basicConfig(
        filename = str(out / "_run.log"),
        level=logging.DEBUG,
        filemode="w",
        format = "%(levelname)s:%(module)s:%(funcName)s:%(lineno)s:MSG:%(message)s",
    ) 

    logging.info(f"merging {name} chains")
    base = Path(f"../results/{name}")
    logging.info(f"base_path: {base}")
    #print("base", base)

    assert "rseed" not in base.name, "merge should exclude '_rseed_x'"
    
    # make the output directory or overwrite
    if not out.is_dir():
        out.mkdir()
    logging.info(f"out_path: {out}")
    
    merge_object = MergeObject(results=None, model_data=None) 
    merge_object = update_merge_object(name, merge_object, base)
    merge_object.write(name)

def update_merge_object(name, merge_object, base):
    for path in base.parent.iterdir():
        if (path.is_dir()) and ("rseed" in path.name) and ("merge" not in path.name) and (name in path.name):
            found_warmup_samples = False
            found_model_pickle = False
            model_base_name = ""
            for fpath in path.iterdir():
                if "model_data" in fpath.name:
                    model_base_name = fpath.name.split("_model_data")[0]
            for fpath in path.iterdir():
                if "warmup_samples" in fpath.name:
                    found_warmup_samples = True
                if fpath.name == model_base_name + ".pkl":
                    found_model_pickle = True
            if found_warmup_samples and found_model_pickle:
                rseed = int(path.name.split("_rseed_")[1])
                merge_object = update_merge_object_from_chain_path(name, rseed, merge_object, path)
    return merge_object

def update_merge_object_from_chain_path(name, rseed, merge_object, chain_path):
    merge_object = update_merge_object_warmup_hmc(merge_object, chain_path)
    merge_object = update_merge_object_results(name, rseed, merge_object, chain_path)
    merge_object = update_merge_object_model_data(merge_object, chain_path)
    return merge_object

def dictionary_expand_dims(dictionary) -> dict:
    return tree_map(lambda x: jnp.expand_dims(x, axis=0), dictionary)

def dictionary_concat_chains(x: dict, y: dict) -> dict:
    return tree_map(lambda a, b: jnp.concatenate([a, b], axis=0), x, y)

def update_merge_object_warmup_hmc(merge_object, chain_path):
    warmup_samples_hmc_path = find_hmc_warmup_samples_file_path(chain_path) 
    logging.info(f"updating merge object with warmup hmc from {warmup_samples_hmc_path}")
    hmc_warmup_samples = dictionary_expand_dims(pklload(warmup_samples_hmc_path))
    if merge_object.hmc_warmup_samples is None:
        merge_object.hmc_warmup_samples = hmc_warmup_samples 
    else:
        merge_object.hmc_warmup_samples = dictionary_concat_chains(merge_object.hmc_warmup_samples, hmc_warmup_samples)
    return merge_object

def update_merge_object_results(name, rseed, merge_object, chain_path):
    results_path = find_results_file_path(name, rseed, chain_path)
    logging.info(f"updating merge object with results from {results_path}")
    results = dictionary_expand_dims(pklload(results_path))
    if merge_object.results is None:
        merge_object.results = results
    else:
        merge_object.results = dictionary_concat_chains(merge_object.results, results)
    return merge_object

def update_merge_object_model_data(merge_object, chain_path):
    model_data_path = find_model_data_file_path(chain_path) 
    logging.info(f"updating merge object with model_data from {model_data_path}")
    model_data = pklload(model_data_path)
    if merge_object.model_data is None:
        merge_object.model_data = model_data
    else:
        for key in model_data.keys():
            assert key in merge_object.model_data.keys(), "model_data should be the same"

            logging.debug(f"except TypeError: chain path{chain_path}")
            a = model_data[key]
            b = merge_object.model_data[key]
            logging.debug(f"key: {key}")
            if isinstance(a, pd.DataFrame):
                 logging.debug(f"a index {a.index}")
                 logging.debug(f"b index {b.index}")
            if key == "composite_table":
                assert np.allclose(a["MSscore"], b["MSscore"])
            elif (a is None) and (b is None):
                pass
            else:
                if isinstance(a, dict):
                    assert a == b
                else:
                    assert np.allclose(a, b)

    return merge_object

def update_rc_dict_rseed(rc_dict, rseed):
    rc_dict["rseed"] = rseed
    rc_dict["model_output_dirpath"] = rc_dict["model_output_dirpath"] + f"_rseed_{rseed}"
    return rc_dict

def jax2numpy(jax2numpy_path):
    x = data_io.pklload(jax2numpy_path)
    temp = {}
    for key, value in x.items():
        if isinstance(value, jaxlib.xla_extension.DeviceArray):
            temp[key] = np.array(value)
        elif isinstance(value, jax.Array):
            temp[key] = np.array(value)
        else:
            temp[key] = value 
    pkl.dump(temp, open(jax2numpy_path, "wb"))

if __name__ == "__main__":
    main()
