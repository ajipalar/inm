"""
0. Define inputs and parameters 
1. Preprocess inputs to an output dir
2. Choose the scoring function
3. Run sampling
4. Generate the sampling figures
5. Generate the sampling tables
6. Generate the benchmark figures
7. Generate the benchmark tables
"""
import json
import logging
import pdb
import shutil
import sys

import click
from pathlib import Path

module_dir = Path(__file__).parent
root_dir = module_dir.parent

sys.path.append((root_dir / "src").as_posix())
print(sys.path)

import preprocess_data
import _model_variations as mv
import generate_sampling_figures as gsf
import generate_benchmark_figures as gbf
import generate_analysis_figures as gaf

logger = logging.getLogger()
def preprocessor_protocol_dispatcher(preprocessor_protocol_name):
    if preprocessor_protocol_name == "cullin_standard":
        return preprocess_data.preprocess_spec_table
    else:
        raise NotImplementedError

def preprocess_model():
    ...

def conditionally_mkdir(d):
    if d.is_dir():
        ...
    else:
        d.mkdir()

def main(
    model_output_dirpath,    
    model_input_fpath,
    preprocessor_protocol_name,
    preprocessor_output_dirpath,
    model_id,
    rseed,
    model_name,
    model_data,
    num_warmup,
    num_samples,
    include_potential_energy,
    include_mean_accept_prob, 
    include_extra_fields, 
    progress_bar, 
    initial_position, 
    save_warmup, 
    load_warmup, 
    jax_profile,
    hyper_param_alpha,
    hyper_param_beta,
    hyper_param_thresholds,
    hyper_param_n_null_bins,
    hyper_param_disconectivity_distance,
    hyper_param_max_distance,
    filter_kw,
    init_strat = "",
    thinning = 1,
    step_size : float = 1.0,
    adapt_step_size : bool = True,
    adapt_mass_matrix  : bool = True,
    target_accept_prob : float = 0.8,
    collect_warmup : bool = False,
    mode = "cullin",
    synthetic_N = None,
    synthetic_Mtrue = None,
    synthetic_rseed = None,
    prey_colname = "PreyGene",
    ms_score_colname = "SaintScore",
    sheet_nums : int = 3,
    n_ctrl_replicates = 12,
):
    """
    Params:
      model_output_dirpath : str - relative path for outputs    
      model_input_fpath : str - relative path to raw .xlsx file from SI
      preprocessor_protocol_name : str - currently supports 'cullin_standard'
      preprocessor_output_dirpath : str - relative path to write prepocessed data
      model_id : int - a numeric identifer for the run
      rseed : int - the random seed
      model_name : str - keyword dispatches the model. 'model23_se', 'model23_se_sc', 'model23_se_sr', 'model23_se_sr_sc', 'model23_ll_lp'
      model_data : None - depricated,
      num_warmup : int - number of MCMC warmup steps,
      num_samples : int - number of MCMC samples (single chain),
      include_potential_energy : bool - save the potential_energy to extra_fields. Note that getting the (pseudo) posterior one can use numpyro.infer.utilities.log_density(model)
      include_mean_accept_prob : bool - save the mean_accept_prob to extra_fields, 
      include_extra_fields : bool - output the extra_fields, 
      progress_bar : bool - visualize progress_bar, 
      initial_position : typically None, 
      save_warmup - bool : pickle the warmup samples, 
      load_warmup - bool : load the warmup state from pickle, 
      jax_profile - bool : run JAX profiling,
      hyper_param_alpha : float - The density of edges is prior distributed Uniform(alpha, beta) 
      hyper_param_beta  : float - beta < alpha
      hyper_param_thresholds : List[float] - an ordered list of mass spec thresholds to define composites. Also used to filter the input nodes based on a minimal score.
      hyper_param_n_null_bins : The number of bins used to represent the null distribution. Modeling can be sensitive to this parameter as some binning produce infinities (TODO fix this). 
      hyper_param_disconectivity_distance : int - only matters for composite_connectivity. The distance at which two nodes are considered disconnected  
      hyper_param_max_distance : int - only matters for composite_connectivity. The number of Warshal algorithm iterations to perform when calculating the shortest paths presence distance matrix.
      filter_kw : str - a flag sent to preprocess_data to define different sets of nodes for modeling
      init_strat : str - currently supports 'uniform' and an empty string.
      thinning = 1 : save chains every thinning steps
      step_size : float = 1.0 : manually set the step size
      adapt_step_size : bool = True : step size adaptation during wamrup -- see numpyro.infer.mcmc.hmc
      adapt_mass_matrix  : bool = True : mass matrix adaptation during wamrup -- see numpyro.infer.mcmc.hmc
      target_accept_prob : float = 0.8 : target acceptance probability -- smaller results in slower more robust sampling in principle
      collect_warmup : bool = False : collect the warmup samples
      mode = "cullin" : a catch all mostly for cullin specific i/o
      synthetic_N : number of nodes in synthetic data 
      synthetic_Mtrue : number of true edges in synthetic data
      synthetic_rseed : the seed to control the synthetic network and data genration independently of the model rseed
    """
    model_output_dirpath = Path(model_output_dirpath)
    model_input_fpath= Path(model_input_fpath)
    preprocessor_output_dirpath = Path(preprocessor_output_dirpath)
    
    conditionally_mkdir(model_output_dirpath)

    #setup_logging(model_output_dirpath)
    logger = logging.basicConfig(filename = str(model_output_dirpath / "_run.log"), filemode="w", level=logging.INFO)

    preprocess_inputs = preprocessor_protocol_dispatcher(preprocessor_protocol_name) 
    logging.info("preprocess_inputs")
    preprocess_inputs(
            input_path = model_input_fpath,
            output_dir = preprocessor_output_dirpath,
            sheet_nums = sheet_nums,
            prey_colname = prey_colname,
            enforce_bait_remapping = True,
            filter_kw = filter_kw,
            ms_score_colname = ms_score_colname,
            n_ctrl_replicates = n_ctrl_replicates,
            )
    logging.info("copy preprocessed inputs to modeling dir")
    # Copy the files from the preprocessed data to the modeling output dir
    shutil.copy(preprocessor_output_dirpath / "spec_table.tsv", model_output_dirpath / "spec_table.tsv")
    shutil.copy(preprocessor_output_dirpath / "composite_table.tsv", model_output_dirpath / "composite_table.tsv")
    shutil.copy(root_dir / "notebooks/shuffled_apms_correlation_matrix.pkl",
                model_output_dirpath / "shuffled_apms_correlation_matrix.pkl")

    modeler_vars = {"alpha" : hyper_param_alpha,
                    "beta" : hyper_param_beta,
                    "thresholds" : hyper_param_thresholds,
                    "n_null_bins" : hyper_param_n_null_bins,
                    "disconectivity_distance" : hyper_param_disconectivity_distance,
                    "max_distance" : hyper_param_max_distance,
                    } 

    logging.info("writing modeler_vars.json")
    with open(str(model_output_dirpath / "modeler_vars.json"), "w") as f:
        json.dump(modeler_vars, f)
    
    save_dir = str(model_output_dirpath)
    logging.info(f"""Params
    rseed {rseed}
    model_name {model_name}
    num_warmup {num_warmup}
    num_samples {num_samples}
    include_potential_energy {include_potential_energy}
    include_extra_fields {include_extra_fields}
    progress_bar {progress_bar}
    save_dir {save_dir}
    initial_position {initial_position}
    save_warmup {save_warmup}
    load_warmup {load_warmup}
    jax_profile {jax_profile}
    init_strat {init_strat}
    thinning {thinning}
    collect_warmup {collect_warmup}
    mode {mode}
    """)
    if mode == "cullin":
        logging.info("enter _model_variation._main")
        mv._main(model_id = model_id,
                 rseed = rseed,
                 model_name = model_name,
                 model_data = model_data,
                 num_warmup = num_warmup,
                 num_samples = num_samples,
                 include_potential_energy = include_potential_energy,
                 include_mean_accept_prob = include_mean_accept_prob,
                 include_extra_fields = include_extra_fields,
                 progress_bar = progress_bar,
                 save_dir = save_dir,
                 initial_position = initial_position,
                 save_warmup = save_warmup,
                 load_warmup = load_warmup,
                 jax_profile = jax_profile,
                 init_strat = init_strat,
                 thinning = thinning,
                 collect_warmup = collect_warmup,
                 synthetic_N = synthetic_N,
                 synthetic_Mtrue = synthetic_Mtrue,
                 synthetic_rseed = synthetic_rseed,)
    
        cullin_figures(model_id = model_id,
                model_name = model_name,
                rseed = rseed,
                model_output_dirpath = model_output_dirpath)
    else:
        raise NotImplementedError

def figures(model_id, model_name, rseed, model_output_dirpath, mode = "cullin", merge = False,**kwargs):
    if mode == "cullin":
        cullin_figures(model_id, model_name, rseed, model_output_dirpath, mode=mode, merge = merge,**kwargs)
    else:
        raise NotImplementedError

def cullin_figures(model_id, model_name, rseed, model_output_dirpath, merge = False, **kwargs):
    if isinstance(model_output_dirpath, str):
        model_output_dirpath = Path(model_output_dirpath)
    fbasename = f"{model_id}_{model_name}_{rseed}"
    input_file = model_output_dirpath / f"{fbasename}.pkl"
    logging.info("enter generate_sampling_figures")
    gsf._main(o = str(model_output_dirpath),
              i = input_file, mode="cullin",
              merge = merge)
    logging.info("enter generate_benchmark_figures")
    gbf.cullin_standard(model_output_dirpath, fbasename)
    logging.info("enter generate_analysis_figures")
    gaf.cullin_standard()

