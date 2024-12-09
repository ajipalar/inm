from typing import NamedTuple, Any, List
from pathlib import Path

module_dir = Path(__file__).parent
root_dir = module_dir.parent

class RunConfiguration(NamedTuple):
    model_output_dirpath : str 
    model_input_fpath : str 
    preprocessor_protocol_name : str 
    preprocessor_output_dirpath : str
    model_id : int 
    rseed : int 
    model_name : int 
    model_data : Any 
    num_warmup : int
    num_samples : int 
    include_potential_energy : bool 
    include_mean_accept_prob : bool 
    include_extra_fields : bool
    progress_bar : bool 
    initial_position : Any
    save_warmup : bool 
    load_warmup : bool 
    jax_profile : bool 
    hyper_param_alpha : float 
    hyper_param_beta : float 
    hyper_param_thresholds : List[float]
    hyper_param_n_null_bins : int 
    hyper_param_disconectivity_distance : int 
    hyper_param_max_distance : int 
    filter_kw : str 
    init_strat : str = ""
    thinning : int = 1
    adapt_step_size : bool = True
    step_size : float = 1.0
    adapt_mass_matrix : bool = True
    target_accept_prob : float = 0.8
    collect_warmup : bool = False
    synthetic_N : int = None
    synthetic_Mtrue : int = None
    synthetic_rseed : int = None
    prey_colname : str = "PreyGene"
    ms_score_colname : str = "SaintScore"
    sheet_nums : int = 3
    n_ctrl_replicates : int = 12


def from_template(template: RunConfiguration, **kwargs) -> RunConfiguration:
    """
    Initialize a RunConfiguration from a 'template' run configuration and keyword arguments.

    e.g., mini_dev_run2 = from_template(mini_dev_run, rseed = 20, num_samples = 500)
    """
    if isinstance(template, RunConfiguration):
        template = template._asdict()  
    assert isinstance(template, dict)
    kwargs = locals()['kwargs']
    for key in kwargs: 
        template[key] = kwargs[key]
    return RunConfiguration(**template)

def from_template_list(template_list, **kwargs) -> RunConfiguration:
    """
    Initialize a run configuration from a list of partial run configurations.
    Each partial run configuration is an dict or NamedTuple with the appropriate keywords.

    Configurations are composed in the order they appear in the list, thus later configurations
    overwrite earlier ones without checking. 

    Optionally apply keyword arguments that appear at the end of the list
    """
    def to_dict(x):
        if isinstance(x, tuple):
            return x._asdict()
        elif isinstance(x, dict):
            return x
        else:
            raise TypeError(f"Expected dict or tuple, got {type(x)}")

    if len(template_list) == 0:
        raise ValueError("Expected non empty list")
    temp = template_list.pop(0)
    temp = to_dict(temp)
    kwargs = locals()['kwargs']

    def recursion(temp, template_list, kwargs):
        if len(template_list) == 0:
            return from_template_list(temp, **kwargs)
        else:
            temp2 = template_list.pop(0)
            for key in temp2:
                temp[key] = temp2[key]
            return recursion(temp, template_list, kwargs)
    return recursion(temp, template_list, kwargs)

mini_template = dict(
    model_input_fpath = root_dir / "data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    preprocessor_protocol_name = "cullin_standard",
    preprocessor_output_dirpath = root_dir / "data/processed/cullin/",
    model_id = 0,
    rseed = 13,
    model_data = None,
    num_warmup = 10,
    num_samples = 10,
    include_potential_energy = True,
    include_mean_accept_prob = False,
    include_extra_fields = True,
    progress_bar = True,
    initial_position = None,
    save_warmup = True,
    load_warmup = False,
    jax_profile = False,
    hyper_param_alpha = 0.02,
    hyper_param_beta = 0.1,
    hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
    hyper_param_n_null_bins = 80,
    hyper_param_disconectivity_distance = 10,
    hyper_param_max_distance = 11,
    collect_warmup = True,
    synthetic_N = None,
    synthetic_Mtrue = None,
    synthetic_rseed = None,
)

mini_dev_run = RunConfiguration(
    model_output_dirpath = root_dir / "results/mini_dev_run",
    model_input_fpath = root_dir  / "data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    preprocessor_protocol_name = "cullin_standard",
    preprocessor_output_dirpath = root_dir / "data/processed/cullin/",
    model_id = 0,
    rseed = 13,
    model_name = "model23_se",
    model_data = None,
    num_warmup = 10,
    num_samples = 10,
    include_potential_energy = True,
    include_mean_accept_prob = False,
    include_extra_fields = True,
    progress_bar = True,
    initial_position = None,
    save_warmup = True,
    load_warmup = True,
    jax_profile = False,
    hyper_param_alpha = 0.02,
    hyper_param_beta = 0.1,
    hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
    hyper_param_n_null_bins = 80,
    hyper_param_disconectivity_distance = 10,
    hyper_param_max_distance = 11,
    filter_kw = "all",
)


mini_model23_p = from_template(
    mini_dev_run,
    model_output_dirpath = root_dir / "results/mini_model23_p",
    model_name = "model23_p",
    num_warmup = 500,
    num_samples = 2_000,
    progress_bar = True,
    load_warmup = True,
    hyper_param_alpha = 0.001, 
    hyper_param_beta = 0.003,  
    init_strat = "uniform",
    filter_kw = "mock",
    collect_warmup = True,
    target_accept_prob = 0.6,
    )

mini_model23_a = from_template(
        mini_model23_p,
        model_name = "model23_a",
        model_output_dirpath = root_dir / "results/mini_model23_a")

mini_model23_b = from_template(
        mini_model23_p,
        model_name = "model23_b",
        model_output_dirpath = root_dir / "results/mini_model23_b")

mini_model23_c = from_template(
        mini_model23_p,
        model_name = "model23_c",
        model_output_dirpath = root_dir / "results/mini_model23_c")

mini_model23_d = from_template(
        mini_model23_p,
        model_name = "model23_d",
        model_output_dirpath = root_dir / "results/mini_model23_d")
    
mini_model23_e = from_template(
        mini_model23_d,
        model_name = "model23_e",
        model_output_dirpath = root_dir / "results/mini_model23_e")

mini_model23_f = from_template(
        mini_model23_d,
        model_name = "model23_f",
        model_output_dirpath = root_dir / "results/mini_model23_f")

mini_model23_g = from_template(
        mini_model23_d,
        model_name = "model23_g",
        model_output_dirpath = root_dir / "results/mini_model23_g")

mini_model23_h = from_template(
        mini_model23_d,
        model_name = "model23_h",
        model_output_dirpath = root_dir / "results/mini_model23_h")

mini_model23_i = from_template(
        mini_model23_d,
        model_name = "model23_i",
        model_output_dirpath = root_dir / "results/mini_model23_i")

mini_model23_k = from_template(
        mini_model23_d,
        model_name = "model23_k",
        model_output_dirpath = root_dir / "results/mini_model23_k")

mini_model23_k_synthetic = from_template(
        mini_model23_k,
        model_name = "model23_k",
        model_output_dirpath = root_dir / "results/mini_model23_k_synthetic",
        synthetic_N = 177,
        synthetic_Mtrue = 49,
        synthetic_rseed = 0,)

mini_model23_l = from_template(
        mini_model23_d,
        model_name = "model23_l",
        model_output_dirpath = root_dir / "results/mini_model23_l")



mini_model23_l_mock = from_template(
    mini_model23_l,
    model_output_dirpath = root_dir / "results/mini_model23_l_mock",
    filter_kw = "mock",
)

mini_model23_l_mock_20k = from_template(
    mini_model23_l_mock,
    model_output_dirpath = root_dir / "results/mini_model23_l_mock_20k",
    num_samples = 20_000,)


mini_model23_l_vif = from_template(
    mini_model23_l,
    model_output_dirpath = root_dir / "results/mini_model23_l_vif",
    filter_kw = "vif",
)

mini_model23_l_wt = from_template(
    mini_model23_l,
    model_output_dirpath = root_dir / "results/mini_model23_l_wt",
    filter_kw = "wt",
)

mini_model23_l_mock_synthetic = from_template(
        mini_model23_l_mock,
        model_output_dirpath = root_dir / "results/mini_model23_l_mock_synthetic",
        synthetic_N = 177,
        synthetic_Mtrue = 49,
        synthetic_rseed = 0,)


mini_model23_l_large_network = from_template(
    mini_model23_l,
    model_output_dirpath = root_dir / "results/mini_model23_l_large_network",
    model_name = "model23_l",
    hyper_param_thresholds = [.4, .5, .6, .7, .8, .9, 1.],
)

mini_model23_m = from_template(
    mini_model23_d,
    model_name = "model23_m",
    model_output_dirpath = root_dir / "results/mini_model23_m")

mini_model23_n = from_template(
    mini_model23_m,
    model_name = "model23_n",
    model_output_dirpath = root_dir / "results/mini_model23_n",
    num_samples = 500,)

mini_model23_n_mock_10k = from_template(
    mini_model23_n,
    num_samples = 10_000,
    num_warmup = 1_000,
    model_output_dirpath = root_dir / "results/mini_model23_n_mock_10k",
    filter_kw = "mock",
)

mini_model23_n_all_500 = from_template(
    mini_model23_n,
    num_samples = 500,
    num_warmup = 50,
    model_output_dirpath = root_dir / "results/mini_model23_n_all_500",
    filter_kw = "all",
    target_accept_prob = 0.8,
)


mini_model23_n_all_2k = from_template(
    mini_model23_n,
    num_samples = 2_000,
    num_warmup = 500,
    model_output_dirpath = root_dir / "results/mini_model23_n_all_2k",
    filter_kw = "all",
    target_accept_prob = 0.95,
)

mini_model23_n_all_10k = from_template(
    mini_model23_n,
    num_samples = 10_000,
    num_warmup = 1_000,
    model_output_dirpath = root_dir / "results/mini_model23_n_all_10k",
    filter_kw = "all",
)

mini_model23_n_all_20k = from_template(
    mini_model23_n,
    num_samples = 20_000,
    num_warmup = 1_000,
    model_output_dirpath = root_dir / "results/mini_model23_n_all_20k",
    filter_kw = "all",
)

mini_model23_n_p_all_5k = from_template(
    mini_model23_n,
    model_name = "model23_n_p",
    model_output_dirpath = root_dir / "results/mini_model23_n_p_all_5k",
    num_samples = 5_000,
    num_warmup = 1_000,
    filter_kw = "all",
)

mini_model23_n_p_s_all_5k = from_template(
    mini_model23_n_p_all_5k,
    model_name = "model23_n_p_s",
    model_output_dirpath = root_dir / "results/mini_model23_n_p_s_all_5k",
)

mini_model23_n_p_r_all_5k = from_template(
    mini_model23_n_p_all_5k,
    model_name = "model23_n_p_r",
    model_output_dirpath = root_dir / "results/mini_model23_n_p_r_all_5k",
)

mini_model23_n_p_d_all_5k = from_template(
    mini_model23_n_p_all_5k,
    model_name = "model23_n_p_d",
    model_output_dirpath = root_dir / "results/mini_model23_n_p_d_all_5k",
)

mini_model23_n_p_ne_all_5k = from_template(
    mini_model23_n_p_all_5k,
    model_name = "model23_n_p_ne",
    model_output_dirpath = root_dir / "results/mini_model23_n_p_ne_all_5k",
)

mini_model23_n_p_s_r_all_5k = from_template(
    mini_model23_n_p_all_5k,
    model_name = "model23_n_p_s_r",
    model_output_dirpath = root_dir / "results/mini_model23_n_p_s_r_all_5k",
)

mini_model23_n_p_s_d_all_5k = from_template(
    mini_model23_n_p_all_5k,
    model_name = "model23_n_p_s_d",
    model_output_dirpath = root_dir / "results/mini_model23_n_p_s_d_all_5k",
)

mini_model23_n_p_s_ne_all_5k = from_template(
    mini_model23_n_p_all_5k,
    model_name = "model23_n_p_s_ne",
    model_output_dirpath = root_dir / "results/mini_model23_n_p_s_ne_all_5k",
)

mini_model23_n_p_r_d_all_5k = from_template(
    mini_model23_n_p_all_5k,
    model_name = "model23_n_p_r_d",
    model_output_dirpath = root_dir / "results/mini_model23_n_p_r_d_all_5k",
)



mini_model23_n_mock_20k = from_template(
    mini_model23_n_all_20k,
    model_output_dirpath = root_dir / "results/mini_model23_n_mock_20k",
    filter_kw = "mock",
)

mini_model23_n_vif_20k = from_template(
    mini_model23_n_all_20k,
    model_output_dirpath = root_dir / "results/mini_model23_n_vif_20k",
    filter_kw = "vif",
)

mini_model23_n_wt_20k = from_template(
    mini_model23_n_all_20k,
    model_output_dirpath = root_dir / "results/mini_model23_n_wt_20k",
    filter_kw = "wt",
)

mini_model23_n__all_20k_synthetic = from_template(
    mini_model23_n_all_20k,
    model_name = "model23_n_",
    synthetic_N = 236,
    synthetic_Mtrue = 49,
    synthetic_rseed = 0,
    model_output_dirpath = root_dir / "results/mini_model23_n__all_20k_synthetic",
    )

mini_model23_o_all = from_template(
    mini_model23_n_all_20k,
    model_name = "model23_o",
    model_output_dirpath = root_dir / "results/mini_model23_o_all",
    num_warmup = 500,
    num_samples = 500,
    filter_kw = "all",
)

mini_model_o_all_10k = from_template(
    mini_model23_o_all,
    model_name = "model23_o",
    model_output_dirpath = root_dir / "results/mini_model_o_all_10k",
    num_samples = 10_000,
    num_warmup = 1_000,
)

mini_model23_m_10k = from_template(
    mini_model23_m,
    model_name = "model23_m",
    model_output_dirpath = root_dir / "results/mini_model23_m_10k",
    num_samples = 10_000)


mini_dev_run_w_thinning = from_template(mini_dev_run,
    model_output_dirpath = root_dir / "results/mini_dev_run_w_thinning",
    thinning = 2,
    )

mini_se_sr_run = RunConfiguration(
    model_output_dirpath = root_dir / "results/mini_se_sr_run",
    model_input_fpath = root_dir / "data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    preprocessor_protocol_name = "cullin_standard",
    preprocessor_output_dirpath = root_dir / "data/processed/cullin/",
    model_id = 0,
    rseed = 13,
    model_name = "model23_se_sr",
    model_data = None,
    num_warmup = 10,
    num_samples = 10,
    include_potential_energy = True,
    include_mean_accept_prob = False,
    include_extra_fields = True,
    progress_bar = True,
    initial_position = None,
    save_warmup = True,
    load_warmup = True,
    jax_profile = False,
    hyper_param_alpha = 0.02,
    hyper_param_beta = 0.1,
    hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
    hyper_param_n_null_bins = 'auto',
    hyper_param_disconectivity_distance = 10,
    hyper_param_max_distance = 11,
    filter_kw = "all",
    collect_warmup = True,
)

mini_se_sr_low_prior_1 = from_template(mini_se_sr_run,
    model_output_dirpath = root_dir / "results/mini_se_sr_low_prior_1",
    hyper_param_alpha = 0.001,
    hyper_param_beta = 0.01,
    )

mini_se_sr_low_prior_1_uniform = from_template(mini_se_sr_run,
    model_output_dirpath = root_dir / "results/mini_se_sr_low_prior_1_uniform",
    hyper_param_alpha = 0.001,
    hyper_param_beta = 0.01,
    init_strat = "uniform",
    )

mini_se_sr_low_prior_1_uniform_num_warmup_1000 = from_template(mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/mini_se_sr_low_prior_1_uniform_num_warmup_1000",
    num_warmup = 1_000,
    )

se_sr_low_prior_1_all_20k = from_template(mini_se_sr_low_prior_1,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_all_20k",
    num_samples = 20_000,
    num_warmup = 1_000,
    )

se_sr_low_prior_1_wt_20k = from_template(mini_se_sr_low_prior_1,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_wt_20k",
    num_samples = 20_000,
    num_warmup = 1_000,
    filter_kw = "wt",
    )

se_sr_low_prior_1_vif_20k = from_template(mini_se_sr_low_prior_1,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_vif_20k",
    num_samples = 20_000,
    num_warmup = 1_000,
    filter_kw = "vif",
    )

se_sr_low_prior_1_uniform_mock_20k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_mock_20k",
    num_samples = 20_000,
    num_warmup = 1_000,
    filter_kw = "mock",
    )

se_low_prior_1_uniform_mock_10k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_low_prior_1_uniform_mock_10k",  
    model_name = "model23_se",
    num_samples = 10_000,
    num_warmup = 1_000,
    filter_kw = "mock",
    )

se_low_prior_1_uniform_vif_10k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_low_prior_1_uniform_vif_10k",  
    model_name = "model23_se",
    num_samples = 10_000,
    num_warmup = 1_000,
    filter_kw = "vif",
    )


se_low_prior_2_uniform_mock_10k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_low_prior_2_uniform_mock_10k",  
    model_name = "model23_se",
    num_samples = 10_000,
    num_warmup = 1_000,
    hyper_param_alpha = 0.0001,
    hyper_param_beta = 0.001,
    )

se_low_prior_3_uniform_mock_10k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_low_prior_3_uniform_mock_10k",  
    model_name = "model23_se",
    num_samples = 10_000,
    num_warmup = 1_000,
    hyper_param_alpha = 0.00001,
    hyper_param_beta = 0.0001,
    )

se_high_prior_1_uniform_mock_10k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_high_prior_1_uniform_mock_10k",  
    model_name = "model23_se",
    num_samples = 10_000,
    num_warmup = 1_000,
    hyper_param_alpha = 0.45,
    hyper_param_beta = 0.55,
    )





se_sr_low_prior_1_uniform_mock_10k_warmup_20k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_mock_10k_20k",
    num_samples = 10_000,
    num_warmup = 20_000,
    filter_kw = "mock",
    )

se_sr_low_prior_1_uniform_mock_100k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_mock_100k",
    num_samples = 100_000,
    num_warmup = 5_000,
    filter_kw = "mock",
    thinning = 5,
    )

se_sr_low_prior_1_uniform_mock_100k_no_thin = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_mock_100k_no_thin",
    num_samples = 100_000,
    num_warmup = 5_000,
    filter_kw = "mock",
    thinning = 1,
    )

se_sr_low_prior_1_uniform_mock_5k_diagnose = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_mock_5k_diagnose",
    num_warmup = 0,
    num_samples = 5000,
    filter_kw = "mock",
    progress_bar = False,
    adapt_step_size = False,
    adapt_mass_matrix = False,
    step_size = 1.,
    )

se_sr_low_prior_1_uniform_mock_2k_diagnose = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_mock_2k_diagnose",
    num_warmup = 0,
    num_samples = 2000,
    filter_kw = "mock",
    progress_bar = False,
    adapt_step_size = False,
    adapt_mass_matrix = False,
    step_size = 1.,
    )

se_sr_low_prior_1_uniform_mock_2k_diagnose_small_step = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_mock_2k_diagnose_small_step",
    num_warmup = 0,
    num_samples = 2000,
    filter_kw = "mock",
    progress_bar = False,
    adapt_step_size = False,
    adapt_mass_matrix = False,
    step_size = 5.2e-4,
    )

se_sr_low_prior_1_uniform_mock_2k_diagnose_small_step_27 = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_mock_2k_diagnose_small_step_27",
    num_warmup = 0,
    num_samples = 2000,
    filter_kw = "mock",
    progress_bar = False,
    adapt_step_size = False,
    adapt_mass_matrix = False,
    step_size = 2.7e-4,
    )

se_sr_low_prior_1_uniform_mock_2k_diagnose_t06 = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_mock_2k_diagnose_t06",
    num_warmup = 1_000,
    num_samples = 2_000,
    filter_kw = "mock",
    progress_bar = False,
    adapt_step_size = True,
    adapt_mass_matrix = True,
    target_accept_prob = 0.6,  
    )

se_sr_low_prior_1_uniform_mock_60k_t06 = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_mock_60k_t06",
    num_warmup = 1_000,
    num_samples = 60_000,
    filter_kw = "mock",
    progress_bar = False,
    adapt_step_size = True,
    adapt_mass_matrix = True,
    target_accept_prob = 0.6,  
    )

se_sr_low_prior_1_uniform_mock_500_diagnose = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_mock_500_diagnose",
    num_warmup = 0,
    num_samples = 500,
    filter_kw = "mock",
    progress_bar = True,
    adapt_step_size = False,
    adapt_mass_matrix = False,
    step_size = 1.,
    )

se_sr_low_prior_1_uniform_vif_100k_no_thin = from_template(
        se_sr_low_prior_1_uniform_mock_100k_no_thin,
        model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_vif_100k_no_thin",
        filter_kw = "vif")

se_sr_low_prior_1_uniform_wt_100k_no_thin = from_template(
        se_sr_low_prior_1_uniform_mock_100k_no_thin,
        model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_wt_100k_no_thin",
        filter_kw = "wt")


se_sr_low_prior_1_uniform_mock_50k_no_thin = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_mock_50k_no_thin",
    num_samples = 50_000,
    num_warmup = 5_000,
    filter_kw = "mock",
    thinning = 1,
    )



se_sr_low_prior_1_uniform_wt_100k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_wt_100k",
    num_samples = 100_000,
    num_warmup = 5_000,
    filter_kw = "wt",
    thinning = 5,
    )

se_sr_low_prior_1_uniform_vif_100k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_vif_100k",
    num_samples = 100_000,
    num_warmup = 5_000,
    filter_kw = "vif",
    thinning = 5,
    )



se_sr_low_prior_2_uniform_mock_5k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_2_uniform_mock_5k",
    num_samples = 20_000,
    num_warmup = 1_000,
    filter_kw = "mock",
    hyper_param_alpha = 0.0001,
    hyper_param_beta = 0.001,
    )

se_sr_low_prior_2_uniform_mock_20k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_2_uniform_mock_20k",
    num_samples = 20_000,
    num_warmup = 1_000,
    filter_kw = "mock",
    hyper_param_alpha = 0.0001,
    hyper_param_beta = 0.001,
    )

se_sr_low_prior_2_uniform_mock_100k = from_template(
    mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_2_uniform_mock_100k",
    num_samples = 100_000,
    num_warmup = 1_000,
    filter_kw = "mock",
    hyper_param_alpha = 0.0001,
    hyper_param_beta = 0.001,
    thinning = 5,
    )

se_sr_low_prior_2_uniform_mock_100k_no_thin = from_template(
    se_sr_low_prior_2_uniform_mock_100k,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_2_uniform_mock_100k_no_thin",
    thinning = 1)

se_sr_low_prior_1_uniform_wt_20k = from_template(mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_wt_20k",
    num_samples = 20_000,
    num_warmup = 1_000,
    filter_kw = "wt",
    )

se_sr_low_prior_1_uniform_vif_20k = from_template(mini_se_sr_low_prior_1_uniform,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_vif_20k",
    num_samples = 20_000,
    num_warmup = 1_000,
    filter_kw = "vif",
    )

se_sr_low_prior_1_all_100k = from_template(se_sr_low_prior_1_all_20k,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_all_100k",
    thinning = 5,
    num_samples = 100_000,
    )

se_sr_low_prior_1_wt_100k = from_template(se_sr_low_prior_1_all_20k,
    model_output_dirpath = root_dir / "results/se_sr_low_prior_1_wt_100k",
    thinning = 5,
    num_samples = 100_000,
    filter_kw = "wt",
    )

se_sr_low_prior_1_uniform_all_20k = from_template(
        mini_se_sr_low_prior_1_uniform,
        model_output_dirpath = root_dir / "results/se_sr_low_prior_1_uniform_all_20k",
        num_warmup = 1_000,
        num_samples = 20_000,
        )

mini_se_sr_run2 = from_template(mini_se_sr_run,
                              model_output_dirpath = "mini_se_sr_run2",
                              init_strat = "uniform")

mini_se_sc_run = RunConfiguration(
    model_output_dirpath = root_dir / "results/mini_se_sc_run",
    model_input_fpath = root_dir / "data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    preprocessor_protocol_name = "cullin_standard",
    preprocessor_output_dirpath = root_dir / "data/processed/cullin/",
    model_id = 0,
    rseed = 13,
    model_name = "model23_se_sc",
    model_data = None,
    num_warmup = 10,
    num_samples = 10,
    include_potential_energy = True,
    include_mean_accept_prob = False,
    include_extra_fields = True,
    progress_bar = True,
    initial_position = None,
    save_warmup = True,
    load_warmup = True,
    jax_profile = False,
    hyper_param_alpha = 0.02,
    hyper_param_beta = 0.1,
    hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
    hyper_param_n_null_bins = 80,
    hyper_param_disconectivity_distance = 10,
    hyper_param_max_distance = 11,
    filter_kw = "all",
)

mini_se_wt = from_template(mini_template,
                           model_output_dirpath = root_dir / "results/mini_se_wt",
                           model_name = "model23_se",
                           filter_kw="wt")

mini_se_vif = from_template(mini_template,
                           model_output_dirpath = root_dir / "results/mini_se_vif",
                           model_name = "model23_se",
                           filter_kw="vif")

mini_se_mock = from_template(mini_template,
                           model_output_dirpath = root_dir / "results/mini_se_mock",
                           model_name = "model23_se",
                           filter_kw="mock")

mini_se_wt_ctrl = from_template(mini_template,
                           model_output_dirpath = root_dir / "results/mini_se_wt_ctrl",
                           model_name = "model23_se",
                           filter_kw="wt_ctrl")

mini_se_vif_ctrl = from_template(mini_template,
                           model_output_dirpath = root_dir / "results/mini_se_vif_ctrl",
                           model_name = "model23_se",
                           filter_kw="vif_ctrl")

mini_se_sr_wt = from_template(mini_template,
                           model_output_dirpath = root_dir / "results/mini_se_sr_wt",
                           model_name = "model23_se_sr",
                           filter_kw="wt")

mini_se_sr_vif = from_template(mini_template,
                           model_output_dirpath = root_dir / "results/mini_se_sr_vif",
                           model_name = "model23_se_sr",
                           filter_kw="vif")

mini_se_sr_mock = from_template(mini_template,
                           model_output_dirpath = root_dir / "results/mini_se_sr_mock",
                           model_name = "model23_se_sr",
                           filter_kw="mock")

mini_se_sr_wt_ctrl = from_template(mini_template,
                           model_output_dirpath = root_dir / "results/mini_se_sr_wt_ctrl",
                           model_name = "model23_se_sr",
                           filter_kw="wt_ctrl")

mini_se_sr_vif_ctrl = from_template(mini_template,
                           model_output_dirpath = root_dir / "results/mini_se_sr_vif_ctrl",
                           model_name = "model23_se_sr",
                           filter_kw="vif_ctrl")

mini_se_sr_mock_ctrl = from_template(mini_template,
                           model_output_dirpath = root_dir / "results/mini_se_sr_mock_ctrl",
                           model_name = "model23_se_sr",
                           filter_kw="mock_ctrl")


se_all_10k =  RunConfiguration(
    model_output_dirpath = root_dir / "results/se_all_10k",
    model_input_fpath = root_dir / "data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    preprocessor_protocol_name = "cullin_standard",
    preprocessor_output_dirpath = root_dir / "data/processed/cullin/",
    model_id = 0,
    rseed = 13,
    model_name = "model23_se",
    model_data = None,
    num_warmup = 1_000,
    num_samples = 10_000,
    include_potential_energy = True,
    include_mean_accept_prob = False,
    include_extra_fields = True,
    progress_bar = True,
    initial_position = None,
    save_warmup = True,
    load_warmup = True,
    jax_profile = False,
    hyper_param_alpha = 0.02,
    hyper_param_beta = 0.1,
    hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
    hyper_param_n_null_bins = 80,
    hyper_param_disconectivity_distance = 10,
    hyper_param_max_distance = 11,
    filter_kw = "all",
)


se_sr_all_10k =  RunConfiguration(
    model_output_dirpath = root_dir / "results/se_sr_all_10k",
    model_input_fpath = root_dir / "data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    preprocessor_protocol_name = "cullin_standard",
    preprocessor_output_dirpath = root_dir / "data/processed/cullin/",
    model_id = 0,
    rseed = 13,
    model_name = "model23_se_sr",
    model_data = None,
    num_warmup = 1_000,
    num_samples = 10_000,
    include_potential_energy = True,
    include_mean_accept_prob = False,
    include_extra_fields = True,
    progress_bar = True,
    initial_position = None,
    save_warmup = True,
    load_warmup = True,
    jax_profile = False,
    hyper_param_alpha = 0.02,
    hyper_param_beta = 0.1,
    hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
    hyper_param_n_null_bins = 100,
    hyper_param_disconectivity_distance = 10,
    hyper_param_max_distance = 11,
    filter_kw = "all",
)

se_sr_all_20k = from_template(se_sr_all_10k,
                              num_samples = 20_000,
                              model_output_dirpath = root_dir / "results/se_sr_all_20k",)

se_sr_all_20k_r16 = from_template(se_sr_all_20k,
                                  model_output_dirpath = root_dir / "results/se_sr_all_20k_r16",
                                  rseed = 16)

se_sr_all_2k =  RunConfiguration(
    model_output_dirpath = root_dir / "results/se_sr_all_2k",
    model_input_fpath = root_dir / "data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    preprocessor_protocol_name = "cullin_standard",
    preprocessor_output_dirpath = root_dir / "data/processed/cullin/",
    model_id = 0,
    rseed = 13,
    model_name = "model23_se_sr",
    model_data = None,
    num_warmup = 500,
    num_samples = 2_000,
    include_potential_energy = True,
    include_mean_accept_prob = False,
    include_extra_fields = True,
    progress_bar = True,
    initial_position = None,
    save_warmup = True,
    load_warmup = True,
    jax_profile = False,
    hyper_param_alpha = 0.02,
    hyper_param_beta = 0.1,
    hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
    hyper_param_n_null_bins = 100,
    hyper_param_disconectivity_distance = 10,
    hyper_param_max_distance = 11,
    filter_kw = "all",
)

se_sr_all_2k_allcorr =  RunConfiguration(
    model_output_dirpath = root_dir / "results/se_sr_all_2k_allcorr",
    model_input_fpath = root_dir / "data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    preprocessor_protocol_name = "cullin_standard",
    preprocessor_output_dirpath = root_dir / "data/processed/cullin/",
    model_id = 0,
    rseed = 13,
    model_name = "model23_se_sr",
    model_data = None,
    num_warmup = 500,
    num_samples = 2_000,
    include_potential_energy = True,
    include_mean_accept_prob = False,
    include_extra_fields = True,
    progress_bar = True,
    initial_position = None,
    save_warmup = True,
    load_warmup = True,
    jax_profile = False,
    hyper_param_alpha = 0.01,
    hyper_param_beta = 0.05,
    hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
    hyper_param_n_null_bins = 100,
    hyper_param_disconectivity_distance = 10,
    hyper_param_max_distance = 11,
    filter_kw = "all",
)

se_sc_all_10k =  RunConfiguration(
    model_output_dirpath = root_dir / "results/se_sc_all_10k",
    model_input_fpath = root_dir / "data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    preprocessor_protocol_name = "cullin_standard",
    preprocessor_output_dirpath = root_dir / "data/processed/cullin/",
    model_id = 0,
    rseed = 13,
    model_name = "model23_se_sc",
    model_data = None,
    num_warmup = 1_000,
    num_samples = 10_000,
    include_potential_energy = True,
    include_mean_accept_prob = False,
    include_extra_fields = True,
    progress_bar = True,
    initial_position = None,
    save_warmup = True,
    load_warmup = True,
    jax_profile = False,
    hyper_param_alpha = 0.02,
    hyper_param_beta = 0.1,
    hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
    hyper_param_n_null_bins = 100,
    hyper_param_disconectivity_distance = 10,
    hyper_param_max_distance = 11,
    filter_kw = "all",
)

se_sr_sc_all_10k =  RunConfiguration(
    model_output_dirpath = root_dir / "results/se_sr_sc_all_10k",
    model_input_fpath = root_dir / "data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    preprocessor_protocol_name = "cullin_standard",
    preprocessor_output_dirpath = root_dir / "data/processed/cullin/",
    model_id = 0,
    rseed = 13,
    model_name = "model23_se_sr_sc",
    model_data = None,
    num_warmup = 1_000,
    num_samples = 10_000,
    include_potential_energy = True,
    include_mean_accept_prob = False,
    include_extra_fields = True,
    progress_bar = True,
    initial_position = None,
    save_warmup = True,
    load_warmup = True,
    jax_profile = False,
    hyper_param_alpha = 0.02,
    hyper_param_beta = 0.1,
    hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
    hyper_param_n_null_bins = 100,
    hyper_param_disconectivity_distance = 10,
    hyper_param_max_distance = 11,
    filter_kw = "all",
)


tenK_template = dict(
   model_input_fpath = root_dir / "data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
   preprocessor_protocol_name = "cullin_standard",
   preprocessor_output_dirpath = root_dir / "data/processed/cullin/",
   model_id = 0,
   rseed = 13,
   model_data = None,
   num_warmup = 1_000,
   num_samples = 10_000,
   include_potential_energy = True,
   include_mean_accept_prob = False,
   include_extra_fields = True,
   progress_bar = True,
   initial_position = None,
   save_warmup = True,
   load_warmup = True,
   jax_profile = False,
   hyper_param_alpha = 0.02,
   hyper_param_beta = 0.1,
   hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.],
   hyper_param_n_null_bins = 80,
   hyper_param_disconectivity_distance = 10,
   hyper_param_max_distance = 11,
)

se_10k_wt = from_template(tenK_template,
                          model_output_dirpath = root_dir / "results/se_10k_wt",
                          model_name = "model23_se",
                          filter_kw = "wt",)

se_sr_10k_wt = from_template(tenK_template,
                          model_output_dirpath = root_dir / "results/se_sr_10k_wt",
                          model_name = "model23_se_sr",
                          filter_kw = "wt",)

se_sr_20k_wt = from_template(tenK_template,
    model_output_dirpath = root_dir / "results/se_sr_20k_wt",
    model_name = "model23_se_sr",
    num_samples = 20_000, 
                          filter_kw = "wt",)

se_sr_sc_10k_wt = from_template(tenK_template,
                          model_output_dirpath = root_dir / "results/se_sr_sc_10k_wt",
                          model_name = "model23_se_sr_sc",
                          filter_kw = "wt",)

se_10k_vif = from_template(tenK_template,
                          model_output_dirpath = root_dir / "results/se_10k_vif",
                          model_name = "model23_se",
                          filter_kw = "vif",)

se_sr_10k_vif  = from_template(tenK_template,
                          model_output_dirpath = root_dir / "results/se_sr_10k_vif",
                          model_name = "model23_se_sr",
                          filter_kw = "vif",)

se_sr_sc_10k_vif = from_template(tenK_template,
                          model_output_dirpath = root_dir / "results/se_sr_sc_10k_vif",
                          model_name = "model23_se_sr_sc",
                          filter_kw = "vif",)

se_10k_mock = from_template(tenK_template,
                          model_output_dirpath = root_dir / "results/se_10k_mock",
                          model_name = "model23_se",
                          filter_kw = "mock",)

se_sr_10k_mock  = from_template(tenK_template,
                          model_output_dirpath = root_dir / "results/se_sr_10k_mock",
                          model_name = "model23_se_sr",
                          filter_kw = "mock",)

se_sr_20k_mock  = from_template(tenK_template,
                          num_samples = 20_000,
                          model_output_dirpath = root_dir / "results/se_sr_20k_mock",
                          model_name = "model23_se_sr",
                          filter_kw = "mock",)

se_sr_wt_ctrl_20k  = from_template(tenK_template,
                          num_samples = 20_000,
                          model_output_dirpath = root_dir / "results/se_sr_wt_ctrl_20k",
                          model_name = "model23_se_sr",
                          filter_kw = "wt_ctrl",)

se_sr_vif_ctrl_20k  = from_template(tenK_template,
                          num_samples = 20_000,
                          model_output_dirpath = root_dir / "results/se_sr_vif_ctrl_20k",
                          model_name = "model23_se_sr",
                          filter_kw = "vif_ctrl",)

se_sr_mock_ctrl_20k  = from_template(tenK_template,
                          num_samples = 20_000,
                          model_output_dirpath = root_dir / "results/se_sr_mock_ctrl_20k",
                          model_name = "model23_se_sr",
                          filter_kw = "mock_ctrl",)

se_sr_sc_10k_mock = from_template(tenK_template,
                          model_output_dirpath = root_dir / "results/se_sr_sc_10k_mock",
                          model_name = "model23_se_sr_sc",
                          filter_kw = "mock",)

p53_dev = from_template(
    mini_model23_n_all_20k,
    model_output_dirpath = "../results/p53_dev",
    model_input_fpath = "../data/p53/MCF7_p53_GFP control.xlsx",
    num_warmup = 0,
    num_samples = 10,
    prey_colname = "PreyGene.x",
    ms_score_colname = "SaintScore.x",
    sheet_nums = 1,
    n_ctrl_replicates = 4,
)
