import _run

_run.main(
        model_output_dirpath = "../results/2024_04_01_dummy",
        model_input_fpath = "../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
        preprocessor_protocol_name = "cullin_standard",
        preprocessor_output_dirpath = "../data/processed/cullin/",
        model_id = 0,
        rseed = 13,
        model_name = "model23_ll_lp",
        model_data = None,
        num_warmup = 1000,
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
        hyper_param_thresholds = [.5, .6, .7, .8, .9, 1.])
        
