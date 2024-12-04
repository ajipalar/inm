if __name__ == "__main__":
    import run_synthetic_benchmark as rb
    
    rb.synthetic_benchmark_fn(
        analysis_name = "_0_99",
        n_prey = 50,
        n_bait = 3,
        d_crit = 21,
        dir_name = "SyntheticBenchmark_0_99_50_from_average",
        rseed = 13,
        edge_probability = 0.99,
        num_warmup = 500,
        num_samples = 1000,
        m_chains = 1000,
        fig_dpi = 300,
        model_name = "model14",
        fit_up_to = 24,
        analyze_next_N = 12,
        generate_cartoon_figures = True,
        n_successive_trajectories_to_analyze = 10, 
        initial_position = "average_network.pkl",
            )
else:
    raise ImportError("Module is meant to run, don't import")
