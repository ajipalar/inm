# examples/cullin_example/rules/postprocess.smk
# Postprocess rules: convergence test, edge probabilities, ROC and summary figures

POSTPROCESS_OUTPUTS = [
    f"results/{MODEL_NAME}/postprocess/10_roc_curve.png",
    f"results/{MODEL_NAME}/postprocess/11_edge_prob_hist.png",
    f"results/{MODEL_NAME}/postprocess/average_edge_scores.tsv",
    f"results/{MODEL_NAME}/postprocess/13_rhat_histogram.png",
    f"results/{MODEL_NAME}/postprocess/14_rhat_rank_plot.png",
    f"results/{MODEL_NAME}/postprocess/15_energy_trace.png",
    f"results/{MODEL_NAME}/postprocess/16_energy_trace_with_warmup.png",
    f"results/{MODEL_NAME}/postprocess/17_ground_truth_matrices.png",
    f"results/{MODEL_NAME}/postprocess/18_example_sampled_matrices.png",
    f"results/{MODEL_NAME}/postprocess/19_average_edge_probability_matrix.png",
    f"results/{MODEL_NAME}/postprocess/20_edge_variance_matrix.png",
    f"results/{MODEL_NAME}/postprocess/21_pooled_binary_degree_posterior.png",
    f"results/{MODEL_NAME}/postprocess/22_pooled_weighted_degree_posterior.png",
    f"results/{MODEL_NAME}/postprocess/23_node_low_binary_degree_posterior.png",
    f"results/{MODEL_NAME}/postprocess/24_node_mid_binary_degree_posterior.png",
    f"results/{MODEL_NAME}/postprocess/25_node_high_binary_degree_posterior.png",
    f"results/{MODEL_NAME}/postprocess/26_node_low_weighted_degree_posterior.png",
    f"results/{MODEL_NAME}/postprocess/27_node_mid_weighted_degree_posterior.png",
    f"results/{MODEL_NAME}/postprocess/28_node_high_weighted_degree_posterior.png",
    f"results/{MODEL_NAME}/postprocess/30_ess_histogram.png",
    f"results/{MODEL_NAME}/postprocess/31_ess_rank_plot.png",
    f"results/{MODEL_NAME}/postprocess/32_pr_curve.png",
    f"results/{MODEL_NAME}/postprocess/33_roc_auc_vs_sample_count.png",
    f"results/{MODEL_NAME}/postprocess/34_split_half_roc.png",
    f"results/{MODEL_NAME}/postprocess/35_best_energy_vs_sample_count.png",
    f"results/{MODEL_NAME}/postprocess/36_sampling_animation.gif",
]

if config.get("analysis", {}).get("degree_distribution", {}).get("enabled", False):
    POSTPROCESS_OUTPUTS.append(f"results/{MODEL_NAME}/postprocess/12_degree_distribution.png")


rule postprocess_all:
    """Run all MCMC postprocessing: convergence test, edge probabilities, ROC."""
    input:
        POSTPROCESS_OUTPUTS,
    message:
        f"Postprocess complete: outputs saved to results/{MODEL_NAME}/postprocess/"


rule postprocess:
    """Postprocess MCMC samples: KS test, edge probabilities, figures."""
    input:
        samples=f"results/{MODEL_NAME}/model/samples.npz",
        extra_fields=f"results/{MODEL_NAME}/model/extra_fields.npz",
        warmup_extra_fields=f"results/{MODEL_NAME}/model/warmup_extra_fields.npz",
        corr=config["data"]["corr"],
        direct_reference=config["data"]["direct_reference"],
        cocomplex_reference=config["data"]["cocomplex_reference"],
        config_file="config.yaml",
    params:
        samples_dir=f"results/{MODEL_NAME}/model",
        output_dir=f"results/{MODEL_NAME}/postprocess",
    output:
        POSTPROCESS_OUTPUTS,
    shell:
        "python scripts/postprocess.py {input.config_file} {params.samples_dir} {params.output_dir}"
