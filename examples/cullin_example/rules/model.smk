# examples/cullin_example/rules/model.smk
# Model rules: filter prey, compute correlations, run MCMC sampling


rule model_all:
    """Run the full model pipeline: filter → correlations → MCMC sampling."""
    input:
        f"results/{MODEL_NAME}/model/samples.npz",
    message:
        f"Model pipeline complete: posterior samples saved to results/{MODEL_NAME}/model/samples.npz"


rule filter_prey:
    """Filter prey to those with SAINT score >= cutoff in at least one purification."""
    input:
        spec_table=config["data"]["formatted_spec_table"],
        composite_table=config["data"]["formatted_composite_table"],
        config_file="config.yaml",
    output:
        spec_table="data/cullin_model/spec_table.tsv",
        composite_table="data/cullin_model/composite_table.tsv",
    shell:
        "python scripts/filter.py {input.config_file} data/cullin_model"


rule compute_correlations:
    """Compute pairwise Pearson correlations on filtered spectral count table."""
    input:
        spec_table="data/cullin_model/spec_table.tsv",
        config_file="config.yaml",
    output:
        corr="data/cullin_model/corr.tsv",
        shuff_corr="data/cullin_model/shuff_corr.tsv",
        shuff_corr_all="data/cullin_model/shuff_corr_all.tsv",
    shell:
        "python scripts/correlations.py {input.config_file} data/cullin_model"


rule run_model:
    """Run NUTS/MCMC sampling and save posterior samples."""
    input:
        spec_table="data/cullin_model/spec_table.tsv",
        composite_table="data/cullin_model/composite_table.tsv",
        corr="data/cullin_model/corr.tsv",
        shuff_corr="data/cullin_model/shuff_corr.tsv",
        shuff_corr_all="data/cullin_model/shuff_corr_all.tsv",
        config_file="config.yaml",
    params:
        output_dir=f"results/{MODEL_NAME}/model",
    output:
        samples=f"results/{MODEL_NAME}/model/samples.npz",
        extra_fields=f"results/{MODEL_NAME}/model/extra_fields.npz",
        warmup_extra_fields=f"results/{MODEL_NAME}/model/warmup_extra_fields.npz",
    shell:
        "python scripts/model.py {input.config_file} {params.output_dir}"
