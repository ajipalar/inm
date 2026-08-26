# examples/cullin_example/rules/predictomes.smk
# Plot the score-column histograms for the Predictomes pair-score dump and
# report unordered-pair overlap with the cullin "full" and "minimal" node sets.
#
# Predictomes paper: Schmidt et al. (2025). Mol Cell.
# https://doi.org/10.1016/j.molcel.2025.01.034 / https://predictomes.org
# The raw CSV is not checked into version control (~116MB); rule
# predictomes_fetch below downloads it on demand.


rule predictomes_fetch:
    """Download the Predictomes pair-score CSV (not checked into version control)."""
    input:
        config_file="config.yaml",
    output:
        config["predictomes"]["raw_path"],
    shell:
        "python scripts/fetch_predictome.py {input.config_file}"


rule predictomes_all:
    """Histograms + full + minimal overlap reports + full + minimal ROC/PRC curves
    + model-vs-spoc threshold agreement analysis."""
    input:
        "results/predictomes_histograms.png",
        "results/predictomes_overlap_full_summary.txt",
        "results/predictomes_overlap_full_covered_pairs.tsv",
        "results/predictomes_overlap_minimal_summary.txt",
        "results/predictomes_overlap_minimal_covered_pairs.tsv",
        "results/predictomes_roc_full.png",
        "results/predictomes_prc_full.png",
        "results/predictomes_roc_prc_full_summary.tsv",
        "results/predictomes_roc_minimal.png",
        "results/predictomes_prc_minimal.png",
        "results/predictomes_roc_prc_minimal_summary.tsv",
        "results/predictomes_spoc_vs_model/spoc_vs_model_roc.png",
        "results/predictomes_spoc_vs_model/spoc_vs_model_prc.png",
        "results/predictomes_spoc_vs_model/spoc_vs_model_threshold_trend.png",
        "results/predictomes_spoc_vs_model/spoc_vs_model_summary.tsv",
    message:
        "Predictomes histograms + overlap reports + ROC/PRC curves + model-vs-spoc "
        "agreement analysis ready under results/"


rule predictomes_histograms:
    """Plot histograms of every score column in the Predictomes dump."""
    input:
        predictomes=config["predictomes"]["raw_path"],
        config_file="config.yaml",
    output:
        "results/predictomes_histograms.png",
    shell:
        "python scripts/predictomes_histograms.py {input.config_file} {input.predictomes} {output}"


rule predictomes_overlap_full:
    """Overlap of Predictomes pairs with the cullin full prey set (~3063 nodes)."""
    input:
        predictomes=config["predictomes"]["raw_path"],
        prey_list="data/cullin_formatted/prey_names.tsv",
        config_file="config.yaml",
    output:
        summary="results/predictomes_overlap_full_summary.txt",
        covered="results/predictomes_overlap_full_covered_pairs.tsv",
    params:
        label="full",
        prey_col="UniProt",
        prey_has_header="true",
    shell:
        "python scripts/predictomes_overlap.py {input.config_file} {input.predictomes} "
        "{input.prey_list} {params.prey_col} {params.prey_has_header} {params.label} results"


rule predictomes_overlap_minimal:
    """Overlap of Predictomes pairs with the cullin minimal node set (236 SAINT>=0.5 nodes used in NUTS)."""
    input:
        predictomes=config["predictomes"]["raw_path"],
        prey_list="data/cullin_model/spec_table.tsv",
        config_file="config.yaml",
    output:
        summary="results/predictomes_overlap_minimal_summary.txt",
        covered="results/predictomes_overlap_minimal_covered_pairs.tsv",
    params:
        label="minimal",
        prey_col="0",
        prey_has_header="false",
    shell:
        "python scripts/predictomes_overlap.py {input.config_file} {input.predictomes} "
        "{input.prey_list} {params.prey_col} {params.prey_has_header} {params.label} results"


rule predictomes_roc_prc_full:
    """ROC + PRC curves: Predictomes spoc_score vs direct/cocomplex PDB references,
    restricted to the cullin full prey set (~3063 nodes)."""
    input:
        predictomes=config["predictomes"]["raw_path"],
        prey_list="data/cullin_formatted/prey_names.tsv",
        direct_reference=config["data"]["direct_reference"],
        cocomplex_reference=config["data"]["cocomplex_reference"],
        config_file="config.yaml",
    output:
        roc="results/predictomes_roc_full.png",
        prc="results/predictomes_prc_full.png",
        summary="results/predictomes_roc_prc_full_summary.tsv",
    params:
        label="full",
        prey_col="UniProt",
        prey_has_header="true",
    shell:
        "python scripts/predictomes_roc_prc.py {input.config_file} {input.predictomes} "
        "{input.prey_list} {params.prey_col} {params.prey_has_header} {params.label} results"


rule predictomes_spoc_vs_model:
    """Agreement analysis: the AP-MS model's average edge score (predictor) vs
    spoc_score thresholded at several cutoffs (reference), restricted to the
    minimal node set. Complements predictomes_roc_prc_* by comparing two
    independent evidence channels (AP-MS spectral counts vs. AF-M + omics)
    instead of a PDB-structural reference that AF-M may have been trained on."""
    input:
        predictomes=config["predictomes"]["raw_path"],
        average_edge_scores=f"results/{MODEL_NAME}/postprocess/average_edge_scores.tsv",
        prey_list="data/cullin_model/spec_table.tsv",
        config_file="config.yaml",
    output:
        roc="results/predictomes_spoc_vs_model/spoc_vs_model_roc.png",
        prc="results/predictomes_spoc_vs_model/spoc_vs_model_prc.png",
        trend="results/predictomes_spoc_vs_model/spoc_vs_model_threshold_trend.png",
        summary="results/predictomes_spoc_vs_model/spoc_vs_model_summary.tsv",
    params:
        prey_col="0",
        prey_has_header="false",
        thresholds="0.05,0.1,0.3,0.5,0.7,0.86,0.95",
    shell:
        "python scripts/predictomes_spoc_vs_model.py {input.config_file} {input.predictomes} "
        "{input.average_edge_scores} {input.prey_list} {params.prey_col} {params.prey_has_header} "
        "{params.thresholds} results/predictomes_spoc_vs_model"


rule predictomes_roc_prc_minimal:
    """ROC + PRC curves: Predictomes spoc_score vs direct/cocomplex PDB references,
    restricted to the cullin minimal node set (236 SAINT>=0.5 nodes used in NUTS)."""
    input:
        predictomes=config["predictomes"]["raw_path"],
        prey_list="data/cullin_model/spec_table.tsv",
        direct_reference=config["data"]["direct_reference"],
        cocomplex_reference=config["data"]["cocomplex_reference"],
        config_file="config.yaml",
    output:
        roc="results/predictomes_roc_minimal.png",
        prc="results/predictomes_prc_minimal.png",
        summary="results/predictomes_roc_prc_minimal_summary.tsv",
    params:
        label="minimal",
        prey_col="0",
        prey_has_header="false",
    shell:
        "python scripts/predictomes_roc_prc.py {input.config_file} {input.predictomes} "
        "{input.prey_list} {params.prey_col} {params.prey_has_header} {params.label} results"
