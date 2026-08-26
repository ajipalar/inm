# examples/cullin_example/rules/analyze.smk
# Analysis rules: generate exploratory figures from preprocessed data

FIGURES = [
    "01_spec_count_distribution.png",
    "02_spec_table_heatmap.png",
    "03_prey_interaction_count.png",
    "04_saint_score_distribution.png",
    "05_pairwise_corr_distribution.png",
    "06_shuffled_corr_distribution.png",
    "07_topn_corr_matrix.png",
    "08_top50_ground_truth_matrix.png",
    "09_topn_shuffled_corr_matrix.png",
]


rule analyze_all:
    """Generate all exploratory figures."""
    input:
        expand("results/{figure}", figure=FIGURES),
    message:
        "Data exploration complete: all figures saved to results/"


rule analyze_data:
    """Generate exploratory figures from preprocessed data."""
    input:
        spec_table="data/cullin_formatted/spec_table.tsv",
        composite_table="data/cullin_formatted/composite_table.tsv",
        prey_names="data/cullin_formatted/prey_names.tsv",
        config_file="config.yaml",
    output:
        expand("results/{figure}", figure=FIGURES),
    shell:
        "python scripts/analyze.py {input.config_file} {input.spec_table} {input.composite_table} {input.prey_names} results"
