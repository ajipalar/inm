# examples/cullin_example/rules/burke.smk
# Fetch Burke et al. 2023 AlphaFold-Multimer human PPI predictions and report
# coverage against the proteins in our network. Pairs are treated as unordered.
#
# Burke et al. 2023, Nat Struct Mol Biol 30, 216-225. DOI: 10.1038/s41594-022-00910-8
# If the URL in config.yaml is wrong or unreachable, download the supplementary
# table manually and place it at config["burke"]["raw_path"]; fetch_burke will
# then become a no-op (Snakemake skips rules whose output already exists).


rule burke_all:
    """Fetch Burke et al. and report coverage over our protein set."""
    input:
        "results/burke_coverage_summary.txt",
        "results/burke_coverage_report.tsv",
        "results/burke_covered_pairs.tsv",
    message:
        "Burke coverage report ready: results/burke_coverage_summary.txt"


rule fetch_burke:
    """Download the Burke et al. 2023 supplementary PPI prediction table."""
    output:
        config["burke"]["raw_path"],
    params:
        url=config["burke"]["url"],
    shell:
        "curl -L --fail -o {output} '{params.url}'"


rule burke_coverage:
    """Compare Burke's predicted edges against the unordered node pairs in our network."""
    input:
        burke=config["burke"]["raw_path"],
        prey_names="data/cullin_formatted/prey_names.tsv",
        config_file="config.yaml",
    output:
        summary="results/burke_coverage_summary.txt",
        report="results/burke_coverage_report.tsv",
        covered="results/burke_covered_pairs.tsv",
    shell:
        "python scripts/burke_coverage.py {input.config_file} {input.burke} {input.prey_names} results"
