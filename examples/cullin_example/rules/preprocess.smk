# examples/cullin_example/rules/preprocess.smk
# Preprocessing rules: fetch cullin xlsx, load, format data

rule preprocess_all:
    """Preprocess cullin AP-MS data."""
    input:
        spec_table="data/cullin_formatted/spec_table.tsv",
        composite_table="data/cullin_formatted/composite_table.tsv",
        prey_names="data/cullin_formatted/prey_names.tsv",
    message:
        "Preprocessing complete: spectral and composite tables ready"


rule fetch_cullin:
    """Download the cullin AP-MS xlsx (Wan et al. 2019).

    The publisher blocks automated downloads (Cloudflare); if this rule
    fails, follow the manual-download instructions in its error output.
    """
    output:
        config["data"]["xlsx_path"],
    params:
        config_file="config.yaml",
    shell:
        "python scripts/fetch_cullin.py {params.config_file} $(dirname {output})"


rule preprocess_cullin:
    """Load cullin xlsx and format using inm.preprocess module."""
    input:
        xlsx=config["data"]["xlsx_path"],
        config_file="config.yaml",
    output:
        spec_table="data/cullin_formatted/spec_table.tsv",
        composite_table="data/cullin_formatted/composite_table.tsv",
        prey_names="data/cullin_formatted/prey_names.tsv",
    shell:
        "python scripts/preprocess.py {input.config_file} data/cullin_formatted"
