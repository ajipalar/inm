import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Updated bait2PreyGene dictionary with the names provided
bait2PreyGene = {
    "ARID1A": "ARID1A",
    "HRAS": "HRAS",
    "SCUBE2": "SCUBE2",
    "BRIP1": "BRIP1",
    "RPA2": "RPA2",
    "EZH2": "EZH2",
    "CCND3": "CCND3",
    "ERBB2": "ERBB2",
    "AKT2": "AKT2",
    "FANCC": "FANCC",
    "MSH2": "MSH2",
    "XPC": "XPC",
    "AKT1": "AKT1",
    "RAD51C": "RAD51C",
    "MTDH": "MTDH",
    "CDH1": "CDH1",
    "SMARCB1": "SMARCB1",
    "CASP8": "CASP8",
    "MLH1": "MLH1",
    "AKT3": "AKT3",
    "FOXA1": "FOXA1",
    "ESR1": "ESR1",
    "PIK3CA": "PIK3CA",
    "XRN2": "XRN2",
    "CDKN1B": "CDKN1B",
    "BRCA1": "BRCA1",
    "SMARCD1": "SMARCD1",
    "EGFR": "EGFR",
    "TBX3": "TBX3",
    "PTEN": "PTEN",
    "RB1": "RB1",
    "RAD51D": "RAD51D",
    "STK11": "STK11",
    "CTCF": "CTCF",
    "TSPYL5": "TSPYL5",
    "TP53": "TP53",
    "GATA3": "GATA3",
    "PALB2": "PALB2",
    "CBFB": "CBFB",
    "CHEK2": "CHEK2"
}

# Initialize the id_map dictionary
id_map = {}

# Load the Excel file
xlsx_path = "./mda231.xlsx"
xls = pd.ExcelFile(xlsx_path)

# Iterate through all available sheets in the Excel file
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    
    # Iterate through each row of the dataframe
    for i, r in df.iterrows():
        bait = r['Bait']
        
        # Use the updated bait2PreyGene dictionary to get the bait_gene
        bait_gene = bait2PreyGene.get(bait)
        if not bait_gene:
            logging.warning(f"Bait '{bait}' not found in bait2PreyGene dictionary.")
            continue
        
        prey_gene = r['PreyGene']
        prey_uid = r['PreyID']
        
        # Add to id_map if the prey_gene is not already present
        if prey_gene not in id_map:
            id_map[prey_gene] = prey_uid
        else:
            # Log a warning if there's a conflicting ID for the same prey_gene
            if id_map[prey_gene] != prey_uid:
                logging.warning(f"Conflicting IDs for {prey_gene}: {id_map[prey_gene]} vs {prey_uid}. Keeping the original.")

# Assertions to check if all bait genes from bait2PreyGene are in id_map
for bait in bait2PreyGene.values():
    if bait not in id_map:
        logging.warning(f"{bait} not found in id_map.")

# Prepare the lists for creating a DataFrame
prey_gene_lst = []
prey_lst = []

for prey_gene, prey in id_map.items():
    prey_gene_lst.append(prey_gene)
    prey_lst.append(prey)

# Check if we have any data before writing
if len(prey_gene_lst) > 0 and len(prey_lst) > 0:
    # Create a DataFrame and save it to a TSV file
    df = pd.DataFrame(data={"PreyGene": prey_gene_lst, "Prey": prey_lst})
    output_path = "./../processed/cullin/id_map.tsv"
    df.to_csv(output_path, sep="\t", index=False, header=False)
    logging.info(f"TSV file successfully saved to {output_path}")
else:
    logging.error("No data available to write to TSV file. Please check the input data and processing logic.")
