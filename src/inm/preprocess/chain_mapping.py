"""Map prey protein sequences to PDB chain sequences via pairwise alignment."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from inm.preprocess.fetch_structures import remap_pdb_id

logger = logging.getLogger(__name__)


def mutate_nonstandard_residues(seq: str) -> str:
    """Replace non-standard one-letter amino acid codes with standard equivalents."""
    seq = seq.replace("U", "C")
    seq = seq.replace("J", "X")
    seq = seq.replace("Z", "X")
    seq = seq.replace("B", "X")
    seq = seq.replace("O", "X")
    return seq


def get_sequence(
    atom_array,
    chain_id: str,
    non_canonical: Dict[str, str],
    min_length: int = 88,
) -> str:
    """Extract a one-letter sequence from an atom array for a given chain.

    Returns an empty string if the chain is shorter than *min_length*.
    """
    import biotite.sequence as seq_mod

    mask = atom_array.chain_id == chain_id
    resids = atom_array.res_id[mask]
    res_names = atom_array.res_name[mask]
    n_atoms = len(resids)
    n_residues = max(resids) - min(resids)

    sequence = ["X"] * n_residues
    if len(sequence) < min_length:
        return ""

    residue_map: Dict[int, str] = {}
    lost: List[str] = []
    for i in range(n_atoms):
        resname = res_names[i]
        if len(resname) == 3:
            if resname in non_canonical:
                resname = non_canonical[resname]
            else:
                try:
                    resname = seq_mod.ProteinSequence.convert_letter_3to1(resname)
                except Exception:
                    lost.append(resname)
                    resname = "X"
        residue_map[resids[i]] = resname

    if lost:
        logger.debug("Unmapped 3-letter codes in chain %s: %s", chain_id, set(lost))

    min_resid = min(resids)
    for i in range(n_residues):
        rid = min_resid + i
        if rid in residue_map:
            sequence[i] = residue_map[rid]

    return "".join(sequence)


def get_sequence_dict(
    atom_array,
    non_canonical: Dict[str, str],
    min_length: int = 88,
) -> Dict[str, str]:
    """Return a mapping of chain_id -> sequence for all chains in the array."""
    chain_ids = set(atom_array.chain_id)
    return {
        cid: get_sequence(atom_array, cid, non_canonical, min_length)
        for cid in chain_ids
    }


def _read_uid_sequences(uids: set, input_seq_dir: Path) -> Dict[str, str]:
    """Read FASTA sequences for each unique prey UID."""
    import biotite.sequence.io.fasta as fasta_io

    uid_sequences: Dict[str, str] = {}
    for uid in sorted(uids):
        fasta_path = input_seq_dir / f"{uid}.fasta"
        if not fasta_path.exists():
            logger.warning("Missing FASTA for %s", uid)
            continue
        fasta_file = fasta_io.FastaFile.read(str(fasta_path))
        if len(fasta_file) != 1:
            logger.warning(
                "Expected 1 entry in %s, got %d", fasta_path, len(fasta_file)
            )
            continue
        _header, sequence = list(fasta_file.items())[0]
        uid_sequences[uid] = sequence
    return uid_sequences


def _load_structure(mmtf_path: Path):
    """Load an MMTF file and return amino-acid-filtered atom array."""
    import biotite.structure as struc
    import biotite.structure.io.mmtf as mmtf

    mmtf_file = mmtf.MMTFFile.read(str(mmtf_path))
    atom_array = mmtf.get_structure(mmtf_file, model=1)
    mask = struc.filter_amino_acids(atom_array)
    return atom_array[mask]


def _align_sequences(
    uid: str,
    uid_seq: str,
    pdb_id: str,
    pdb_chain_seqs: Dict[str, str],
    min_length: int,
) -> List[Dict[str, Any]]:
    """Align a prey sequence against all chains in a PDB structure."""
    import biotite.sequence as seq_mod
    import biotite.sequence.align as align_mod

    blosum62 = align_mod.SubstitutionMatrix.std_protein_matrix()
    rows: List[Dict[str, Any]] = []

    if len(uid_seq) < min_length:
        return rows

    for chain_id, pdb_seq in pdb_chain_seqs.items():
        if len(pdb_seq) < min_length:
            continue

        logger.debug("Aligning %s to %s chain %s", uid, pdb_id, chain_id)
        useq = seq_mod.ProteinSequence(uid_seq)
        pdbseq = seq_mod.ProteinSequence(pdb_seq)

        alignments = align_mod.align_optimal(useq, pdbseq, blosum62, local=False)
        aln = alignments[0]

        gapped_q, gapped_t = aln.get_gapped_sequences()
        percent_seq_id = align_mod.get_sequence_identity(aln, mode="not_terminal")

        rows.append(
            {
                "QueryID": uid,
                "ChainID": chain_id,
                "PDBID": pdb_id,
                "bt_aln_score": aln.score,
                "bt_aln_evalue": None,
                "bt_aln_percent_seq_id": percent_seq_id,
                "bt_aln_Q": str(gapped_q),
                "bt_aln_T": str(gapped_t),
                "Q": uid_seq,
                "T": pdb_seq,
            }
        )

    return rows


def run(cfg: Dict[str, Any], output_dir: Any) -> None:
    """Build chain_mapping_all.csv by aligning prey sequences to PDB chains."""
    output_dir = Path(output_dir)
    remapping = cfg.get("obsolete_pdb_remapping", {})
    non_canonical = cfg.get("non_canonical_residues", {})
    filter_cfg = cfg.get("filter", {})
    min_length = filter_cfg.get("min_seq_length", 88)

    sig_path = output_dir / "hhblits_out" / "SignificantPreyPDB70PairAlign.csv"
    if not sig_path.exists():
        logger.warning("No significant hits file at %s", sig_path)
        return

    sig_alns = pd.read_csv(sig_path)
    logger.info("Loaded %d significant alignments", len(sig_alns))

    # Extract and remap PDB IDs and chain IDs from PDB70ID column
    pdb_ids_col = []
    chain_ids_col = []
    for pdb70id in sig_alns["PDB70ID"].values:
        parts = str(pdb70id).split("_")
        pdb_id = parts[0].lower()
        chain_id = parts[1] if len(parts) > 1 else ""
        pdb_id = remap_pdb_id(pdb_id, remapping)
        pdb_ids_col.append(pdb_id)
        chain_ids_col.append(chain_id)

    sig_alns = sig_alns.copy()
    sig_alns["pdb_id"] = pdb_ids_col
    sig_alns["chain_id"] = chain_ids_col

    # Read prey FASTA sequences
    input_seq_dir = output_dir / "input_sequences"
    uids = set(sig_alns["QueryUID"].values)
    uid_sequences = _read_uid_sequences(uids, input_seq_dir)
    logger.info("Loaded %d prey sequences", len(uid_sequences))

    # Mutate non-standard residues in prey sequences
    uid_sequences = {
        uid: mutate_nonstandard_residues(seq) for uid, seq in uid_sequences.items()
    }

    # Process each PDB
    struct_dir = output_dir / "significant_cifs"
    pdb_ids = sorted(set(sig_alns["pdb_id"].values))
    logger.info("Processing %d PDB structures", len(pdb_ids))

    all_rows: List[Dict[str, Any]] = []

    for pdb_id in pdb_ids:
        mmtf_path = struct_dir / f"{pdb_id}.bio.mmtf"
        if not mmtf_path.exists():
            logger.warning("Missing MMTF for %s", pdb_id)
            continue

        logger.info("Processing %s", pdb_id)
        atom_array = _load_structure(mmtf_path)
        seq_dict = get_sequence_dict(atom_array, non_canonical, min_length)

        # Mutate non-standard residues in PDB sequences
        seq_dict = {
            cid: mutate_nonstandard_residues(seq)
            for cid, seq in seq_dict.items()
            if len(seq) >= min_length
        }

        # Find prey UIDs that map to this PDB
        sub_uids = set(sig_alns.loc[sig_alns["pdb_id"] == pdb_id, "QueryUID"].values)

        for uid in sorted(sub_uids):
            if uid not in uid_sequences:
                continue
            rows = _align_sequences(
                uid, uid_sequences[uid], pdb_id, seq_dict, min_length
            )
            all_rows.extend(rows)

    if not all_rows:
        logger.warning("No alignments produced")
        return

    out_df = pd.DataFrame(all_rows)
    out_path = struct_dir / "chain_mapping_all.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote %d chain mappings to %s", len(out_df), out_path)
