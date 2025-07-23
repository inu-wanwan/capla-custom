import os
import csv
import sys
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.DSSP import DSSP, ss_to_index
from collections import defaultdict

AMINO_ACID_CLUSTER = {
    1: {'C'},                          # Cluster 1: ++++, Cys       
    2: {'D', 'E'},                     # Cluster 2: +++', +, Asp, Glu
    3: {'R', 'K'},                     # Cluster 3: +++, +, Arg, Lys
    4: {'H', 'N', 'Q', 'W'},           # Cluster 4: ++, +, His, Asn, Gln, Trp
    5: {'Y', 'M', 'T', 'S'},           # Cluster 5: +, +, Tyr, Met, Thr, Ser
    6: {'I', 'L', 'F', 'P'},           # Cluster 6: -, +, Ile, Leu, Phe, Pro
    7: {'A', 'G', 'V'},                # Cluster 7: -, -, Ala, Gly, Val
}

GROUP_FLAGS = {
    "non_polar": {'G', 'A', 'V', 'L', 'I', 'M', 'F', 'P', 'W'},
    "polar":     {'S', 'T', 'Y', 'C', 'Q', 'N'},
    "acidic":    {'D', 'E'},
    "basic":     {'K', 'R', 'H'}
}

def get_physicochemical_features(aa: str) -> dict:
    """
    Get physicochemical features for a given amino acid.
    """

    # initialize features dictionary
    features = {
        "non_polar": 0.0,
        "polar": 0.0,
        "acidic": 0.0,
        "basic": 0.0,
        **{f"c2_{i}": 0.0 for i in range(1, 8)},
    }

    # Check if the amino acid is in the group flags
    for group, residues in AMINO_ACID_CLUSTER.items():
        if aa in residues:
            features[f"c2_{group}"] = 1.0

    # Check for group flags
    for key, residues in GROUP_FLAGS.items():
        if aa in residues:
            features[key] = 1.0
            break

    return features

def get_dssp_sse_onehot(sse: str) -> dict:
    """
    get one-hot encoding for secondary structure elements.
    input:
        sse: secondary structure element (ex: 'H', 'E', 'C' ...)
    output:
        one-hot encoded dictionary with keys 'H', 'E', 'C' ...
    """

    sse_mapping = {
        'H': 's2_H',
        'B': 's2_B',
        'E': 's2_E',
        'G': 's2_G',
        'I': 's2_I',
        'T': 's2_T',
        'S': 's2_S',
        'C': 's2_C',
        ' ': 's2_C',  # 空白も Coil 扱い
    }

    one_hot = {f"s2_{c}": 0 for c in ['B', 'C', 'E', 'G', 'H', 'I', 'S', 'T']}

    if sse in sse_mapping:
        one_hot[sse_mapping[sse]] = 1

    return one_hot

def extract_atom_lines(input_path: str, output_path: str):
    """
    PDBファイルからATOM行のみを抽出して新しいファイルに保存する
    """
    with open(input_path, "r") as infile:
        lines = infile.readlines()

    atom_lines = [line for line in lines if line.startswith("ATOM")]

    # DSSPが必要とするHEADER行を追加（任意）
    atom_lines.insert(0, "HEADER    GENERATED_BY_SCRIPT_FOR_MKDSSP\n")

    with open(output_path, "w") as outfile:
        outfile.writelines(atom_lines)

    print(f"✅ Extracted {len(atom_lines)} ATOM lines to: {output_path}")


if __name__ == "__main__":
    print(get_physicochemical_features('A'))  # Example usage
    print(get_dssp_sse_onehot('H'))  # Example usage

    # 入力と出力パス
    input_pdb = "data/v2020-other-PL/1a0q/1a0q_protein.pdb"
    output_pdb = "data/v2020-other-PL/1a0q/fixed_1a0q.pdb"

    # ATOM行のみを抽出して新しいファイルに保存
    extract_atom_lines(input_pdb, output_pdb)

    # 構造を読み込む
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("1A0Q", output_pdb)

    model = structure[0]
    dssp = DSSP(model, output_pdb)
    for key in dssp.keys():
        print("Chain:", key[0], "ResID:", key[1], "AA:", dssp[key], "SS:", dssp[key])
        break  # 最初の1件だけ確認
