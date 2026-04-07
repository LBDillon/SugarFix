#!/usr/bin/env python3
"""
Fix multi-chain PDB files by extracting the sequence-containing chain
and creating single-chain PDB files.

This handles cases where the manifest specifies chain "A" but the PDB file
contains multi-chain complexes (antibodies, viruses, etc.) where chain "A"
is empty or doesn't exist.
"""

from Bio.PDB import PDBParser, is_aa, PDBIO
from Bio import SeqIO
import os
from pathlib import Path

problematic = {
    '1HZH_A': {'manifest_chain': 'A', 'correct_chain': 'H', 'reason': 'Antibody-antigen complex'},
    '1RUZ_A': {'manifest_chain': 'A', 'correct_chain': 'H', 'reason': 'Trimeric hemagglutinin'},
    '3TIH_G': {'manifest_chain': 'G', 'correct_chain': 'A', 'reason': 'Tetrameric gp120, G does not exist'},
}

pdb_dir = 'data/glyco_benchmark/raw/glycoproteins'

for pdb_name, info in problematic.items():
    pdb_file = os.path.join(pdb_dir, f'{pdb_name}.pdb')
    
    print(f"\n{'='*70}")
    print(f"{pdb_name}")
    print(f"{'='*70}")
    print(f"Issue: {info['reason']}")
    print(f"Manifest specifies: chain {info['manifest_chain']}")
    print(f"Actual protein chain: chain {info['correct_chain']}")
    
    if not os.path.exists(pdb_file):
        print(f"File not found: {pdb_file}")
        continue
    
    # Load structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_file)
    
    # Extract the correct chain
    model = structure[0]
    if info['correct_chain'] not in [c.id for c in model]:
        print(f"ERROR: Chain {info['correct_chain']} not found in structure!")
        continue
    
    chain_to_extract = model[info['correct_chain']]
    
    # Get amino acid count
    residues = [r for r in chain_to_extract.get_residues() if is_aa(r)]
    print(f"✓ Chain {info['correct_chain']} contains {len(residues)} amino acids")
    
    # Create new structure with just this chain
    new_structure = structure.copy()
    new_model = new_structure[0]
    
    # Remove all other chains
    chains_to_remove = [c.id for c in new_model if c.id != info['correct_chain']]
    for chain_id in chains_to_remove:
        new_model.detach_child(chain_id)
    
    # Save new PDB file (overwrite original)
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(pdb_file)
    
    print(f"✓ Saved single-chain PDB (chain {info['correct_chain']}) to {pdb_file}")

print(f"\n{'='*70}")
print("Multi-chain PDB files have been fixed!")
print("The PDB files now contain only the sequence-bearing chain.")
print("="*70)
