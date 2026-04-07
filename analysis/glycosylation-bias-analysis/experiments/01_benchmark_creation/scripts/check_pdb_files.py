#!/usr/bin/env python3
"""Examine problematic PDB files to diagnose issues"""

from Bio.PDB import PDBParser, is_aa
import os

problematic = ['1HZH_A', '1RUZ_A', '3TIH_G']
pdb_dir = 'data/glyco_benchmark/raw/glycoproteins'

for pdb_name in problematic:
    pdb_file = os.path.join(pdb_dir, f'{pdb_name}.pdb')
    print(f"\n{'='*70}")
    print(f"{pdb_name}.pdb")
    print(f"{'='*70}")
    
    if not os.path.exists(pdb_file):
        print(f"FILE NOT FOUND: {pdb_file}")
        continue
    
    # Check file size and basic content
    size = os.path.getsize(pdb_file)
    print(f"File size: {size:,} bytes")
    
    # Try parsing with BioPython
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_name, pdb_file)
        print(f"Structure loaded successfully")
        print(f"Number of models: {len(structure)}")
        
        for model_idx, model in enumerate(structure):
            print(f"  Model {model_idx}: {len(list(model))} chains")
            for chain in model:
                residues = [r for r in chain.get_residues() if is_aa(r)]
                print(f"    Chain {chain.id}: {len(residues)} amino acids")
                
    except Exception as e:
        print(f"Error parsing with BioPython: {e}")
    
    # Check raw file content
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
        
    atom_lines = [l for l in lines if l.startswith('ATOM')]
    model_lines = [l for l in lines if l.startswith('MODEL')]
    hetatm_lines = [l for l in lines if l.startswith('HETATM')]
    endmdl_lines = [l for l in lines if l.startswith('ENDMDL')]
    
    print(f"\nRaw content:")
    print(f"  ATOM records: {len(atom_lines)}")
    print(f"  HETATM records: {len(hetatm_lines)}")
    print(f"  MODEL records: {len(model_lines)}")
    print(f"  ENDMDL records: {len(endmdl_lines)}")
    
    # Show first few ATOM lines
    if atom_lines:
        print(f"\nFirst ATOM line:")
        print(f"  {atom_lines[0][:80]}")
    
    # Check for NMR structures
    if endmdl_lines:
        print(f"\n⚠️  NMR STRUCTURE DETECTED")
        print(f"  This is an NMR structure with {len(endmdl_lines)} models")
        print(f"  ColabDesign requires single-model X-ray structures")
