#!/usr/bin/env python3
"""
Process design FASTA files into a comprehensive CSV with metadata.

Parses FASTA headers to extract design metrics and merges with wild-type
protein information from the benchmark manifest.
"""

import pandas as pd
from pathlib import Path
import re

# Load benchmark manifest for WT protein information
manifest_path = 'data/glyco_benchmark/manifests/benchmark_manifest_simple.csv'
manifest_df = pd.read_csv(manifest_path)

# Map (pdb_id, chain_id) -> protein info
protein_metadata = {}
for _, row in manifest_df.iterrows():
    key = (row['pdb_id'], row['chain_id'])
    protein_metadata[key] = {
        'protein_name': row['protein_name'],
        'organism': row['organism'],
        'protein_class': row['protein_class'],
        'wt_sequence': row['sequence'],
        'wt_sequence_length': len(row['sequence']) if pd.notna(row['sequence']) else None,
        'n_glycosites': row['n_glycosites'],
        'glycosite_positions': row['glycosite_positions'],
        'uniprot_id': row['uniprot_id'],
    }

# Also map by PDB ID only for proteins where chain changed after multi-chain extraction
pdb_only_metadata = {}
for _, row in manifest_df.iterrows():
    pdb_id = row['pdb_id']
    if pdb_id not in pdb_only_metadata:
        pdb_only_metadata[pdb_id] = {
            'protein_name': row['protein_name'],
            'organism': row['organism'],
            'protein_class': row['protein_class'],
            'wt_sequence': row['sequence'],
            'wt_sequence_length': len(row['sequence']) if pd.notna(row['sequence']) else None,
            'n_glycosites': row['n_glycosites'],
            'glycosite_positions': row['glycosite_positions'],
            'uniprot_id': row['uniprot_id'],
        }

print(f"Loaded {len(protein_metadata)} proteins from manifest")

# Find all FASTA files in designs directory
designs_dir = Path('data/glyco_benchmark/designs')
fasta_files = sorted(designs_dir.glob('*.fasta'))

print(f"Found {len(fasta_files)} FASTA files")

all_designs = []

for fasta_file in fasta_files:
    # Parse filename: {pdb_id}_{chain_id}_{condition}.fasta
    parts = fasta_file.stem.split('_')
    if len(parts) < 3:
        print(f"Skipping {fasta_file.name} (unexpected format)")
        continue
    
    pdb_id = parts[0]
    chain_id = parts[1]
    condition = '_'.join(parts[2:])  # Handle case where condition might have underscores
    
    print(f"\nProcessing: {fasta_file.name}")
    print(f"  PDB: {pdb_id}, Chain: {chain_id}, Condition: {condition}")
    
    # Get protein metadata
    protein_key = (pdb_id, chain_id)
    if protein_key not in protein_metadata:
        # Try looking up by PDB ID only (handles chain changes from multi-chain extraction)
        if pdb_id in pdb_only_metadata:
            print(f"  Note: Using metadata from PDB ID only (chain was {protein_metadata.get((pdb_id, 'A'), {}).get('protein_name', 'Unknown')} in manifest)")
            protein_info = pdb_only_metadata[pdb_id]
        else:
            print(f"  Warning: {pdb_id} {chain_id} not in manifest")
            protein_info = {
                'protein_name': 'Unknown',
                'organism': 'Unknown',
                'protein_class': 'Unknown',
                'wt_sequence': None,
                'wt_sequence_length': None,
                'n_glycosites': None,
                'glycosite_positions': None,
                'uniprot_id': None,
            }
    else:
        protein_info = protein_metadata[protein_key]
    
    # Parse FASTA file
    design_count = 0
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Header format: >{pdb_id}_{condition}_design{N}|score={score}|seqid={seqid}
                header = line[1:]  # Remove '>'
                
                # Extract design number
                design_match = re.search(r'design(\d+)', header)
                if design_match:
                    design_num = int(design_match.group(1))
                else:
                    design_num = design_count
                
                # Extract score
                score_match = re.search(r'score=([-\d.]+)', header)
                score = float(score_match.group(1)) if score_match else None
                
                # Extract seqid
                seqid_match = re.search(r'seqid=([\d.]+)', header)
                seqid = float(seqid_match.group(1)) if seqid_match else None
                
                all_designs.append({
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'protein_name': protein_info['protein_name'],
                    'organism': protein_info['organism'],
                    'protein_class': protein_info['protein_class'],
                    'uniprot_id': protein_info['uniprot_id'],
                    'n_glycosites': protein_info['n_glycosites'],
                    'glycosite_positions': protein_info['glycosite_positions'],
                    'wt_sequence_length': protein_info['wt_sequence_length'],
                    'condition': condition,
                    'design_number': design_num,
                    'design_score': score,
                    'design_seqid': seqid,
                    'fasta_file': fasta_file.name,
                })
                
                design_count += 1
    
    print(f"  ✓ Extracted {design_count} designs")

# Create DataFrame
designs_df = pd.DataFrame(all_designs)

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Total designs: {len(designs_df)}")
print(f"Unique proteins: {designs_df[['pdb_id', 'chain_id']].drop_duplicates().shape[0]}")
print(f"Conditions: {designs_df['condition'].unique()}")
print(f"\nDesigns by condition:")
for condition in sorted(designs_df['condition'].unique()):
    count = len(designs_df[designs_df['condition'] == condition])
    print(f"  {condition}: {count}")

# Save to CSV
output_path = 'data/glyco_benchmark/designs/design_summary.csv'
designs_df.to_csv(output_path, index=False)
print(f"\n✓ Saved to {output_path}")

# Show sample
print(f"\n{'='*70}")
print("Sample rows:")
print(f"{'='*70}")
print(designs_df.head(10).to_string())

# Compute basic statistics
print(f"\n{'='*70}")
print("Design Quality Statistics")
print(f"{'='*70}")

for condition in sorted(designs_df['condition'].unique()):
    cond_data = designs_df[designs_df['condition'] == condition]
    print(f"\n{condition.upper()}:")
    print(f"  Mean score:  {cond_data['design_score'].mean():.4f} ± {cond_data['design_score'].std():.4f}")
    print(f"  Mean seqid:  {cond_data['design_seqid'].mean():.4f} ± {cond_data['design_seqid'].std():.4f}")
    print(f"  Score range: [{cond_data['design_score'].min():.4f}, {cond_data['design_score'].max():.4f}]")
    print(f"  Seqid range: [{cond_data['design_seqid'].min():.4f}, {cond_data['design_seqid'].max():.4f}]")

# Protein-level statistics
print(f"\n{'='*70}")
print("Statistics by Protein Class")
print(f"{'='*70}")

for pclass in sorted(designs_df['protein_class'].unique()):
    class_data = designs_df[designs_df['protein_class'] == pclass]
    print(f"\n{pclass.upper()} ({len(class_data)} designs):")
    print(f"  Mean score:  {class_data['design_score'].mean():.4f}")
    print(f"  Mean seqid:  {class_data['design_seqid'].mean():.4f}")
