#!/usr/bin/env python3
"""
STEP 3: Run ProteinMPNN Designs

Runs ProteinMPNN under three conditions:
1. Unconstrained - no positions fixed
2. N-only fixed - just asparagine (N) of each sequon
3. Full sequon fixed - all N-X-S/T positions

Uses ProteinMPNN's own sequence parsing to ensure correct position indexing.

Usage:
    python 03_run_proteinmpnn.py --pdb_dir ./results/1EO8 --proteinmpnn_path /path/to/protein_mpnn_run.py

Outputs:
    - designs/unconstrained/seqs/{pdb_id}.fa
    - designs/n_only_fixed/seqs/{pdb_id}.fa
    - designs/full_sequon_fixed/seqs/{pdb_id}.fa
    - designs/design_summary.csv
    - designs/figures/design_scores.png
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import shared utilities
from mpnn_utils import (
    get_mpnn_chain_seqs_and_order,
    write_fixed_positions_jsonl,
    verify_sequon_positions,
    sanity_check_full_fixed,
    is_functional_sequon
)


def run_proteinmpnn(pdb_path, output_dir, proteinmpnn_path, fixed_pos_jsonl=None,
                    num_seqs=32, temp=0.1, seed=42, chains=None):
    """Run ProteinMPNN."""
    proteinmpnn_dir = Path(proteinmpnn_path).parent
    proteinmpnn_script = Path(proteinmpnn_path).name
    
    cmd = [
        "python", str(proteinmpnn_script),
        "--pdb_path", str(pdb_path),
        "--num_seq_per_target", str(num_seqs),
        "--sampling_temp", str(temp),
        "--out_folder", str(output_dir),
        "--seed", str(seed)
    ]

    if chains:
        cmd.extend(["--pdb_path_chains", chains])

    if fixed_pos_jsonl:
        cmd.extend(["--fixed_positions_jsonl", str(fixed_pos_jsonl)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(proteinmpnn_dir)
        )
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e.stderr[:500] if e.stderr else str(e)


def read_fasta_scores(fasta_path):
    """Read scores from FASTA headers."""
    scores = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Parse header for score and seq_recovery
                parts = line.strip().split(', ')
                score_dict = {}
                for part in parts:
                    if '=' in part:
                        key, val = part.split('=', 1)
                        try:
                            score_dict[key] = float(val)
                        except ValueError:
                            score_dict[key] = val
                scores.append(score_dict)
    return scores


def create_visualizations(design_results, output_dir, pdb_id):
    """Create design summary visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Collect all scores
    all_scores = []
    for condition, results in design_results.items():
        if results['success'] and results['scores']:
            for i, score_dict in enumerate(results['scores'][1:], 1):  # Skip WT
                all_scores.append({
                    'condition': condition.replace('_', ' ').title(),
                    'design': i,
                    'score': score_dict.get('score', 0),
                    'seq_recovery': score_dict.get('seq_recovery', 0)
                })

    if not all_scores:
        print("  No scores to visualize")
        return

    scores_df = pd.DataFrame(all_scores)

    # Figure 1: Score distribution by condition
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ProteinMPNN scores
    sns.boxplot(data=scores_df, x='condition', y='score', ax=axes[0], palette='Set2')
    axes[0].set_xlabel('Condition', fontsize=12)
    axes[0].set_ylabel('ProteinMPNN Score', fontsize=12)
    axes[0].set_title('Design Scores by Condition', fontsize=12)
    axes[0].tick_params(axis='x', rotation=15)

    # Right: Sequence recovery
    sns.boxplot(data=scores_df, x='condition', y='seq_recovery', ax=axes[1], palette='Set2')
    axes[1].set_xlabel('Condition', fontsize=12)
    axes[1].set_ylabel('Sequence Recovery', fontsize=12)
    axes[1].set_title('Sequence Recovery by Condition', fontsize=12)
    axes[1].tick_params(axis='x', rotation=15)

    plt.suptitle(f'{pdb_id}: ProteinMPNN Design Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / "design_scores.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Score vs Recovery scatter
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in scores_df['condition'].unique():
        subset = scores_df[scores_df['condition'] == condition]
        ax.scatter(subset['seq_recovery'], subset['score'],
                  label=condition, alpha=0.6, s=50)

    ax.set_xlabel('Sequence Recovery', fontsize=12)
    ax.set_ylabel('ProteinMPNN Score', fontsize=12)
    ax.set_title(f'{pdb_id}: Score vs Sequence Recovery', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "score_vs_recovery.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save scores CSV
    scores_df.to_csv(output_dir / "design_scores.csv", index=False)

    print(f"  Saved figures to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Step 3: Run ProteinMPNN designs')
    parser.add_argument('--pdb_dir', required=True, help='Directory from Steps 1-2')
    parser.add_argument('--proteinmpnn_path', required=True, help='Path to protein_mpnn_run.py')
    parser.add_argument('--num_designs', type=int, default=32, help='Number of designs per condition')
    parser.add_argument('--temp', type=float, default=0.1, help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--unconstrained_only', action='store_true', help='Only run unconstrained designs (skip fixed position designs)')
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    structure_dir = pdb_dir / "structure"
    sequons_dir = pdb_dir / "sequons"
    designs_dir = pdb_dir / "designs"
    designs_dir.mkdir(exist_ok=True)

    proteinmpnn_path = Path(args.proteinmpnn_path)
    if not proteinmpnn_path.exists():
        print(f"ERROR: ProteinMPNN not found at {proteinmpnn_path}")
        sys.exit(1)

    # Load structure info
    with open(structure_dir / "structure_info.json") as f:
        structure_info = json.load(f)

    pdb_id = structure_info['pdb_id']
    # Support both 'protein_only_path' (newer) and 'path' (older format)
    asymmetric_unit = structure_info['asymmetric_unit']
    pdb_path = Path(asymmetric_unit.get('protein_only_path', asymmetric_unit.get('path')))
    # ProteinMPNN names output files based on input PDB filename
    pdb_stem = pdb_path.stem  # e.g., "1I9E_protein" or "1I9E"

    # Load sequons (now uses position_0idx from MPNN-consistent parsing)
    with open(sequons_dir / "sequons_by_chain.json") as f:
        sequons_by_chain = json.load(f)

    # Load MPNN chain order (saved by step 2)
    chain_order_path = sequons_dir / "mpnn_chain_order.json"
    if chain_order_path.exists():
        with open(chain_order_path) as f:
            all_chain_ids = json.load(f)["chain_order"]
    else:
        # Fallback to structure_info.json for older results
        all_chain_ids = [c['chain_id'] for c in structure_info['asymmetric_unit']['chains']
                         if c['model'] == 1]

    # Get MPNN chain sequences for verification
    chain_seqs, _ = get_mpnn_chain_seqs_and_order(pdb_path)

    # Verify sequon positions before running
    print("\n1. Verifying sequon positions match MPNN indexing...")
    try:
        verify_sequon_positions(chain_seqs, sequons_by_chain, pdb_id)
        print("   Verified successfully")
    except AssertionError as e:
        print(f"   ERROR: {e}")
        print("   Re-run Step 2 to regenerate sequon positions with MPNN parsing.")
        sys.exit(1)

    print("=" * 70)
    print(f"STEP 3: PROTEINMPNN DESIGNS - {pdb_id}")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  Designs per condition: {args.num_designs}")
    print(f"  Sampling temperature: {args.temp}")
    print(f"  Random seed: {args.seed}")

    design_results = {}

    # Condition 1: Unconstrained
    print(f"\n{'=' * 70}")
    print("CONDITION 1: UNCONSTRAINED")
    print("=" * 70)

    unconstrained_dir = designs_dir / "unconstrained"
    unconstrained_dir.mkdir(exist_ok=True)

    print("Running ProteinMPNN with no constraints...")
    success, error = run_proteinmpnn(
        pdb_path, unconstrained_dir, proteinmpnn_path,
        num_seqs=args.num_designs, temp=args.temp, seed=args.seed,
        # ProteinMPNN expects whitespace-separated chain IDs
        chains=' '.join(all_chain_ids)
    )

    if success:
        fa_path = unconstrained_dir / "seqs" / f"{pdb_stem}.fa"
        scores = read_fasta_scores(fa_path) if fa_path.exists() else []
        print(f"  ✓ Generated {len(scores) - 1} designs")
        design_results['unconstrained'] = {'success': True, 'scores': scores}
    else:
        print(f"  ✗ FAILED: {error}")
        design_results['unconstrained'] = {'success': False, 'error': error}

    # Condition 2: N-only fixed
    if not args.unconstrained_only:
        print(f"\n{'=' * 70}")
        print("CONDITION 2: N-ONLY FIXED")
        print("=" * 70)

        n_only_dir = designs_dir / "n_only_fixed"
        n_only_dir.mkdir(exist_ok=True)

        # Get N positions (use position_0idx from MPNN-consistent parsing)
        n_only_positions = {}
        total_fixed = 0
        for chain_id, sequons in sequons_by_chain.items():
            if sequons:
                positions = [s['position_0idx'] for s in sequons]
                n_only_positions[chain_id] = positions
                total_fixed += len(positions)
                print(f"  Chain {chain_id}: fixing N at positions {[p+1 for p in positions]} (1-indexed)")

        if total_fixed > 0:
            fixed_pos_file = designs_dir / "fixed_positions_n_only.jsonl"
            write_fixed_positions_jsonl(pdb_stem, n_only_positions, all_chain_ids, fixed_pos_file)

            print(f"\nRunning ProteinMPNN with {total_fixed} N positions fixed...")
            success, error = run_proteinmpnn(
                pdb_path, n_only_dir, proteinmpnn_path,
                fixed_pos_jsonl=fixed_pos_file,
                num_seqs=args.num_designs, temp=args.temp, seed=args.seed,
                # ProteinMPNN expects whitespace-separated chain IDs
                chains=' '.join(all_chain_ids)
            )

            if success:
                fa_path = n_only_dir / "seqs" / f"{pdb_stem}.fa"
                scores = read_fasta_scores(fa_path) if fa_path.exists() else []
                print(f"  ✓ Generated {len(scores) - 1} designs")
                design_results['n_only_fixed'] = {'success': True, 'scores': scores}
            else:
                print(f"  ✗ FAILED: {error}")
                design_results['n_only_fixed'] = {'success': False, 'error': error}
        else:
            print("  No sequons found - skipping N-only fixed designs")
            design_results['n_only_fixed'] = {'success': False, 'error': 'No sequons found'}
    else:
        print(f"\nSkipping N-only fixed designs (--unconstrained_only flag set)")
        design_results['n_only_fixed'] = {'success': False, 'error': 'Skipped (--unconstrained_only)'}

    # Condition 3: Full sequon fixed
    print(f"\n{'=' * 70}")
    print("CONDITION 3: FULL SEQUON FIXED")
    print("=" * 70)

    full_sequon_dir = designs_dir / "full_sequon_fixed"
    full_sequon_dir.mkdir(exist_ok=True)

    # Get full sequon positions (N, X, S/T) using position_0idx
    full_sequon_positions = {}
    total_fixed = 0
    for chain_id, sequons in sequons_by_chain.items():
        if sequons:
            positions = []
            seq_len = len(chain_seqs.get(chain_id, ""))
            for s in sequons:
                pos = s['position_0idx']
                # Only include positions that are within sequence bounds
                for offset in [0, 1, 2]:
                    if pos + offset < seq_len:
                        positions.append(pos + offset)
            full_sequon_positions[chain_id] = sorted(set(positions))
            total_fixed += len(positions)
            print(f"  Chain {chain_id}: fixing {len(positions)} positions")

    if total_fixed > 0:
        fixed_pos_file = designs_dir / "fixed_positions_full_sequon.jsonl"
        write_fixed_positions_jsonl(pdb_stem, full_sequon_positions, all_chain_ids, fixed_pos_file)

        print(f"\nRunning ProteinMPNN with {total_fixed} full sequon positions fixed...")
        success, error = run_proteinmpnn(
            pdb_path, full_sequon_dir, proteinmpnn_path,
            fixed_pos_jsonl=fixed_pos_file,
            num_seqs=args.num_designs, temp=args.temp, seed=args.seed,
            # ProteinMPNN expects whitespace-separated chain IDs
            chains=' '.join(all_chain_ids)
        )

        if success:
            fa_path = full_sequon_dir / "seqs" / f"{pdb_stem}.fa"
            scores = read_fasta_scores(fa_path) if fa_path.exists() else []
            print(f"  ✓ Generated {len(scores) - 1} designs")

            # CRITICAL: Verify that fixing actually worked
            print("  Verifying fixed positions preserved WT triplets...")
            if sanity_check_full_fixed(fa_path, all_chain_ids, sequons_by_chain):
                print("  ✓ All WT triplets preserved correctly")
                design_results['full_sequon_fixed'] = {'success': True, 'scores': scores}
            else:
                print("  ✗ WARNING: Some WT triplets were NOT preserved!")
                print("    This indicates an indexing mismatch. Check sequon positions.")
                design_results['full_sequon_fixed'] = {'success': True, 'scores': scores, 'warning': 'sanity_check_failed'}
        else:
            print(f"  ✗ FAILED: {error}")
            design_results['full_sequon_fixed'] = {'success': False, 'error': error}
    else:
        print("  No sequons to fix - skipping")
        design_results['full_sequon_fixed'] = {'success': False, 'error': 'No sequons'}

    # Create visualizations
    print(f"\n{'=' * 70}")
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    create_visualizations(design_results, designs_dir, pdb_id)

    # Save summary
    summary = {
        'pdb_id': pdb_id,
        'num_designs': args.num_designs,
        'temperature': args.temp,
        'seed': args.seed,
        'conditions': {k: {'success': v['success']} for k, v in design_results.items()}
    }
    with open(designs_dir / "design_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE")
    print("=" * 70)
    print(f"\nOutputs in {designs_dir}/:")
    for condition in ['unconstrained', 'n_only_fixed', 'full_sequon_fixed']:
        if design_results.get(condition, {}).get('success'):
            print(f"  ✓ {condition}/seqs/{pdb_id}.fa")
        else:
            print(f"  ✗ {condition} (failed)")
    print(f"  - design_summary.json")
    print(f"  - design_scores.csv")
    print(f"  - figures/design_scores.png")

    print(f"\n→ Next: python 04_analyze_retention.py --pdb_dir {pdb_dir}")


if __name__ == "__main__":
    main()
