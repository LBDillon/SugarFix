#!/usr/bin/env python3
"""
Master Pipeline Runner

Runs the complete sequon analysis pipeline for a given PDB structure.

Usage:
    python run_pipeline.py --pdb_id 1EO8 --output_dir ./results/1EO8 --proteinmpnn_path /path/to/protein_mpnn_run.py

This will execute:
    0. Baseline amino acid retention analysis (after designs)
    1. Structure preparation
    2. Sequon identification
    3. ProteinMPNN designs (3 conditions)
    4. Retention analysis
    5. De novo sequon analysis
    6. Structural context analysis
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(script_name, args_dict, scripts_dir):
    """Run a pipeline step."""
    script_path = scripts_dir / script_name

    cmd = ["python", str(script_path)]
    for key, value in args_dict.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'#' * 70}")
    print(f"# Running: {script_name}")
    print(f"{'#' * 70}\n")

    result = subprocess.run(cmd, cwd=str(scripts_dir.parent))

    if result.returncode != 0:
        print(f"\n❌ FAILED: {script_name}")
        return False

    print(f"\n✓ Completed: {script_name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run complete sequon analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python run_pipeline.py --pdb_id 1EO8 --output_dir ./results/1EO8 --proteinmpnn_path ../ProteinMPNN/protein_mpnn_run.py

    # Run only analysis steps (if designs already exist)
    python run_pipeline.py --pdb_id 1EO8 --output_dir ./results/1EO8 --skip_design

    # Run specific steps
    python run_pipeline.py --pdb_id 1EO8 --output_dir ./results/1EO8 --steps 1,2,4,5
        """
    )

    parser.add_argument('--pdb_id', required=True, help='PDB ID (e.g., 1EO8)')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--proteinmpnn_path', help='Path to protein_mpnn_run.py')
    parser.add_argument('--skip_design', action='store_true', help='Skip ProteinMPNN design step')
    parser.add_argument('--steps', help='Comma-separated list of steps to run (e.g., 1,2,4,5)')
    parser.add_argument('--num_designs', type=int, default=32, help='Number of designs per condition')

    args = parser.parse_args()

    # Determine which steps to run
    if args.steps:
        steps_to_run = [int(s) for s in args.steps.split(',')]
    elif args.skip_design:
        steps_to_run = [1, 2, 4, 5, 6]
    else:
        steps_to_run = [1, 2, 3, 4, 5, 6]

    # Check ProteinMPNN path if needed
    if 3 in steps_to_run and not args.proteinmpnn_path:
        print("ERROR: --proteinmpnn_path required for design step (step 3)")
        print("Use --skip_design to skip design generation if designs already exist")
        sys.exit(1)

    scripts_dir = Path(__file__).parent / "scripts"
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("SEQUON ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"\nPDB ID: {args.pdb_id}")
    print(f"Output: {output_dir}")
    print(f"Steps to run: {steps_to_run}")
    print()

    # Step 1: Structure Preparation
    if 1 in steps_to_run:
        success = run_step(
            "01_prepare_structure.py",
            {"pdb_id": args.pdb_id, "output_dir": output_dir},
            scripts_dir
        )
        if not success:
            sys.exit(1)

    # Step 2: Sequon Identification
    if 2 in steps_to_run:
        success = run_step(
            "02_identify_sequons.py",
            {"pdb_dir": output_dir},
            scripts_dir
        )
        if not success:
            sys.exit(1)

    # Step 3: ProteinMPNN Design
    if 3 in steps_to_run:
        success = run_step(
            "03_run_proteinmpnn.py",
            {
                "pdb_dir": output_dir,
                "proteinmpnn_path": args.proteinmpnn_path,
                "num_designs": args.num_designs
            },
            scripts_dir
        )
        if not success:
            sys.exit(1)

    # Step 4: Retention Analysis
    if 4 in steps_to_run:
        success = run_step(
            "04_analyze_retention.py",
            {"pdb_dir": output_dir},
            scripts_dir
        )
        if not success:
            sys.exit(1)

    # Step 5: De Novo Analysis
    if 5 in steps_to_run:
        success = run_step(
            "05_analyze_denovo.py",
            {"pdb_dir": output_dir},
            scripts_dir
        )
        if not success:
            sys.exit(1)

    # Step 6: Structural Context
    if 6 in steps_to_run:
        success = run_step(
            "06_structural_context.py",
            {"pdb_dir": output_dir},
            scripts_dir
        )
        if not success:
            sys.exit(1)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(f"\nAll outputs saved to: {output_dir}/")
    print(f"\nKey output files:")
    print(f"  Structure:  {output_dir}/structure/")
    print(f"  Sequons:    {output_dir}/sequons/")
    print(f"  Designs:    {output_dir}/designs/")
    print(f"  Analysis:   {output_dir}/analysis/")

    print(f"\nVisualization files:")
    print(f"  - structure/figures/chain_lengths.png")
    print(f"  - sequons/figures/sequon_map.png")
    print(f"  - designs/figures/design_scores.png")
    print(f"  - analysis/retention/figures/retention_heatmap.png")
    print(f"  - analysis/denovo/figures/hotspot_barplot.png")
    print(f"  - analysis/structural/figures/retention_vs_rsa.png")

    print(f"\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
