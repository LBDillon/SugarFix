#!/usr/bin/env python3
"""Collect AF3 confidence metrics and merge with RMSD alignments.

Copied into the portable case-study pipeline bundle.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_PROTEINS = ["1ATJ", "1C1Z", "1GQV", "1RUZ", "5EQG"]
CONDITIONS = [
    "full_sequon_fixed_full_glycans",
    "full_sequon_fixed_with_glycans",
    "full_sequon_fixed",
    "unconstrained_denovo_glycans",
    "unconstrained",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    pipeline_root = script_dir.parent
    data_dir = pipeline_root / "data"
    default_output = data_dir / "af3_structural_validation.csv"
    default_rmsd = data_dir / "rmsd_values_template.csv"

    parser = argparse.ArgumentParser(
        description="Extract AF3 confidence metrics and merge with RMSD values."
    )
    parser.add_argument(
        "--af3-dir",
        required=True,
        help="Directory containing AF3 outputs organized by protein (e.g. /path/to/AF3_Structures).",
    )
    parser.add_argument(
        "--rmsd-csv",
        default=str(default_rmsd),
        help="CSV with columns: protein,condition,rmsd",
    )
    parser.add_argument(
        "--output-csv",
        default=str(default_output),
        help="Output CSV path (default: data/af3_structural_validation.csv)",
    )
    parser.add_argument(
        "--proteins",
        nargs="+",
        default=DEFAULT_PROTEINS,
        help="Protein IDs to process (default: 1ATJ 1C1Z 1GQV 1RUZ 5EQG)",
    )
    return parser.parse_args()


def parse_condition(folder_name: str) -> str | None:
    """Extract condition label from an AF3 folder name."""
    name = folder_name.lower()
    if "full_sequon_fixed" in name and "full_glycans" in name:
        return "full_sequon_fixed_full_glycans"
    if "full_sequon_fixed" in name and ("with_glycans" in name or "with_glycan" in name):
        return "full_sequon_fixed_with_glycans"
    if "unconstrained" in name and "denovo_glycan" in name:
        return "unconstrained_denovo_glycans"
    if "full_sequon_fixed" in name:
        return "full_sequon_fixed"
    if "unconstrained" in name:
        return "unconstrained"
    return None


def load_confidences(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def load_rmsd_map(rmsd_csv: Path) -> dict[tuple[str, str], float]:
    if not rmsd_csv.exists():
        raise FileNotFoundError(f"RMSD CSV not found: {rmsd_csv}")

    result: dict[tuple[str, str], float] = {}
    with open(rmsd_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {"protein", "condition", "rmsd"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{rmsd_csv} must contain columns: protein, condition, rmsd"
            )
        for row in reader:
            protein = row["protein"].strip().upper()
            condition = row["condition"].strip()
            rmsd_raw = row["rmsd"].strip()
            if not protein or not condition or not rmsd_raw:
                continue
            result[(protein, condition)] = float(rmsd_raw)
    return result


def find_prediction_folders(protein_dir: Path) -> dict[str, Path]:
    """Select one AF3 folder per condition for a protein."""
    candidates: dict[str, list[tuple[Path, int, float]]] = defaultdict(list)

    for entry in sorted(protein_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in {"compare", "msas", "templates"}:
            continue

        condition = parse_condition(entry.name)
        if condition is None:
            continue

        conf_files = glob.glob(str(entry / "*summary_confidences_*.json"))
        if not conf_files:
            continue

        candidates[condition].append((entry, len(conf_files), entry.stat().st_mtime))

    selected: dict[str, Path] = {}
    for condition, items in candidates.items():
        # Choose the folder with the most confidence files, break ties by mtime.
        items.sort(key=lambda x: (x[1], x[2], str(x[0])))
        selected_folder = items[-1][0]
        selected[condition] = selected_folder
        if len(items) > 1:
            print(
                f"  NOTE: {protein_dir.name}/{condition} had {len(items)} candidates; "
                f"selected {selected_folder.name}"
            )

    return selected


def extract_metrics_for_prediction(folder_path: Path) -> dict | None:
    conf_files = sorted(glob.glob(str(folder_path / "*summary_confidences_*.json")))
    if not conf_files:
        return None

    all_metrics = []
    top_model = None

    for cf in conf_files:
        match = re.search(r"_(\d+)\.json$", cf)
        model_idx = int(match.group(1)) if match else -1

        data = load_confidences(cf)
        metrics = {
            "model_idx": model_idx,
            "ptm": data.get("ptm"),
            "iptm": data.get("iptm"),
            "ranking_score": data.get("ranking_score"),
            "fraction_disordered": data.get("fraction_disordered"),
            "has_clash": data.get("has_clash"),
            "chain_ptm": data.get("chain_ptm", []),
        }
        all_metrics.append(metrics)

        if model_idx == 0:
            top_model = metrics

    if top_model is None and all_metrics:
        top_model = all_metrics[0]

    ptm_values = [m["ptm"] for m in all_metrics if m["ptm"] is not None]
    iptm_values = [m["iptm"] for m in all_metrics if m["iptm"] is not None]
    ranking_values = [
        m["ranking_score"]
        for m in all_metrics
        if m["ranking_score"] is not None and m["ranking_score"] > -1
    ]

    result = {
        "n_models": len(all_metrics),
        "ptm": top_model["ptm"],
        "iptm": top_model["iptm"],
        "ranking_score": top_model["ranking_score"],
        "fraction_disordered": top_model["fraction_disordered"],
        "has_clash": top_model["has_clash"],
        "mean_chain_ptm": None,
        "ptm_mean": statistics.mean(ptm_values) if ptm_values else None,
        "ptm_std": statistics.stdev(ptm_values) if len(ptm_values) > 1 else None,
        "iptm_mean": statistics.mean(iptm_values) if iptm_values else None,
        "iptm_std": statistics.stdev(iptm_values) if len(iptm_values) > 1 else None,
        "ranking_score_mean": statistics.mean(ranking_values) if ranking_values else None,
        "ranking_score_std": statistics.stdev(ranking_values) if len(ranking_values) > 1 else None,
    }

    chain_ptms = [v for v in top_model["chain_ptm"] if v is not None]
    if chain_ptms:
        result["mean_chain_ptm"] = statistics.mean(chain_ptms)

    return result


def main() -> int:
    args = parse_args()

    af3_dir = Path(args.af3_dir).resolve()
    rmsd_csv = Path(args.rmsd_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    proteins = [p.upper() for p in args.proteins]

    if not af3_dir.exists():
        print(f"ERROR: AF3 directory not found: {af3_dir}")
        return 1

    try:
        rmsd_map = load_rmsd_map(rmsd_csv)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR loading RMSD CSV: {exc}")
        return 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print("=" * 70)
    print("COLLECTING AF3 METRICS")
    print("=" * 70)
    print(f"AF3 dir: {af3_dir}")
    print(f"RMSD CSV: {rmsd_csv}")
    print(f"Output CSV: {output_csv}")
    print()

    for protein in proteins:
        protein_dir = af3_dir / protein
        if not protein_dir.is_dir():
            print(f"  WARNING: {protein_dir} not found, skipping")
            continue

        prediction_folders = find_prediction_folders(protein_dir)
        for condition in CONDITIONS:
            folder = prediction_folders.get(condition)
            if folder is None:
                print(f"  WARNING: {protein}/{condition} not found")
                continue

            metrics = extract_metrics_for_prediction(folder)
            if metrics is None:
                print(f"  WARNING: no confidence files for {protein}/{condition}")
                continue

            rmsd = rmsd_map.get((protein, condition))

            rows.append(
                {
                    "protein": protein,
                    "condition": condition,
                    "rmsd": rmsd,
                    "ptm": metrics["ptm"],
                    "iptm": metrics["iptm"],
                    "ranking_score": metrics["ranking_score"],
                    "fraction_disordered": metrics["fraction_disordered"],
                    "has_clash": metrics["has_clash"],
                    "mean_chain_ptm": metrics["mean_chain_ptm"],
                    "n_models": metrics["n_models"],
                    "ptm_mean": metrics["ptm_mean"],
                    "ptm_std": metrics["ptm_std"],
                    "iptm_mean": metrics["iptm_mean"],
                    "iptm_std": metrics["iptm_std"],
                    "ranking_score_mean": metrics["ranking_score_mean"],
                    "ranking_score_std": metrics["ranking_score_std"],
                    "af3_folder": folder.name,
                }
            )

            print(
                f"  {protein:<6} {condition:<38} "
                f"pTM={str(metrics['ptm']):<6} "
                f"ipTM={str(metrics['iptm']):<6} "
                f"rank={str(metrics['ranking_score']):<6} "
                f"RMSD={str(rmsd) if rmsd is not None else 'N/A'}"
            )

    fieldnames = [
        "protein",
        "condition",
        "rmsd",
        "ptm",
        "iptm",
        "ranking_score",
        "fraction_disordered",
        "has_clash",
        "mean_chain_ptm",
        "n_models",
        "ptm_mean",
        "ptm_std",
        "iptm_mean",
        "iptm_std",
        "ranking_score_mean",
        "ranking_score_std",
        "af3_folder",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
