"""High-level workflows for SugarFix tutorial notebooks.

These functions are intentionally notebook-shaped: each one performs a complete
researcher-facing step and returns tables/paths that can be displayed directly.
The implementation lives here so tutorial notebooks can stay small and readable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class StructurePreparationResult:
    pdb_path: Path
    protein_pdb_path: Path
    annotation_structure_path: Path
    chains_info: List[dict]
    n_models: int
    missing_residues: Dict[str, list]
    glycan_trees: Dict[str, dict]
    glycan_trees_path: Optional[Path] = None


@dataclass
class SequonMappingResult:
    session: object
    chain_seqs: Dict[str, str]
    chain_order: List[str]
    glyco_sites: List[object]
    site_inventory_df: pd.DataFrame
    evidence_audit_df: pd.DataFrame
    remapped_annotations_df: pd.DataFrame
    conflict_df: pd.DataFrame
    assessment: dict
    tier_summary: dict
    glycan_trees: Dict[str, dict]
    dbref_ranges: Dict[str, list]
    pdb_to_mpnn_by_chain: Dict[str, Dict[int, int]]
    fetched_evidence_by_chain: Dict[str, Dict[int, str]]


@dataclass
class DesignWorkflowResult:
    constraints_by_condition: Dict[str, object]
    condition_manifest_df: pd.DataFrame
    retention_df: pd.DataFrame
    site_summary_df: pd.DataFrame
    condition_summary_df: pd.DataFrame
    top_designs_df: pd.DataFrame
    top_designs: Dict[str, object]
    af3_outputs: Dict[str, dict] = field(default_factory=dict)


def repo_root_from(start: Optional[Path] = None) -> Path:
    """Return the repository root for notebook use."""
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pipeline").is_dir() and (candidate / "sugarfix_helpers.py").exists():
            return candidate
    return start


def setup_tutorial_environment(
    repo_root: Optional[Path] = None,
    proteinmpnn_dir: Optional[Path] = None,
    auto_clone_proteinmpnn: bool = False,
) -> dict:
    """Prepare import paths for a SugarFix tutorial notebook."""
    repo_root = repo_root_from(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    candidates = [
        proteinmpnn_dir,
        Path(os.environ["PROTEINMPNN_DIR"]) if os.environ.get("PROTEINMPNN_DIR") else None,
        repo_root / "ProteinMPNN",
        repo_root.parent / "ProteinMPNN",
    ]
    found = None
    for candidate in candidates:
        if candidate and (Path(candidate) / "protein_mpnn_utils.py").exists():
            found = Path(candidate).resolve()
            break

    if found is None and auto_clone_proteinmpnn:
        found = repo_root / "ProteinMPNN"
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/dauparas/ProteinMPNN.git",
                str(found),
            ],
            check=True,
        )

    if found is not None:
        os.environ["PROTEINMPNN_DIR"] = str(found)
        if str(found) not in sys.path:
            sys.path.insert(0, str(found))

    return {"repo_root": repo_root, "proteinmpnn_dir": found}


def new_design_session(
    pdb_id: str,
    repo_root: Optional[Path] = None,
    run_label: str = "",
    num_seqs: int = 64,
    sampling_temp: float = 0.1,
    seed: int = 42,
):
    from sugarfix_helpers import DesignSession

    session = DesignSession(
        pdb_id=pdb_id.upper(),
        run_label=run_label,
        num_seqs=num_seqs,
        sampling_temp=sampling_temp,
        seed=seed,
        pipeline_root=repo_root_from(repo_root),
    )
    session.setup_paths()
    return session


def prepare_structure_for_tutorial(
    session,
    alphafold_uniprot: str = "",
) -> StructurePreparationResult:
    """Download/prepare the structure and extract resolved glycan trees."""
    from pipeline.extract_pdb_glycans import extract_glycan_trees
    from pipeline.prepare_structure import (
        create_protein_only_pdb,
        download_pdb,
        parse_missing_residues,
        parse_structure,
    )

    session.structure_dir.mkdir(parents=True, exist_ok=True)
    session.output_root.mkdir(parents=True, exist_ok=True)
    session.figure_dir.mkdir(parents=True, exist_ok=True)
    session.af3_dir.mkdir(parents=True, exist_ok=True)

    alphafold_uniprot = _clean_alphafold_accession(alphafold_uniprot)
    if alphafold_uniprot:
        session.pdb_id = f"AF-{alphafold_uniprot}-F1"
        session.structure_dir = session.data_dir / "prep" / session.pdb_id / "structure"
        session.output_root = session.data_dir / "outputs" / session.pdb_id
        session.run_dir = session.output_root / session.run_label if session.run_label else session.output_root
        session.figure_dir = session.run_dir / "figures"
        session.af3_dir = session.run_dir / "af3"
        session.structure_dir.mkdir(parents=True, exist_ok=True)
        session.figure_dir.mkdir(parents=True, exist_ok=True)
        session.af3_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = session.structure_dir / f"{session.pdb_id}.pdb"
    protein_pdb_path = session.structure_dir / f"{session.pdb_id}_protein.pdb"
    annotation_structure_path = pdb_path

    if alphafold_uniprot and not pdb_path.exists():
        _download_alphafold_model(alphafold_uniprot, pdb_path)
    elif not pdb_path.exists():
        if not download_pdb(session.pdb_id, pdb_path):
            raise FileNotFoundError(f"Could not download {session.pdb_id} from RCSB")

    mmcif_path = pdb_path.with_suffix(".cif")
    if mmcif_path.exists():
        annotation_structure_path = mmcif_path

    chains_info, n_models = parse_structure(pdb_path)
    missing = parse_missing_residues(pdb_path)
    create_protein_only_pdb(pdb_path, protein_pdb_path, chains_info)

    glycan_trees = extract_glycan_trees(annotation_structure_path)
    glycan_trees_path = None
    if glycan_trees:
        glycan_trees_path = session.structure_dir / "glycan_trees.json"
        with open(glycan_trees_path, "w") as handle:
            json.dump(glycan_trees, handle, indent=2)

    return StructurePreparationResult(
        pdb_path=pdb_path,
        protein_pdb_path=protein_pdb_path,
        annotation_structure_path=annotation_structure_path,
        chains_info=chains_info,
        n_models=n_models,
        missing_residues=missing,
        glycan_trees=glycan_trees,
        glycan_trees_path=glycan_trees_path,
    )


def run_sequon_mapping(
    session,
    protein_pdb_path: Path,
    annotation_structure_path: Path,
    glycan_trees: Optional[Dict[str, dict]] = None,
    candidates_csv: Optional[Path] = None,
    extra_sites: str = "",
) -> SequonMappingResult:
    """Identify N-X-S/T sites and annotate each with evidence/provenance."""
    from pipeline.identify_sequons import (
        annotate_evidence_tiers,
        build_pdb_resnum_to_mpnn_idx,
        extract_uniprot_accessions,
        fetch_uniprot_glycosylation,
        load_uniprot_evidence,
        parse_pdb_dbref,
    )
    from pipeline.mpnn_utils import (
        find_sequons,
        get_mpnn_chain_seqs_and_order,
        verify_sequon_positions,
    )
    from sugarfix_helpers import (
        build_glyco_site,
        snap_to_nearest_sequon,
        summarize_glycoprotein_status,
    )

    try:
        chain_seqs, chain_order = get_mpnn_chain_seqs_and_order(protein_pdb_path)
    except ImportError:
        chain_seqs, chain_order = _get_chain_seqs_and_order_biopython(protein_pdb_path)
    sequons_by_chain = {}
    for chain_id in chain_order:
        sequons_by_chain[chain_id] = [
            {"position_0idx": record["position_0idx"], "sequon": record["sequon"]}
            for record in find_sequons(chain_seqs[chain_id])
        ]
    verify_sequon_positions(chain_seqs, sequons_by_chain, session.pdb_id)

    glycan_trees = glycan_trees or {}
    uniprot_evidence = load_uniprot_evidence(candidates_csv, session.pdb_id)
    dbref_ranges = parse_pdb_dbref(annotation_structure_path)
    chain_accessions = extract_uniprot_accessions(annotation_structure_path)

    fetched_evidence_by_chain = {}
    for chain_id, accession in chain_accessions.items():
        fetched = fetch_uniprot_glycosylation(accession)
        if not fetched:
            continue
        fetched_evidence_by_chain[chain_id] = fetched
        _merge_evidence(uniprot_evidence, fetched)

    tier_summary = annotate_evidence_tiers(
        sequons_by_chain,
        chain_seqs,
        annotation_structure_path,
        uniprot_evidence,
        glycan_trees,
    )

    pdb_to_mpnn_by_chain = {
        chain_id: build_pdb_resnum_to_mpnn_idx(protein_pdb_path, chain_id)
        for chain_id in chain_order
    }

    glyco_sites = []
    for chain_id in chain_order:
        for record in sequons_by_chain[chain_id]:
            site = build_glyco_site(
                chain_id,
                record["position_0idx"],
                chain_seqs,
                pdb_to_mpnn_by_chain,
                dbref_ranges,
                uniprot_evidence,
                glycan_trees,
                source="motif",
            )
            if site:
                glyco_sites.append(site)

    _add_uniprot_only_sites(
        glyco_sites,
        fetched_evidence_by_chain,
        chain_order,
        chain_seqs,
        pdb_to_mpnn_by_chain,
        dbref_ranges,
        uniprot_evidence,
        glycan_trees,
        snap_to_nearest_sequon,
        build_glyco_site,
    )
    _add_manual_sites(
        glyco_sites,
        extra_sites,
        chain_order,
        chain_seqs,
        pdb_to_mpnn_by_chain,
        dbref_ranges,
        uniprot_evidence,
        glycan_trees,
        build_glyco_site,
    )

    glyco_sites = sorted(glyco_sites, key=lambda site: (site.chain, site.position_0idx))
    session.glyco_sites = glyco_sites
    session.chain_seqs = chain_seqs
    session.chain_order = chain_order
    session.assessment = summarize_glycoprotein_status(glyco_sites)

    site_inventory_df = _site_inventory_df(glyco_sites)
    evidence_audit_df = _evidence_audit_df(glyco_sites)
    remapped_annotations_df = _remapped_annotations_df(glyco_sites, chain_accessions)
    conflict_df = _conflict_df(glyco_sites, dbref_ranges, chain_accessions)

    return SequonMappingResult(
        session=session,
        chain_seqs=chain_seqs,
        chain_order=chain_order,
        glyco_sites=glyco_sites,
        site_inventory_df=site_inventory_df,
        evidence_audit_df=evidence_audit_df,
        remapped_annotations_df=remapped_annotations_df,
        conflict_df=conflict_df,
        assessment=session.assessment,
        tier_summary=tier_summary,
        glycan_trees=glycan_trees,
        dbref_ranges=dbref_ranges,
        pdb_to_mpnn_by_chain=pdb_to_mpnn_by_chain,
        fetched_evidence_by_chain=fetched_evidence_by_chain,
    )


def write_tutorial01_outputs(
    result: SequonMappingResult,
    output_dir: Optional[Path] = None,
    structure_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Write the tutorial 01 tables/report/PyMOL annotation script."""
    output_dir = Path(output_dir or result.session.run_dir / "tutorial_01_sequon_mapping")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "sequon_table": output_dir / f"{result.session.pdb_id}_sequon_table.csv",
        "evidence_audit": output_dir / f"{result.session.pdb_id}_evidence_audit.csv",
        "remapped_annotations": output_dir / f"{result.session.pdb_id}_remapped_annotations.csv",
        "conflicts": output_dir / f"{result.session.pdb_id}_numbering_conflicts.csv",
        "confidence_report": output_dir / f"{result.session.pdb_id}_confidence_report.html",
        "pymol_annotation": output_dir / f"{result.session.pdb_id}_sequons.pml",
    }

    result.site_inventory_df.to_csv(paths["sequon_table"], index=False)
    result.evidence_audit_df.to_csv(paths["evidence_audit"], index=False)
    result.remapped_annotations_df.to_csv(paths["remapped_annotations"], index=False)
    result.conflict_df.to_csv(paths["conflicts"], index=False)
    paths["confidence_report"].write_text(_confidence_report_html(result))
    paths["pymol_annotation"].write_text(
        _pymol_annotation_script(result.glyco_sites, structure_path)
    )
    return paths


def choose_preservation_strategy(
    session,
    strategy: str = "evidence_aware",
    target_role: Optional[str] = None,
    run_baselines: bool = True,
) -> pd.DataFrame:
    """Apply a design-preservation strategy without notebook-side branching."""
    from sugarfix_helpers import build_site_policy_map

    target_role = target_role or session.assessment.get("recommended_role", "glycoprotein")
    site_policy_map = build_site_policy_map(session.glyco_sites, strategy, interactive=False)
    session.apply_decisions(target_role, strategy, run_baselines, site_policy_map)
    return session.decision_df


def build_design_conditions(session) -> Dict[str, object]:
    """Build SugarFix and optional baseline ProteinMPNN constraint bundles."""
    from sugarfix_helpers import build_constraint_bundle

    constraints_by_condition = {}
    if session.selected_site_policies:
        constraints_by_condition["designer_selected"] = build_constraint_bundle(
            session.glyco_sites,
            session.selected_site_policies,
            "designer_selected",
            "User-selected/evidence-aware glycosite preservation plan",
        )
    if session.run_baselines and session.glyco_sites:
        soft_filter_map = {site.label: "soft_filter" for site in session.glyco_sites}
        constraints_by_condition["soft_filter"] = build_constraint_bundle(
            session.glyco_sites,
            soft_filter_map,
            "soft_filter",
            "ProteinMPNN baseline without hard sequon fixing",
        )
    return constraints_by_condition


def run_proteinmpnn_designs(
    session,
    protein_pdb_path: Path,
    proteinmpnn_dir: Optional[Path] = None,
    constraints_by_condition: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Run ProteinMPNN for every requested design condition."""
    from pipeline.mpnn_utils import write_fixed_positions_jsonl
    from sugarfix_helpers import build_condition_manifest

    proteinmpnn_dir = _resolve_proteinmpnn_dir(proteinmpnn_dir)
    run_script = proteinmpnn_dir / "protein_mpnn_run.py"
    constraints_by_condition = constraints_by_condition or build_design_conditions(session)
    session.run_dir.mkdir(parents=True, exist_ok=True)

    pdb_label = Path(protein_pdb_path).stem
    for condition_name, bundle in constraints_by_condition.items():
        condition_dir = session.run_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)
        working_pdb = session.run_dir / f"{pdb_label}.pdb"
        if not working_pdb.exists():
            shutil.copy(protein_pdb_path, working_pdb)

        fixed_jsonl = None
        if bundle.fixed_positions_0idx:
            fixed_jsonl = session.run_dir / f"fixed_positions_{condition_name}.jsonl"
            write_fixed_positions_jsonl(
                pdb_label,
                bundle.fixed_positions_0idx,
                session.chain_order,
                fixed_jsonl,
            )

        cmd = [
            sys.executable,
            str(run_script),
            "--pdb_path",
            str(working_pdb),
            "--out_folder",
            str(condition_dir),
            "--num_seq_per_target",
            str(session.num_seqs),
            "--sampling_temp",
            str(session.sampling_temp),
            "--seed",
            str(session.seed),
            "--pdb_path_chains",
            " ".join(session.chain_order),
        ]
        if fixed_jsonl is not None:
            cmd.extend(["--fixed_positions_jsonl", str(fixed_jsonl)])
        subprocess.run(cmd, cwd=str(proteinmpnn_dir), check=True)

    return build_condition_manifest(session.glyco_sites, constraints_by_condition)


def score_design_workflow(
    session,
    constraints_by_condition: Optional[Dict[str, object]] = None,
    condition_manifest_df: Optional[pd.DataFrame] = None,
) -> DesignWorkflowResult:
    """Score ProteinMPNN FASTAs against the selected glycosite plan."""
    from pipeline.mpnn_utils import (
        is_functional_sequon,
        read_fasta_sequences,
        split_mpnn_concat_seq,
    )
    from sugarfix_helpers import (
        build_condition_manifest,
        score_designs,
        select_top_design,
    )

    constraints_by_condition = constraints_by_condition or build_design_conditions(session)
    condition_manifest_df = condition_manifest_df if condition_manifest_df is not None else (
        build_condition_manifest(session.glyco_sites, constraints_by_condition)
    )

    all_results = []
    top_designs = {}
    pdb_label = f"{session.pdb_id}_protein"
    for condition_name in constraints_by_condition:
        fasta_path = session.run_dir / condition_name / "seqs" / f"{pdb_label}.fa"
        if not fasta_path.exists():
            fasta_candidates = sorted((session.run_dir / condition_name / "seqs").glob("*.fa"))
            if not fasta_candidates:
                top_designs[condition_name] = None
                continue
            fasta_path = fasta_candidates[0]
        results = score_designs(
            fasta_path,
            session.glyco_sites,
            session.chain_order,
            constraints_by_condition[condition_name].site_policies,
            condition_name,
            read_fasta_sequences,
            split_mpnn_concat_seq,
            is_functional_sequon,
        )
        all_results.extend(results)
        top_designs[condition_name] = select_top_design(results)

    retention_rows = []
    for result in all_results:
        for status in result.site_statuses:
            retention_rows.append(
                {
                    "design_condition": result.design_condition,
                    "design_id": result.design_id,
                    "sample_idx": result.sample_idx,
                    "mpnn_score": result.mpnn_score,
                    "site_label": status.site.label,
                    "chain": status.site.chain,
                    "position_1idx": status.site.position_1idx,
                    "pdb_resnum": status.site.pdb_resnum,
                    "wt_motif": status.site.motif,
                    "design_triplet": status.design_triplet,
                    "evidence_tier": status.site.evidence_tier,
                    "selected_policy": status.selected_policy,
                    "n_retained": status.n_retained,
                    "exact_match": status.exact_match,
                    "functional": status.functional,
                    "meets_selected_policy": status.meets_selected_policy,
                }
            )
    retention_df = pd.DataFrame(retention_rows)
    site_summary_df, condition_summary_df, top_designs_df = _summarise_design_tables(
        retention_df,
        all_results,
        top_designs,
    )
    return DesignWorkflowResult(
        constraints_by_condition=constraints_by_condition,
        condition_manifest_df=condition_manifest_df,
        retention_df=retention_df,
        site_summary_df=site_summary_df,
        condition_summary_df=condition_summary_df,
        top_designs_df=top_designs_df,
        top_designs=top_designs,
    )


def score_and_export_designs(
    session,
    constraints_by_condition: Optional[Dict[str, object]] = None,
    condition_manifest_df: Optional[pd.DataFrame] = None,
    af3_export_mode: str = "alphafold3",
    model_seeds: Optional[List[int]] = None,
) -> DesignWorkflowResult:
    """Score ProteinMPNN FASTAs and export AF3 JSONs in one call."""
    design = score_design_workflow(
        session,
        constraints_by_condition=constraints_by_condition,
        condition_manifest_df=condition_manifest_df,
    )
    design.af3_outputs = export_top_designs_for_af3(
        session,
        design.top_designs,
        mode=af3_export_mode,
        model_seeds=model_seeds,
    )
    return design


def export_top_designs_for_af3(
    session,
    top_designs: Dict[str, object],
    mode: str = "alphafold3",
    model_seeds: Optional[List[int]] = None,
) -> Dict[str, dict]:
    """Export AF3 JSON inputs for top designs.

    ``mode`` may be ``"alphafold3"``, ``"alphafoldserver"``, or ``"both"``.
    Standalone AF3 exports include ligand entities and covalent bonds for
    glycans; server exports intentionally keep single-CCD glycan stubs.
    """
    from pipeline.af3_json import make_af3_full_json, make_af3_server_json, write_json

    if mode not in {"alphafold3", "alphafoldserver", "both"}:
        raise ValueError("mode must be 'alphafold3', 'alphafoldserver', or 'both'")

    session.af3_dir.mkdir(parents=True, exist_ok=True)
    model_seeds = model_seeds or [session.seed]
    outputs = {}

    for condition_name, top_design in top_designs.items():
        if top_design is None:
            continue
        chain_sequences = [top_design.chain_sequences[cid] for cid in session.chain_order]
        job_name = f"{session.pdb_id}_{condition_name}"
        glycan_positions = _glycan_positions_for_design(session, top_design)
        outputs[condition_name] = {
            "n_glycans": int(sum(len(records) for records in glycan_positions.values()))
        }

        if mode in {"alphafold3", "both"}:
            plain_path = write_json(
                make_af3_full_json(
                    job_name,
                    chain_sequences,
                    chain_ids=session.chain_order,
                    model_seeds=model_seeds,
                ),
                session.af3_dir / f"{job_name}_AF3_full.json",
            )
            glycan_path = write_json(
                make_af3_full_json(
                    f"{job_name}_glycans",
                    chain_sequences,
                    glycan_positions,
                    chain_ids=session.chain_order,
                    model_seeds=model_seeds,
                ),
                session.af3_dir / f"{job_name}_AF3_full_with_glycans.json",
            )
            outputs[condition_name].update(
                {
                    "standalone_plain": str(plain_path),
                    "standalone_glycans": str(glycan_path),
                }
            )

        if mode in {"alphafoldserver", "both"}:
            plain_path = write_json(
                make_af3_server_json(job_name, chain_sequences),
                session.af3_dir / f"{job_name}_AF3_server.json",
            )
            glycan_path = write_json(
                make_af3_server_json(
                    f"{job_name}_glycan_stubs",
                    chain_sequences,
                    glycan_positions,
                ),
                session.af3_dir / f"{job_name}_AF3_server_glycan_stubs.json",
            )
            outputs[condition_name].update(
                {
                    "server_plain": str(plain_path),
                    "server_glycan_stubs": str(glycan_path),
                }
            )

    return outputs


def write_design_outputs(
    session,
    mapping: SequonMappingResult,
    design: DesignWorkflowResult,
) -> Dict[str, Path]:
    """Persist the core Tutorial 04 output tables."""
    paths = {
        "site_inventory": session.run_dir / f"{session.pdb_id}_site_inventory.csv",
        "evidence_audit": session.run_dir / f"{session.pdb_id}_evidence_audit.csv",
        "decisions": session.run_dir / f"{session.pdb_id}_site_decisions.csv",
        "condition_manifest": session.run_dir / f"{session.pdb_id}_condition_manifest.csv",
        "retention": session.run_dir / f"{session.pdb_id}_retention.csv",
        "site_summary": session.run_dir / f"{session.pdb_id}_site_summary.csv",
        "condition_summary": session.run_dir / f"{session.pdb_id}_condition_summary.csv",
        "top_designs": session.run_dir / f"{session.pdb_id}_top_designs.csv",
        "session": session.run_dir / f"{session.pdb_id}_designer_session.json",
    }
    mapping.site_inventory_df.to_csv(paths["site_inventory"], index=False)
    mapping.evidence_audit_df.to_csv(paths["evidence_audit"], index=False)
    session.decision_df.to_csv(paths["decisions"], index=False)
    design.condition_manifest_df.to_csv(paths["condition_manifest"], index=False)
    design.retention_df.to_csv(paths["retention"], index=False)
    design.site_summary_df.to_csv(paths["site_summary"], index=False)
    design.condition_summary_df.to_csv(paths["condition_summary"], index=False)
    design.top_designs_df.to_csv(paths["top_designs"], index=False)

    session_payload = {
        **session.decision_payload,
        "assessment": session.assessment,
        "af3_outputs": design.af3_outputs,
    }
    paths["session"].write_text(json.dumps(session_payload, indent=2))
    return paths


def _clean_alphafold_accession(accession: str) -> str:
    accession = (accession or "").strip().upper()
    if accession.startswith("AF-"):
        accession = accession[3:]
    if accession.endswith("-F1"):
        accession = accession[:-3]
    return accession


def _download_alphafold_model(accession: str, output_path: Path) -> None:
    import json as _json
    import urllib.request

    meta_url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    with urllib.request.urlopen(meta_url, timeout=30) as response:
        metadata = _json.loads(response.read())
    if not metadata:
        raise FileNotFoundError(f"No AlphaFold DB prediction found for {accession}")
    urllib.request.urlretrieve(metadata[0]["pdbUrl"], str(output_path))


def _merge_evidence(target: Dict[int, str], source: Dict[int, str]) -> None:
    rank = {"experimental": 3, "pdb_evidence": 2, "curator_inferred": 1, "motif_only": 0}
    for pos, tier in source.items():
        if rank.get(tier, 0) > rank.get(target.get(pos), 0):
            target[pos] = tier


def _add_uniprot_only_sites(
    glyco_sites: List[object],
    fetched_evidence_by_chain: Dict[str, Dict[int, str]],
    chain_order: List[str],
    chain_seqs: Dict[str, str],
    pdb_to_mpnn_by_chain: Dict[str, Dict[int, int]],
    dbref_ranges: Dict[str, list],
    uniprot_evidence: Dict[int, str],
    glycan_trees: Dict[str, dict],
    snap_to_nearest_sequon,
    build_glyco_site,
) -> None:
    seen = {(site.chain, site.position_0idx) for site in glyco_sites}
    for chain_id, fetched in fetched_evidence_by_chain.items():
        if chain_id not in chain_order:
            continue
        pdb_to_mpnn = pdb_to_mpnn_by_chain.get(chain_id, {})
        for unp_pos, tier in fetched.items():
            pdb_resnum = None
            for rng in dbref_ranges.get(chain_id, []):
                if rng["unp_start"] <= unp_pos <= rng["unp_end"]:
                    pdb_resnum = rng["pdb_start"] + (unp_pos - rng["unp_start"])
                    break
            if pdb_resnum is None:
                continue
            mpnn_idx = pdb_to_mpnn.get(pdb_resnum)
            if mpnn_idx is None:
                continue
            if not _is_sequon_like(chain_seqs[chain_id], mpnn_idx):
                snapped = snap_to_nearest_sequon(chain_seqs[chain_id], mpnn_idx, window=50)
                if snapped is None:
                    continue
                mpnn_idx = snapped
            if (chain_id, mpnn_idx) in seen:
                continue
            site = build_glyco_site(
                chain_id,
                mpnn_idx,
                chain_seqs,
                pdb_to_mpnn_by_chain,
                dbref_ranges,
                uniprot_evidence,
                glycan_trees,
                source="uniprot_only",
            )
            if site:
                site.uniprot_position = unp_pos
                site.uniprot_tier = tier
                site.evidence_tier = tier
                site.evidence_ok = True
                site.evidence_reasons = [
                    f"UniProt {tier} at position {unp_pos}",
                    "recovered from UniProt annotation",
                ]
                glyco_sites.append(site)
                seen.add((chain_id, mpnn_idx))


def _add_manual_sites(
    glyco_sites: List[object],
    extra_sites: str,
    chain_order: List[str],
    chain_seqs: Dict[str, str],
    pdb_to_mpnn_by_chain: Dict[str, Dict[int, int]],
    dbref_ranges: Dict[str, list],
    uniprot_evidence: Dict[int, str],
    glycan_trees: Dict[str, dict],
    build_glyco_site,
) -> None:
    if not extra_sites.strip():
        return
    seen = {(site.chain, site.position_0idx) for site in glyco_sites}
    for raw_item in extra_sites.split(","):
        item = raw_item.strip()
        if not item or ":" not in item:
            continue
        chain, pos_str = item.split(":", 1)
        chain = chain.strip()
        pos_str = pos_str.strip()
        if chain not in chain_order:
            continue
        if pos_str.upper().startswith("M"):
            mpnn_idx = int(pos_str[1:]) - 1
        else:
            pdb_resnum = int(pos_str)
            mpnn_idx = pdb_to_mpnn_by_chain.get(chain, {}).get(pdb_resnum)
            if mpnn_idx is None:
                continue
        if (chain, mpnn_idx) in seen:
            continue
        site = build_glyco_site(
            chain,
            mpnn_idx,
            chain_seqs,
            pdb_to_mpnn_by_chain,
            dbref_ranges,
            uniprot_evidence,
            glycan_trees,
            source="user_specified",
        )
        if site:
            glyco_sites.append(site)
            seen.add((chain, mpnn_idx))


def _is_sequon_like(sequence: str, mpnn_idx: int) -> bool:
    triplet = sequence[mpnn_idx:mpnn_idx + 3]
    return len(triplet) == 3 and triplet[0] == "N" and triplet[1] != "P" and triplet[2] in "ST"


def _site_inventory_df(glyco_sites: List[object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "site_label": site.label,
                "chain": site.chain,
                "position_1idx": site.position_1idx,
                "pdb_resnum": site.pdb_resnum,
                "uniprot_position": site.uniprot_position,
                "motif": site.motif,
                "evidence_tier": site.evidence_tier,
                "resolved_glycan": site.glycan_tree is not None,
                "default_policy": site.default_policy,
                "why_this_site_matters": "; ".join(site.evidence_reasons),
            }
            for site in glyco_sites
        ]
    )


def _evidence_audit_df(glyco_sites: List[object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "site_label": site.label,
                "assigned_tier": site.evidence_tier,
                "expected_tier": site.expected_tier,
                "tier_ok": site.evidence_ok,
                "uniprot_position": site.uniprot_position,
                "uniprot_tier": site.uniprot_tier,
                "has_glycan_tree": site.glycan_tree is not None,
            }
            for site in glyco_sites
        ]
    )


def _remapped_annotations_df(
    glyco_sites: List[object],
    chain_accessions: Dict[str, str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "site_label": site.label,
                "chain": site.chain,
                "uniprot_accession": chain_accessions.get(site.chain, ""),
                "uniprot_position": site.uniprot_position,
                "pdb_resnum": site.pdb_resnum,
                "mpnn_position_1idx": site.position_1idx,
                "motif": site.motif,
                "evidence_tier": site.evidence_tier,
                "resolved_glycan": site.glycan_tree is not None,
            }
            for site in glyco_sites
            if site.uniprot_position is not None or chain_accessions.get(site.chain)
        ]
    )


def _conflict_df(
    glyco_sites: List[object],
    dbref_ranges: Dict[str, list],
    chain_accessions: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    for site in glyco_sites:
        has_mapping = site.uniprot_position is not None
        rows.append(
            {
                "site_label": site.label,
                "chain": site.chain,
                "pdb_resnum": site.pdb_resnum,
                "uniprot_accession": chain_accessions.get(site.chain, ""),
                "uniprot_position": site.uniprot_position,
                "mapping_status": "mapped" if has_mapping else "unmapped",
                "low_alignment_confidence": not has_mapping and bool(dbref_ranges),
                "note": ""
                if has_mapping
                else "No DBREF/mmCIF mapping covered this resolved residue",
            }
        )
    return pd.DataFrame(rows)


def _confidence_report_html(result: SequonMappingResult) -> str:
    inventory = result.site_inventory_df.to_html(index=False, escape=False)
    conflicts = result.conflict_df.to_html(index=False, escape=False)
    assessment = result.assessment
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{result.session.pdb_id} glycosylation confidence report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.45rem; font-size: 0.9rem; }}
    th {{ background: #f4f4f4; text-align: left; }}
    .metric {{ display: inline-block; margin-right: 1.5rem; }}
  </style>
</head>
<body>
  <h1>{result.session.pdb_id} glycosylation confidence report</h1>
  <p>{assessment.get("headline", "")}</p>
  <p>
    <span class="metric"><b>Sites:</b> {assessment.get("n_sites", 0)}</span>
    <span class="metric"><b>Validated:</b> {assessment.get("validated_sites", 0)}</span>
    <span class="metric"><b>Resolved glycans:</b> {assessment.get("glycan_tree_sites", 0)}</span>
  </p>
  <h2>Site inventory</h2>
  {inventory}
  <h2>Numbering and mapping checks</h2>
  {conflicts}
</body>
</html>
"""


def _pymol_annotation_script(glyco_sites: List[object], structure_path: Optional[Path]) -> str:
    lines = [
        "reinitialize",
    ]
    if structure_path:
        lines.append(f"load {Path(structure_path).as_posix()}, glycoprotein")
    lines.extend(
        [
            "hide everything",
            "show cartoon, polymer.protein",
            "color slate, polymer.protein",
        ]
    )
    for site in glyco_sites:
        if site.pdb_resnum is None:
            continue
        selection = f"chain {site.chain} and resi {site.pdb_resnum}"
        name = f"sequon_{site.chain}_{site.pdb_resnum}"
        color = {
            "experimental": "green",
            "pdb_evidence": "cyan",
            "curator_inferred": "yellow",
            "motif_only": "gray70",
        }.get(site.evidence_tier, "gray70")
        lines.extend(
            [
                f"select {name}, {selection}",
                f"show sticks, {name}",
                f"color {color}, {name}",
                f"label {name} and name CA, \"{site.label} {site.evidence_tier}\"",
            ]
        )
    lines.append("zoom polymer.protein")
    return "\n".join(lines) + "\n"


def _resolve_proteinmpnn_dir(proteinmpnn_dir: Optional[Path] = None) -> Path:
    candidates = [
        proteinmpnn_dir,
        Path(os.environ["PROTEINMPNN_DIR"]) if os.environ.get("PROTEINMPNN_DIR") else None,
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "protein_mpnn_run.py").exists():
            return Path(candidate).resolve()
    raise FileNotFoundError("ProteinMPNN not found; run setup_tutorial_environment first")


def _get_chain_seqs_and_order_biopython(pdb_path: Path):
    from Bio.PDB import MMCIFParser, PDBParser, is_aa
    from Bio.SeqUtils import seq1

    pdb_path = Path(pdb_path)
    if pdb_path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    model = next(structure.get_models())

    chain_seqs = {}
    chain_order = []
    for chain in model:
        residues = [residue for residue in chain.get_residues() if is_aa(residue)]
        if not residues:
            continue
        chain_order.append(chain.id)
        chain_seqs[chain.id] = "".join(seq1(residue.get_resname()) for residue in residues)
    return chain_seqs, chain_order


def _summarise_design_tables(retention_df, all_results, top_designs):
    if retention_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    site_summary_df = (
        retention_df
        .groupby(["design_condition", "site_label", "wt_motif", "evidence_tier", "selected_policy"])
        .agg(
            n_designs=("design_id", "nunique"),
            functional_retention=("functional", "mean"),
            exact_retention=("exact_match", "mean"),
            selected_policy_satisfaction=("meets_selected_policy", "mean"),
        )
        .reset_index()
    )

    condition_summary_df = (
        retention_df
        .groupby("design_condition")
        .agg(
            mean_functional_retention=("functional", "mean"),
            mean_exact_retention=("exact_match", "mean"),
            mean_selected_policy_satisfaction=("meets_selected_policy", "mean"),
            mean_mpnn_score=("mpnn_score", "mean"),
        )
        .reset_index()
    )

    top_rows = []
    for condition_name, top_design in top_designs.items():
        if top_design is None:
            continue
        top_rows.append(
            {
                "design_condition": condition_name,
                "top_design_id": top_design.design_id,
                "mpnn_score": top_design.mpnn_score,
                "sites_satisfied": top_design.n_sites_satisfied,
                "required_sites": top_design.n_sites_required,
                "plan_satisfaction_rate": top_design.plan_satisfaction_rate,
            }
        )
    top_designs_df = pd.DataFrame(top_rows)
    return site_summary_df, condition_summary_df, top_designs_df


def _glycan_positions_for_design(session, top_design) -> Dict[int, List[dict]]:
    glycan_positions = {}
    for status in top_design.site_statuses:
        if status.selected_policy not in ("full_sequon", "functional_preserve"):
            continue
        chain_idx = session.chain_order.index(status.site.chain)
        record = {"residues": "NAG", "position": status.site.position_1idx}
        if status.site.glycan_tree:
            record.update(status.site.glycan_tree)
            record["position"] = status.site.position_1idx
        glycan_positions.setdefault(chain_idx, []).append(record)
    return glycan_positions
