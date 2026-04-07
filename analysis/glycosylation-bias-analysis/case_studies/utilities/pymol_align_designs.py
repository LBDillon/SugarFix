"""
PyMOL script: Align all loaded AF3 design structures against a reference crystal structure.

Usage (inside PyMOL):
  1. Load your reference crystal structure (e.g., fetch a PDB and name it REF_OBJ)
  2. Load all AF3 predicted .cif files
  3. Run this script:
       run /path/to/pymol_align_designs.py
  4. Call the function:
       align_designs REF_OBJ                              # auto-detect single vs multi-chain
       align_designs REF_OBJ, chain=each                  # align each chain independently
       align_designs REF_OBJ, chain=all                   # force full-structure alignment
       align_designs REF_OBJ, chain=A                     # force chain A only
       align_designs REF_OBJ, chain=each, log=results     # save to results.txt + results.csv

For multi-chain proteins, the script automatically aligns only the first chain
to avoid quaternary arrangement artifacts inflating RMSD.
Use chain=each to iterate over all chains and get per-chain RMSDs.

When chain letters differ between reference and target (e.g., reference H,I,J,K,L,M
vs AF3 model A,B,C,D,E,F), chains are matched by positional order.

The log parameter saves output to:
  - <log>.txt  -- Human-readable table (same as console output)
  - <log>.csv  -- Machine-readable CSV (structure, ref_chain, target_chain, rmsd, n_atoms)
"""

import os
from pymol import cmd, stored


def _get_chains(obj_name):
    """Get sorted list of unique chain IDs for an object."""
    stored.chains = []
    cmd.iterate(f"{obj_name} and name CA", "stored.chains.append(chain)")
    return sorted(set(stored.chains))


def _map_chain(ref_chain, ref_chains, obj_chains):
    """Map a reference chain to the corresponding target chain.

    If the target has the same chain letter, use it directly.
    Otherwise, map by positional index (1st->1st, 2nd->2nd, etc.).
    Returns the target chain letter, or None if no mapping possible.
    """
    if ref_chain in obj_chains:
        return ref_chain
    # Map by position
    if ref_chain in ref_chains:
        idx = ref_chains.index(ref_chain)
        if idx < len(obj_chains):
            return obj_chains[idx]
    return None


def _align_one(mob_sel, ref_sel):
    """Run alignment and return (rmsd, n_atoms) or (None, error_str)."""
    try:
        result = cmd.align(mob_sel, ref_sel, cycles=5, cutoff=2.0)
        return result[0], result[1]
    except Exception as e:
        return None, str(e)


class _Logger:
    """Captures output to both console and optionally a file."""

    def __init__(self, log_path=None):
        self.log_path = log_path
        self.lines = []
        self.csv_rows = []  # (structure, ref_chain, target_chain, rmsd, n_atoms)

    def log(self, text=""):
        print(text)
        self.lines.append(text)

    def add_csv_row(self, structure, ref_chain, target_chain, rmsd, n_atoms):
        self.csv_rows.append((structure, ref_chain, target_chain, rmsd, n_atoms))

    def save(self):
        if not self.log_path:
            return
        base = self.log_path.rstrip(".txt").rstrip(".csv")
        txt_path = base + ".txt"
        csv_path = base + ".csv"

        with open(txt_path, "w") as f:
            f.write("\n".join(self.lines) + "\n")
        print(f"\n  Saved text log: {txt_path}")

        with open(csv_path, "w") as f:
            f.write("structure,ref_chain,target_chain,rmsd,n_atoms\n")
            for row in self.csv_rows:
                struct, rc, tc, rmsd, natoms = row
                rmsd_str = f"{rmsd:.4f}" if rmsd is not None else "NA"
                natoms_str = str(natoms) if isinstance(natoms, int) else "NA"
                f.write(f"{struct},{rc},{tc},{rmsd_str},{natoms_str}\n")
        print(f"  Saved CSV:      {csv_path}")


def align_all_to_ref(ref_name, chain="auto", log=""):
    """Align all loaded objects to a reference structure.

    Args:
        ref_name: Name of the reference object (crystal structure).
        chain:    "auto" = detect multi-chain and align first chain only if needed.
                  "all"  = align full structure (all chains).
                  "each" = align each chain independently, report per-chain RMSDs.
                  "A"    = force chain A only alignment.
                  Any other chain letter to force that chain.
        log:      Base filename for saving output (e.g. "results" -> results.txt + results.csv).
                  Leave empty to only print to console.
    """
    # Strip any surrounding quotes PyMOL may pass through
    ref_name = ref_name.strip("'\"")
    chain = chain.strip("'\"")
    log = log.strip("'\" ")

    out = _Logger(log if log else None)

    ref_chains = _get_chains(ref_name)
    n_chains = len(ref_chains)

    # Get all objects except the reference
    all_objects = cmd.get_object_list()
    targets = sorted(obj for obj in all_objects if obj != ref_name)

    if not targets:
        out.log("  No other objects loaded to align.")
        out.save()
        return

    # --- "each" mode: iterate over all chains ---
    if chain == "each":
        out.log(f"\n  Reference '{ref_name}' has {n_chains} chains: {ref_chains}")
        out.log(f"  -> Aligning each chain independently\n")

        # Show chain mapping for first target so user can verify
        first_obj_chains = _get_chains(targets[0])
        if first_obj_chains != ref_chains:
            out.log(f"  Chain mapping (by position):")
            for i, rc in enumerate(ref_chains):
                tc = first_obj_chains[i] if i < len(first_obj_chains) else "?"
                out.log(f"    ref {rc} -> target {tc}")
            out.log("")

        for i, ch in enumerate(ref_chains):
            ref_sel = f"{ref_name} and chain {ch}"
            results = []

            for obj in targets:
                obj_chains = _get_chains(obj)
                mapped = _map_chain(ch, ref_chains, obj_chains)

                if mapped is None:
                    results.append((obj, None, "no matching chain"))
                    out.add_csv_row(obj, ch, "NA", None, "NA")
                    continue

                mob_sel = f"{obj} and chain {mapped}"
                rmsd, n_atoms = _align_one(mob_sel, ref_sel)
                results.append((obj, rmsd, n_atoms))
                out.add_csv_row(obj, ch, mapped, rmsd, n_atoms if isinstance(n_atoms, int) else None)

            mapped_label = _map_chain(ch, ref_chains, _get_chains(targets[0])) if targets else ch
            out.log(f"  Chain {ch} (ref) -> chain {mapped_label} (target):" if mapped_label else f"  Chain {ch}:")
            out.log(f"    {'Structure':<58} {'RMSD (A)':>10} {'Atoms':>8}")
            out.log(f"    {'-' * 78}")
            for obj, rmsd, n_atoms in sorted(results, key=lambda x: x[1] if x[1] is not None else 999):
                if rmsd is not None:
                    out.log(f"    {obj:<58} {rmsd:>10.3f} {n_atoms:>8}")
                else:
                    out.log(f"    {obj:<58} {'FAILED':>10} {n_atoms}")
            out.log("")

        # Summary: average RMSD across chains per target
        out.log(f"  Average RMSD across all {n_chains} chains:")
        out.log(f"    {'Structure':<58} {'Mean RMSD':>10}")
        out.log(f"    {'-' * 70}")

        for obj in targets:
            obj_chains = _get_chains(obj)
            rmsds = []
            for ch in ref_chains:
                ref_sel = f"{ref_name} and chain {ch}"
                mapped = _map_chain(ch, ref_chains, obj_chains)
                if mapped is None:
                    continue
                mob_sel = f"{obj} and chain {mapped}"
                rmsd, _ = _align_one(mob_sel, ref_sel)
                if rmsd is not None:
                    rmsds.append(rmsd)

            if rmsds:
                mean_rmsd = sum(rmsds) / len(rmsds)
                out.log(f"    {obj:<58} {mean_rmsd:>10.3f}")
                out.add_csv_row(obj, "mean", "mean", mean_rmsd, None)
            else:
                out.log(f"    {obj:<58} {'FAILED':>10}")
        out.log("")
        out.save()
        return

    # --- Single-chain or all-chain mode ---
    if chain == "auto":
        if n_chains > 1:
            align_chain = ref_chains[0]
            out.log(f"\n  Reference '{ref_name}' has {n_chains} chains: {ref_chains}")
            out.log(f"  -> Using chain {align_chain} for alignment (avoids quaternary RMSD artifacts)")
            out.log(f"     Tip: use chain=each to see per-chain RMSDs\n")
        else:
            align_chain = None
            out.log(f"\n  Reference '{ref_name}' is single-chain -> aligning full structure\n")
    elif chain == "all":
        align_chain = None
    else:
        align_chain = chain

    if align_chain:
        ref_sel = f"{ref_name} and chain {align_chain}"
    else:
        ref_sel = ref_name

    results = []
    for obj in targets:
        if align_chain:
            obj_chains = _get_chains(obj)
            mapped = _map_chain(align_chain, ref_chains, obj_chains)
            if mapped:
                mob_sel = f"{obj} and chain {mapped}"
            else:
                mob_sel = obj
                mapped = "all"
        else:
            mob_sel = obj
            mapped = "all"

        rmsd, n_atoms = _align_one(mob_sel, ref_sel)
        results.append((obj, rmsd, n_atoms))
        out.add_csv_row(obj, align_chain or "all", mapped, rmsd, n_atoms if isinstance(n_atoms, int) else None)

    out.log("  " + "-" * 80)
    out.log(f"  {'Structure':<60} {'RMSD (A)':>10} {'Atoms':>8}")
    out.log("  " + "-" * 80)

    for obj, rmsd, n_atoms in sorted(results, key=lambda x: x[1] if x[1] is not None else 999):
        if rmsd is not None:
            out.log(f"  {obj:<60} {rmsd:>10.3f} {n_atoms:>8}")
        else:
            out.log(f"  {obj:<60} {'FAILED':>10} {n_atoms}")

    out.log("  " + "-" * 80)
    if align_chain:
        out.log(f"  Aligned on: chain {align_chain}")
    else:
        out.log(f"  Aligned on: all chains")
    out.log("")
    out.save()


# Make it available as a PyMOL command
cmd.extend("align_designs", align_all_to_ref)
print("\n  Script loaded. Usage:")
print('    align_designs REF_OBJ                              # auto-detect')
print('    align_designs REF_OBJ, chain=each                  # per-chain RMSDs')
print('    align_designs REF_OBJ, chain=each, log=results     # save to results.txt + results.csv')
print('    align_designs REF_OBJ, chain=all                   # full-structure alignment')
print('    align_designs REF_OBJ, chain=A                     # force chain A only\n')
