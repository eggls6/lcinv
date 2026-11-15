"""Names of the hybrid spin states (formerly "modes") and backward compatibility.

A hybrid run explores each distinct pole solution of the global layer separately; these are
called spin states: "spin state 1", "spin state 2", ... with file tag spin_state_1, ... :

    posterior_spin_state_1.npz, viewer_spin_state_1.html, shape_viewer_spin_state_1.html,
    obj/spin_state_1/, summary_hybrid.json["spin_states"]

Results written before the rename (posterior_mode1.npz, summary["modes"], "mode 1") are still
read by every script. To rename an old results folder in place:

    python -m lcinv.hybrid.naming results/566130_X05_hybrid
"""
import glob
import json
import os
import re
import shutil
import sys


def name(k):
    """Display name of spin state `k`: ``'spin state k'``."""
    return f"spin state {int(k)}"


def tag(k):
    """File tag of spin state `k`: ``'spin_state_k'``."""
    return f"spin_state_{int(k)}"


def index_of(label):
    """'spin state 2' / 'spin_state_2' / 'mode 2' / 'mode2' -> 2 (None if no number)."""
    m = re.search(r"(\d+)\s*$", str(label))
    return int(m.group(1)) if m else None


def tag_of(label):
    """File tag of a label in any spelling (see `index_of`)."""
    k = index_of(label)
    return tag(k) if k is not None else str(label).replace(" ", "_")


def posterior_files(outdir):
    """[(k, path)] sorted by k: new spin_state files, else legacy mode files."""
    new = glob.glob(os.path.join(outdir, "posterior_spin_state_*.npz"))
    old = glob.glob(os.path.join(outdir, "posterior_mode*.npz"))
    files = new or old
    return sorted(((index_of(os.path.basename(f)[:-4]), f) for f in files), key=lambda t: t[0])


def has_hybrid(outdir):
    """True if a folder holds hybrid posterior files."""
    return bool(posterior_files(outdir))


def states(summary):
    """Per-spin-state records of a summary_hybrid.json (new or legacy key)."""
    if not summary:
        return []
    return summary.get("spin_states") or summary.get("modes") or []


def state_info(summary, k):
    """Summary record of spin state `k` from a ``summary_hybrid.json`` dict (empty if missing)."""
    for m in states(summary):
        if index_of(m.get("name", "")) == int(k):
            return m
    return {}


def migrate(outdir, dry_run=False):
    """Rename legacy mode files/keys of a hybrid results folder to spin states."""
    moves = []
    for f in sorted(os.listdir(outdir)):
        m = re.match(r"^(posterior_|viewer_|shape_viewer_)mode(\d+)(\.\w+)$", f)
        if m:
            moves.append((f, f"{m.group(1)}{tag(m.group(2))}{m.group(3)}"))
    od = os.path.join(outdir, "obj")
    if os.path.isdir(od):
        for f in sorted(os.listdir(od)):
            m = re.match(r"^mode(\d+)$", f)
            if m:
                moves.append((os.path.join("obj", f), os.path.join("obj", tag(m.group(1)))))
    for a, b in moves:
        print(f"{a} -> {b}")
        if not dry_run:
            shutil.move(os.path.join(outdir, a), os.path.join(outdir, b))
    sp = os.path.join(outdir, "summary_hybrid.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        if "modes" in s and "spin_states" not in s:
            for m in s["modes"]:
                k = index_of(m.get("name", ""))
                if k is not None:
                    m["name"] = name(k)
            s["spin_states"] = s.pop("modes")
            print("summary_hybrid.json: modes -> spin_states")
            if not dry_run:
                with open(sp, "w") as fh:
                    json.dump(s, fh, indent=2)
    return moves


if __name__ == "__main__":
    for d in sys.argv[1:]:
        if d.startswith("-"):
            continue
        migrate(d, dry_run="--dry-run" in sys.argv)
        print(f"{d}: renamed. Rebuild the HTML viewers (text inside them) with\n"
              f"  python make_hybrid_report.py {d} && python make_viewer.py {d}")
