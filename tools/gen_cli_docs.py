#!/usr/bin/env python3
"""Regenerate docs/reference/cli/*.md from the argparse parsers of the entry scripts.

Run from the project root after changing any command-line option:

    python tools/gen_cli_docs.py

Each script is executed with ``--help`` while ``ArgumentParser.parse_args`` is intercepted, so
the pages always list the options, defaults and choices the code actually has. Scripts that
read ``sys.argv`` directly (no argparse) are documented from their module docstring.
"""
import argparse
import ast
import io
import os
import runpy
import sys
from contextlib import redirect_stdout, redirect_stderr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "reference", "cli")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# (name, kind, one-line purpose); kind: "script" (file in the project root) or "module" (python -m)
SCRIPTS = [
    ("run_inversion", "script", "Global layer: convex shape, nested sampling over the spin state"),
    ("run_hybrid", "script", "Local layer: radius-field shape and spin, NUTS within each spin state"),
    ("run_binary", "script", "Contact-binary model (two ellipsoidal lobes), nested sampling"),
    ("batch_shapes", "script", "Whole pipeline for every row of a target table"),
    ("check_convergence", "script", "Convergence report for finished runs, agreement between repeats"),
    ("compare_models", "script", "Evidence-based comparison of runs on the same data"),
    ("complexity_scan", "script", "Evidence versus shape resolution (lmax) and prior scale (tau0)"),
    ("export_obj", "script", "Watertight .obj meshes of the posterior shapes"),
    ("make_folded_plots", "script", "Phase-folded lightcurve figures for finished runs"),
    ("make_hybrid_report", "script", "Rebuild all hybrid figures and viewers without resampling"),
    ("make_viewer", "script", "Self-contained interactive 3D shape viewer (HTML)"),
    ("make_uncertainty_viewer", "script", "3D viewer of the shape uncertainty (radius spread)"),
    ("reweight_resolution", "script", "Facet-resolution check by importance reweighting"),
    ("par_to_targets", "script", "Convert a .par target list into a batch table"),
    ("lcinv.mpc", "module", "Download ADES observations from the MPC"),
    ("lcinv.size", "module", "Diameter from H (CNEOS size estimator)"),
    ("lcinv.hybrid.naming", "module", "Rename legacy hybrid results ('mode' to 'spin state')"),
]


class _Captured(Exception):
    pass


def _capture(name, kind):
    """Return the parser of a script, or None if it does not use argparse."""
    def grab(self, *a, **k):
        raise _Captured(self)
    saved = (argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args, sys.argv)
    argparse.ArgumentParser.parse_args = grab
    argparse.ArgumentParser.parse_known_args = grab
    sys.argv = [name, "--help"]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            if kind == "module":
                runpy.run_module(name, run_name="__main__")
            else:
                runpy.run_path(os.path.join(ROOT, f"{name}.py"), run_name="__main__")
    except _Captured as c:
        return c.args[0]
    except BaseException:
        return None
    finally:
        argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args, sys.argv = saved
    return None


def _module_doc(name, kind):
    path = (os.path.join(ROOT, f"{name}.py") if kind == "script"
            else os.path.join(ROOT, *name.split(".")) + ".py")
    return ast.get_docstring(ast.parse(open(path).read())) or ""


def _md_escape(s):
    return (s or "").replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def _default(a):
    if isinstance(a, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "off"
    d = a.default
    if d is None or d is argparse.SUPPRESS:
        return "–"
    if isinstance(d, (list, tuple)):
        return "`" + " ".join(str(x) for x in d) + "`"
    if isinstance(d, str):
        return f"`{d}`" if d else '`""`'
    return f"`{d}`"


def _flag(a):
    if not a.option_strings:
        return f"`{a.metavar or a.dest}`"
    names = ", ".join(f"`{o}`" for o in a.option_strings)
    if isinstance(a, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._CountAction)):
        return names
    mv = a.metavar
    if isinstance(mv, tuple):
        mv = " ".join(mv)
    if mv is None:
        mv = a.dest.upper()
    if a.nargs in ("+", "*"):
        mv = f"{mv} [...]"
    elif isinstance(a.nargs, int) and a.nargs > 1 and not isinstance(a.metavar, tuple):
        mv = " ".join([mv] * a.nargs)
    return f"{names}&nbsp;<code>{_md_escape(str(mv))}</code>"


def _help(a):
    h = (a.help or "").replace("%(default)s", str(a.default))
    h = " ".join(h.split())
    if a.choices:
        h = (h + " " if h else "") + "One of " + ", ".join(f"`{c}`" for c in a.choices) + "."
    if isinstance(a, argparse._StoreFalseAction):
        h = (h + " " if h else "") + "(Flag: turns the behaviour off.)"
    if getattr(a, "required", False) and a.option_strings:
        h = "**Required.** " + h
    return _md_escape(h) or "&nbsp;"


def render(name, kind, purpose, parser):
    cmd = f"python {name}.py" if kind == "script" else f"python -m {name}"
    lines = [f"# `{name}`", "", f"*{purpose}.*", ""]
    if parser is None:
        doc = _module_doc(name, kind)
        lines += ["This command reads its arguments directly (no option parser).", "",
                  "```text", doc.strip(), "```", ""]
        return "\n".join(lines)
    usage = parser.format_usage().replace("usage: ", "").strip()
    usage = usage.replace(parser.prog, cmd, 1)
    lines += ["```text", usage, "```", ""]
    if parser.description:
        lines += ["```text", parser.description.strip(), "```", ""]
    n = 0
    for g in parser._action_groups:
        acts = [a for a in g._group_actions if not isinstance(a, argparse._HelpAction)]
        if not acts:
            continue
        title = {"options": "Options", "optional arguments": "Options",
                 "positional arguments": "Positional arguments"}.get(g.title, g.title[:1].upper() + g.title[1:])
        lines += [f"## {title}", ""]
        if g.description:
            lines += [g.description, ""]
        lines += ["| Option | Default | Description |", "|---|---|---|"]
        for a in acts:
            lines.append(f"| {_flag(a)} | {_default(a)} | {_help(a)} |")
            n += 1
        lines.append("")
    if parser.epilog:
        lines += ["```text", parser.epilog.strip(), "```", ""]
    lines += [f"<small>{n} options. Generated from the argparse definition by "
              "`tools/gen_cli_docs.py`.</small>", ""]
    return "\n".join(lines)


def main():
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    index = ["# Command-line reference", "",
             "Every entry point with all of its options and defaults. The pages are generated from "
             "the code (`python tools/gen_cli_docs.py`), so they match the installed version.", "",
             "| Command | Purpose |", "|---|---|"]
    for name, kind, purpose in SCRIPTS:
        parser = _capture(name, kind)
        page = name.replace(".", "_")
        with open(os.path.join(OUT, f"{page}.md"), "w") as f:
            f.write(render(name, kind, purpose, parser))
        cmd = f"{name}.py" if kind == "script" else f"python -m {name}"
        index.append(f"| [`{cmd}`]({page}.md) | {purpose} |")
        print(f"{name:26s} {'argparse' if parser else 'sys.argv (docstring)'}")
    with open(os.path.join(OUT, "index.md"), "w") as f:
        f.write("\n".join(index) + "\n")


if __name__ == "__main__":
    main()
