"""LaTeX tables for the runs written by running_experiments.py.

Nine tables: PLAIN, FLAUG and FLEV for each of T, TD and TDN. Each cell is the mean over
seeds of the per-seed mean over the 45 topics, with the 95% t-interval half-width over seeds;
seed-independent runs (the plain / FL-aug ALLFL rows) are a single run and carry no interval.
Base in FLAUG and FLEV is PLAIN.BASE, repeated as the reference. Bold marks the row maximum;
no significance testing is applied.

Usage (project root):
  python -m src.dspy_experiments.build_luna_tables
  python -m src.dspy_experiments.build_luna_tables --metric map --preview
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import scipy.stats as st

from src.dspy_experiments.common import PROJECT_ROOT

# Mirrors running_experiments.py (kept here so building tables does not load the models).
QT_TAGS = {"T": "T--", "TD": "TD-", "TDN": "TDN"}
QT_LABELS = {"T": "Title (T)", "TD": "Title+Desc (TD)", "TDN": "Title+Desc+Narr (TDN)"}
BLOCKS = [
    ["B.--F-.-.-"],
    ["B.TOFS.-.-", "C.TOFS.-.-", "E.TOFS.-.-", "W.TOFS.-.-"],
    ["W.TOFS.-.c", "W.TOFS.s---2.c"],
    ["-.----.-.b", "-.----.-.c", "-.----.-.e"],
]
ALLFL_ROWS = BLOCKS[-1]
# table -> [(column header, (table, column) of the run behind it)]
TABLES = {
    "PLAIN": [("Base", ("PLAIN", "BASE")), ("DOC", ("PLAIN", "DOC")), ("FL", ("PLAIN", "FL")),
              ("DOC+FL", ("PLAIN", "DOCFL")), ("CT", ("PLAIN", "CT"))],
    "FLAUG": [("Base", ("PLAIN", "BASE")), ("FL-aug", ("FLAUG", "BASENQ")), ("+FL", ("FLAUG", "FL")),
              ("+DOC+FL", ("FLAUG", "DOCFL")), ("+CT", ("FLAUG", "CT"))],
    "FLEV": [("Base", ("PLAIN", "BASE")), ("FL-ev", ("FLEV", "BASENQ")), ("+FL", ("FLEV", "FL")),
             ("+CT", ("FLEV", "CT"))],
}
TITLES = {
    "PLAIN": "plain folder label, query expansions",
    "FLAUG": "folder label augmented from its classification (no evidence)",
    "FLEV": "folder label augmented with document evidence (per seed)",
}
CAPTIONS = {
    "PLAIN": ("Document side untouched. \\textbf{DOC} appends the three hypothetical documents to the "
              "query, \\textbf{FL} the predicted filing (catalogue meaning of the predicted codes and "
              "subject terms), \\textbf{DOC+FL} both, \\textbf{CT} the core themes and related concepts."),
    "FLAUG": ("Every folder label is augmented with an LLM description written from its classification "
              "alone: in the main ranker's \\texttt{folderlabel} field and in the ALLFL ranker (the "
              "hybrid partner, or the only ranker in the ALLFL rows). \\textbf{FL-aug} uses the plain "
              "query; the \\textbf{+} columns add the query expansions of the PLAIN table. "
              "\\textbf{Base} is PLAIN's Base, repeated as the reference."),
    "FLEV": ("ALLFL rows only. Each seed's folder descriptions are grounded in that seed's sampled "
             "documents (own folder, else same SNC, similar SNC, same box), so every seed has its own "
             "index. \\textbf{Base} is PLAIN's Base (seed-independent), repeated as the reference."),
}


def run_values(run_dir, metric):
    """Per-run mean over topics of `metric`: one value per seed, or one for a seed-independent run."""
    values = []
    for path in sorted(glob.glob(os.path.join(run_dir, "Random*_TopicsFolderMetrics.json")) +
                       glob.glob(os.path.join(run_dir, "AllFolderLabel_TopicsFolderMetrics.json"))):
        with open(path, encoding="utf-8") as f:
            topics = json.load(f)
        values.append(float(np.mean([t.get(metric, 0.0) for t in topics.values()])))
    return values


def run_seeds(run_dir):
    return sorted(int(m.group(1)) for p in os.listdir(run_dir)
                  if (m := re.fullmatch(r"Random(\d+)_TopicsFolderMetrics\.json", p)))


def summarize(values):
    """(mean, 95% t half-width or None, n)"""
    if not values:
        return None
    n, mean = len(values), float(np.mean(values))
    if n < 2:
        return mean, None, n
    lo, hi = st.t.interval(0.95, df=n - 1, loc=mean, scale=st.sem(values))
    return mean, float(hi - mean), n


def tt(code):
    return code.replace("-", "{-}")


def cell(stat, bold):
    if stat is None:
        return "---"
    mean, margin, _ = stat
    text = f"\\textbf{{{mean:.4f}}}" if bold else f"{mean:.4f}"
    return text if margin is None else f"{text} {{\\scriptsize$\\pm${margin:.3f}}}"


def render(qt, table, runs_dir, metric):
    columns = TABLES[table]
    blocks = [ALLFL_ROWS] if table == "FLEV" else BLOCKS
    body, seed_counts, flev_seeds = [], set(), None
    for block in blocks:
        for config in block:
            stats = []
            for _, (tab, col) in columns:
                run_dir = os.path.join(runs_dir, f"U5.{QT_TAGS[qt]}.{config}.{tab}.{col}")
                stat = summarize(run_values(run_dir, metric)) if os.path.isdir(run_dir) else None
                stats.append(stat)
                seeds = run_seeds(run_dir) if stat is not None else []
                if seeds:
                    seed_counts.add(len(seeds))
                if seeds and tab == "FLEV" and flev_seeds is None:
                    flev_seeds = seeds
            present = [s[0] for s in stats if s is not None]
            best = max(present) if present else None
            body.append(f"\\texttt{{{tt(config)}}} & "
                        + " & ".join(cell(s, s is not None and s[0] == best) for s in stats) + " \\\\")
        body.append("\\hline")

    metric_tex = metric.replace("_", "\\_")
    seeds_text = (", ".join(str(n) for n in sorted(seed_counts)) or "no") + " seeds"
    caption = (f"\\texttt{{{metric_tex}}}, {QT_LABELS[qt]} queries, {TITLES[table]}. U5 uniform sampling, "
               f"mean over {seeds_text} $\\pm$ 95\\% t-interval half-width over seeds; cells without an "
               f"interval are a single seed-independent run. {CAPTIONS[table]} Bold marks the row maximum; "
               f"no significance testing is applied. \\texttt{{---}}: no run yet.")
    if table == "FLEV" and flev_seeds:
        caption += f" Seeds ({len(flev_seeds)}): {', '.join(map(str, flev_seeds))}."

    spec = "| l || " + " | ".join("c" for _ in columns) + " |"
    head = " & ".join(f"\\textbf{{{name}}}" for name, _ in columns)
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}",
             f"\\label{{tab:luna-{table.lower()}-{qt.lower()}}}", "\\resizebox{\\textwidth}{!}{",
             f"\\begin{{tabular}}{{{spec}}}", "\\hline", f"\\textbf{{Configuration}} & {head} \\\\", "\\hline",
             *body, "\\end{tabular}", "}", "\\end{table*}"]
    return "\n".join(lines)


PREVIEW = r"""%% Generated by src/dspy_experiments/build_luna_tables.py -- standalone preview of %(fragment)s.
\documentclass[a4paper]{article}
\usepackage[margin=1.5cm,landscape]{geometry}
\usepackage{graphicx}
\pagestyle{empty}
\renewcommand{\topfraction}{1.0}
\renewcommand{\bottomfraction}{1.0}
\renewcommand{\textfraction}{0.0}
\renewcommand{\floatpagefraction}{0.0}
\setcounter{topnumber}{10}
\setcounter{totalnumber}{10}
\begin{document}
\input{%(fragment)s}
\end{document}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(PROJECT_ROOT, "all_runs_luna"))
    ap.add_argument("--metric", default="ndcg_cut_5",
                    help="any key of the per-seed metrics JSON: ndcg_cut_5, ndcg_cut_10, map, recip_rank, ...")
    ap.add_argument("--out", default=None, help="default: <runs>/tables_luna[_<metric>].tex")
    ap.add_argument("--preview", action="store_true", help="also write a standalone wrapper next to it")
    args = ap.parse_args()

    suffix = "" if args.metric == "ndcg_cut_5" else f"_{args.metric}"
    out = args.out or os.path.join(args.runs, f"tables_luna{suffix}.tex")
    parts = ["% Generated by src/dspy_experiments/build_luna_tables.py -- do not edit by hand.",
             f"% Source: {args.runs}, metric {args.metric}."]
    for qt in QT_TAGS:
        for table in TABLES:
            parts.append(render(qt, table, args.runs, args.metric))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts) + "\n")
    print(f"wrote {out}")
    if args.preview:
        preview = out.replace(".tex", "_preview.tex")
        with open(preview, "w", encoding="utf-8") as f:
            f.write(PREVIEW % {"fragment": os.path.basename(out)})
        print(f"wrote {preview} (pdflatex it from {os.path.dirname(out)})")


if __name__ == "__main__":
    main()
