#!/usr/bin/env python3
"""
Find complete 4-D ANOVA grids inside the SUSHI runs you already have, and emit the
input CSV for sushi_anova.py.

The six-element run name (Table 6 of the paper) is parsed into six dimensions:

    sample . query . scoring . fields . prop . labels
    U5     . TD-   . W       . TOFS   . ----  . -

Topic and training set come free with every run, so a 4-D ANOVA needs exactly one
complete 2-D rectangle of runs: query type x (one other dimension), with the remaining
three dimensions held fixed.

Usage
-----
    # what complete grids do I already have?
    python build_grid.py inventory --trec-eval-dir results/
    python build_grid.py inventory --scores my_scores.csv

    # write the ANOVA input for a chosen grid
    python build_grid.py emit --trec-eval-dir results/ \
        --factor4 scoring --fix sample=U5 fields=TOFS prop=---- labels=- \
        --out scores.csv

    # self-test on a synthetic run tree
    python build_grid.py demo
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

DIMS = ["sample", "query", "scoring", "fields", "prop", "labels"]

# Matches e.g. U5.TD-.W.TOFS.mx-2.b  /  A-.T--.L.-O--.----.-  /  U5.T--.B.TOFS.-.-
# prop and labels are variable-length in the actual run trees (e.g. "-", "mx--2",
# "s---2"), not the fixed 4/1-char tokens the original regex assumed.
RUN_RE = re.compile(
    r"([A-Z0-9-]{2})\.([TDN-]{3})\.([A-Z-])\.([TOFS-]{4})\.([^.\s]+)\.([^.\s]+)"
)
SEED_RE = re.compile(r"(?:seed|ecf|s)[_\-.]?(\d+)", re.IGNORECASE)
BARE_INT_RE = re.compile(r"(?:^|[/_\-.])(\d{1,3})(?:[/_\-.]|$)")

# Per-training-set score files inside a run directory, e.g. "Random12345_TopicsFolderMetrics.json".
# "AllFolderLabel_TopicsFolderMetrics.json" (All-Labels Scoring) and "OfficialECF_..." (the
# single-training-set A- condition) are deliberately NOT matched by this.
TOPIC_METRICS_RE = re.compile(r"^Random(\d+)_TopicsFolderMetrics\.json$")


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------

def parse_run_name(name: str, strict: bool = False) -> dict | None:
    """Parse the six-element run name out of `name`.

    strict=True requires the ENTIRE string to be the six-part name (used for run
    directory basenames, so that e.g. "U5.T--.B.TOFS.-.-.OfficialECF" -- a different,
    unrelated condition that happens to share the same six-part prefix -- is not
    silently parsed as if it were "U5.T--.B.TOFS.-.-").
    """
    m = RUN_RE.fullmatch(name) if strict else RUN_RE.search(name)
    if not m:
        return None
    return dict(zip(DIMS, m.groups()))


def extract_seed(path: str) -> str | None:
    m = SEED_RE.search(path)
    if m:
        return str(int(m.group(1)))
    # fall back to a bare integer directory or filename component
    for part in reversed(Path(path).parts):
        if RUN_RE.search(part):
            continue
        m = BARE_INT_RE.search(part)
        if m:
            return str(int(m.group(1)))
    return None


def read_trec_eval(path: str, measure: str = "ndcg_cut_5") -> list[tuple[str, float]]:
    """Parse `trec_eval -q -m ndcg_cut.5` output; returns [(topic, score), ...]."""
    out = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 3:
                continue
            meas, topic, val = parts
            if meas != measure or topic == "all":
                continue
            out.append((topic, float(val)))
    return out


def _looks_like_json_run_dirs(root: str) -> bool:
    """Detect the actual SUSHI layout: one directory per six-part run name, containing
    per-training-set 'Random<seed>_TopicsFolderMetrics.json' files -- as opposed to the
    trec_eval plaintext files the rest of this module was originally written for."""
    for p in Path(root).iterdir():
        if p.is_dir() and parse_run_name(p.name, strict=True) is not None:
            if any(TOPIC_METRICS_RE.match(f.name) for f in p.iterdir() if f.is_file()):
                return True
    return False


def _zero_fill_missing_topics(df: pd.DataFrame) -> pd.DataFrame:
    """A topic missing from a (run, training set)'s JSON means no relevant training
    document was sampled for it -- per the guide (sec. 2.3) that is a real ndcg=0.0
    observation, not a missing cell, and must be filled in rather than dropped."""
    key_cols = [c for c in DIMS + ["training_set"] if c in df.columns]
    all_topics = sorted(df["topic"].unique())
    keys = df[key_cols].drop_duplicates()
    full = keys.merge(pd.DataFrame({"topic": all_topics}), how="cross")
    out = full.merge(df, on=key_cols + ["topic"], how="left")
    n_filled = int(out["ndcg"].isna().sum())
    out["ndcg"] = out["ndcg"].fillna(0.0)
    if n_filled:
        print(f"  zero-filled {n_filled} (run, training set, topic) cells with no relevant "
              f"training document (ndcg=0.0), per guide section 2.3")
    return out


def load_from_json_run_dirs(root: str, measure: str = "ndcg_cut_5") -> pd.DataFrame:
    rows, run_dirs, skipped_dirs = [], 0, 0
    for p in sorted(Path(root).iterdir()):
        if not p.is_dir():
            continue
        dims = parse_run_name(p.name, strict=True)
        if dims is None:
            skipped_dirs += 1
            continue
        matched_any = False
        for f in sorted(p.iterdir()):
            m = TOPIC_METRICS_RE.match(f.name)
            if not m:
                continue
            matched_any = True
            seed = str(int(m.group(1)))
            with open(f) as fh:
                data = json.load(fh)
            for topic, metrics in data.items():
                if measure in metrics:
                    rows.append({**dims, "training_set": seed, "topic": topic,
                                "ndcg": metrics[measure]})
        if matched_any:
            run_dirs += 1

    print(f"parsed {len(rows)} score rows from {run_dirs} run directories under {root}")
    if skipped_dirs:
        print(f"  skipped {skipped_dirs} entries whose basename was not a bare six-element "
              f"run name (e.g. '...OfficialECF' / '...WRRF065' variants)")
    if not rows:
        raise SystemExit("nothing parsed from JSON run directories; check the layout")
    return _zero_fill_missing_topics(pd.DataFrame(rows))


def load_from_dir(root: str, glob: str = "**/*", measure: str = "ndcg_cut_5",
                  run_regex: str | None = None, seed_regex: str | None = None,
                  fmt: str = "auto") -> pd.DataFrame:
    global RUN_RE, SEED_RE
    if run_regex:
        RUN_RE = re.compile(run_regex)
    if seed_regex:
        SEED_RE = re.compile(seed_regex, re.IGNORECASE)

    if fmt == "auto":
        fmt = "json" if _looks_like_json_run_dirs(root) else "trec_eval"

    if fmt == "json":
        return load_from_json_run_dirs(root, measure)

    rows, skipped_run, skipped_seed, skipped_empty = [], 0, 0, 0
    for p in sorted(Path(root).glob(glob)):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        dims = parse_run_name(rel)
        if dims is None:
            skipped_run += 1
            continue
        seed = extract_seed(rel)
        if seed is None:
            skipped_seed += 1
            continue
        scores = read_trec_eval(str(p), measure)
        if not scores:
            skipped_empty += 1
            continue
        for topic, val in scores:
            rows.append({**dims, "training_set": seed, "topic": topic, "ndcg": val})

    print(f"parsed {len(rows)} score rows from {root}")
    if skipped_run:
        print(f"  skipped {skipped_run} files whose path did not contain a 6-element run name")
    if skipped_seed:
        print(f"  skipped {skipped_seed} files with a run name but no recognisable seed")
    if skipped_empty:
        print(f"  skipped {skipped_empty} files containing no '{measure}' per-topic lines")
    if not rows:
        raise SystemExit(
            "nothing parsed. Use --run-regex / --seed-regex to match your naming, "
            "or use --scores with a tidy CSV instead."
        )
    return pd.DataFrame(rows)


def load_from_csv(path: str, run_col: str | None, seed_col: str, topic_col: str,
                  score_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if run_col and run_col in df.columns:
        parsed = df[run_col].astype(str).map(parse_run_name)
        bad = parsed.isna().sum()
        if bad:
            print(f"  warning: {bad} rows had an unparseable run name and were dropped")
        keep = parsed.notna()
        df = df[keep].reset_index(drop=True)
        dims = pd.DataFrame(list(parsed[keep]))
        df = pd.concat([df.reset_index(drop=True), dims], axis=1)
    missing = [d for d in DIMS if d not in df.columns]
    if missing:
        raise SystemExit(f"CSV lacks columns {missing} and no parseable run-name column")
    df = df.rename(columns={seed_col: "training_set", topic_col: "topic", score_col: "ndcg"})
    return df[DIMS + ["training_set", "topic", "ndcg"]].copy()


# --------------------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------------------

def cell_coverage(df: pd.DataFrame):
    """Per (6-dim run) coverage of topics and training sets."""
    g = df.groupby(DIMS, observed=True).agg(
        n_topics=("topic", "nunique"),
        n_tsets=("training_set", "nunique"),
        n_rows=("ndcg", "size"),
    ).reset_index()
    g["complete"] = g["n_rows"] == g["n_topics"] * g["n_tsets"]
    return g


def inventory(df: pd.DataFrame, min_levels: int = 3):
    n_topics = df["topic"].nunique()
    n_tsets = df["training_set"].nunique()
    print(f"\n{n_topics} distinct topics, {n_tsets} distinct training sets overall")

    cov = cell_coverage(df)
    full = cov[(cov["n_topics"] == n_topics) & (cov["n_tsets"] == n_tsets) & cov["complete"]]
    print(f"{len(full)} of {len(cov)} runs have full topic x training-set coverage "
          f"({n_topics} x {n_tsets})")
    if len(full) < len(cov):
        short = cov[~cov.index.isin(full.index)]
        print("\n  runs with partial coverage (these cannot enter a balanced grid):")
        print(short.head(15).to_string(index=False))
        if len(short) > 15:
            print(f"  ... and {len(short) - 15} more")

    results = []
    for f4 in ["scoring", "fields", "prop", "labels", "sample"]:
        held = [d for d in DIMS if d not in ("query", f4)]
        for fixed_vals, sub in full.groupby(held, observed=True):
            fixed = dict(zip(held, fixed_vals if isinstance(fixed_vals, tuple) else (fixed_vals,)))
            present = {}
            for _, r in sub.iterrows():
                present.setdefault(r["query"], set()).add(r[f4])
            queries = sorted(present)
            best = None
            for k in range(len(queries), 0, -1):
                for qs in itertools.combinations(queries, k):
                    common = set.intersection(*(present[q] for q in qs))
                    if not common:
                        continue
                    size = len(qs) * len(common)
                    if best is None or size > best[0]:
                        best = (size, qs, sorted(common))
                if best and best[0] >= k * max(len(v) for v in present.values()):
                    break
            if not best:
                continue
            size, qs, lv = best
            if len(qs) < 2 or len(lv) < min_levels:
                continue
            results.append({
                "factor4": f4,
                "n_query": len(qs),
                "n_levels": len(lv),
                "n_runs": size,
                "N_obs": size * n_topics * n_tsets,
                "queries": ",".join(qs),
                "levels": ",".join(lv),
                **{f"fix_{k}": v for k, v in fixed.items()},
            })

    if not results:
        print("\nNo complete query x (other dimension) rectangle found with "
              f">= {min_levels} levels. Lower --min-levels, or check parsing.")
        return pd.DataFrame()

    out = pd.DataFrame(results).sort_values(
        ["n_runs", "n_levels"], ascending=False).reset_index(drop=True)
    print("\n=== Complete grids available (best first) ===")
    cols = ["factor4", "n_query", "n_levels", "n_runs", "N_obs", "queries", "levels"] + \
           [c for c in out.columns if c.startswith("fix_")]
    print(out[cols].head(20).to_string(index=False))
    print("\nPick a row and pass its factor4 / fix_* values to `emit`.")
    return out


# --------------------------------------------------------------------------------------
# Emit
# --------------------------------------------------------------------------------------

def emit(df: pd.DataFrame, factor4: str, fixed: dict, queries=None, levels=None,
         out_path: str = "scores.csv", pair_dim: str | None = None,
         pair_map: dict | None = None):
    """Write the ANOVA input CSV.

    `pair_dim` / `pair_map` handle an ASYMMETRIC rectangle where each factor4 level was
    only run under one specific value of another dimension (e.g. search-fields levels
    T---/-O--/--F-/---S exist only under scoring=L, and TOFS only under scoring=B --
    guide sec. 8bis's Table 7 + Table 8 combination). Without this, `--fix` can only
    hold every non-varying dimension to a SINGLE value, which cannot express that.
    """
    sub = df.copy()
    for k, v in fixed.items():
        if k not in DIMS:
            raise SystemExit(f"unknown dimension '{k}'; valid: {DIMS}")
        sub = sub[sub[k] == v]
    if queries:
        sub = sub[sub["query"].isin(queries)]
    if levels:
        sub = sub[sub[factor4].isin(levels)]

    if pair_dim:
        if pair_dim not in DIMS:
            raise SystemExit(f"unknown --pair-dim '{pair_dim}'; valid: {DIMS}")
        if not pair_map or set(pair_map) != set(levels or sub[factor4].unique()):
            raise SystemExit(
                "--pair-dim requires --pair-map to give exactly one "
                f"{pair_dim} value for every level passed via --levels")
        parts = [sub[(sub[factor4] == lvl) & (sub[pair_dim] == val)]
                 for lvl, val in pair_map.items()]
        sub = pd.concat(parts, ignore_index=True) if parts else sub.iloc[0:0]

    if sub.empty:
        raise SystemExit("filter selected no rows; check --fix values against `inventory` output")

    keep = sub[["topic", "training_set", "query", factor4, "ndcg"]].copy()
    keep = keep.rename(columns={factor4: "fields"})   # sushi_anova.py expects this column name

    nt, ng = keep["topic"].nunique(), keep["training_set"].nunique()
    nq, nf = keep["query"].nunique(), keep["fields"].nunique()
    expect = nt * ng * nq * nf
    dupes = keep.duplicated(["topic", "training_set", "query", "fields"]).sum()

    print(f"\ngrid: {nt} topics x {ng} training sets x {nq} query types x {nf} "
          f"{factor4} levels = {expect} cells")
    print(f"rows written: {len(keep)}  duplicates: {dupes}  missing: {expect - len(keep) + dupes}")
    if dupes:
        print("  WARNING: duplicate cells. Deduplicate before running the ANOVA.")
    if len(keep) - dupes != expect:
        print("  WARNING: design is NOT balanced. sushi_anova.py will refuse it; "
              "use the statsmodels Type III path (guide section 5.3).")

    keep.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(f"\nnext:  python sushi_anova.py --input {out_path} --outdir results/ "
          f"--factor4-label {factor4}")
    return keep


# --------------------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------------------

def demo(tmpdir="/tmp/sushi_demo_runs"):
    import numpy as np
    rng = np.random.default_rng(0)
    topics = [f"T18Eval-{i:05d}" for i in range(1, 46)]
    runs = [f"U5.{q}.{m}.TOFS.----.-" for q in ("T--", "TD-", "TDN")
            for m in ("B", "C", "E", "Z", "W")]
    runs += [f"U5.{q}.L.{f}.----.-" for q in ("T--", "TD-", "TDN")
             for f in ("T---", "-O--", "--F-", "---S")]
    os.makedirs(tmpdir, exist_ok=True)
    for seed in range(1, 31):
        d = os.path.join(tmpdir, f"seed{seed:02d}")
        os.makedirs(d, exist_ok=True)
        for r in runs:
            with open(os.path.join(d, f"{r}.eval"), "w") as fh:
                for t in topics:
                    fh.write(f"ndcg_cut_5\t{t}\t{max(0.0, rng.normal(0.21, 0.12)):.4f}\n")
                fh.write("ndcg_cut_5\tall\t0.2100\n")
    print(f"built synthetic run tree at {tmpdir}")
    df = load_from_dir(tmpdir)
    inventory(df)
    emit(df, "scoring", {"sample": "U5", "fields": "TOFS", "prop": "----", "labels": "-"},
         out_path="/tmp/sushi_demo_scores.csv")


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["inventory", "emit", "demo"])
    ap.add_argument("--trec-eval-dir")
    ap.add_argument("--glob", default="**/*")
    ap.add_argument("--measure", default="ndcg_cut_5")
    ap.add_argument("--run-regex", help="override the 6-element run-name regex (6 groups)")
    ap.add_argument("--seed-regex", help="override the training-set/seed regex (1 group)")
    ap.add_argument("--scores", help="tidy CSV instead of a directory")
    ap.add_argument("--run-col", default="run", help="run-name column in --scores")
    ap.add_argument("--seed-col", default="seed")
    ap.add_argument("--topic-col", default="topic")
    ap.add_argument("--score-col", default="ndcg")
    ap.add_argument("--format", choices=["auto", "trec_eval", "json"], default="auto",
                    help="run-tree layout: trec_eval plaintext files, or one directory "
                         "per run holding 'Random<seed>_TopicsFolderMetrics.json' files "
                         "(the actual SUSHI layout); auto-detected by default")
    ap.add_argument("--min-levels", type=int, default=3)
    ap.add_argument("--factor4", choices=DIMS)
    ap.add_argument("--fix", nargs="*", default=[], metavar="DIM=VALUE")
    # comma-separated, not nargs="*": several real level values (e.g. "--F-", "---S")
    # start with "-" and argparse's nargs="*" misparses them as new options.
    ap.add_argument("--queries", help="comma-separated, e.g. T--,TD-,TDN")
    ap.add_argument("--levels", help="comma-separated, e.g. T---,-O--,--F-,---S,TOFS")
    ap.add_argument("--pair-dim", choices=DIMS,
                    help="dimension whose value co-varies with --factor4, one-to-one "
                         "(e.g. scoring), for an asymmetric rectangle; see --pair-map")
    ap.add_argument("--pair-map",
                    help="comma-separated LEVEL=VALUE, one per --levels entry, e.g. "
                         "'T---=L,-O--=L,--F-=L,---S=L,TOFS=B'")
    ap.add_argument("--out", default="scores.csv")
    args = ap.parse_args()

    if args.mode == "demo":
        return demo()

    if args.trec_eval_dir:
        df = load_from_dir(args.trec_eval_dir, args.glob, args.measure,
                           args.run_regex, args.seed_regex, args.format)
    elif args.scores:
        df = load_from_csv(args.scores, args.run_col, args.seed_col,
                           args.topic_col, args.score_col)
    else:
        ap.error("give --trec-eval-dir or --scores")

    if args.mode == "inventory":
        inv = inventory(df, args.min_levels)
        if not inv.empty:
            inv.to_csv("grid_inventory.csv", index=False)
            print("\nwrote grid_inventory.csv")
    else:
        if not args.factor4:
            ap.error("emit needs --factor4")
        fixed = dict(kv.split("=", 1) for kv in args.fix)
        queries = args.queries.split(",") if args.queries else None
        levels = args.levels.split(",") if args.levels else None
        pair_map = (dict(kv.split("=", 1) for kv in args.pair_map.split(","))
                   if args.pair_map else None)
        emit(df, args.factor4, fixed, queries, levels, args.out,
             args.pair_dim, pair_map)


if __name__ == "__main__":
    sys.exit(main())
