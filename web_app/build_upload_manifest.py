"""
build_upload_manifest.py
------------------------
Lists exactly the files the web app needs, so a repository copy can run it without the 4 GB of luna runs and
2.4 GB of LLM calls. Writes, under upload_manifest/ (project root):

  manifest_git.txt      files to commit: per-run summary files, seed_ndcg5.json, the LLM text the 🤖 LLM
                        Experiments page reads (per-seed files for the core seeds only), rerank rankings for the
                        core seeds, the metadata and code the app reads.
  manifest_extra.txt    the per-seed files of the other seeds (FL-ev / own-only / web descriptions, rerank
                        rankings and answers). Optional: pack them into one archive and host it outside GitHub;
                        once extracted in the repo root, the page's seed pickers offer those seeds too.
  manifest_untrack.txt  files that are tracked in git today but belong to the extra set (stale copies would be
                        offered by the seed pickers): remove them from the index, not from disk.

Run `python3 build_seed_summaries.py` first (the summaries replace the per-seed metric files).

    cd web_app && python3 build_upload_manifest.py [--core-seeds 42] [--out ../upload_manifest]
"""

import argparse
import glob
import os
import subprocess

import llm_registry as reg
import utils_llm as ul
import utils_runs as ur

ROOT = ur.PROJECT_ROOT
LUNA = os.path.relpath(ul.LUNA_ROOT, ROOT)          # data/llm_calls_luna

# What the app reads in a run folder (utils_runs / utils_experiments_viz / run_picker / app_sushi).
RUN_FILES = set(ur.VALID_METRIC_FILES) | {"info.json", "topics_relevant_count_stats.json", "rerank_summary.json",
                                          ur.SEED_SUMMARY}

CODE = [
    "web_app/app_sushi.py", "web_app/utils_experiments_viz.py", "web_app/utils_topics_viz.py",
    "web_app/utils_retrieval.py", "web_app/utils_ecf_inspector.py", "web_app/utils_runs.py",
    "web_app/utils_llm.py", "web_app/llm_registry.py", "web_app/run_picker.py",
    "web_app/build_seed_summaries.py", "web_app/build_upload_manifest.py",
    "web_app/.streamlit/config.toml", "web_app/Ntcir18SushiOfficialExperimentControlFileV1.1.json",
    "src/augmentation_text.py",
    "src/dspy_experiments/signatures.py", "src/dspy_experiments/collection_context.py",
    "src/dspy_experiments/generation/web_notes.py",
    "src/data_creation/topics_output.txt",
    "README_DATA.md",
]
METADATA = [
    "data/folders_metadata/FoldersV1.3.json",
    "qrels/formal-folder-qrel.txt",
    "ecf/random_generated/ECF_ALL_TRAINING_SET.json",
]


def rel(path):
    return os.path.relpath(path, ROOT)


def exists(p):
    return os.path.isfile(os.path.join(ROOT, p))


def llm_files(core_seeds):
    """(core, extra) paths under data/llm_calls_luna that the LLM Experiments page reads."""
    core, extra = set(), set()
    seeded = []                                       # templates with {seed}
    for s in reg.STRATEGIES:
        d = s.data
        for k in ("path", "raw", "notes"):
            if k in d:
                core.add(d[k])
        for k in ("template", "template_uneven"):
            if k in d:
                seeded.append(d[k])
        core.update(d.get("templates", {}).values())
        if "variant" in d:
            core.add(f"{d['variant']}/folder_label_augmentation/2_folder_context/folders.json")
        if "set" in d:
            for qt in ("T", "TD", "TDN"):
                core.update(ul.query_files(d["set"], qt))
        for _, items in s.prompt:
            for _, spec in items:
                if spec.startswith("program:"):
                    core.add(f"programs/{spec.split(':', 1)[1]}.txt")
    for qt in ("T", "TD", "TDN"):                     # the column strategies read the hw set
        core.update(ul.query_files("hw", qt))
    core.add("folds.json")
    core.add("d2q/folder_label_augmentation/2_folder_context/folders.json")   # n_topics for Doc2Query
    for tmpl in seeded:
        for seed in ul.SEEDS:
            (core if seed in core_seeds else extra).add(tmpl.format(seed=seed))
    for run in sorted(n for n, i in ur.scan_runs().items() if i.slots.get("reranker")):
        for seed in ul.SEEDS:
            (core if seed in core_seeds else extra).add(f"rerank/{run}/Random{seed}.json")
        core.add(f"rerank/{run}/AllFolderLabel.json")              # seed-independent runs (exists() filters)
    to_paths = lambda xs: {os.path.join(LUNA, x) for x in xs if exists(os.path.join(LUNA, x))}
    return to_paths(core), to_paths(extra)


def run_files(core_seeds):
    core, extra = set(), set()
    runs = ur.scan_runs()
    for name, info in runs.items():
        for f in os.listdir(info.path):
            if f in RUN_FILES:
                core.add(rel(os.path.join(info.path, f)))
        if info.slots.get("reranker"):                # rerank example: both stages' per-seed top 20
            first = ur.load_info_json(name).get("first_stage")
            for path in (info.path, runs[first].path if first in runs else None):
                if not path:
                    continue
                for tsv in glob.glob(os.path.join(path, "runs", "Random*.tsv")):
                    seed = int(os.path.basename(tsv)[len("Random"):-len(".tsv")])
                    (core if seed in core_seeds else extra).add(rel(tsv))
                if os.path.isfile(os.path.join(path, "runs", "AllFolderLabel.tsv")):   # seed-independent: one file
                    core.add(rel(os.path.join(path, "runs", "AllFolderLabel.tsv")))
    return core, extra


def size(paths):
    return sum(os.path.getsize(os.path.join(ROOT, p)) for p in paths)


def git_lines(*args):
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return [l for l in out.stdout.splitlines() if l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-seeds", nargs="+", type=int, default=[42],
                    help="seeds whose per-seed LLM files and rerank rankings go into git (default: 42)")
    ap.add_argument("--out", default=os.path.join(ROOT, "upload_manifest"))
    args = ap.parse_args()
    core_seeds = set(args.core_seeds)

    missing_summaries = [n for n, i in ur.scan_runs().items()
                         if ur._seed_files(i.path) and not os.path.isfile(os.path.join(i.path, ur.SEED_SUMMARY))]
    if missing_summaries:
        raise SystemExit(f"{len(missing_summaries)} runs have no {ur.SEED_SUMMARY}: run build_seed_summaries.py first")

    llm_core, llm_extra = llm_files(core_seeds)
    run_core, run_extra = run_files(core_seeds)
    code = {p for p in CODE + METADATA if exists(p)}
    code |= {rel(p) for s in core_seeds for p in
             glob.glob(os.path.join(ROOT, "ecf", "random_generated", f"ECF_UNIFORM_Random_Seed_{s}.json"))}
    extra_ecf = {rel(p) for p in glob.glob(os.path.join(ROOT, "ecf", "random_generated", "ECF_UNIFORM_Random_Seed_*.json"))} - code
    core = sorted(code | llm_core | run_core)
    extra = sorted(llm_extra | run_extra | extra_ecf)

    tracked = set(git_lines("ls-files"))
    untrack = sorted(p for p in extra if p in tracked)
    ignored = git_lines("check-ignore", *core) if core else []
    big = [p for p in core if os.path.getsize(os.path.join(ROOT, p)) > 50e6]

    os.makedirs(args.out, exist_ok=True)
    for fname, paths in (("manifest_git.txt", core), ("manifest_extra.txt", extra), ("manifest_untrack.txt", untrack)):
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
            f.write("".join(p + "\n" for p in paths))

    mb = lambda ps: f"{size(ps) / 1e6:,.0f} MB"
    print(f"manifest_git.txt     {len(core):>6} files  {mb(core)}")
    print(f"   runs (summaries)  {len([p for p in run_core if not p.endswith('.tsv')]):>6} files  "
          f"{mb([p for p in run_core if not p.endswith('.tsv')])}")
    print(f"   rerank rankings   {len([p for p in run_core if p.endswith('.tsv')]):>6} files  "
          f"{mb([p for p in run_core if p.endswith('.tsv')])}  (seeds {sorted(core_seeds)})")
    print(f"   LLM text          {len(llm_core):>6} files  {mb(llm_core)}")
    print(f"   code + metadata   {len(code):>6} files  {mb(code)}")
    print(f"   not yet tracked   {len([p for p in core if p not in tracked]):>6} files")
    print(f"manifest_extra.txt   {len(extra):>6} files  {mb(extra)}")
    print(f"manifest_untrack.txt {len(untrack):>6} files tracked today that belong to the extra set")
    if ignored:
        print(f"WARNING: {len(ignored)} files are gitignored (add them with git add -f): {ignored[:5]}")
    if big:
        print(f"WARNING: files over 50 MB (GitHub warns at 50 MB, refuses 100 MB): {big}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
