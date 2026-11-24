# Data for the web app

The repository holds everything the web app (`streamlit run web_app/app_sushi.py`) needs, but not the full
experiment outputs: those are about 6 GB. This file says what was left out and how to add it.

## What is in the repository

- **Every run's results** (`all_runs/`, `all_runs_luna/`, `all_runs_luna_rerank/`): the summary files the app reads
  (`info.json`, `*overall_stats.json`, `topics_mean_margin.json`, `topics_values.json`,
  `topics_relevant_count_stats.json`, `rerank_summary.json`), plus `seed_ndcg5.json`, the per-topic nDCG@5 of each
  seed. That file replaces the per-seed `Random<s>_TopicsFolderMetrics.json` files: the significance test, the
  charts and the seed-variance plot give the same numbers with either.
- **The LLM text** the 🤖 LLM Experiments page shows (`data/llm_calls_luna/`): every seed-independent description
  file, all query-side files, the GEPA instructions, the web notes and `folds.json`. Per-seed files (FL-ev,
  FL-ev own-only and FL-ev + web descriptions, and the rerank rankings and answers) are included for **seed 42
  only**.

## Optional: the other 29 seeds (LLM Experiments examples)

Without this archive, the seed pickers on the LLM Experiments page offer seed 42 only. Everything else works.

1. Download `sushi_extra_seeds.tar.gz` from <!-- FILL IN: the release / Zenodo / Drive link --> **[link]**.
2. Extract it in the repository root (it holds paths such as `data/llm_calls_luna/...` and
   `all_runs_luna_rerank/.../runs/...`):

   ```bash
   tar -xzf sushi_extra_seeds.tar.gz
   ```

3. Restart the app, or press **🔄 Reload runs** in the sidebar.

## For the maintainer: rebuilding the upload

```bash
cd web_app
python3 build_seed_summaries.py           # seed_ndcg5.json in every run folder (--check verifies them)
python3 build_upload_manifest.py          # upload_manifest/manifest_{git,extra,untrack}.txt
cd ..
git rm --cached -q --pathspec-from-file=upload_manifest/manifest_untrack.txt   # per-seed files now in the archive
git add --pathspec-from-file=upload_manifest/manifest_git.txt
tar -czf sushi_extra_seeds.tar.gz -T upload_manifest/manifest_extra.txt         # upload outside GitHub
```

Never use `git add -A` or `git add .`: `all_runs_luna/` (4 GB) and `data/llm_calls_luna/` (2.4 GB) are not
gitignored.
