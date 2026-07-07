"""
Phase 1 — Results Reporter.

Reads all experiment results from all_runs/Phase1/ and generates a concise
Markdown report at results/phase1_report.md.

The report includes:
  - Set 1A: nDCG@5 per Config × Model (heatmap-style table)
  - F_best per model (derived from 1A)
  - Set 1B: nDCG@5 per run with Δ vs Q0 baseline
  - Set 1B-R: nDCG@5 per run with Δ vs original counterpart
  - Top-10 runs overall

Usage:
  cd <repo_root>
  python scripts/report_phase1.py
"""
import sys
import os
import json
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_BASE = os.path.join(PROJECT_ROOT, 'all_runs', 'Phase1')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results')
REPORT_PATH  = os.path.join(RESULTS_DIR, 'phase1_report.md')


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_ndcg(run_folder: str) -> float | None:
    """Reads nDCG@5 from all_documents_model_overall_stats.json."""
    stats_file = os.path.join(run_folder, 'all_documents_model_overall_stats.json')
    if not os.path.exists(stats_file):
        return None
    try:
        with open(stats_file) as f:
            return json.load(f)['model_global_ndcg']['mean']
    except Exception:
        return None


def load_margin(run_folder: str) -> float | None:
    """Reads margin (95% CI) from all_documents_model_overall_stats.json."""
    stats_file = os.path.join(run_folder, 'all_documents_model_overall_stats.json')
    if not os.path.exists(stats_file):
        return None
    try:
        with open(stats_file) as f:
            return json.load(f)['model_global_ndcg'].get('margin', 0.0)
    except Exception:
        return None


def find_run_folder(phase_dir: str, run_id_prefix: str) -> str | None:
    """Finds the first subdirectory starting with run_id_prefix."""
    if not os.path.isdir(phase_dir):
        return None
    for name in os.listdir(phase_dir):
        if name.startswith(run_id_prefix):
            return os.path.join(phase_dir, name)
    return None


def fmt(val: float | None, baseline: float | None = None) -> str:
    """Formats a score value with optional delta."""
    if val is None:
        return '—'
    s = f'{val:.4f}'
    if baseline is not None and baseline > 0:
        delta = val - baseline
        sign = '+' if delta >= 0 else ''
        s += f' ({sign}{delta:.4f})'
    return s


def bold_max(values: list, val: float | None) -> bool:
    valid = [v for v in values if v is not None]
    return val is not None and valid and val == max(valid)


def generate_1a_table(lines: list):
    dir_1a = os.path.join(RESULTS_BASE, '1A')
    if not os.path.isdir(dir_1a):
        lines.append('> ⚠️ No 1A results found.\n')
        return {}, {}

    configs = ['F1', 'F2', 'F3', 'F4']
    models  = ['b', 'e', 'c', 'bce', 'bc', 'be', 'ce']
    model_labels = {'b': 'B', 'e': 'E', 'c': 'C',
                    'bce': 'BCE', 'bc': 'BC', 'be': 'BE', 'ce': 'CE'}

    # Load all scores
    scores = {}  # {(config, model): ndcg}
    for i in range(1, len(configs) * len(models) + 1):
        run_id = f'1A-{i:02d}'
        config = configs[(i - 1) // len(models)]
        model  = models[(i - 1) % len(models)]
        folder = find_run_folder(dir_1a, run_id)
        if folder:
            scores[(config, model)] = load_ndcg(folder)

    # Header
    header = '| Config | ' + ' | '.join(model_labels[m] for m in models) + ' |'
    sep    = '|--------|' + '|'.join(['--------:'] * len(models)) + '|'
    lines.append(header)
    lines.append(sep)

    for config in configs:
        row_vals = [scores.get((config, m)) for m in models]
        row = f'| **{config}** | '
        cells = []
        for m, v in zip(models, row_vals):
            col_vals = [scores.get((c, m)) for c in configs]
            cell = f'**{v:.4f}**' if v is not None and bold_max(col_vals, v) else (f'{v:.4f}' if v is not None else '—')
            cells.append(cell)
        row += ' | '.join(cells) + ' |'
        lines.append(row)

    lines.append('')

    # F_best per model
    f_best = {}
    f_best_path = os.path.join(dir_1a, 'f_best.json')
    if os.path.exists(f_best_path):
        with open(f_best_path) as f:
            f_best = json.load(f)
        lines.append('**F_best per model** (highest nDCG@5 across F1–F4):')
        lines.append('')
        lines.append('| Model | F_best | nDCG@5 |')
        lines.append('|-------|--------|--------|')
        for m in models:
            fb = f_best.get(m, '?')
            v  = scores.get((fb, m))
            lines.append(f'| {model_labels[m]} | {fb} | {v:.4f if v else "—"} |')
        lines.append('')
    return scores, f_best


def generate_1b_table(lines: list, scores_1a: dict, f_best: dict):
    dir_1b = os.path.join(RESULTS_BASE, '1B')
    if not os.path.isdir(dir_1b):
        lines.append('> ⚠️ No 1B results found.\n')
        return {}

    SET_1B_LOCAL = [
        ('1B-01','Q1','b',None),('1B-02','Q1','e',None),('1B-03','Q1','c',None),
        ('1B-04','Q1','bce',None),('1B-05','Q1','bc',None),('1B-06','Q1','be',None),('1B-07','Q1','ce',None),
        ('1B-08','Q2','b',None),('1B-09','Q2','bc',None),('1B-10','Q2','bce',None),
        ('1B-11','Q3','e',None),('1B-12','Q3','ce',None),('1B-13','Q3','be',None),('1B-14','Q3','bce',None),
        ('1B-15','Q4','b',None),('1B-16','Q4','e',None),('1B-17','Q4','c',None),
        ('1B-18','Q4','bce',None),('1B-19','Q4','bc',None),('1B-20','Q4','be',None),('1B-21','Q4','ce',None),
        ('1B-22','Q0+Q1','b',None),('1B-23','Q0+Q1','e',None),('1B-24','Q0+Q1','bce',None),
        ('1B-25','Q0+Q2','b',None),('1B-26','Q0+Q2','bce',None),
        ('1B-27','Q0+Q3','e',None),('1B-28','Q0+Q3','bce',None),
        ('1B-29','Q0+Q4','b',None),('1B-30','Q0+Q4','e',None),('1B-31','Q0+Q4','bce',None),
        ('1B-32','QALL','bce',None),('1B-33','Q0+QALL','bce',None),
        ('1B-34','Q0+QALL','b',None),('1B-35','Q0+QALL','e',None),
        ('1B-36','Q0+QALL','bce','F1'),('1B-37','Q0+QALL','bce','F2'),
        ('1B-38','Q0+QALL','bce','F3'),('1B-39','Q0+QALL','bce','F4'),
        ('1B-40','Q0+QALL','bce','F5'),
    ]

    lines.append('| Run ID | Query | Model | Config | nDCG@5 | Δ vs Q0 |')
    lines.append('|--------|-------|-------|--------|--------|---------|')

    run_scores = {}
    for run_id, query_type, model_key, config_override in SET_1B_LOCAL:
        folder = find_run_folder(dir_1b, run_id)
        ndcg   = load_ndcg(folder) if folder else None
        config = config_override if config_override else f_best.get(model_key, '?')
        # Q0 baseline for this model
        q0_baseline = scores_1a.get((config, model_key)) if scores_1a else None
        delta_str = ''
        if ndcg is not None and q0_baseline is not None:
            delta = ndcg - q0_baseline
            delta_str = f'{"+" if delta >= 0 else ""}{delta:.4f}'
        lines.append(f'| {run_id} | {query_type} | {model_key.upper()} | {config} | '
                     f'{"—" if ndcg is None else f"{ndcg:.4f}"} | {delta_str} |')
        run_scores[run_id] = ndcg

    lines.append('')
    return run_scores


def generate_1br_table(lines: list, scores_1b: dict, f_best: dict):
    dir_1br = os.path.join(RESULTS_BASE, '1BR')
    if not os.path.isdir(dir_1br):
        lines.append('> ⚠️ No 1BR results found.\n')
        return {}

    SET_1BR_LOCAL = [
        ('1BR-01','Q5','b','1B-08'),('1BR-02','Q5','e','1B-11'),('1BR-03','Q5','bce','1B-10'),
        ('1BR-04','Q6','b','1B-01'),('1BR-05','Q6','e','1B-02'),('1BR-06','Q6','bce','1B-04'),
        ('1BR-07','Q7','e','1B-11'),('1BR-08','Q7','bce','1B-14'),
        ('1BR-09','Q2R','b','1B-08'),('1BR-10','Q2R','bce','1B-10'),
        ('1BR-11','Q0+Q5','b','1B-25'),('1BR-12','Q0+Q5','e','1B-27'),('1BR-13','Q0+Q5','bce','1B-26'),
        ('1BR-14','Q0+Q6','b','1B-22'),('1BR-15','Q0+Q6','bce','1B-24'),
        ('1BR-16','Q0+Q5+Q6','bce',None),
        ('1BR-17','Q0+QALL-R','bce','1B-33'),('1BR-18','Q0+QALL-R','b','1B-34'),('1BR-19','Q0+QALL-R','e','1B-35'),
    ]

    lines.append('| Run ID | Query | Model | nDCG@5 | vs Original |')
    lines.append('|--------|-------|-------|--------|-------------|')

    run_scores = {}
    for run_id, query_type, model_key, compare_id in SET_1BR_LOCAL:
        folder   = find_run_folder(dir_1br, run_id)
        ndcg     = load_ndcg(folder) if folder else None
        baseline = scores_1b.get(compare_id) if compare_id else None
        delta_str = ''
        if ndcg is not None and baseline is not None:
            delta = ndcg - baseline
            delta_str = f'{"+" if delta >= 0 else ""}{delta:.4f}'
        lines.append(f'| {run_id} | {query_type} | {model_key.upper()} | '
                     f'{"—" if ndcg is None else f"{ndcg:.4f}"} | {delta_str} |')
        run_scores[run_id] = ndcg

    lines.append('')
    return run_scores


def main():
    _log("Phase 1 Report Generator")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lines = [
        '# Phase 1 Results Report',
        '',
        f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*',
        '',
        '> **Metric:** Folder nDCG@5, averaged over 45 topics — single run per experiment (no ECF masking in Phase 1).',
        '> Bold values = best in column.',
        '',
        '---',
        '',
        '## Set 1A — Normal Query × Folder Index Configurations',
        '',
        '**Query:** Q0 = Title + Description  ',
        '**Goal:** Find F_best (best folder index config) per model.',
        '',
    ]

    scores_1a, f_best = generate_1a_table(lines)

    lines += [
        '---',
        '',
        '## Set 1B — Original Expanded Queries',
        '',
        '**Index:** F_best per model (from Set 1A)  ',
        '**Goal:** Find Q_best using original entity/event-focused LLM expansions.',
        '',
    ]

    scores_1b = generate_1b_table(lines, scores_1a, f_best)

    lines += [
        '---',
        '',
        '## Set 1B-R — Revised Concept-Focused Queries',
        '',
        '**Index:** F_best per model (from Set 1A)  ',
        '**Goal:** Test whether concept-focused expansions outperform entity-focused ones.',
        '',
    ]

    scores_1br = generate_1br_table(lines, scores_1b, f_best)

    # ── Top-10 overall ──────────────────────────────────────────────────────────
    all_scores = {}
    for run_id, v in {**scores_1a, **scores_1b, **scores_1br}.items():
        if v is not None:
            all_scores[str(run_id)] = v

    top10 = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    if top10:
        lines += [
            '---',
            '',
            '## Top-10 Runs Overall',
            '',
            '| Rank | Run ID | nDCG@5 |',
            '|------|--------|--------|',
        ]
        for rank, (run_id, score) in enumerate(top10, 1):
            lines.append(f'| {rank} | {run_id} | {score:.4f} |')
        lines.append('')

    lines += [
        '---',
        '',
        '## Phase 1 Decisions (fill after reviewing)',
        '',
        '| Decision | Value |',
        '|----------|-------|',
        '| D1 — F_best (overall) | ___ |',
        '| D1a — F_best for B | ___ |',
        '| D1b — F_best for E | ___ |',
        '| D1c — F_best for C | ___ |',
        '| D2 — Q_best_B (original) | ___ |',
        '| D3 — Q_best_E (original) | ___ |',
        '| D4 — Q_best_C (original) | ___ |',
        '| D5 — Q_best_BCE (original) | ___ |',
        '| D20 — Q_best_B (revised or original?) | ___ |',
        '| D21 — Q_best_E (revised or original?) | ___ |',
        '| D22 — Q_best_BCE (revised or original?) | ___ |',
        '| D23 — Revised > Original? (per model) | ___ |',
        '',
    ]

    report_text = '\n'.join(lines)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_text)

    _log(f"Report saved to {REPORT_PATH}")
    print(f"\n  Open: file://{REPORT_PATH}")


if __name__ == '__main__':
    main()
