# Folder-Label Augmentation Result Tables

Documents `build_results_tables.py`: the generator for
`all_runs/tables_folder_label.tex`, which reports the folder-label-augmentation
experiment set (`run_all_folders_augmented_experiments.py`) as three
per-query-field LaTeX tables plus a cross-query-field summary.

The script reads `model_overall_stats.json` (`model_global_ndcg`) out of the run
folders in `all_runs/`. It runs no experiments and needs no ML dependencies --
plain Python, standard library only.

## Running it

From the project root:

```bash
python src/stats_test/build_results_tables.py            # tables fragment only
python src/stats_test/build_results_tables.py --preview  # + standalone wrapper
python src/stats_test/build_results_tables.py --compile  # + render it to PDF
```

| Output | Purpose |
|---|---|
| `all_runs/tables_folder_label.tex` | The fragment. This is what the paper `\input`s. |
| `all_runs/tables_folder_label_preview.tex` | Throwaway wrapper that renders the fragment. Not part of the paper. |
| `all_runs/tables_folder_label_preview.pdf` | The rendered preview, 4 pages. |

All three are generated -- do not hand-edit them; edit the script and re-run.
`--compile` needs `pdflatex` on `PATH`, builds in a temp directory so no
`.aux`/`.log` clutter lands in `all_runs/`, and prints the LaTeX errors if the
build fails.

Re-run the script whenever runs are added to `all_runs/`. Cells fill in
automatically; nothing needs editing by hand.

## Why three tables

Query field (T / TD / TDN) used to be a column axis, which made a single table
too wide to read once folder-label augmentation added variants. Splitting by
query field frees the columns to carry the axis that actually varies within this
experiment set: how much folder-label augmentation is applied on the document
side, and whether the query is expanded.

## Column layout

Seven numeric columns plus one shared margin column:

```
Base | +HyDE || FL-aug: ALLFL partner (Base, +CT) || FL-aug: doc. field (Base, +HyDE, +CT) | ±
```

| Column | Meaning | Run-name shape |
|---|---|---|
| `Base` | Plain query, plain folder label | `U5.<QF>.<suffix>` |
| `+HyDE` | Query + hypothetical documents + hypothetical folder label | `U5.<QF>AUG.<suffix>` |
| `FL-aug: partner / Base` | Only the ALLFL partner ranker's labels are augmented; the main ranker's own `folderlabel` field is not | `U5.<QF>.<suffix>.FLAug` |
| `FL-aug: partner / +CT` | As above, with a core-themes-expanded query | `U5.<QF>CT.<suffix>.FLAug` |
| `FL-aug: doc. field / Base` | The main ranker's own `folderlabel` field is augmented (and, for hybrid rows, the partner too) | `U5.<QF>.<TOFAUGS suffix>[.FLAug]` |
| `FL-aug: doc. field / +HyDE` | As above, with a HyDE-expanded query | `U5.<QF>AUG.<TOFAUGS suffix>[.FLAug]` |
| `FL-aug: doc. field / +CT` | As above, with a core-themes-expanded query | `U5.<QF>CT.<TOFAUGS suffix>[.FLAug]` |
| `±` | Range of the 95% CI half-widths across that row's cells | -- |

### `+HyDE` and `+CT` are different expansions

They are two distinct query augmentations, not two names for one thing:

- **`+HyDE`** -- original query + three hypothetical documents + a hypothetical
  folder label, from `run_query_augmentated_experiments.py`'s `AUG` variant
  (`data/llm_calls/query_expansion/1_doc_folder_hip/`).
- **`+CT`** -- original query + the topic's `core_themes` + `related_concepts`
  (`data/llm_calls/query_expansion/2_core_themes_and_related_concepts/`).

> **Note:** the captions in the older `all_runs/tables_of_results.tex` describe
> its `+AUG` column as "expanded with core themes and related concepts". That is
> CT's definition, not HyDE's, and that column holds HyDE runs. The caption is
> wrong; this file's tables are unaffected. That file is hand-maintained and was
> deliberately left untouched.

## The grid is sparse on purpose

The experiment set is not a full factorial, so several cells have no run behind
them and print as `---` rather than zero:

| Row group | Base | +HyDE | FL-part | FL-part +CT | FL-doc | FL-doc +HyDE | FL-doc +CT |
|---|---|---|---|---|---|---|---|
| **A** hybrid (main ranker + separate ALLFL partner) | ✅ | ✅ | ✅ | ✅ | ✅ | ⏳ | ✅ |
| **B** standalone (main ranker only, no partner) | ✅ | ✅ | -- | -- | ✅ | ⏳ | ✅ |
| **C** pure ALLFL baselines (the ALLFL ranker *is* the ranker) | ✅ `.c`, ⏳ `.b`/`.e` | ✅ `.c`, ⏳ `.b`/`.e` | ✅ | ✅ | -- | -- | -- |

✅ populated &nbsp;&nbsp; ⏳ pending, see [Known gaps](#known-gaps) &nbsp;&nbsp; `--` structurally impossible

- `+CT` was only ever run against an augmented folder label.
- Group B has no partner ranker, so the partner columns cannot exist.
- Group C has no main ranker, so the doc-field columns cannot exist; its
  augmented runs are partner-side augmentation.
- `+HyDE` against the *partner-only* augmentation is not planned -- see the
  note at the end of Known gaps.

A `---` in a rendered table therefore means "no run behind this cell", which is
either structural or pending. The generated captions say which.

## Rows

Fourteen configurations, emitted in four `\hline`-separated blocks:

| Block | Configurations |
|---|---|
| single-field | `B.--F-.-.-` |
| standalone | `B.TOFS.-.-`, `L.TOFS.mx--2.-`, `C.TOFS.-.-`, `E.TOFS.-.-`, `Z.TOFS.-.-`, `W.TOFS.-.-` |
| hybrid | `V.TOFS.mx--2.b`, `X.TOFS.mx--2.b`, `W.TOFS.-.c`, `W.TOFS.s---2.c` |
| pure ALLFL | `-.----.-.b`, `-.----.-.c`, `-.----.-.e` |

Columns escalate left-to-right by how much augmentation is applied; rows
escalate top-to-bottom by how much retrieval machinery is involved.

### The `B.--FAUG-` naming

The folder-label-only ablation is `CONFIGS` offset 8, whose suffix is
`L.--F-.-.-`. Its augmented runs were originally written as `L.--FAUG-.-.-`,
but `L` (tuned BM25F) and `B` (unweighted BM25F) denote the *same*
configuration when only one field is searched -- see `models.py`: *"Only affects
multi-field BM25F; single-field BM25 is unaffected."* The runs were renamed to
`B.--FAUG-.-.-` so they pair by name with the existing `B.--F-.-.-` baselines,
and `run_all_folders_augmented_experiments.py` normalises the scoring letter
(`single_field_scoring_code()`) so re-runs agree.

## Reading the tables

Three conventions to be aware of, all stated in the generated captions:

**Bold marks the highest value in each row -- nothing more.** No significance
testing is applied, so bold indicates the larger number, not a demonstrated
difference.

**The `±` column is a range, not an average.** It gives the smallest and largest
95% CI half-width across that row's cells, collapsing to a single number only
when they agree to three decimals. Margins are reported once per row rather than
in all seven columns, which would spend most of the table width repeating them.
They are *not* near-constant within a row -- the ALLFL baselines span ~0.04 --
so averaging them would understate the spread. The script prints the widest
observed spread on every run.

**The half-widths are large relative to the column-to-column differences.** Most
gaps in these tables are well inside the confidence intervals printed beside
them. Per-topic paired testing is possible -- every run folder carries a
`topics_values.json` of 45 topics x 30 seeds, aligned across runs -- but is
deliberately not done here. If any of this reaches the thesis, run the paired
tests first.

## Known gaps

43 runs the tables have room for but `all_runs/` does not hold yet. The affected
cells print `---`, and each table carries a deduplicated `% TODO:` comment
naming the command that fills them.

`run_table_completion_experiments.py` is the single entry point for all of them:

```bash
python -m src.llm_experiments.runs.run_table_completion_experiments          # all 43
python -m src.llm_experiments.runs.run_table_completion_experiments ALLFL    # groups 1+2 (10)
python -m src.llm_experiments.runs.run_table_completion_experiments FLDOC    # group 3 (33)
python src/stats_test/build_results_tables.py --compile                      # then refresh
```

| Group | What | Count | Also reachable as |
|---|---|---|---|
| 1 | Plain ALLFL baselines `.b`/`.e` at T and TDN (the `Base` those rows are measured against) | 4 | `run_new_experiments.py 16 17 76 77` |
| 2 | HyDE ALLFL baselines `.b`/`.e`, all query fields | 6 | `run_query_augmentated_experiments.py 1046 1047 1146 1147 1246 1247` |
| 3 | `FL-aug: doc. field` × `+HyDE` -- the `TOFAUGS` runs with a HyDE query | 33 | *(new -- no equivalent)* |

Only group 3 needed new run logic. Groups 1 and 2 were always reachable through
the existing scripts' registered ids; the completion script delegates to their
callables rather than reimplementing them, so each run has exactly one
definition.

Group 3 is the one that adds a comparison rather than filling a hole: without
it, `+HyDE` and `+CT` never appear under the same document-side condition, so
the two query expansions cannot be compared directly.

**Not planned:** `+HyDE` against the *partner-only* augmentation (`FL-part
+HyDE`, for the hybrid and pure-ALLFL rows). Adding it would make all three
column blocks a full `{Base, +HyDE, +CT}` and complete the factorial, at the
cost of 21 further runs. It is a one-line change -- add the group A and group C
offsets to `FL_DOC_OFFSETS` in `run_table_completion_experiments.py`.

## Preview wrapper details

The preview is deliberately **one-column**, though the paper is two-column. In a
twocolumn document a `table*` (full-width float) can never be placed on the page
that declares it, so LaTeX defers all of them -- the preview opens on a blank
page and the tables drift out of order. One-column makes `table*` degrade to
`table`, and each lands where declared.

The wrapper also relaxes the float placement parameters (`\textfraction`,
`\topfraction`, `topnumber`, ...). The defaults reserve 20% of every page for
body text, and the wrapper has two lines of it, so without the relaxation all
four tables defer to the end regardless of column count.

Both choices live in the wrapper's preamble only. The tables keep their own
`[t]` placement for the paper. One consequence to keep in mind: the summary
table sizes itself to `\columnwidth`, which in the one-column preview is the
full text width -- in the two-column paper it will render at half that.

## Related

- `run_table_completion_experiments.py` -- generates the runs listed under Known gaps
- `run_all_folders_augmented_experiments.py` / `runs_all_folders.md` -- generates the FLAug / `TOFAUGS` runs
- `run_query_augmentated_experiments.py` / `runs.md` -- generates the `+HyDE` (`AUG`) runs
- `run_new_experiments.py` -- generates the `Base` runs and the ALLFL baselines
- `RUN_NOTATION.md` -- the six-slot run-name notation
- `anova/build_grid.py` -- **does not** see this experiment set; its run-name regex
  matches neither the 7-char `TOFAUGS` fields code nor the 5-char `T--CT` query
  tags, so these runs are silently absent from its `inventory` output
