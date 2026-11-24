"""
llm_registry.py
---------------
Plain-language names for every run-name slot value (used by the run picker and run card), and one Strategy
entry per LLM experiment (used by the 🤖 LLM Experiments page). All page content comes from here; prompts and
examples themselves are read at runtime by utils_llm.py.

Sources: reports/run_catalog.md §1-§5, src/dspy_experiments/running_experiments.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Slot labels (picker + run card)
# ──────────────────────────────────────────────────────────────────────────────
FAMILY = {"legacy": "Baselines (no LLM)", "luna": "LLM experiments (GPT-6-luna)"}

QF = {"T--": "Title", "TD-": "Title + description", "TDN": "Title + description + narrative"}

SAMPLE = {
    "U5": "uniform sample, 5 docs/box", "U1": "uniform, 1 doc/box", "U2": "uniform, 2 docs/box",
    "U3": "uniform, 3 docs/box", "U4": "uniform, 4 docs/box", "K5": "uneven sample", "A-": "all documents (oracle)",
}

# sample + config -> label. Anything missing is decoded by describe_config().
CONFIG = {
    "U5 · W.TOFS.s---2.c": "RRF(BM25F tuned, Emb, ColBERT) + similar-SNC k=2 + ALLFL ColBERT",
    "U5 · W.TOFS.-.c": "RRF(3) + ALLFL ColBERT",
    "U5 · W.TOFS.-.-": "RRF(3)",
    "U5 · B.TOFS.-.-": "BM25 (TOFS)",
    "U5 · C.TOFS.-.-": "ColBERT (TOFS)",
    "U5 · E.TOFS.-.-": "Embeddings (TOFS)",
    "U5 · B.--F-.-.-": "BM25 on folder-label field",
    "U5 · -.----.-.b": "ALLFL BM25",
    "U5 · -.----.-.c": "ALLFL ColBERT",
    "U5 · -.----.-.e": "ALLFL Embeddings",
    "U5 · -.----.-.x": "ALLFL BM25 + ColBERT",
    "U5 · -.----.-.y": "ALLFL BM25 + Embeddings",
    "U5 · -.----.-.z": "ALLFL BM25 + Embeddings + ColBERT",
    "K5 · -.----.-.b": "ALLFL BM25, uneven-sample evidence",
    "K5 · -.----.-.c": "ALLFL ColBERT, uneven-sample evidence",
    "K5 · -.----.-.e": "ALLFL Embeddings, uneven-sample evidence",
    "K5 · W.TOFS.-.-": "RRF(3), uneven sample",
    "A- · W.TOFS.-.-": "RRF(3), all documents indexed (oracle)",
    "U1 · W.TOFS.-.-": "RRF(3), 1 doc/box",
    "U2 · W.TOFS.-.-": "RRF(3), 2 docs/box",
    "U3 · W.TOFS.-.-": "RRF(3), 3 docs/box",
    "U4 · W.TOFS.-.-": "RRF(3), 4 docs/box",
}

_SCORING = {
    "B": "BM25", "L": "BM25F tuned", "C": "ColBERT", "E": "Embeddings", "W": "RRF(BM25F tuned, Emb, ColBERT)",
    "Z": "RRF(BM25F, Emb, ColBERT)", "V": "RRF(BM25F tuned, ColBERT)", "X": "RRF(BM25F, ColBERT)",
    "Y": "RRF(BM25F, Emb)",
}
_FIELD_NAMES = {"T": "title", "O": "OCR", "F": "folder label", "S": "summary"}
_EXP_REL = {"m": "same SNC", "s": "similar SNC", "x": "same box", "d": "close date"}
_PARTNER = {"b": "ALLFL BM25", "c": "ALLFL ColBERT", "e": "ALLFL Embeddings", "x": "ALLFL BM25+ColBERT",
            "y": "ALLFL BM25+Emb", "z": "ALLFL BM25+Emb+ColBERT"}


def describe_config(sample: str, config: str) -> str:
    """Plain-language label for `sample · config`, decoded from the slots (run_catalog §1)."""
    key = f"{sample} · {config}"
    if key in CONFIG:
        return CONFIG[key]
    try:
        scoring, fields, exp, partner = config.split(".")
    except ValueError:
        return config
    if scoring == "-":
        text = _PARTNER.get(partner, f"ALLFL {partner}")
    else:
        text = _SCORING.get(scoring, scoring)
        if fields != "TOFS":
            used = [_FIELD_NAMES[c] for c in fields if c in _FIELD_NAMES]
            text += f" on {' + '.join(used) if used else 'no field'}"
        if exp != "-":
            rels = [_EXP_REL[c] for c in exp[:-1] if c in _EXP_REL]
            text += f" + {' & '.join(rels)} k={exp[-1]}"
        if partner != "-":
            text += f" + {_PARTNER.get(partner, partner)}"
    if sample != "U5":
        text += f" ({SAMPLE.get(sample, sample)})"
    return text


TABLE = {"PLAIN": "Plain label (no folder text)", "FLAUG": "FL-aug: classification-only description",
         "FLEV": "FL-ev: grounded in sampled docs", "FLEVOWN": "FL-ev own-only", "FLOPTCV": "FL-aug, GEPA (CV)",
         "FLOPTALL": "FL-aug, GEPA (all topics)", "FDQ": "Doc2Query headings",
         "FDQK": "Doc2Query + keywords", "FDQF": "Doc2Query + facets", "FDQKF": "Doc2Query + keywords + facets",
         "FW": "FL-web", "FWEV": "FL-ev + web", None: "— (no LLM)"}
COLUMN = {"BASE": "Plain query", "DOC": "+ HyDE passages", "FL": "+ predicted filing", "DOCFL": "+ HyDE + filing",
          "CT": "+ core themes", None: "— (no LLM)"}
QUERY_SET = {"hw": "Hand-written prompts", "QCV": "GEPA-optimized (2-fold CV)",
             "QALL": "GEPA-optimized (all topics)", "QWEB": "Hand-written + web notes", None: "—"}
RERANKER = {None: "None", "RRN": "LLM listwise rerank (reasoning off)", "RRM": "LLM listwise rerank (reasoning medium)"}
PROTOCOL = {"30 seeds": "30 random seeds", "OfficialECF": "NTCIR-18 official ECF (3 fixed sets)",
            "ALLFL": "seed-independent (ALLFL)"}

SLOT_LABELS = {"table": TABLE, "column": COLUMN, "query_set": QUERY_SET, "reranker": RERANKER}

RUN_NAME_DIAGRAM = """\
U5 . TD- . W . TOFS . s---2 . c . FLEV . CT . QCV . RRN
│     │    └────────┬───────────┘   │     │    │     └ RERANKER   (optional) LLM listwise rerank of the top 20
│     │             │               │     │    └ QUERY SET        (optional) who wrote the query text: hw / QCV / QALL / QWEB
│     │             │               │     └ COLUMN                query-side text: BASE / DOC / FL / DOCFL / CT
│     │             │               └ TABLE                       folder-side text: PLAIN / FLAUG / FLEV / ...
│     │             └ retrieval config: scoring . fields . expansion . ALLFL partner
│     └ QF: topic title / + description / + narrative
└ sample: U5 = uniform 5 docs per box, K5 = uneven sample"""


# ──────────────────────────────────────────────────────────────────────────────
# Strategies (LLM Experiments page)
# ──────────────────────────────────────────────────────────────────────────────
# A prompt spec is a string:
#   "sig:<Class>"            a DSPy signature of src/dspy_experiments/signatures.py (instructions + fields)
#   "sig:<Class>Web"         CoreThemesWeb / DescribeFolderWeb / DescribeFolderWithEvidenceWeb (composed)
#   "d2q"                    Doc2QueryFolder with search_topics = n_topics of d2q/.../folders.json
#   "program:<arm>_<split>"  a GEPA instruction, data/llm_calls_luna/programs/<arm>_<split>.txt
#   "web:<NAME>"             a web-search prompt of src/dspy_experiments/generation/web_notes.py
# Strategy.prompt is a list of groups (title, [(column title, spec), ...]); a group with several columns is
# shown side by side.

PromptGroup = Tuple[str, List[Tuple[str, str]]]

GEPA_NOTE = (
    "GEPA set-up: the metric is mean nDCG@10 over 5 optimization seeds (7, 13, 21, 56, 88) of `W.TOFS.s---2.c` "
    "plus `-.----.-.b`; the folder side is scored by re-describing all folders and ranking on ALLFL BM25. "
    "A generalizing reflection prompt plus a guard keep topic-specific terms out of the instructions. "
    "The 45 topics are split into folds A and B (`folds.json`): the fold-A program is trained on fold-A topics "
    "and applied to fold-B topics, and vice versa, so every topic's text is out-of-sample."
)
INVALID_NOTE = "Invalid generations are dropped by the runner; those items fall back to the plain label / plain query."


@dataclass
class Strategy:
    code: str                     # "FLEV"
    slot: str                     # "table" | "column" | "query_set" | "reranker" | "sample"
    name: str                     # long name
    aim: str                      # one sentence: the question this experiment answers
    how: List[str]                # bullets
    prompt: List[PromptGroup]     # see above; [] = no LLM call
    example: str                  # loader key: folder, folder_seed, folder_cv, d2q, query, query_cv, rerank, none
    data: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


_FOLDER = "folder_label_augmentation/2_folder_context"
_FOLDER_EV = "folder_label_augmentation_with_evidence/2_folder_context"

STRATEGIES: List[Strategy] = [
    # ── folder side (TABLE) ──────────────────────────────────────────────────
    Strategy(
        "PLAIN", "table", "Plain label: no folder augmentation",
        "The baseline: how well do the rankers do on the catalogue's own folder text?",
        ["The folder-label field holds the classification text only: the SNC meaning plus the truncated scope note "
         "(or the raw label when there is none).",
         "`PLAIN.BASE` is the no-LLM baseline of the luna study; `PLAIN.<column>` isolates a query-side strategy."],
        [], "none", {}),
    Strategy(
        "FLAUG", "table", "FL-aug: a classification-only description of each folder",
        "Does an LLM description of what a folder's classification would contain help match queries to folders?",
        ["One description per folder (core themes + related concepts), written from the classification only: "
         "label, SNC, meaning, scope note, dates. No documents are shown.",
         "The text is appended to the folder-label field in the sampled docs **and** in the ALLFL index.",
         "One description set for all seeds (seed-independent)."],
        [("Hand-written prompt", [("DescribeFolder", "sig:DescribeFolder")])],
        "folder", {"path": f"hw/{_FOLDER}/folders.json"}, [INVALID_NOTE]),
    Strategy(
        "FLEV", "table", "FL-ev: a folder description grounded in sampled documents",
        "Does grounding the folder description in the seed's sampled documents beat a classification-only description?",
        ["Evidence: all the folder's own sampled docs if it has any; otherwise ≤5 docs borrowed from the first "
         "non-empty pool of same SNC → similar SNC → same box, with a 'NOT in this folder' warning.",
         "One description set **per seed** (the sample changes with the seed), so FL-ev runs vary by seed.",
         "ALLFL rows only by default (`-.----.-.{b,c,e}`)."],
        [("Hand-written prompt", [("DescribeFolderWithEvidence", "sig:DescribeFolderWithEvidence")])],
        "folder_seed", {"template": f"hw/{_FOLDER_EV}/seed_{{seed}}.json",
                        "template_uneven": f"hw/folder_label_augmentation_with_evidence/2_folder_context_uneven/seed_{{seed}}.json"},
        [INVALID_NOTE]),
    Strategy(
        "FLEVOWN", "table", "FL-ev own-only: FL-ev where the evidence is the folder's own, FL-aug elsewhere",
        "Is it the grounding that helps, or does borrowed evidence hurt?",
        ["No LLM call: for each seed, FL-ev's text where the folder's evidence was its own sampled docs, "
         "FL-aug's text everywhere else.",
         "Separates 'grounding helps' from 'borrowed grounding hurts'."],
        [], "folder_seed", {"template": f"hw_ownonly/{_FOLDER_EV}/seed_{{seed}}.json"},
        ["No LLM call: FL-ev text where the evidence is the folder's own, FL-aug text elsewhere."]),
    Strategy(
        "FLOPTCV", "table", "FL-aug with a GEPA-optimized instruction, 2-fold cross-validated",
        "Can prompt optimization (GEPA) write better folder descriptions than the hand-written prompt, out-of-sample?",
        ["GEPA optimizes the DescribeFolder instruction on the topics of one fold; the other fold's topics are "
         "searched on the index the program wrote.",
         "Two description sets (programs A and B); each topic uses the one trained on the *other* fold."],
        [("Folder instruction", [("Hand-written", "sig:DescribeFolder"), ("GEPA fold A", "program:folder_A"),
                                 ("GEPA fold B", "program:folder_B")])],
        "folder_cv", {"templates": {"A": f"opt_cv/{_FOLDER}/folders_A.json", "B": f"opt_cv/{_FOLDER}/folders_B.json"}},
        [GEPA_NOTE, INVALID_NOTE]),
    Strategy(
        "FLOPTALL", "table", "FL-aug with a GEPA instruction optimized on all 45 topics",
        "How far can an optimized folder instruction go when it is trained on the evaluation topics (upper bound)?",
        ["Same as FLOPTCV, but GEPA trains on all 45 topics."],
        [("Folder instruction", [("Hand-written", "sig:DescribeFolder"), ("GEPA all topics", "program:folder_ALL")])],
        "folder", {"path": f"opt_all/{_FOLDER}/folders.json"}, [GEPA_NOTE]),
    Strategy(
        "FDQ", "table", "Doc2Query: subject headings that would lead a searcher to the folder",
        "Does indexing short query-like subject headings work better than descriptive prose?",
        ["From the classification only, the LLM writes 3-5 facets, 10 keywords and N=25 search topics "
         "(noun-phrase subject headings), each tagged with its facet.",
         "FDQ indexes the search topics; FDQK / FDQF / FDQKF add the keywords / facets / both from the same answer."],
        [("Hand-written prompt", [("Doc2QueryFolder", "d2q")])],
        "d2q", {"raw": "d2q/raw.json", "variant": "d2q"}, [INVALID_NOTE]),
    Strategy(
        "FDQK", "table", "Doc2Query + keywords", "Do the answer's keywords add to the headings?",
        ["Same LLM answer as FDQ; the indexed text is the search topics plus the 10 keywords."],
        [("Hand-written prompt", [("Doc2QueryFolder", "d2q")])],
        "d2q", {"raw": "d2q/raw.json", "variant": "d2q_k"}, [INVALID_NOTE]),
    Strategy(
        "FDQF", "table", "Doc2Query + facets", "Do the answer's broad facets add to the headings?",
        ["Same LLM answer as FDQ; the indexed text is the search topics plus the 3-5 facets."],
        [("Hand-written prompt", [("Doc2QueryFolder", "d2q")])],
        "d2q", {"raw": "d2q/raw.json", "variant": "d2q_f"}, [INVALID_NOTE]),
    Strategy(
        "FDQKF", "table", "Doc2Query + keywords + facets", "Does adding both keywords and facets help?",
        ["Same LLM answer as FDQ; the indexed text is the search topics plus keywords and facets."],
        [("Hand-written prompt", [("Doc2QueryFolder", "d2q")])],
        "d2q", {"raw": "d2q/raw.json", "variant": "d2q_kf"}, [INVALID_NOTE]),
    Strategy(
        "FW", "table", "FL-web: FL-aug's prompt plus web-search notes on the folder's subject",
        "Does outside knowledge from a web search improve classification-only folder descriptions?",
        ["A web search (SUSHI/NTCIR domains blocked) writes 100-150 words of notes per distinct folder label.",
         "FL-aug's prompt gets those notes plus one paragraph on how to use them; nothing else changes.",
         "One description set for all seeds."],
        [("Web-search prompt", [("FOLDER_PROMPT", "web:FOLDER_PROMPT")]),
         ("Description prompt", [("DescribeFolderWeb", "sig:DescribeFolderWeb")])],
        "folder", {"path": f"web/{_FOLDER}/folders.json", "notes": "web/notes/folders.json"}, [INVALID_NOTE]),
    Strategy(
        "FWEV", "table", "FL-ev + web: FL-ev's grounded prompt plus FW's web notes",
        "Do web notes add to evidence-grounded descriptions?",
        ["FL-ev's evidence (per seed) plus FW's web notes in one prompt.", "One description set per seed."],
        [("Web-search prompt", [("FOLDER_PROMPT", "web:FOLDER_PROMPT")]),
         ("Description prompt", [("DescribeFolderWithEvidenceWeb", "sig:DescribeFolderWithEvidenceWeb")])],
        "folder_seed", {"template": f"web/{_FOLDER_EV}/seed_{{seed}}.json",
                        "template_uneven": f"web/folder_label_augmentation_with_evidence/2_folder_context_uneven/seed_{{seed}}.json",
                        "notes": "web/notes/folders.json"},
        [INVALID_NOTE]),

    # ── query side (COLUMN) ──────────────────────────────────────────────────
    Strategy(
        "BASE", "column", "Plain query: no query expansion",
        "The query-side baseline: the topic text alone.",
        ["The query is the title, `title. description` or `title. description. narrative`.",
         "On any table other than PLAIN, `<TABLE>.BASE` isolates that table's folder text."],
        [], "query", {"arm": None}),
    Strategy(
        "DOC", "column", "HyDE: the query plus 3 hypothetical document passages",
        "Does adding passages written as if they were the sought documents help?",
        ["The LLM writes 3 passages of 60-90 words, each as if it were a document of the collection.",
         "Query text = `query. passage 1. passage 2. passage 3`."],
        [("Hand-written prompt", [("HypotheticalDocuments", "sig:HypotheticalDocuments")])],
        "query", {"arm": "DOC"}, [INVALID_NOTE]),
    Strategy(
        "FL", "column", "Filing prediction: the query plus the predicted filing headings",
        "Does predicting where the documents would have been filed (SNC codes) help?",
        ["The LLM picks 3-5 SNC codes from the 378 used in the collection, plus 10-15 subject terms.",
         "Query text = `query. <meanings of the codes> <subject terms>` (raw codes are dropped)."],
        [("Hand-written prompt", [("PredictFiling", "sig:PredictFiling")])],
        "query", {"arm": "FL"}, [INVALID_NOTE]),
    Strategy(
        "DOCFL", "column", "HyDE + filing (the legacy 'AUG')",
        "Do the two expansions combine?",
        ["Query text = `query. <DOC text>. <FL text>`."],
        [("Hand-written prompts", [("HypotheticalDocuments", "sig:HypotheticalDocuments"),
                                   ("PredictFiling", "sig:PredictFiling")])],
        "query", {"arm": "DOCFL"}, [INVALID_NOTE]),
    Strategy(
        "CT", "column", "Core themes: the query restated in the records' language",
        "Does restating the need in the vocabulary of the period's records help?",
        ["The LLM writes one 80-120 word paragraph plus 15-20 related concepts.",
         "Query text = `query. <core themes> <related concepts, comma-joined>`."],
        [("Hand-written prompt", [("CoreThemes", "sig:CoreThemes")])],
        "query", {"arm": "CT"}, [INVALID_NOTE]),

    # ── query sets ───────────────────────────────────────────────────────────
    Strategy(
        "hw", "query_set", "Hand-written query prompts (no suffix)",
        "The default: the query text comes from the hand-written DSPy prompts.",
        ["Runs without a query-set suffix use these prompts (see the 🔎 Query side tab)."],
        [("Hand-written prompts", [("HypotheticalDocuments", "sig:HypotheticalDocuments"),
                                   ("PredictFiling", "sig:PredictFiling"), ("CoreThemes", "sig:CoreThemes")])],
        "query", {"set": "hw"}),
    Strategy(
        "QCV", "query_set", "GEPA-optimized query prompts, 2-fold cross-validated (.QCV)",
        "Can GEPA-optimized query instructions beat the hand-written ones, out-of-sample?",
        ["One GEPA program per arm (docs, filing, core themes) and fold.",
         "Each topic's text comes from the program trained on the *other* fold; the entry's `program` field says which."],
        [("HyDE (docs)", [("Hand-written", "program:docs_hw"), ("GEPA fold A", "program:docs_A"),
                          ("GEPA fold B", "program:docs_B")]),
         ("Filing", [("Hand-written", "program:filing_hw"), ("GEPA fold A", "program:filing_A"),
                     ("GEPA fold B", "program:filing_B")]),
         ("Core themes", [("Hand-written", "program:core_themes_hw"), ("GEPA fold A", "program:core_themes_A"),
                          ("GEPA fold B", "program:core_themes_B")])],
        "query_cv", {"set": "opt_cv"}, [GEPA_NOTE, INVALID_NOTE]),
    Strategy(
        "QALL", "query_set", "GEPA-optimized query prompts, trained on all 45 topics (.QALL)",
        "How far can optimized query instructions go when trained on the evaluation topics (upper bound)?",
        ["Same as QCV, but each arm has one program trained on all 45 topics."],
        [("HyDE (docs)", [("Hand-written", "program:docs_hw"), ("GEPA all topics", "program:docs_ALL")]),
         ("Filing", [("Hand-written", "program:filing_hw"), ("GEPA all topics", "program:filing_ALL")]),
         ("Core themes", [("Hand-written", "program:core_themes_hw"), ("GEPA all topics", "program:core_themes_ALL")])],
        "query", {"set": "opt_all"}, [GEPA_NOTE]),
    Strategy(
        "QWEB", "query_set", "Hand-written core themes plus web notes on the topic (.QWEB)",
        "Does outside knowledge from a web search improve the core-themes restatement?",
        ["A web search writes 120-200 words of notes per topic and formulation.",
         "CoreThemes' prompt gets the notes plus one paragraph on how to use them.",
         "`CT` column only, on PLAIN / FLAUG / FLEV / FW / FWEV."],
        [("Web-search prompt", [("TOPIC_PROMPT", "web:TOPIC_PROMPT")]),
         ("Core-themes prompt", [("CoreThemesWeb", "sig:CoreThemesWeb")])],
        "query", {"set": "web", "notes": "web/notes/topics.json"}, [INVALID_NOTE]),

    # ── reranker ─────────────────────────────────────────────────────────────
    Strategy(
        "RRN", "reranker", "LLM listwise reranking of the top 20 (reasoning off)",
        "Can the LLM reorder a strong first stage's top 20 better than the rankers did?",
        ["Each first stage's per-seed top 20 is reordered by GPT-6-luna in one listwise call, reasoning off.",
         "The model sees the topic's plain text and, per candidate, the label, SNC meaning and scope note "
         "(≤40 words).",
         "First stages that searched sampled docs (`W.TOFS…`) also show up to 3 of the folder's sampled docs "
         "('title - summary'): `RerankFolders`. ALLFL first stages use `RerankFoldersByLabel`.",
         "Invalid orders are repaired: repeats and out-of-range numbers dropped, missing candidates appended "
         "in first-stage order.",
         "7 first stages × T/TD/TDN = 21 runs, picked by a rule fixed beforehand (best family per kind)."],
        [("Reranking prompts", [("RerankFolders (doc-side first stages)", "sig:RerankFolders"),
                                ("RerankFoldersByLabel (ALLFL first stages)", "sig:RerankFoldersByLabel")])],
        "rerank", {}),

]

STRATEGY_BY_CODE = {(s.slot, s.code): s for s in STRATEGIES}


def strategy(slot: str, code: Optional[str]) -> Optional[Strategy]:
    return STRATEGY_BY_CODE.get((slot, code))


def strategies_for(slot: str) -> List[Strategy]:
    return [s for s in STRATEGIES if s.slot == slot]


def check_registry(runs) -> List[str]:
    """Slot values present in the scan that have no registry entry (should be empty)."""
    missing = set()
    for info in runs.values():
        if info.family != "luna":
            continue
        for slot in ("table", "column", "query_set", "reranker"):
            v = info.slots.get(slot)
            if v is not None and strategy(slot, v) is None:
                missing.add(f"{slot}={v}")
    return sorted(missing)
