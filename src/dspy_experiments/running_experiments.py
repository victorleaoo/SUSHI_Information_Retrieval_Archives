"""Retrieval runs for the luna generations, from scratch, for T, TD and TDN.

Three tables per query type, over 10 configurations:

  PLAIN  plain folder label     columns BASE, DOC, FL, DOCFL, CT
  FLAUG  folder-label aug (no   columns BASENQ, FL, DOCFL, CT        (Base = PLAIN.BASE)
         evidence) on the doc field and on the ALLFL ranker/partner
  FLEV   folder-label aug with  columns BASENQ, FL, CT               (Base = PLAIN.BASE)
         evidence, one set per seed; ALLFL rows only, seeds that have a seed_<s>.json

Query columns: DOC = the 3 hypothetical passages, FL = the predicted filing (label_text +
subject_terms), DOCFL = both (run_generator's 'AUG'), CT = core themes + related concepts.

Output: <out>/U5.<QF>.<config>.<TABLE>.<COL>/ with one metrics JSON per seed (the resume
marker), the top-20 folder ranking per seed under runs/, and the evaluator's aggregates.

How it differs from run_new_experiments.py (same numbers, much less work):
  - Loop order is seed -> index -> (query type x query variant). Each distinct index
    (BM25 untuned/tuned TOFS, BM25 F, Embeddings TOFS, ColBERT TOFS; plain and FL-aug label)
    is built once per seed and answers every query; C/E/W/W.c/W.s---2.c are derived from
    the same raw rankings by the existing RRF / expansion / hybrid-fusion code.
  - The ALLFL rankers (plain, FL-aug) are seed-independent: computed once, and reused as
    the hybrid partner for every seed.
  - ColBERT is scored exactly (MaxSim against every document, top 100), not through a PLAID
    index: PLAID's build is random even with seeds fixed, so one ALLFL ColBERT index gave
    nDCG@5 anywhere in 0.118-0.161 across the old 30 identical runs.
  - Model weights are loaded once; query encodings are cached.

Usage (project root):
  python -m src.dspy_experiments.running_experiments --dry-run
  python -m src.dspy_experiments.running_experiments --parity-check
  nohup python -m src.dspy_experiments.running_experiments \
      > data/llm_calls_luna/logs/running_experiments.out 2>&1 &
  python -m src.dspy_experiments.running_experiments --query-types T --tables plain \
      --configs=W.TOFS.-.c,-.----.-.b --seeds 1 42
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
import torch

from src.dspy_experiments.common import LUNA_ROOT, PROJECT_ROOT, load_json, log

# run_generator.py uses bare imports and a cwd-relative RGdistribution.xlsx, so import it from src/.
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
LAUNCH_DIR = os.getcwd()  # relative --llm-root / --out resolve against this, not src/
sys.path.insert(0, SRC_DIR)
os.chdir(SRC_DIR)
from hybrid_models import perform_hybrid_fusion  # noqa: E402
from models import BM25Model, EmbeddingsModel, get_best_device  # noqa: E402
from pylate import models as pylate_models  # noqa: E402
from run_generator import (RANDOM_SEED_LIST, RunGenerator, build_augmentation_text,  # noqa: E402
                           build_core_themes_augmentation_text)
from sentence_transformers import util  # noqa: E402

try:  # the per-model "Loading weights" bars only clutter the nohup log
    from transformers.utils import logging as _hf_logging
    _hf_logging.disable_progress_bar()
except ImportError:
    pass

TOFS =["title", "ocr", "folderlabel", "summary"]
ALLFL = ["folderlabel"]
QUERY_TYPES = ["T", "TD", "TDN"]
QT_TAGS = {"T": "T--", "TD": "TD-", "TDN": "TDN"}
DOCS_PER_BOX = 5
EXPANSION_CEILING_K = 2
TOP_RUN_DEPTH = 20
COLBERT_K = 100  # same depth ColBERTModel.search returned

# column -> query variant (None = plain query); the variant names are run_generator's
QUERY_VARIANTS = {"BASE": None, "BASENQ": None, "DOC": "DOC", "FL": "FL", "DOCFL": "AUG", "CT": "CT"}
TABLES = {
    "PLAIN": {"label": "plain", "columns": ["BASE", "DOC", "FL", "DOCFL", "CT"]},
    "FLAUG": {"label": "flaug", "columns": ["BASENQ", "FL", "DOCFL", "CT"]},
    "FLEV": {"label": "flev", "columns": ["BASENQ", "FL", "CT"]},
}

# Doc-side rankers: key -> (model, index fields, tuned BM25F weights). All share one training set.
DOC_RANKERS = {
    "bm25F": ("bm25", ["folderlabel"], False),
    "bm25u": ("bm25", TOFS, False),
    "bm25t": ("bm25", TOFS, True),
    "emb": ("embeddings", TOFS, None),
    "colbert": ("colbert", TOFS, None),
}
# ALLFL rankers, keyed by the config's last slot.
ALLFL_RANKERS = {"b": ("bm25", ALLFL, False), "c": ("colbert", ALLFL, None), "e": ("embeddings", ALLFL, None)}
# The name each ranker has in RunGenerator.apply_document_level_rrf (its weights are keyed by it).
RRF_NAME = {"bm25F": "bm25", "bm25u": "bm25", "bm25t": "bm25", "emb": "embeddings", "colbert": "colbert"}

W = ["bm25t", "emb", "colbert"]  # same order as run_new_experiments' models list
CONFIGS = {
    "B.--F-.-.-": {"rankers": ["bm25F"]},
    "B.TOFS.-.-": {"rankers": ["bm25u"]},
    "C.TOFS.-.-": {"rankers": ["colbert"]},
    "E.TOFS.-.-": {"rankers": ["emb"]},
    "W.TOFS.-.-": {"rankers": W},
    "W.TOFS.-.c": {"rankers": W, "partner": "c"},
    "W.TOFS.s---2.c": {"rankers": W, "expansion": ["similar_snc"], "partner": "c"},
    "-.----.-.b": {"allfl": "b"},
    "-.----.-.c": {"allfl": "c"},
    "-.----.-.e": {"allfl": "e"},
}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    qt: str
    table: str
    config: str
    column: str
    seeds: list = field(default_factory=list)  # empty = one seed-independent run

    @property
    def name(self):
        return f"U5.{QT_TAGS[self.qt]}.{self.config}.{self.table}.{self.column}"

    @property
    def query(self):
        return self.qt, QUERY_VARIANTS[self.column]

    @property
    def label(self):
        return TABLES[self.table]["label"]

    @property
    def cfg(self):
        return CONFIGS[self.config]


def build_experiments(query_types, tables, configs, seeds, evidence_seeds):
    exps = []
    for qt in query_types:
        for table in tables:
            for config in configs:
                cfg = CONFIGS[config]
                if table == "FLEV" and "allfl" not in cfg:
                    continue  # evidence descriptions only exist for the ALLFL rows
                if table == "FLEV":
                    run_seeds = [s for s in seeds if s in evidence_seeds]
                else:
                    run_seeds = [] if "allfl" in cfg else list(seeds)
                for column in TABLES[table]["columns"]:
                    exps.append(Experiment(qt, table, config, column, run_seeds))
    return exps


def metrics_name(seed):
    return "AllFolderLabel" if seed is None else f"Random{seed}"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def hms(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s"


@contextlib.contextmanager
def timed(what):
    """Logs `what` with its duration once the block finishes."""
    t0 = time.time()
    yield
    log(f"{what} ({time.time() - t0:.1f}s)")


def query_names(queries):
    return ", ".join(f"{qt}/{variant or 'plain'}" for qt, variant in sorted(queries, key=str))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ExactColBERT:
    """ColBERT late interaction, scored exhaustively: MaxSim of the query against every document.

    Deterministic, unlike ColBERTModel's PLAID index, and cheap at this scale (a few thousand
    documents at most). Returns the top COLBERT_K, the depth ColBERTModel.search returned.
    """

    def __init__(self):
        self.device = get_best_device()
        self.model = pylate_models.ColBERT(model_name_or_path="lightonai/colbertv2.0", device=self.device)
        self._queries = {}

    def train(self, training_data):
        self.docnos = [str(d["docno"]) for d in training_data]
        self.folders = [d["folder"] for d in training_data]
        embs = self.model.encode([d["text_blob"] for d in training_data], batch_size=512,
                                 is_query=False, show_progress_bar=False)
        embs = [torch.as_tensor(e, dtype=torch.float32) for e in embs]
        lengths = torch.tensor([len(e) for e in embs])
        self.docs = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True).to(self.device)  # [N, L, dim]
        self.pad = (torch.arange(self.docs.shape[1])[None, :] >= lengths[:, None]).to(self.device)

    def encode_query(self, query):
        if query not in self._queries:
            q = self.model.encode([query], is_query=True, show_progress_bar=False)[0]
            self._queries[query] = torch.as_tensor(q, dtype=torch.float32).to(self.device)
        return self._queries[query]

    def scores(self, query):
        sim = torch.einsum("qd,nld->nql", self.encode_query(query), self.docs)
        sim = sim.masked_fill(self.pad[:, None, :], float("-inf"))
        return sim.max(dim=-1).values.sum(dim=-1)  # [N]

    def search(self, query):
        scores = self.scores(query)
        top = torch.topk(scores, k=min(COLBERT_K, len(self.docnos)))
        idx, vals = top.indices.tolist(), top.values.tolist()
        return pd.DataFrame({"docno": [self.docnos[i] for i in idx],
                             "folder": [self.folders[i] for i in idx], "score": vals})


class CachedEmbeddingsModel(EmbeddingsModel):
    """EmbeddingsModel with query encodings cached; same scores and ordering."""

    def __init__(self):
        super().__init__()
        self._queries = {}

    def search(self, query):
        if query not in self._queries:
            self._queries[query] = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(self._queries[query], self.doc_embeddings)[0].tolist()
        df = pd.DataFrame({"docno": [m["docno"] for m in self.metadata_map],
                           "folder": [m["folder"] for m in self.metadata_map], "score": scores})
        return df.sort_values(by="score", ascending=False)


_last_bm25_second = [0]


def _terrier_dirs():
    root = os.path.join(SRC_DIR, "terrierindex")
    return set(os.listdir(root)) if os.path.isdir(root) else set()


def fresh_bm25(fields, tuned, data):
    """BM25Model names its index dir after the current second: wait for a new one so two
    builds never share (and overwrite) a directory. Returns (model, dirs it created)."""
    while int(time.time()) <= _last_bm25_second[0]:
        time.sleep(0.05)
    before = _terrier_dirs()
    model = BM25Model(fields, tuned_weights=tuned)
    model.train(data)
    _last_bm25_second[0] = int(time.time())
    return model, _terrier_dirs() - before


def drop_terrier_dirs(dirs):
    for d in dirs:
        shutil.rmtree(os.path.join(SRC_DIR, "terrierindex", d), ignore_errors=True)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, llm_root, out_root):
        self.llm_root = llm_root
        self.out_root = out_root
        self.tmp_dir = os.path.join(out_root, "_tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

        # One RunGenerator supplies the loader, training-data builder, relations, RRF and
        # expansion, so the numbers come from the same code as every earlier run.
        self.gen = RunGenerator()
        self.gen.expansion_ceiling_k = EXPANSION_CEILING_K
        self.evaluator = self.gen.evaluator
        self.topics = self.gen.loader.get_topics()
        self.topic_ids = [t["ID"] for t in self.topics]

        self.hip = {qt: self._valid_arms(self._load(f"query_expansion/1_doc_folder_hip/{qt}.json")["topics"])
                    for qt in QUERY_TYPES}
        self.ct = {qt: self._valid_arms(self._load(
            f"query_expansion/2_core_themes_and_related_concepts/{qt}.json")["topics"]) for qt in QUERY_TYPES}
        self.flaug = self._valid_folders(self._load("folder_label_augmentation/2_folder_context/folders.json"),
                                         "folders.json")
        self._queries = {}
        self._emb = None
        self._colbert = None

    # -- inputs -------------------------------------------------------------

    def _load(self, rel):
        path = os.path.join(self.llm_root, rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return load_json(path, None)

    @staticmethod
    def _valid_arms(topics):
        """Drops invalid generations so build_*_augmentation_text falls back to the plain query."""
        out = {}
        for tid, entry in topics.items():
            out[tid] = {k: v for k, v in entry.items() if not isinstance(v, dict) or v.get("valid", True)}
            dropped = [k for k in entry if k not in out[tid]]
            if dropped:
                log(f"  {tid}: invalid {dropped}, using the plain query for them")
        return out

    @staticmethod
    def _valid_folders(data, what):
        """Invalid descriptions are left out, so those folders keep their plain label."""
        folders = data["folders"]
        valid = {fid: e for fid, e in folders.items() if e.get("valid")}
        if len(valid) < len(folders):
            log(f"{what}: {len(folders) - len(valid)} invalid descriptions, plain label used for them")
        return valid

    def evidence_path(self, seed):
        return os.path.join(self.llm_root, "folder_label_augmentation_with_evidence", "2_folder_context",
                            f"seed_{seed}.json")

    def evidence_seeds(self):
        return {s for s in RANDOM_SEED_LIST if os.path.isfile(self.evidence_path(s))}

    def folder_aug(self, label, seed=None):
        if label == "plain":
            return None
        if label == "flaug":
            return self.flaug
        return self._valid_folders(load_json(self.evidence_path(seed), None), f"seed_{seed}.json")

    def query_text(self, qt, variant, topic):
        """Same text RunGenerator.produce_topics_results / get_augmented_query build."""
        key = (qt, variant, topic["ID"])
        if key not in self._queries:
            title, desc, narr = (topic.get(f, "") for f in ("TITLE", "DESCRIPTION", "NARRATIVE"))
            if qt == "TDN":
                query = f"{title}. {desc}. {narr}".strip()
            elif qt == "TD":
                query = f"{title}. {desc}".strip()
            else:
                query = title.strip()
            if variant == "CT":
                aug = build_core_themes_augmentation_text(self.ct[qt].get(topic["ID"], {}))
            elif variant:
                aug = build_augmentation_text(self.hip[qt].get(topic["ID"], {}), variant)
            else:
                aug = ""
            self._queries[key] = f"{query}. {aug}".strip() if aug else query
        return self._queries[key]

    def training_data(self, allfl, folder_aug):
        g = self.gen
        g.current_searching_field = ALLFL if allfl else TOFS
        g.all_folders_folder_label = allfl
        g.all_folders_folder_label_augmented = allfl and folder_aug is not None
        g.folder_label_augmented = (not allfl) and folder_aug is not None
        g._folder_label_augmentation_cache = folder_aug
        return g.prepare_training_data()

    # -- rankers ------------------------------------------------------------

    @property
    def emb(self):
        if self._emb is None:
            with timed("loaded Embeddings model all-mpnet-base-v2 (once for the whole run)"):
                self._emb = CachedEmbeddingsModel()
        return self._emb

    @property
    def colbert(self):
        if self._colbert is None:
            with timed(f"loaded ColBERT model lightonai/colbertv2.0 on {get_best_device()} (once for the whole run)"):
                self._colbert = ExactColBERT()
        return self._colbert

    def raw_rankings(self, spec, data, queries, where):
        """{(qt, variant): [doc DataFrame per topic]} for one index built from `data`."""
        kind, fields, tuned = spec
        name = f"{kind}{'-tuned' if tuned else ''} on {'+'.join(fields)}"
        dirs = set()
        t0 = time.time()
        if kind == "bm25":
            model, dirs = fresh_bm25(fields, tuned, data)
        else:
            model = self.emb if kind == "embeddings" else self.colbert
            t0 = time.time()  # model loading is logged on its own
            model.train(data)
        t_index = time.time() - t0
        out = {q: [model.search(self.query_text(q[0], q[1], t)) for t in self.topics] for q in sorted(queries, key=str)}
        drop_terrier_dirs(dirs)
        log(f"  {where} | {name}: index {len(data)} docs ({t_index:.1f}s), searched {len(queries)} queries "
            f"x {len(self.topics)} topics ({time.time() - t0 - t_index:.1f}s) [{query_names(queries)}]")
        return out

    def folder_ranking(self, docs_df, expansion):
        """Doc scores -> ranked folder list, as RunGenerator.produce_topics_results does it."""
        if expansion:
            self.gen.expansion = expansion
            df = self.gen.produce_expansion_results(docs_df)
        else:
            df = docs_df.groupby("folder", as_index=False)["score"].max()
        return df.sort_values("score", ascending=False)["folder"].drop_duplicates().tolist()

    def allfl_lists(self, needed, folder_aug, where):
        """needed: {allfl key: {query}} -> {(allfl key, query): results}"""
        if not needed:
            return {}
        data = self.training_data(allfl=True, folder_aug=folder_aug)
        out = {}
        for key, queries in sorted(needed.items()):
            raw = self.raw_rankings(ALLFL_RANKERS[key], data, queries, f"{where} ALLFL-{key}")
            for q, dfs in raw.items():
                out[(key, q)] = [{"Id": tid, "RankedList": self.folder_ranking(df, [])}
                                 for tid, df in zip(self.topic_ids, dfs)]
        return out

    def derive_doc(self, cfg, raw, query, partner=None):
        results = []
        for i, tid in enumerate(self.topic_ids):
            maps = {RRF_NAME[k]: raw[k][query][i] for k in cfg["rankers"]}
            if len(maps) > 1:
                self.gen.models = [RRF_NAME[k] for k in cfg["rankers"]]
                docs = self.gen.apply_document_level_rrf(maps)
            else:
                docs = next(iter(maps.values()))
            results.append({"Id": tid, "RankedList": self.folder_ranking(docs, cfg.get("expansion", []))})
        return perform_hybrid_fusion(results, partner) if partner is not None else results

    # -- output -------------------------------------------------------------

    def exp_dir(self, exp):
        return os.path.join(self.out_root, exp.name)

    def is_done(self, exp, seed):
        return os.path.isfile(os.path.join(self.exp_dir(exp), f"{metrics_name(seed)}_TopicsFolderMetrics.json"))

    def write(self, exp, seed, results):
        folder = self.exp_dir(exp)
        name = metrics_name(seed)
        info_path = os.path.join(folder, "info.json")
        if not os.path.isfile(info_path):
            os.makedirs(folder, exist_ok=True)
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump({"query_type": exp.qt, "table": exp.table, "config": exp.config,
                           "column": exp.column, "query_variant": QUERY_VARIANTS[exp.column],
                           "folder_label": exp.label, "seeds": exp.seeds or None,
                           "llm_root": os.path.relpath(self.llm_root, PROJECT_ROOT),
                           "colbert": "exact MaxSim, top 100"}, f, indent=2)
        top = [{"Id": r["Id"], "RankedList": r["RankedList"][:TOP_RUN_DEPTH]} for r in results]
        self.evaluator.save_run_file(top, os.path.join(folder, "runs", f"{name}.tsv"), exp.name)

        run_tmp = os.path.join(self.tmp_dir, f"run_{os.getpid()}.tsv")
        json_tmp = os.path.join(self.tmp_dir, f"metrics_{os.getpid()}.json")
        self.evaluator.save_run_file(results, run_tmp, exp.name)
        self.evaluator.evaluate(run_tmp, json_tmp)
        os.replace(json_tmp, os.path.join(folder, f"{name}_TopicsFolderMetrics.json"))  # last: resume marker

    def aggregate(self, exp):
        """Writes the evaluator's aggregate files once every seed is in; returns whether it did."""
        seeds = exp.seeds or [None]
        if not all(self.is_done(exp, s) for s in seeds):
            return False
        with contextlib.redirect_stdout(io.StringIO()):  # it prints the stats dict for every run
            self.evaluator.generate_aggregated_metrics(self.exp_dir(exp), "random" if exp.seeds else "all_folder_label")
        return True


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def pending_by_seed(engine, exps, seeds):
    """{seed or None: [experiments still missing that seed]}"""
    out = defaultdict(list)
    for exp in exps:
        for seed in exp.seeds or [None]:
            if (seed is None or seed in seeds) and not engine.is_done(exp, seed):
                out[seed].append(exp)
    return out


def plan_seed(exps):
    """For one seed's pending experiments: doc rankers needed per label, ALLFL keys needed for
    FLEV, and partner queries needed per label."""
    doc = defaultdict(lambda: defaultdict(set))  # label -> ranker -> {query}
    flev = defaultdict(set)  # allfl key -> {query}
    partners = defaultdict(lambda: defaultdict(set))  # label -> allfl key -> {query}
    for exp in exps:
        if "allfl" in exp.cfg:
            flev[exp.cfg["allfl"]].add(exp.query)
            continue
        for r in exp.cfg["rankers"]:
            doc[exp.label][r].add(exp.query)
        if "partner" in exp.cfg:
            partners[exp.label][exp.cfg["partner"]].add(exp.query)
    return doc, flev, partners


def run(engine, exps, seeds):
    t_start = time.time()
    pending = pending_by_seed(engine, exps, seeds)
    seeded = [s for s in seeds if pending.get(s)]
    n_pairs = sum(len(v) for v in pending.values())
    log(f"plan: {len(exps)} experiments, {n_pairs} (experiment, seed) pairs pending: "
        f"{len(pending.get(None, []))} seed-independent + {n_pairs - len(pending.get(None, []))} over "
        f"{len(seeded)} seeds {seeded}")
    if not n_pairs:
        log("nothing to do: every selected (experiment, seed) already has its metrics JSON")

    # 1. Seed-independent ALLFL rankers: the ALLFL rows, and the partners every seed's hybrids use.
    global_needed = defaultdict(lambda: defaultdict(set))
    for exp in pending.get(None, []):
        global_needed[exp.label][exp.cfg["allfl"]].add(exp.query)
    for seed in seeded:
        for label, by_key in plan_seed(pending[seed])[2].items():
            for key, queries in by_key.items():
                global_needed[label][key] |= queries
    allfl = {}
    if global_needed:
        log("=== phase 1/2: seed-independent ALLFL rankers (ALLFL rows + hybrid partners) ===")
        for label, needed in sorted(global_needed.items()):
            with timed(f"[global|{label}] ALLFL rankers done"):
                allfl[label] = engine.allfl_lists(needed, engine.folder_aug(label), f"[global|{label}]")
        if pending.get(None):
            with timed(f"[global] evaluated + wrote {len(pending[None])} seed-independent experiments"):
                for exp in pending[None]:
                    engine.write(exp, None, allfl[exp.label][(exp.cfg["allfl"], exp.query)])
                    engine.aggregate(exp)

    # 2. Per seed: every index once, every query against it, every config derived from it.
    if seeded:
        log(f"=== phase 2/2: {len(seeded)} seeds ===")
    t_seeds = time.time()
    for n, seed in enumerate(seeded, 1):
        t0 = time.time()
        tag = f"[seed {seed} | {n}/{len(seeded)}]"
        doc, flev, _ = plan_seed(pending[seed])
        log(f"{tag} start: {len(pending[seed])} experiments | doc-side labels: {sorted(doc) or '-'} | "
            f"FLEV: {'ALLFL-' + ','.join(sorted(flev)) if flev else '-'}")
        try:
            with timed(f"{tag} ECF sampled"):
                engine.gen.ecf = engine.gen.loader.create_random_ecf(seed, "uniform", docs_per_box=DOCS_PER_BOX)
            if any("expansion" in e.cfg for e in pending[seed]):
                with timed(f"{tag} folder relations for expansion built"):
                    engine.gen.relations = engine.gen.create_folder_relations_for_expansion(
                        engine.training_data(allfl=False, folder_aug=None))
            for label, rankers in sorted(doc.items()):
                where = f"{tag}[{label}]"
                data = engine.training_data(allfl=False, folder_aug=engine.folder_aug(label))
                log(f"{where} training set: {len(data)} sampled docs, rankers {sorted(rankers)}")
                raw = {r: engine.raw_rankings(DOC_RANKERS[r], data, queries, where)
                       for r, queries in sorted(rankers.items())}
                todo = [e for e in pending[seed] if e.label == label and "allfl" not in e.cfg]
                with timed(f"{where} derived (RRF/expansion/hybrid) + evaluated + wrote {len(todo)} experiments"):
                    for exp in todo:
                        partner = allfl[label][(exp.cfg["partner"], exp.query)] if "partner" in exp.cfg else None
                        engine.write(exp, seed, engine.derive_doc(exp.cfg, raw, exp.query, partner))
                del raw
            if flev:
                where = f"{tag}[flev]"
                lists = engine.allfl_lists(flev, engine.folder_aug("flev", seed), where)
                todo = [e for e in pending[seed] if "allfl" in e.cfg]
                with timed(f"{where} evaluated + wrote {len(todo)} experiments"):
                    for exp in todo:
                        engine.write(exp, seed, lists[(exp.cfg["allfl"], exp.query)])
        except Exception:
            log(f"{tag} FAILED; seeds before it are saved, rerun the same command to resume.\n"
                f"{traceback.format_exc()}")
            raise
        avg = (time.time() - t_seeds) / n
        log(f"{tag} done in {hms(time.time() - t0)} | elapsed {hms(time.time() - t_start)} | "
            f"avg {hms(avg)}/seed | ETA {hms(avg * (len(seeded) - n))}")

    with timed("aggregated metrics written"):
        complete = sum(engine.aggregate(exp) for exp in exps)
    log(f"{complete}/{len(exps)} experiments have every seed; "
        f"{len(exps) - complete} still incomplete for their seed list")
    log(f"finished {n_pairs} (experiment, seed) pairs in {hms(time.time() - t_start)}")


def dry_run(engine, exps, seeds):
    pending = pending_by_seed(engine, exps, seeds)
    by_table = defaultdict(int)
    for exp in exps:
        by_table[(exp.qt, exp.table)] += 1
    for (qt, table), n in sorted(by_table.items()):
        seed_counts = sorted({len(e.seeds) for e in exps if e.qt == qt and e.table == table})
        print(f"{qt:>3} {table:<5} {n:>3} experiments, seeds per experiment: "
              f"{', '.join(str(c) if c else '1 (seed-independent ALLFL)' for c in seed_counts)}")
    builds = 0
    for seed in seeds:
        if pending.get(seed):
            doc, flev, _ = plan_seed(pending[seed])
            builds += sum(len(r) for r in doc.values()) + len(flev)
    print(f"\n{len(exps)} experiments; {sum(len(v) for v in pending.values())} (experiment, seed) pairs pending")
    print(f"index builds: ~{builds} seeded + up to 6 seed-independent ALLFL")
    print(f"evidence seeds ({len(engine.evidence_seeds())}): {sorted(engine.evidence_seeds())}")
    for exp in exps[:3] + exps[-3:]:
        print(f"  e.g. {exp.name}")


# ---------------------------------------------------------------------------
# Parity check
# ---------------------------------------------------------------------------

def parity_check(engine, seed, qt="TD"):
    """The new derivation against RunGenerator.run_single_seed on one seed, for the
    deterministic rankers (BM25, Embeddings), plus exact MaxSim against a per-document loop."""

    def old_gen(fields, folder_aug=False, allfl=False, variant=None, **kw):
        g = RunGenerator(searching_fields=[fields], query_fields=[qt], all_folders_folder_label=allfl,
                         all_folders_folder_label_augmented=allfl and folder_aug,
                         folder_label_augmented=folder_aug and not allfl, query_augmentation=variant, **kw)
        g.expansion_ceiling_k = EXPANSION_CEILING_K
        g._augmentation_data_cache[qt] = engine.hip[qt]  # luna, not the Llama files
        g._core_themes_data_cache[qt] = engine.ct[qt]
        g._folder_label_augmentation_cache = engine.flaug
        return g

    def old_run(g, fields):
        while int(time.time()) <= _last_bm25_second[0]:
            time.sleep(0.05)
        before = _terrier_dirs()
        res = g.run_single_seed(seed, fields, qt)
        _last_bm25_second[0] = int(time.time())
        drop_terrier_dirs(_terrier_dirs() - before)
        return res

    def compare(name, old, new):
        same = [o["RankedList"] == n["RankedList"] for o, n in zip(old, new)]
        print(f"  {'OK  ' if all(same) else 'DIFF'} {name}: {sum(same)}/{len(same)} topics identical")
        return all(same)

    engine.gen.ecf = engine.gen.loader.create_random_ecf(seed, "uniform", docs_per_box=DOCS_PER_BOX)
    plain = engine.training_data(allfl=False, folder_aug=None)
    engine.gen.relations = engine.gen.create_folder_relations_for_expansion(plain)
    flaug = engine.training_data(allfl=False, folder_aug=engine.flaug)
    ok = True

    def new_doc(rankers, data, variant, expansion=None, partner=None):
        q = (qt, variant)
        raw = {r: engine.raw_rankings(DOC_RANKERS[r], data, {q}, "[parity]") for r in rankers}
        cfg = {"rankers": rankers, **({"expansion": expansion} if expansion else {})}
        return engine.derive_doc(cfg, raw, q, partner)

    print(f"parity check, seed {seed}, {qt}:")
    ok &= compare("B.TOFS PLAIN BASE", old_run(old_gen(TOFS, models=["bm25"], bm25_tuned=False), TOFS),
                  new_doc(["bm25u"], plain, None))
    ok &= compare("B.--F- FLAUG CT", old_run(old_gen(ALLFL, True, models=["bm25"], bm25_tuned=False,
                                                     variant="CT"), ALLFL),
                  new_doc(["bm25F"], flaug, "CT"))
    ok &= compare("E.TOFS PLAIN DOCFL", old_run(old_gen(TOFS, models=["embeddings"], variant="AUG"), TOFS),
                  new_doc(["emb"], plain, "AUG"))
    ok &= compare("-.----.-.e FLAUG FL", old_run(old_gen(ALLFL, True, True, "FL", models=["embeddings"]), ALLFL),
                  engine.allfl_lists({"e": {(qt, "FL")}}, engine.flaug, "[parity]")[("e", (qt, "FL"))])
    # RRF + similar_snc expansion + hybrid fusion, with deterministic rankers in place of ColBERT.
    gen_a = old_gen(TOFS, True, models=["bm25", "embeddings"], bm25_tuned=True, expansion=["similar_snc"],
                    variant="FL")
    gen_b = old_gen(ALLFL, True, True, "FL", models=["bm25"], bm25_tuned=False)
    old = perform_hybrid_fusion(old_run(gen_a, TOFS), old_run(gen_b, ALLFL))
    partner = engine.allfl_lists({"b": {(qt, "FL")}}, engine.flaug, "[parity]")[("b", (qt, "FL"))]
    ok &= compare("bm25t+emb s---2 + ALLFL bm25, FLAUG FL", old,
                  new_doc(["bm25t", "emb"], flaug, "FL", ["similar_snc"], partner))

    # Exact MaxSim: the batched, padded scores against a plain per-document loop. (Not against
    # pylate's rank.rerank: it zero-pads documents without masking, so short documents gain up
    # to ~1.2 whenever a query token's real similarities are all negative.)
    allfl_data = engine.training_data(allfl=True, folder_aug=None)
    cb = engine.colbert
    cb.train(allfl_data)
    doc_embs = [torch.as_tensor(e, dtype=torch.float32) for e in cb.model.encode(
        [d["text_blob"] for d in allfl_data], batch_size=512, is_query=False, show_progress_bar=False)]
    worst = 0.0
    for topic in engine.topics[:5]:
        q = engine.query_text(qt, None, topic)
        qe = cb.encode_query(q).cpu()
        ref = [float((qe @ e.T).max(dim=1).values.sum()) for e in doc_embs]
        worst = max(worst, max(abs(r - s) for r, s in zip(ref, cb.scores(q).tolist())))
    exact_ok = worst < 1e-3
    print(f"  {'OK  ' if exact_ok else 'DIFF'} exact MaxSim vs per-document loop: max |score diff| {worst:.2e}")
    ok &= exact_ok
    print("parity: all identical" if ok else "parity: MISMATCH (see above)")
    return ok


# ---------------------------------------------------------------------------

def config_list(value):
    configs = [c.strip() for c in value.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown configs {unknown}; choices: {','.join(CONFIGS)}")
    return configs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-types", nargs="+", default=QUERY_TYPES, choices=QUERY_TYPES)
    ap.add_argument("--tables", nargs="+", default=list(TABLES), type=str.upper, choices=list(TABLES))
    ap.add_argument("--configs", type=config_list, default=list(CONFIGS),
                    help="comma-separated; write it as --configs=... since the ALLFL names start with '-' "
                         f"(choices: {','.join(CONFIGS)})")
    ap.add_argument("--seeds", nargs="+", type=int, default=RANDOM_SEED_LIST)
    ap.add_argument("--llm-root", default=os.path.join(LUNA_ROOT, "hw"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "all_runs_luna"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parity-check", action="store_true")
    args = ap.parse_args()

    unknown = set(args.seeds) - set(RANDOM_SEED_LIST)
    if unknown:
        ap.error(f"seeds not in RANDOM_SEED_LIST: {sorted(unknown)}")
    seeds = [s for s in RANDOM_SEED_LIST if s in args.seeds]
    llm_root, out_root = (os.path.join(LAUNCH_DIR, p) for p in (args.llm_root, args.out))

    log(f"running_experiments pid {os.getpid()} | device {get_best_device()}"
        f"{' (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else ''}")
    log(f"  llm root : {llm_root}")
    log(f"  out      : {out_root}")
    log(f"  query    : {' '.join(args.query_types)} | tables: {' '.join(args.tables)}")
    log(f"  configs  : {' '.join(args.configs)}")
    log(f"  seeds    : {len(seeds)} {seeds}")
    with timed("loaded collection, topics and luna generations"):
        engine = Engine(llm_root, out_root)
    evidence = engine.evidence_seeds()
    missing = [s for s in seeds if s not in evidence]
    if "FLEV" in args.tables:
        log(f"  evidence : {len(evidence)} seeds with seed_<s>.json; FLEV skips {missing or 'none'}")
    if args.parity_check:
        sys.exit(0 if parity_check(engine, seeds[0]) else 1)
    exps = build_experiments(args.query_types, args.tables, args.configs, seeds, evidence)
    if args.dry_run:
        dry_run(engine, exps, seeds)
        return
    run(engine, exps, seeds)
    log("done")


if __name__ == "__main__":
    main()
