import collections, json
from typing import Literal
import dspy
from src.dspy_experiments.common import CTX, FOLDERS_PATH     # common.py: Step 2.3

########################
## CRIA LISTA DE SNCs ##
########################
_folders = json.load(open(FOLDERS_PATH, encoding="utf-8"))
_meanings = collections.defaultdict(collections.Counter)
for f in _folders.values():
    if f["snc"] != "Unknown" and f.get("label_parent_expanded"):
        _meanings[f["snc"]][f["label_parent_expanded"]] += 1

MEANING = {code: c.most_common(1)[0][0] for code, c in _meanings.items()}   # 378 codes
CODE_LIST = [f"{code} | {m}" for code, m in sorted(MEANING.items())]        # ~4,600 tokens
SNCCode = Literal[tuple(sorted(MEANING))]

#############################
## CLASSES PARA OS PROMPTS ##
#############################
class HypotheticalDocuments(dspy.Signature):
    """You are an expert historian of the records described in the collection context,
    reconstructing the documentary record. A researcher is looking for documents matching
    the information need. Write short passages that read as if they were excerpts from
    actual documents in this collection that would satisfy this need. Write each passage
    as the document itself, not as a description of it. Use the vocabulary, naming
    conventions, abbreviations and phrasing of the records' creators in that period. Name
    the posts, officials, ministries, agencies, programs, parties and places that such a
    document would name. Do not hedge, do not use conditional language, and do not refer
    to the researcher or to the search."""
    collection_context: str = dspy.InputField()
    information_need: str = dspy.InputField()
    passages: list[str] = dspy.OutputField(
        desc="exactly 3 passages of 60-90 words, each starting 'This document is about'; "
             "never invent telegram numbers, file references or document identifiers")


class PredictFiling(dspy.Signature):
    """You are an expert archivist of the filing system described in the collection
    context. A researcher is looking for documents matching the information need. Predict
    where in the filing system such documents would have been filed. Records were filed at
    the time of creation according to the administrative subject of the document, not
    according to its later historical interest. Choose codes on that basis: under what
    routine subject heading would it have been filed."""
    collection_context: str = dspy.InputField()
    information_need: str = dspy.InputField()
    code_list: list[str] = dspy.InputField(desc="every code used in this collection, as 'CODE | MEANING'")
    snc_codes: list[SNCCode] = dspy.OutputField(desc="3 to 5 codes from code_list, most likely first")
    subject_terms: list[str] = dspy.OutputField(
        desc="10-15 terms in the register of folder labels and filing headings: subject "
             "themes; not sentences")


class CoreThemes(dspy.Signature):
    """You are an expert archivist and historian helping a researcher search the collection
    described in the collection context. Restate the information need as it would be
    expressed in the language of the records themselves: the terminology of the records'
    creators in that period, naming the institutions, actors, policies and events that
    documents satisfying this need would discuss. Write it as a description of the
    documents being sought, not as a question or a request. Then list specific historical
    entities, institutions, programs, treaties, places, technologies and diplomatic terms
    that such documents would name, using local-language forms where those were the forms
    actually used. Do not include generic words or terms unrelated to the need."""
    collection_context: str = dspy.InputField()
    information_need: str = dspy.InputField()
    core_themes: str = dspy.OutputField(desc="one paragraph, 80-120 words")
    related_concepts: list[str] = dspy.OutputField(desc="15-20 terms")


class DescribeFolder(dspy.Signature):
    """You are an expert archivist and historian. Below is the archival classification of
    one physical folder of the collection described in the collection context; no documents
    from it are available. Describe what records filed under this classification, in this
    collection and period, would concern, so that a researcher's query can be matched
    against the folder although its contents are not digitized. Draw on established
    historical knowledge of that subject in that period: the institutions, actors, policies
    and events such records would concern. Do not invent specific documents, telegram
    numbers or incidents you cannot ground in the classification and the historical
    record. List the entities, institutions, programs, treaties, places, technologies and
    diplomatic terms such records would plausibly name, using local-language forms where
    those were the forms actually used."""
    collection_context: str = dspy.InputField()
    folder_label: str = dspy.InputField()
    snc: str = dspy.InputField()
    meaning: str = dspy.InputField()
    scope_note: str = dspy.InputField(desc="'None' when the catalogue has none")
    date_range: str = dspy.InputField(desc="'None to None' when unknown")
    core_themes: str = dspy.OutputField(desc="one paragraph of 100-150 words beginning 'This folder contains'")
    related_concepts: list[str] = dspy.OutputField(desc="15-20 terms")

class DescribeFolderWithEvidence(dspy.Signature):
    """You are an expert archivist and historian. Below is the archival classification of
    one physical folder of the collection described in the collection context, together
    with document evidence. Merge them into a single folder-level description that will be
    indexed for retrieval. Combine the classification's scope with the concrete evidence,
    in the terminology of the records' creators in that period. Draw on established
    historical knowledge to make explicit the context, institutions and policies implicit
    in this material, so that queries which do not use the folder's exact wording can still
    match it. Do not invent specific incidents, telegram numbers or named individuals that
    neither the evidence nor the historical record supports. List the entities,
    institutions, programs, treaties, places and diplomatic terms that are either present
    in the evidence or historically bound to this subject and period, using local-language
    forms where those were the forms actually used."""
    collection_context: str = dspy.InputField()
    folder_label: str = dspy.InputField()
    snc: str = dspy.InputField()
    meaning: str = dspy.InputField()
    scope_note: str = dspy.InputField(desc="'None' when the catalogue has none")
    date_range: str = dspy.InputField(desc="'None to None' when unknown")
    evidence_kind: Literal["own", "same_snc", "similar_snc", "same_box"] = dspy.InputField()
    evidence: list[str] = dspy.InputField(
        desc="'title - summary' of sampled documents. Unless evidence_kind is 'own', these "
             "documents are NOT in this folder: they come from folders nearby in the archive "
             "or sharing its classification, only show the kind of material this part of the "
             "collection holds, and must not be stated or implied to be in this folder")
    core_themes: str = dspy.OutputField(desc="one paragraph of 100-150 words beginning 'This folder contains'")
    related_concepts: list[str] = dspy.OutputField(desc="15-20 terms")

#############################
## VERIFICADORES DE SAÍDAS ##
#############################
def _words(s): return len(s.split())

def ok_docs(args, pred):
    p = pred.passages
    return float(len(p) == 3 and all(x.startswith("This document is about")
                                     and 40 <= _words(x) <= 120 for x in p))

def ok_filing(args, pred):
    return float(3 <= len(pred.snc_codes) <= 7 and all(c in MEANING for c in pred.snc_codes)
                 and 5 <= len(pred.subject_terms) <= 20)

def ok_core_themes(args, pred):
    return float(60 <= _words(pred.core_themes) <= 150 and 10 <= len(pred.related_concepts) <= 30)

def ok_folder(args, pred):
    return float(pred.core_themes.startswith("This folder contains")
                 and 70 <= _words(pred.core_themes) <= 185
                 and 10 <= len(pred.related_concepts) <= 30)

# Garante que retorna o valor dentro do resultado esperado do verificador
def refined(signature_or_module, reward):
    module = signature_or_module if isinstance(signature_or_module, dspy.Module) \
        else dspy.Predict(signature_or_module)
    return dspy.Refine(module=module, N=3, reward_fn=reward, threshold=1.0)

def run_batch(module, reward, inputs, rollout_start=0, num_threads=10):
    """Runs `module` over a list of input dicts. Returns [(pred_or_None, valid), ...] in order.

    rollout_start > 0 gives fresh (uncached) attempts: use it for the repair pass.
    """
    examples = [dspy.Example(**x).with_inputs(*x.keys()) for x in inputs]
    lm = dspy.settings.lm.copy(rollout_id=rollout_start)
    with dspy.context(lm=lm):
        preds = module.batch(examples, num_threads=num_threads, max_errors=len(examples),
                             disable_progress_bar=False)
    out = []
    for x, p in zip(inputs, preds):
        valid = bool(p is not None and reward(x, p) == 1.0)
        out.append((p, valid))
    return out