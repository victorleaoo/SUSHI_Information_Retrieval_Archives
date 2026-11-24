import collections, json, re
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


class Doc2QueryFolder(dspy.Signature):
    """You are an expert archivist and historian. Below is the archival classification of
    one physical folder of the collection described in the collection context; no documents
    from it are available. Researchers search this collection by subject: each search names
    one subject briefly, like a heading in a subject index, and the archive must point them
    to the folders that hold records on it. Anticipate the subjects that would lead a
    researcher to this folder. First identify the broad facets of the subject that records
    filed under this classification, in this collection and period, would cover, including
    less obvious matters that the filing practice of the period would still have placed
    here. Then list keywords for those facets: the terminology of the records' creators in
    that period, with local-language forms where those were used, and the plain words a
    present-day researcher would use for the same things. Finally, under the facets, write
    search topics: narrower subject headings, each naming one specific matter that records
    in this folder would document, such as a program, event, place, institution, agreement,
    commodity, piece of equipment or activity, with Brazilian states, regions or years where
    they apply. Together the search topics must cover every facet. Write each as a noun
    phrase, not as a question or a sentence. Draw on established historical knowledge of the
    subject in that period, and name only programs, events and places attested for Brazil in
    that period; do not invent documents, telegram numbers or incidents."""
    collection_context: str = dspy.InputField()
    folder_label: str = dspy.InputField()
    snc: str = dspy.InputField()
    meaning: str = dspy.InputField()
    scope_note: str = dspy.InputField(desc="'None' when the catalogue has none")
    date_range: str = dspy.InputField(desc="'None to None' when unknown")
    facets: list[str] = dspy.OutputField(desc="3 to 5 broad facets, each a short subject heading")
    keywords: list[str] = dspy.OutputField(desc="10 terms, in period and present-day forms")
    search_topics: list[str] = dspy.OutputField(desc="N search topics, each a noun phrase of 3-10 words")  # N: d2q_signature
    topic_facets: list[int] = dspy.OutputField(
        desc="for each search topic, in the same order, the number (1 = first) of the facet it belongs to")


def d2q_signature(n_topics):
    return Doc2QueryFolder.with_updated_fields(
        "search_topics", desc=f"exactly {n_topics} search topics, each a noun phrase of 3-10 words")


# Step 9: the same prompts as .QH and .FH, plus the web notes of generation/web_notes.py and one paragraph on
# using them, so the notes are the only difference. Checked with ok_core_themes and ok_folder, as in Step 2.
WEB_NOTES_QUERY = (
    "Web notes are also given (see web_notes): specifics a web search found for this need. Take "
    "from them the names and terms that records satisfying the need would use (institutions, "
    "programs, laws, events, places, positions, commodities, technical terms), in the records' own "
    "forms, and put them in related_concepts and, where they fit, in core_themes. The need decides "
    "what is relevant: ignore anything in the notes that is unrelated to it or outside the "
    "collection's period, and do not copy their sentences.")
WEB_NOTES_FOLDER = (
    "Web notes are also given (see web_notes): specifics a web search found for this "
    "classification's subject in this period. Take from them the names and terms that records "
    "under this classification would use (institutions, programs, laws, events, places, "
    "positions, commodities, technical terms), and put them in related_concepts and, where they "
    "fit, in core_themes. The notes describe the subject in general, not this folder: never state "
    "or imply that what they describe is in this folder, and ignore anything in them that is "
    "unrelated to the classification or outside this folder's date range. Do not copy their "
    "sentences.")
_NOTES = dspy.InputField(desc="100-200 words of background from a web search, in English")

CoreThemesWeb = (CoreThemes.append("web_notes", _NOTES, type_=str)
                 .with_instructions(CoreThemes.instructions + "\n\n" + WEB_NOTES_QUERY))
DescribeFolderWeb = (DescribeFolder.append("web_notes", _NOTES, type_=str)
                     .with_instructions(DescribeFolder.instructions + "\n\n" + WEB_NOTES_FOLDER))
# dissertation_plan.md E3: FL-ev's prompt with the same notes and paragraph (generation/generate_web_grounded.py).
DescribeFolderWithEvidenceWeb = (DescribeFolderWithEvidence.append("web_notes", _NOTES, type_=str)
                                 .with_instructions(DescribeFolderWithEvidence.instructions + "\n\n"
                                                    + WEB_NOTES_FOLDER))


# Step 10: listwise reranking of a first stage's top 20 (rerank.py). RerankFolders for the first stages that
# searched the sampled documents (candidates show their title and summary), RerankFoldersByLabel for the others:
# the same prompt without the sentences about documents.
class RerankFolders(dspy.Signature):
    """You are an expert archivist of the collection described in the collection context.
    Given a researcher's information need and a numbered list of candidate folders, order
    the candidates from most to least likely to contain documents that satisfy the need.
    Judge each folder by its classification and by the titles and summaries of any of its
    documents that happen to be digitized. Most folders have no digitized documents, and a
    folder without documents can still be the best match."""
    collection_context: str = dspy.InputField()
    information_need: str = dspy.InputField()
    candidates: list[str] = dspy.InputField(
        desc="'[n] LABEL | MEANING | SCOPE NOTE | document: TITLE - SUMMARY' for n = 1..20, with one "
             "'document:' part for each of up to 3 of the folder's digitized documents, or 'documents: none'")
    ranking: list[int] = dspy.OutputField(desc="every candidate number, each exactly once, best first")


class RerankFoldersByLabel(dspy.Signature):
    """You are an expert archivist of the collection described in the collection context.
    Given a researcher's information need and a numbered list of candidate folders, order
    the candidates from most to least likely to contain documents that satisfy the need.
    Judge each folder by its classification: its label, the meaning of its subject code and
    its scope note."""
    collection_context: str = dspy.InputField()
    information_need: str = dspy.InputField()
    candidates: list[str] = dspy.InputField(desc="'[n] LABEL | MEANING | SCOPE NOTE' for n = 1..20")
    ranking: list[int] = dspy.OutputField(desc="every candidate number, each exactly once, best first")

# dissertation_plan.md day 2: the folder-text audit (analysis/audit.py). Judges one folder text against the folder's
# classification and a sample of its documents; the text's version (FL-aug, FL-ev, FW, FL-ev+web) is never shown.
class AuditFolderText(dspy.Signature):
    """You are an expert archivist of the collection described in the collection context.
    A short text was written to describe one folder so that researchers can find it by
    search. Check it against the folder's classification and against a sample of the
    documents the folder actually holds. Judge what the text says the folder contains or
    is about, and set each flag that applies:
    wrong_era_or_place: the text puts the folder's contents in a period outside the
    folder's dates and its documents' dates, or in a country or region that neither the
    classification nor the documents are about.
    unsupported_specifics: the text presents specific events, people, organizations,
    programs, agreements, places or figures as the folder's contents, and neither the
    classification nor the documents make them likely. The documents are only a sample:
    a specific is not unsupported merely because no shown document names it, but it is
    when it belongs to a different subject than the folder's documents. General background
    on the period, not presented as the folder's contents, is not a flag.
    generic_filler: beyond restating the classification, the text says nothing that would
    tell this folder apart from any other folder with the same subject code, such as
    boilerplate about correspondence, reports and memoranda.
    Several flags can apply; a text with none is faithful."""
    collection_context: str = dspy.InputField()
    folder_label: str = dspy.InputField()
    meaning: str = dspy.InputField(desc="the meaning of the folder's subject code")
    scope_note: str = dspy.InputField(desc="'None' when the catalogue has none")
    dates: str = dspy.InputField(desc="the folder's date range and the range of its documents' dates")
    documents: list[str] = dspy.InputField(desc="'TITLE - SUMMARY' of up to 10 of the folder's documents")
    text: str = dspy.InputField(desc="the text to check")
    wrong_era_or_place: bool = dspy.OutputField()
    unsupported_specifics: bool = dspy.OutputField()
    generic_filler: bool = dspy.OutputField()
    note: str = dspy.OutputField(desc="at most 25 words: the phrase behind each flag set, or 'none'")

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

# Step 8: a search topic must be a subject heading: not a question, and no words about searching or wanting
_NOT_A_HEADING = re.compile(r"\s*(please\s+)?(find|search|look|locate|show|tell|give|retrieve|i|we|my|our|"
                            r"how|what|which|why|when|where|who|whom|whose|did|does|do|is|are|was|were|can|could)\b",
                            re.I)


def ok_d2q(n_topics):
    def reward(args, pred):
        facets, topics = pred.facets or [], pred.search_topics or []
        covers = pred.topic_facets or []
        return float(3 <= len(facets) <= 5 and 8 <= len(pred.keywords or []) <= 12
                     and len(topics) == n_topics and all(2 <= _words(t) <= 12 for t in topics)
                     and not any("?" in t or _NOT_A_HEADING.match(t) for t in topics)   # headings only
                     and len({t.strip().lower() for t in topics}) == n_topics            # no repeats
                     and len(covers) == n_topics
                     and set(covers) == set(range(1, len(facets) + 1)))   # every facet used, no other number
    return reward

# Step 10: a reranking must be a permutation of the candidate numbers
def ok_permutation(args, pred):
    return float(sorted(pred.ranking or []) == list(range(1, len(args["candidates"]) + 1)))


def repair_permutation(ranking, n):
    """A valid order of 1..n from a model answer: drop repeats and out-of-range numbers, then append
    the missing candidates in their first-stage order. Returns (order, repaired?)."""
    seen = []
    for x in ranking or []:
        if isinstance(x, int) and 1 <= x <= n and x not in seen:
            seen.append(x)
    order = seen + [i for i in range(1, n + 1) if i not in seen]
    return order, order != list(ranking or [])

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