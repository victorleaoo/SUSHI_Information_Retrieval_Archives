"""Step 9a: web-search notes for topics and folder labels (OpenAI Responses API, built-in web search).

One call per item: the model searches the web and writes short English notes. The notes, every URL
the search opened and every URL cited are cached in data/llm_calls_luna/web/notes/{kind}.json, so each
item is searched once. Items:
  topics : one per topic and formulation (T, TD, TDN), searched with that formulation's text only
  folders: one per distinct folder label; folders sharing a label share its notes

Leakage control: the search blocks BLOCKED_DOMAINS, and every answer is checked afterwards. If a source
is on a blocked domain, or the notes mention SUSHI or NTCIR, the item is searched once more; if that
also fails, the item is marked "dropped" and Step 9b gives it its no-web (hw/) text. An answer with no
notes, an incomplete response or no search at all is retried and dropped the same way: notes written
without a search are not web notes.

Resumes: items already ok or dropped are kept, unless their prompt changed since (each record keeps the
sha256 of the prompt it was searched with), so a pilot (--limit) continues with the full run, and a
prompt edited after the pilot is searched again. An item that ended in an API error is searched again by
the next run of the same command. The first item is searched alone: a systematic API error (blocked_domains
or the effort rejected) stops the run there, before the other calls.

Usage (project root):
  python3 -m src.dspy_experiments.generation.web_notes --kind folders --dry-run    (items, searches, maximum fee)
  python3 -m src.dspy_experiments.generation.web_notes --kind topics --limit 5     (pilot)
  python3 -m src.dspy_experiments.generation.web_notes --kind topics --show        (notes, queries, sources; no call)
  nohup python3 -m src.dspy_experiments.generation.web_notes --kind topics > data/llm_calls_luna/logs/web_topics.out 2>&1 &
  nohup python3 -m src.dspy_experiments.generation.web_notes --kind folders > data/llm_calls_luna/logs/web_folders.out 2>&1 &
  python3 -m src.dspy_experiments.generation.web_notes --kind folders --summary     (statuses, leaks, source sites; no call)
"""
import argparse
import collections
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv

from src.dspy_experiments.common import (CTX, LUNA_ROOT, MODEL_NAME, QUERY_TYPE_FIELDS, build_folder_fields,
                                         build_query_text, load_folders, load_json, load_topics, log,
                                         normalize_field, save_json)

NOTES_DIR = os.path.join(LUNA_ROOT, "web", "notes")
BLOCKED_DOMAINS = ["catalog.archives.gov", "history.state.gov", "research.nii.ac.jp", "arxiv.org", "github.com"]
LEAK_WORDS = re.compile(r"\b(SUSHI|NTCIR)\b", re.I)
SETTINGS = {"topics": {"max_tool_calls": 2, "search_context_size": "medium"},
            "folders": {"max_tool_calls": 1, "search_context_size": "low"}}
PRICE_PER_SEARCH = 0.01                                  # $10 per 1,000 web search calls
KINDS = list(SETTINGS)

# ---------------------------------------------------------------- hand-written instructions (review these)
TOPIC_PROMPT = """You are helping a historian search the archival collection described below.

{ctx}

Research need:
{need}

Search the web for this research need as it concerns that collection's subject and period. Sources \
in English or Portuguese are both useful. Then write notes in English, 120-200 words, made of the \
specifics that records about this need would name: institutions and agencies, programs, laws and \
agreements, named events with their years, places (states, cities, regions), officials' positions, \
commodities and technical terms, each in the form used at the time, with the Portuguese name or \
acronym where one was used. Leave out general background, assessments, and anything after the \
collection's period. Report only what your sources support. Write plain prose: no URLs, no citations, \
no headings, no comments about the search."""

FOLDER_PROMPT = """You are helping an archivist describe folders of the archival collection described below.

{ctx}

The folders are filed under this classification:
Label: {label}
Subject: {meaning}
Scope note: {scope}

Search the web for what this subject involved in relations between the United States and Brazil, \
and in Brazil itself, between 1963 and 1973. Sources in English or Portuguese are both useful. Then \
write notes in English, 100-150 words, made of the specifics that records filed under this subject \
would name: institutions and agencies, programs, laws and agreements, named events with their years, \
places (states, cities, regions), officials' positions, commodities and technical terms, each in the \
form used at the time, with the Portuguese name or acronym where one was used. Leave out general \
background, assessments, and anything after 1973. Report only what your sources support. Write plain \
prose: no URLs, no citations, no headings, no comments about the search."""


# ---------------------------------------------------------------- items
def topic_key(topic_id, query_type):
    return f"{topic_id}|{query_type}"


def topic_items():
    items = {}
    for t in load_topics():
        for qf, fields in QUERY_TYPE_FIELDS.items():
            items[topic_key(t["ID"], qf)] = TOPIC_PROMPT.format(ctx=CTX, need=build_query_text(t, fields))
    return items


def label_of(folder_id, folder):
    return normalize_field(folder.get("label")) or folder_id


def folder_items():
    """{label: prompt}, one per distinct label; the first folder with that label gives meaning and scope
    (every folder of a label has the same ones)."""
    items = {}
    for fid, folder in sorted(load_folders().items()):
        label = label_of(fid, folder)
        if label not in items:
            f = build_folder_fields(folder)
            items[label] = FOLDER_PROMPT.format(ctx=CTX, label=label, meaning=f["parent_expanded_snc"] or "None",
                                                scope=f["scope_note"] or "None")
    return items


def items_of(kind):
    return topic_items() if kind == "topics" else folder_items()


def notes_path(kind):
    return os.path.join(NOTES_DIR, f"{kind}.json")


def prompt_sha(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def is_done(rec, prompt):
    """Searched for good (ok or dropped) with this very prompt."""
    return rec.get("status") in ("ok", "dropped") and rec.get("prompt_sha256") == prompt_sha(prompt)


# ---------------------------------------------------------------- one call
def make_client():
    load_dotenv()
    from openai import OpenAI
    return OpenAI(max_retries=6)


def domain_blocked(url):
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)


_CITATION = re.compile(r"\s*\((?:\s*\[[^\]]*\]\(https?://[^)]*\)\s*[,;]?)+\s*\)")   # ([site](url)), the API's citations
_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)]*)\)")


def clean_notes(text):
    """Citations and URLs out; a markdown link keeps its words unless they are only a site name."""
    text = _CITATION.sub("", text or "")
    text = _LINK.sub(lambda m: "" if re.fullmatch(r"[\w.-]+\.[a-z]{2,}", m.group(1).strip()) else m.group(1), text)
    text = re.sub(r"https?://\S+", "", text)
    return " ".join(text.split())


def parse(response):
    """Notes, every URL the search listed or opened, cited URLs, search queries, number of tool calls,
    the response status and usage, from a Responses object."""
    d = response.model_dump() if hasattr(response, "model_dump") else response
    notes, sources, cited, queries, searches = "", set(), set(), [], 0
    for item in d.get("output") or []:
        if item.get("type") == "web_search_call":
            searches += 1
            action = item.get("action") or {}
            queries += [q for q in [action.get("query"), *(action.get("queries") or [])] if q]
            sources.update(s.get("url") for s in action.get("sources") or [] if s.get("url"))
            if action.get("url"):                                   # open_page, find_in_page
                sources.add(action["url"])
        elif item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    notes += c.get("text") or ""
                    cited.update(a.get("url") for a in c.get("annotations") or []
                                 if a.get("type") == "url_citation" and a.get("url"))
    return (notes, sorted(sources), sorted(cited), list(dict.fromkeys(queries)), searches, d.get("status"),
            d.get("usage"))


def search(client, kind, prompt, effort, use_filter):
    tool = {"type": "web_search", "search_context_size": SETTINGS[kind]["search_context_size"]}
    if use_filter:
        tool["filters"] = {"blocked_domains": BLOCKED_DOMAINS}
    return client.responses.create(model=MODEL_NAME, input=prompt, tools=[tool], tool_choice="required",
                                   max_tool_calls=SETTINGS[kind]["max_tool_calls"],
                                   include=["web_search_call.action.sources"], reasoning={"effort": effort})


def attempt_status(notes, leaked, mentions, searches, response_status):
    if leaked or mentions:
        return "leak"
    if not searches:
        return "no_search"
    if not notes.strip() or response_status != "completed":
        return "empty"
    return "ok"


def notes_for(client, kind, item_id, prompt, effort, use_filter):
    attempts = []
    for _ in (1, 2):
        notes, sources, cited, queries, searches, response_status, usage = parse(
            search(client, kind, prompt, effort, use_filter))
        in_text = re.findall(r"https?://[^\s)\]]+", notes)
        leaked = sorted(u for u in set(sources) | set(cited) | set(in_text) if domain_blocked(u))
        mentions = sorted({m.upper() for m in LEAK_WORDS.findall(notes)})
        status = attempt_status(notes, leaked, mentions, searches, response_status)
        attempts.append({"status": status, "leaked": leaked, "mentions": mentions, "searches": searches,
                         "queries": queries, "sources": sources, "cited": cited, "response_status": response_status,
                         "usage": usage, "notes": clean_notes(notes)})
        if status == "ok":
            break
    last = attempts[-1]
    ok = last["status"] == "ok"
    return {"item": item_id, "status": "ok" if ok else "dropped", "drop_reason": None if ok else last["status"],
            "notes": last["notes"] if ok else "", "sources": last["sources"], "cited": last["cited"],
            "searches": sum(a["searches"] for a in attempts), "attempts": attempts, "model": MODEL_NAME,
            "effort": effort, "filter": use_filter, "prompt_sha256": prompt_sha(prompt),
            "created_at": datetime.now().isoformat(timespec="seconds")}


# ---------------------------------------------------------------- run
def rejected_filter(error_text):
    return f"The API rejected blocked_domains ({error_text}). Rerun with --no-filter: the check after each " \
           f"call still removes blocked sources."


def run(kind, effort, use_filter, threads, dry_run, limit=None):
    items = items_of(kind)
    path = notes_path(kind)
    cache = load_json(path, {})
    todo = [i for i in items if not is_done(cache.get(i, {}), items[i])][:limit]
    older = sum(1 for i in items if cache.get(i, {}).get("status") in ("ok", "dropped") and not is_done(cache[i], items[i]))
    max_searches = len(todo) * SETTINGS[kind]["max_tool_calls"] * 2          # worst case: every item retried
    log(f"{kind}: {len(items)} items, {len(todo)} to search"
        + (f" ({older} were searched with an older prompt)" if older else "")
        + f"; at most {max_searches} searches (${max_searches * PRICE_PER_SEARCH:.2f} in search fees; usually "
          f"about half that), plus the model's tokens")
    if dry_run or not todo:
        return
    client = make_client()

    def one(item_id):
        try:
            return notes_for(client, kind, item_id, items[item_id], effort, use_filter)
        except Exception as e:                      # left for the next run of the same command
            return {"item": item_id, "status": "error", "error": str(e)[:300],
                    "created_at": datetime.now().isoformat(timespec="seconds")}

    probe = one(todo[0])                            # alone: a systematic error stops here
    cache[todo[0]] = probe
    save_json(path, cache)
    if probe["status"] == "error":
        if use_filter and "blocked_domains" in probe["error"]:
            raise SystemExit(rejected_filter(probe["error"]))
        raise SystemExit(f"The first call failed: {probe['error']}\nNothing else was searched; fix it "
                         f"(e.g. another --effort) and rerun.")
    log(f"  first item {todo[0]!r}: {probe['status']}, {probe['searches']} searches, {len(probe['sources'])} sources")

    done = 1
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(one, i) for i in todo[1:]]
        for future in as_completed(futures):
            rec = future.result()
            cache[rec["item"]] = rec
            done += 1
            if done % 20 == 0 or done == len(todo):
                save_json(path, cache)
                log(f"  {done}/{len(todo)}")
    save_json(path, cache)
    s = summary(kind)
    log(f"{kind}: {s['status']}; {s['leaked_once']} items had a blocked source or a SUSHI/NTCIR mention at least "
        f"once; {s['searches']} searches in all (${s['searches'] * PRICE_PER_SEARCH:.2f})")
    if s["status"].get("error"):
        log(f"{s['status']['error']} items ended in an API error: rerun the same command")
    save_json(os.path.join(LUNA_ROOT, "web", "usage", f"notes_{kind}.json"),
              {k: s[k] for k in ("items", "status", "searches", "tokens", "searched_from", "searched_to")})


# ---------------------------------------------------------------- reading the cache (no call)
def summary(kind):
    """What the paper reports about one kind's notes: statuses, leaks, sources, searches, tokens, dates."""
    cache = load_json(notes_path(kind), {})
    recs = list(cache.values())
    attempts = [a for r in recs for a in r.get("attempts", [])]
    tokens = collections.Counter()
    for a in attempts:
        u = a.get("usage") or {}
        tokens["input"] += u.get("input_tokens") or 0
        tokens["output"] += u.get("output_tokens") or 0
        tokens["reasoning"] += (u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
    hosts = collections.Counter(urlparse(u).hostname for r in recs for u in r.get("sources", []))
    dates = sorted(r["created_at"] for r in recs if r.get("created_at") and r.get("status") != "error")
    ok = [r for r in recs if r.get("status") == "ok"]
    return {"items": len(recs), "of": len(items_of(kind)),
            "status": dict(collections.Counter(r.get("status") for r in recs)),
            "drop_reasons": dict(collections.Counter(r.get("drop_reason") for r in recs if r.get("status") == "dropped")),
            "leaked_once": sum(any(a["status"] == "leak" for a in r.get("attempts", [])) for r in recs),
            "blocked_hosts": dict(collections.Counter(urlparse(u).hostname for a in attempts for u in a.get("leaked", []))),
            "searches": sum(r.get("searches", 0) for r in recs),
            "tokens": dict(tokens),
            "mean_words": (sum(len(r["notes"].split()) for r in ok) / len(ok)) if ok else None,
            "mean_sources": (sum(len(r.get("sources", [])) for r in ok) / len(ok)) if ok else None,
            "hosts": hosts.most_common(15),
            "models": sorted({r.get("model") for r in recs if r.get("model")}),
            "efforts": sorted({r.get("effort") for r in recs if r.get("effort")}),
            "filters": sorted({r.get("filter") for r in recs if "filter" in r}),
            "searched_from": dates[0] if dates else None, "searched_to": dates[-1] if dates else None}


def print_summary(kind):
    s = summary(kind)
    print(f"{kind}: {s['items']}/{s['of']} items searched | {s['status']}"
          + (f" | dropped for {s['drop_reasons']}" if s["drop_reasons"] else ""))
    print(f"  {s['leaked_once']} items had a blocked source or a SUSHI/NTCIR mention at least once"
          + (f" (blocked sources by site, over all attempts: {s['blocked_hosts']})" if s["blocked_hosts"] else ""))
    print(f"  {s['searches']} searches (${s['searches'] * PRICE_PER_SEARCH:.2f}); tokens {s['tokens']}")
    if s["mean_words"]:
        print(f"  ok notes: mean {s['mean_words']:.0f} words, {s['mean_sources']:.1f} sources")
    print(f"  model {s['models']}, effort {s['efforts']}, blocked_domains sent {s['filters']}; "
          f"searched {s['searched_from']} to {s['searched_to']}")
    print("  most frequent source sites:")
    for host, n in s["hosts"]:
        print(f"    {n:>5}  {host}")


def show(kind, ids):
    cache = load_json(notes_path(kind), {})
    for item in ids or list(cache):
        r = cache.get(item)
        if r is None:
            print(f"\n=== {item}: not searched")
            continue
        print(f"\n=== {item} | {r['status']}{' (' + r['drop_reason'] + ')' if r.get('drop_reason') else ''} | "
              f"{r.get('searches', 0)} searches | {r.get('created_at')}")
        if r["status"] == "error":
            print(f"error: {r.get('error')}")
            continue
        for n, a in enumerate(r.get("attempts", []), 1):
            print(f"attempt {n}: {a['status']} | queries {a['queries']}"
                  + (f" | blocked {a['leaked']}" if a["leaked"] else "") + (f" | mentions {a['mentions']}" if a["mentions"] else ""))
        print(f"notes ({len(r['notes'].split())} words): {r['notes'] or '-'}")
        hosts = collections.Counter(urlparse(u).hostname for u in r.get("sources", []))
        print(f"sources ({len(r.get('sources', []))}): {', '.join(f'{h} x{n}' if n > 1 else h for h, n in hosts.most_common())}")
        print(f"cited: {r.get('cited')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--effort", default="low", help="reasoning effort of the search call (record which you used)")
    ap.add_argument("--no-filter", action="store_true", help="do not send blocked_domains (the check still runs)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="search only the first N items still to do (pilot)")
    ap.add_argument("--dry-run", action="store_true", help="items, searches and the maximum fee; no call")
    ap.add_argument("--show", nargs="*", default=None, metavar="ITEM",
                    help="print the cached notes (default: every item; a topic item is '<topic id>|<T|TD|TDN>'); no call")
    ap.add_argument("--summary", action="store_true", help="statuses, leaks, searches, source sites; no call")
    args = ap.parse_args()
    if args.show is not None:
        show(args.kind, args.show)
    elif args.summary:
        print_summary(args.kind)
    else:
        run(args.kind, args.effort, not args.no_filter, args.threads, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
