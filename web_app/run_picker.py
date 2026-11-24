"""
run_picker.py
-------------
The run picker (a row of filters following the run-name slots) and the run card.

Family → QF → Retrieval config → (legacy) Protocol | (luna) Folder-side → Query-side → Query set → Reranker.
Each filter offers the values present among the runs that match every filter before it, so the last filter
always leaves exactly one run.
"""

import hashlib
from typing import Dict, List, Optional, Tuple

import streamlit as st

import llm_registry as reg
import utils_runs as ur

LEGACY_SLOTS = ["family", "qf", "config", "protocol"]
LUNA_SLOTS = ["family", "qf", "config", "table", "column", "query_set", "reranker"]

SLOT_TITLES = {"family": "Family", "qf": "QF", "config": "Retrieval config", "protocol": "Protocol",
               "table": "Folder-side", "column": "Query-side", "query_set": "Query set", "reranker": "Reranker"}

_ORDER = {
    "family": list(reg.FAMILY),
    "qf": list(reg.QF),
    "table": list(reg.TABLE),
    "column": list(reg.COLUMN),
    "query_set": list(reg.QUERY_SET),
    "reranker": list(reg.RERANKER),
    "protocol": list(reg.PROTOCOL),
}
_CONFIG_ORDER = list(reg.CONFIG)
_SAMPLE_ORDER = ["U5", "K5", "U1", "U2", "U3", "U4", "A-"]


def slot_value(info: ur.RunInfo, slot: str):
    if slot == "family":
        return info.family
    if slot == "config":
        return f"{info.slots['sample']} · {info.slots['config']}"
    return info.slots.get(slot)


def _sort_key(slot: str, value):
    if slot == "config":
        if value in _CONFIG_ORDER:
            return (0, _CONFIG_ORDER.index(value), "")
        sample, _, cfg = value.partition(" · ")
        s = _SAMPLE_ORDER.index(sample) if sample in _SAMPLE_ORDER else 99
        return (1, s, cfg)
    order = _ORDER.get(slot, [])
    return (0, order.index(value), "") if value in order else (1, 0, str(value))


def format_value(slot: str, value) -> str:
    if slot == "family":
        return f"{reg.FAMILY.get(value, value)}"
    if slot == "qf":
        return f"{reg.QF.get(value, value)} ({value})"
    if slot == "config":
        sample, _, cfg = value.partition(" · ")
        return f"{reg.describe_config(sample, cfg)} ({value})"
    if slot == "protocol":
        return reg.PROTOCOL.get(value, value)
    label = reg.SLOT_LABELS.get(slot, {}).get(value, value)
    return label if value is None else f"{label} ({value})"


def slots_for(family: str) -> List[str]:
    return LEGACY_SLOTS if family == "legacy" else LUNA_SLOTS


def filter_chain(runs: Dict[str, ur.RunInfo], selections: Dict) -> Tuple[List[Tuple[str, list, object]], List[str]]:
    """Pure filter logic. Returns ([(slot, options, chosen), ...], names of the runs left).
    A selection that is not among a slot's options is replaced by the first option."""
    candidates = list(runs.values())
    steps = []
    slot_list = LUNA_SLOTS
    i = 0
    while i < len(slot_list):
        slot = slot_list[i]
        options = sorted({slot_value(r, slot) for r in candidates}, key=lambda v: _sort_key(slot, v))
        if not options:
            break
        chosen = selections.get(slot)
        if chosen not in options:
            chosen = options[0]
        steps.append((slot, options, chosen))
        candidates = [r for r in candidates if slot_value(r, slot) == chosen]
        if slot == "family":
            slot_list = slots_for(chosen)
        i += 1
    return steps, [r.name for r in candidates]


def _selections(key: str, default: Optional[str], runs: Dict[str, ur.RunInfo]) -> Dict:
    """The picker's chosen slot values, kept in session state apart from the widgets (a keyed selectbox
    forgets its value when its options change; the picker keeps any value that is still valid)."""
    state_key = f"{key}__sel"
    if state_key not in st.session_state:
        info = runs.get(default) if default else None
        st.session_state[state_key] = ({s: slot_value(info, s) for s in slots_for(info.family)} if info else {})
    return st.session_state[state_key]


def _widget_key(key: str, slot: str, options: list) -> str:
    """One widget per (slot, option list): a new option list makes a new widget, created at the stored value."""
    digest = hashlib.md5(repr(options).encode()).hexdigest()[:8]
    return f"{key}_{slot}_{digest}"


def run_picker(key: str, default: Optional[str] = None) -> Optional[str]:
    """A row of slot filters; returns the selected run name (or None if there are no runs)."""
    runs = ur.scan_runs()
    if not runs:
        st.error("No valid run folders found.")
        return None
    sel = _selections(key, default, runs)

    # Draw the widgets one at a time: each widget's value narrows the next one's options. A stored value
    # that an upstream change made invalid is reset to the first option.
    candidates = list(runs.values())
    family_options = sorted({slot_value(r, "family") for r in candidates}, key=lambda v: _sort_key("family", v))
    if sel.get("family") not in family_options:
        sel["family"] = family_options[0]
    slot_list = slots_for(sel["family"])
    row1 = st.columns([1.3, 1.3, 3])
    row2 = st.columns(4) if len(slot_list) > 3 else []
    cells = list(row1) + list(row2)
    for i, slot in enumerate(slot_list):
        options = sorted({slot_value(r, slot) for r in candidates}, key=lambda v: _sort_key(slot, v))
        if sel.get(slot) not in options:
            sel[slot] = options[0]
        with cells[i]:
            value = st.selectbox(SLOT_TITLES[slot], options, index=options.index(sel[slot]),
                                 key=_widget_key(key, slot, options),
                                 format_func=lambda v, s=slot: format_value(s, v))
        if value != sel[slot]:
            sel[slot] = value
            if slot == "family":                      # the slot list changes: draw again
                st.rerun()
        candidates = [r for r in candidates if slot_value(r, slot) == value]

    if len(candidates) != 1:
        st.warning(f"{len(candidates)} runs match these filters.")
        return candidates[0].name if candidates else None
    name = candidates[0].name
    st.code(name, language=None)
    return name


# ──────────────────────────────────────────────────────────────────────────────
# Run card
# ──────────────────────────────────────────────────────────────────────────────
def render_run_card(name: Optional[str]):
    """Decoded slots of the run, with the registry's one-line aim for each LLM slot."""
    if not name:
        return
    info = ur.run_info(name)
    if info is None:
        return
    s = info.slots
    with st.container(border=True):
        sample, cfg = s.get("sample"), s.get("config")
        lines = [f"**QF:** {reg.QF.get(s.get('qf'), s.get('qf'))}",
                 f"**Retrieval config:** {reg.describe_config(sample, cfg)} (`{sample} · {cfg}`)"]
        if info.family == "legacy":
            lines.append(f"**Protocol:** {reg.PROTOCOL.get(s.get('protocol'), s.get('protocol'))}")
        else:
            for slot, title in (("table", "Folder-side"), ("column", "Query-side"),
                                ("query_set", "Query set"), ("reranker", "Reranker")):
                v = s.get(slot)
                if v is None:
                    continue
                label = reg.SLOT_LABELS[slot].get(v, v)
                lines.append(f"**{title}:** {label} (`{v}`)")
            if s.get("sample") == "K5":
                lines.append(f"**Sample:** {reg.SAMPLE['K5']} (FL-ev evidence drawn from the uneven sample)")
            if s.get("reranker"):
                first = ur.load_info_json(name).get("first_stage")
                if first:
                    lines.append(f"**First stage:** `{first}`")
        if info.seed_independent:
            lines.append("**Seeds:** seed-independent (paired by topic in the significance test)")
        st.markdown("  \n".join(lines))
        if info.family == "luna":
            st.caption("Details, prompts and examples: 🤖 LLM Experiments page")
