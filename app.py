import html
import json
import os
import re
import time
from datetime import date, datetime, timezone

import pandas as pd
import pymupdf
import streamlit as st
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEADLINES_FILE = "deadlines.json"
MODELS = {"Gemma 4 26B (fast)": "gemma-4-26b-a4b-it", "Gemma 4 31B (stronger)": "gemma-4-31b-it"}
LANGS = ["English", "Urdu", "Hindi", "Arabic", "Spanish", "French"]


def default_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


def h(x) -> str:
    return html.escape("" if x is None else str(x))


def as_list(x) -> list:
    if isinstance(x, list):
        return x
    return [x] if x else []


def dicts(x) -> list:
    return [i for i in as_list(x) if isinstance(i, dict)]


# ------------------------------------------------------------ agent prompts
EXTRACTOR_PROMPT = """You are the Extractor Agent in a report-reading pipeline.
You receive a few pages of an official report. Pages are marked [Page N]; some pages are attached as images.
Extract only what is written. Keep the report's original language. Copy numbers and dates exactly.
Reply with ONLY valid JSON (no markdown, no code fences) using exactly these keys:
{
 "title": "report title or null",
 "issuer": "who issued it or null",
 "report_date": "date of the report as written, or null",
 "facts": [
  {"type": "decision | figure | obligation | deadline | finding | other",
   "text": "one short factual statement",
   "page": 3,
   "date_iso": "YYYY-MM-DD if a complete date is stated, else null"}
 ]
}
Never invent details. If the pages contain nothing relevant, return an empty facts list."""


def analyst_prompt(lang: str) -> str:
    return f"""You are the Analyst Agent. You receive facts extracted from an official report as JSON.
Use ONLY those facts. Write every text value in {lang}, in simple words a 12-year-old could follow.
Reply with ONLY valid JSON (no markdown, no code fences) using exactly these keys:
{{
 "document_type": "what kind of report this is",
 "summary": "3-4 sentence plain explanation",
 "key_points": ["the most important points"],
 "who_is_affected": "who this report matters to, or null",
 "actions": ["short steps the reader should take"],
 "urgency": "low | medium | high",
 "urgency_reason": "one sentence why",
 "risks": "what happens if ignored, or null",
 "unclear": "anything missing or uncertain, or null"
}}
Do not add information that is not in the facts."""


VERIFIER_PROMPT = """You are the Verifier Agent. You receive a JSON list of items, each with an id, a claim and a source_excerpt.
For each item, decide whether the source_excerpt supports the claim. Numbers and dates must match exactly.
Reply with ONLY valid JSON (no markdown, no code fences):
{"results": [{"id": 0, "status": "supported | unsupported | unclear", "note": "one short reason"}]}
Judge only against the excerpt given. Do not use outside knowledge."""


RESEARCH_FOCI = {
    "laws": "the laws, regulations and legal rights that apply (statutes, acts, ordinances, tax or municipal rules)",
    "procedure": "the official procedure: how to comply, appeal or object, required forms, fees, offices and time limits",
}


def research_prompt(jurisdiction: str, lang: str, today: str, focus: str) -> str:
    return f"""You are the Legal Research Agent. Today is {today}.
You receive facts extracted from an official report as JSON.
Jurisdiction given by the user: "{jurisdiction or 'not given'}". If none was given, infer the most likely country and region from the facts (issuer, currency, language, laws named) and say so.
Focus of this search: {focus}.
Use Google Search at least twice with different queries. Prefer official government, court and regulator websites over news or blogs.
Base every rule ONLY on pages you found in this search, never on memory.
Write text values in {lang}. Reply with ONLY valid JSON (no markdown, no code fences) using exactly these keys:
{{
 "jurisdiction_used": "country and region you applied",
 "jurisdiction_source": "user | inferred",
 "topic_area": "for example tax, tenancy, employment, traffic fine, public notice, immigration",
 "rules": [
  {{"name": "name of the law, rule or procedure",
    "what_it_says": "plain explanation in 1-2 sentences",
    "effect_on_this_case": "how it affects the situation in the report",
    "official_source": "authority or website where it can be confirmed"}}
 ],
 "authorities": ["bodies the reader can contact or appeal to"],
 "gaps": "what you could not find or confirm, or null"
}}
Never invent law names, section numbers, fees or deadlines. If you cannot confirm a rule, leave it out and mention it in "gaps"."""


def advisor_prompt(lang: str, today: str) -> str:
    return f"""You are the Advisor Agent. Today is {today}.
You receive extracted facts, an analysis, the rules found by the Legal Research Agent, and deadlines with days left (calculated by code).
Think step by step and show your reasoning: connect each fact to a rule, then to a consequence, then to an action.
Write every text value in {lang}, in simple words. Use ONLY the facts and rules provided. If something needed is missing, list it under "assumptions" instead of guessing.
Reply with ONLY valid JSON (no markdown, no code fences) using exactly these keys:
{{
 "situation": "2-3 sentences: what is happening to the reader and why it matters",
 "reasoning": [{{"fact": "...", "rule": "...", "conclusion": "what follows for the reader"}}],
 "options": [{{"title": "...", "when_it_fits": "...", "steps": ["..."], "pros": "...", "cons": "..."}}],
 "recommended_path": [{{"step": "...", "why": "...", "by_when": "date or relative time, or null"}}],
 "documents_to_prepare": ["..."],
 "who_to_contact": ["..."],
 "risks_if_ignored": "...",
 "see_a_lawyer_if": "specific situations where professional help is needed",
 "assumptions": ["..."],
 "confidence": "low | medium | high"
}}
Order recommended_path by urgency, nearest deadline first. Do not invent section numbers, fees or deadlines.
This is legal information, not legal advice."""


def qa_prompt(lang: str) -> str:
    return f"""You are the Q&A and Advisor Agent for one official report. Today is {date.today().isoformat()}. Answer in {lang}.
Use ONLY the page excerpts, page images, extracted facts, legal rules and action plan provided (plus web results if search is enabled).
For "what should I do" or "how do I" questions, reason step by step: 1) the relevant fact from the report [p. N], 2) the rule that applies, 3) what follows for the reader, 4) the concrete steps in order, with deadlines.
State any assumption you make. If the material does not contain the answer, say what is missing. Never invent laws, section numbers, fees or dates.
This is legal information, not legal advice."""


# ------------------------------------------------------------ model helpers
def parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def sources_from(resp):
    out = []
    try:
        chunks = resp.candidates[0].grounding_metadata.grounding_chunks or []
    except (AttributeError, IndexError, TypeError):
        return out
    for c in chunks:
        web = getattr(c, "web", None)
        if web and getattr(web, "uri", None):
            out.append({"title": getattr(web, "title", None), "url": web.uri})
    return out


def ask_full(client, model, instruction, parts, search=False, tries=3):
    cfg = types.GenerateContentConfig(
        system_instruction=instruction,
        tools=[{"google_search": {}}] if search else None,
    )
    for attempt in range(tries):
        try:
            resp = client.models.generate_content(model=model, contents=parts, config=cfg)
            return resp.text or "", sources_from(resp)
        except Exception as e:
            transient = any(c in str(e) for c in ("500", "503", "INTERNAL", "UNAVAILABLE"))
            if transient and attempt < tries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise


def ask(client, model, instruction, parts, tries=3) -> str:
    return ask_full(client, model, instruction, parts, tries=tries)[0]


def run_agent(client, model, instruction, parts, log, retries=1, search=False) -> dict:
    for _ in range(retries + 1):
        raw, sources = ask_full(client, model, instruction, parts, search=search)
        log.append(raw)
        try:
            data = parse_json(raw)
            if search:
                data["_sources"] = sources
            return data
        except ValueError:
            continue
    raise ValueError("An agent replied in the wrong format.")


# ------------------------------------------------------------ retrieval (plain Python)
def load_pages(file, max_pages):
    data = file.getvalue()
    if file.type == "application/pdf":
        doc = pymupdf.open(stream=data, filetype="pdf")
        pages = []
        for i, pg in enumerate(doc, start=1):
            if i > max_pages:
                break
            text = pg.get_text().strip()
            img = pg.get_pixmap(dpi=110).tobytes("png") if len(text) < 40 else None
            pages.append({"n": i, "text": text, "img": img, "mime": "image/png"})
        return pages, len(doc)
    return [{"n": 1, "text": "", "img": data, "mime": file.type}], 1


def chunk_pages(pages, size):
    return [pages[i:i + size] for i in range(0, len(pages), size)]


def chunk_parts(chunk):
    text = "\n\n".join(f"[Page {p['n']}]\n{p['text']}" for p in chunk if p["text"])
    parts = [text] if text else []
    for p in chunk:
        if p["img"]:
            parts.append(f"[The next image is Page {p['n']}]")
            parts.append(types.Part.from_bytes(data=p["img"], mime_type=p["mime"]))
    return parts


def page_corpus(pages, facts):
    """Searchable text per page. Scanned pages use the facts the Extractor found on them."""
    by_page = {}
    for f in facts:
        if f.get("page") is not None:
            by_page.setdefault(f["page"], []).append(f["text"])
    return [{**p, "search": p["text"] or " ".join(by_page.get(p["n"], []))} for p in pages]


def top_pages(question, corpus, k=3):
    docs = [p for p in corpus if p["search"].strip()]
    if not docs:
        return corpus[:k]
    try:
        m = TfidfVectorizer(stop_words="english").fit_transform([p["search"] for p in docs] + [question])
    except ValueError:
        return docs[:k]
    sims = cosine_similarity(m[-1], m[:-1]).ravel()
    return [docs[i] for i in sims.argsort()[::-1][:k]]


def qa_parts(hits, facts, question, legal=None, plan=None):
    parts = []
    for p in hits:
        if p["text"]:
            parts.append(f"[Page {p['n']}]\n{p['text'][:3000]}")
        else:
            parts.append(f"[Page {p['n']} is attached as an image]")
        if p["img"]:
            parts.append(types.Part.from_bytes(data=p["img"], mime_type=p["mime"]))
    ctx = ""
    if legal:
        ctx += "Applicable rules found:\n" + json.dumps(slim_legal(legal), ensure_ascii=False) + "\n\n"
    if plan:
        ctx += "Action plan already prepared:\n" + json.dumps(plan, ensure_ascii=False) + "\n\n"
    known = "\n".join(f"- p. {f.get('page')}: {f['text']}" for f in facts["facts"][:60])
    parts.append(f"{ctx}Facts already extracted from the report:\n{known}\n\nQuestion: {question}")
    return parts


def merge(results):
    out = {"title": None, "issuer": None, "report_date": None, "facts": []}
    seen = set()
    for r in results:
        for k in ("title", "issuer", "report_date"):
            out[k] = out[k] or r.get(k)
        for f in r.get("facts") or []:
            key = str(f.get("text", "")).strip().lower() if isinstance(f, dict) else ""
            if key and key not in seen:
                seen.add(key)
                try:
                    f["page"] = int(f.get("page"))
                except (TypeError, ValueError):
                    f["page"] = None
                out["facts"].append(f)
    for i, f in enumerate(out["facts"]):
        f["id"] = i
    return out


def verify(client, model, facts, pages, log, batch=15):
    by_n = {p["n"]: p["text"] for p in pages}
    items = [{"id": f["id"], "claim": f["text"], "source_excerpt": by_n[f["page"]][:3000]}
             for f in facts if f.get("page") in by_n and by_n[f["page"]]]
    verdicts = {}
    for i in range(0, len(items), batch):
        r = run_agent(client, model, VERIFIER_PROMPT, [json.dumps(items[i:i + batch], ensure_ascii=False)], log)
        for v in r.get("results") or []:
            try:
                verdicts[int(v["id"])] = v
            except (KeyError, TypeError, ValueError):
                pass
    return verdicts


# ------------------------------------------------------------ legal research (search-grounded)
OFFICIAL_RE = re.compile(
    r"(^|\.)(gov|gob|gouv|mil|edu)(\.[a-z]{2})?$|(^|\.)go\.[a-z]{2}$|(^|\.)europa\.eu$|(^|\.)un\.org$")


def is_official(src) -> bool:
    for field in (src.get("title") or "", src.get("url") or ""):
        host = re.sub(r"^https?://", "", field.strip().lower()).split("/")[0]
        if OFFICIAL_RE.search(host):
            return True
    return False


def research(client, model, facts, summary, jurisdiction, lang, today, log):
    merged = {"jurisdiction_used": None, "jurisdiction_source": None, "topic_area": None,
              "rules": [], "authorities": [], "gaps": [], "_sources": []}
    seen_rules, seen_urls = set(), set()
    for focus in RESEARCH_FOCI.values():
        r = run_agent(client, model,
                      research_prompt(jurisdiction or merged["jurisdiction_used"] or "", lang, today, focus),
                      [json.dumps({"facts": facts, "summary": summary}, ensure_ascii=False)],
                      log, search=True)
        srcs = r.get("_sources", [])
        for k in ("jurisdiction_used", "jurisdiction_source", "topic_area"):
            merged[k] = merged[k] or r.get(k)
        for rule in r.get("rules") or []:
            name = rule.get("name") if isinstance(rule, dict) else None
            if not name or name in seen_rules:
                continue
            seen_rules.add(name)
            if srcs:  # keep a rule only if this search returned sources
                rule["sources"] = srcs
                merged["rules"].append(rule)
            else:
                merged["gaps"].append(f"No source found for: {name}")
        merged["authorities"] += [a for a in as_list(r.get("authorities")) if a not in merged["authorities"]]
        if r.get("gaps"):
            merged["gaps"].append(str(r["gaps"]))
        for s in srcs:
            if s.get("url") and s["url"] not in seen_urls:
                seen_urls.add(s["url"])
                merged["_sources"].append(s)
    merged["gaps"] = "; ".join(merged["gaps"]) or None
    return merged


def slim_legal(legal):
    out = {k: v for k, v in legal.items() if k != "_sources"}
    out["rules"] = [{k: v for k, v in r.items() if k != "sources"} for r in dicts(legal.get("rules"))]
    return out


# ------------------------------------------------------------ action stage (no AI)
def collect_deadlines(facts, verdicts):
    out = []
    for f in facts:
        d = f.get("date_iso")
        try:
            date.fromisoformat(d)
        except (TypeError, ValueError):
            continue
        if verdicts.get(f["id"], {}).get("status") == "unsupported":
            continue
        out.append({"date": d, "text": f["text"], "page": f.get("page")})
    return sorted(out, key=lambda x: x["date"])


def save_deadlines(new, title):
    try:
        with open(DEADLINES_FILE, encoding="utf-8") as fh:
            saved = json.load(fh)
    except (OSError, ValueError):
        saved = []
    have = {(s["date"], s["text"]) for s in saved}
    saved += [{**d, "report": title} for d in new if (d["date"], d["text"]) not in have]
    with open(DEADLINES_FILE, "w", encoding="utf-8") as fh:
        json.dump(saved, fh, ensure_ascii=False, indent=2)


def ics_escape(s):
    return str(s).replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def make_ics(deadlines, title):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//DocDecode//EN"]
    for i, d in enumerate(deadlines):
        day = d["date"].replace("-", "")
        lines += ["BEGIN:VEVENT", f"UID:orr-{day}-{i}@reportreader", f"DTSTAMP:{stamp}",
                  f"DTSTART;VALUE=DATE:{day}", f"SUMMARY:{ics_escape(d['text'][:80])}",
                  f"DESCRIPTION:{ics_escape(title or 'Official report')}"]
        for n in (7, 3, 1):
            lines += ["BEGIN:VALARM", "ACTION:DISPLAY", "DESCRIPTION:Reminder", f"TRIGGER:-P{n}D", "END:VALARM"]
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)


# ------------------------------------------------------------ UI helpers
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
  --bg: #050706;
  --surface: #101312;
  --surface-raised: #171b19;
  --surface-soft: #0b100e;
  --ink: #f1f4f1;
  --muted: #8d9690;
  --line: #2a332e;
  --accent: #25e6a1;
  --accent-dark: #13bd82;
  --accent-soft: #12392d;
  --danger-soft: #3b2429;
  --warning-soft: #3b331e;
  --success-soft: #173c30;
}

html, body, .stApp, button, input, textarea, select, [class*="css"] {font-family: 'DM Sans', sans-serif !important;}
.stApp {background: radial-gradient(circle at 50% -10%, #123c2b 0, #07130e 24rem, var(--bg) 52rem); color: var(--ink);}
.stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"] {color: var(--ink);}
header[data-testid="stHeader"] {display: block !important; height: 0 !important; background: transparent !important;
  z-index: 1000;}
[data-testid="stSidebarCollapsedControl"] {display: flex !important; visibility: visible !important;
  position: fixed !important; top: 1rem !important; left: 1rem !important; z-index: 1001 !important;}
[data-testid="stSidebarCollapsedControl"] button {display: flex !important; visibility: visible !important;
  background: var(--surface-raised) !important; border: 1px solid var(--line) !important;
  color: var(--accent) !important; border-radius: 9px !important; box-shadow: 0 6px 16px rgba(0, 0, 0, .25);}
#MainMenu, footer, [data-testid="stDecoration"], .stDeployButton {display: none !important;}
.block-container {padding-top: 2.25rem; padding-bottom: 6rem; max-width: 920px;}

/* ---------- brand ---------- */
.brand {display: flex; align-items: center; gap: 11px; margin-bottom: 24px; padding-bottom: 18px;
        border-bottom: 1px solid var(--line);}
.brand svg {width: 36px; height: 36px; filter: drop-shadow(0 0 10px rgba(37, 230, 161, .28));}
.brand span {font-weight: 700; font-size: 1.05rem; letter-spacing: -.01em; color: var(--ink) !important;}

/* ---------- chat ---------- */
[data-testid="stChatMessage"] {background: transparent; padding: .75rem 0;}
[data-testid="stChatMessageAvatarAssistant"] {background: var(--accent-soft) !important; color: var(--accent) !important;}
[data-testid="stChatMessageAvatarUser"] {background: #27312c !important; color: var(--ink) !important;}
[data-testid="stChatInput"] {background: var(--surface-raised); border: 1px solid var(--line); border-radius: 14px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, .24);}
[data-testid="stChatInput"]:focus-within {border-color: var(--accent);}
[data-testid="stChatInput"] textarea, [data-testid="stChatInput"] textarea::placeholder {color: var(--ink) !important;}
[data-testid="stChatInput"] textarea::placeholder {opacity: .58;}

/* ---------- cards ---------- */
.card {border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; background: var(--surface);
       box-shadow: 0 8px 24px rgba(0, 0, 0, .2); margin-bottom: 12px;}
.card h4 {margin: 0 0 6px 0; font-size: .72rem; font-weight: 600; letter-spacing: .06em;
          text-transform: uppercase; color: var(--muted) !important;}
.card p {margin: 0; line-height: 1.7;}
.card ul {margin: 0; padding-left: 18px; line-height: 1.7;}
.card li::marker {color: var(--accent);}

/* ---------- urgency pills ---------- */
.pill {display: inline-block; padding: 5px 12px; border-radius: 999px; font-weight: 600; font-size: .78rem;
       letter-spacing: .02em;}
.pill-high {background: var(--danger-soft); color: #f0a4ad !important;}
.pill-medium {background: var(--warning-soft); color: #e7cc7e !important;}
.pill-low {background: var(--success-soft); color: #76e0b2 !important;}

/* ---------- facts ---------- */
.fact {border: 1px solid var(--line); border-left: 4px solid #8aa5af; padding: 12px 15px; margin: 8px 0;
       background: var(--surface); border-radius: 0 10px 10px 0;}
.fact.ok {border-left-color: #25e6a1;}
.fact.bad {border-left-color: #dc7d8b;}
.fact.unk {border-left-color: #d0af5f;}
.small {color: var(--muted) !important; font-size: .82rem; margin-top: 2px;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0a1712 0%, #07100d 100%); border-right: 1px solid #294c3c;
  box-shadow: 8px 0 30px rgba(0, 0, 0, .18);}
[data-testid="stSidebar"] .block-container {padding: 2rem 1.25rem;}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {letter-spacing: -.02em;}

/* ---------- buttons ---------- */
.stButton > button[kind="primary"], .stDownloadButton > button {
  background: var(--accent); color: #04120d !important; border: 1px solid var(--accent);
  border-radius: 10px; font-weight: 700; box-shadow: 0 6px 18px rgba(37, 230, 161, .16);}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
  background: var(--accent-dark); border-color: var(--accent-dark);}
.stButton > button[kind="primary"]:disabled {background: #294038; border-color: #294038; color: #7d968b !important;}
.stButton > button:not([kind="primary"]) {border: 1px solid var(--line); color: var(--ink); border-radius: 10px;
  background: var(--surface-raised);}
.stButton > button:not([kind="primary"]):hover {border-color: var(--accent); color: var(--accent);}

/* ---------- tabs ---------- */
[data-baseweb="tab-list"] {gap: 4px; border-bottom: 1px solid var(--line);}
[data-baseweb="tab"] {color: var(--muted);}
[data-baseweb="tab"][aria-selected="true"] {color: var(--accent) !important; font-weight: 600;}
[data-baseweb="tab-highlight"] {background-color: var(--accent) !important;}

/* ---------- other widgets ---------- */
[data-testid="stExpander"] {background: var(--surface); border: 1px solid var(--line); border-radius: 12px;}
[data-testid="stFileUploaderDropzone"] {background: var(--surface-soft); border: 1.5px dashed #40584b; border-radius: 12px;}
[data-testid="stFileUploaderDropzone"] button {color: var(--ink); background: var(--surface-raised); border-color: var(--line);}
[data-testid="stAlert"] {border-radius: 10px; background: var(--surface-raised);}
[data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {background: var(--accent) !important; border-color: var(--accent) !important;}
[data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] svg {fill: #04120d !important;}
[data-baseweb="select"] > div, [data-testid="stTextInput"] input {background: var(--surface-raised); color: var(--ink); border-color: var(--line);}
[data-testid="stTextInput"] input::placeholder {color: var(--muted); opacity: .8;}
[data-testid="stDataFrame"] {border: 1px solid var(--line); border-radius: 10px; overflow: hidden;}
[data-testid="stCaptionContainer"] p {color: var(--muted) !important;}
a {color: var(--accent) !important;}
</style>
"""

APP_NAME = "DocDecode"
ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.svg")
with open(ICON_PATH, encoding="utf-8") as icon_file:
    LOGO = icon_file.read().replace("Official Report Reader", APP_NAME)

STATUS = {"supported": ("ok", "Verified against source"),
          "unsupported": ("bad", "Not found in source"),
          "unclear": ("unk", "Unclear")}


def card(label, value):
    if value:
        st.markdown(f'<div class="card"><h4>{h(label)}</h4><p>{h(value)}</p></div>', unsafe_allow_html=True)


def card_list(label, items):
    items = as_list(items)
    if items:
        li = "".join(f"<li>{h(i)}</li>" for i in items)
        st.markdown(f'<div class="card"><h4>{h(label)}</h4><ul>{li}</ul></div>', unsafe_allow_html=True)


# ------------------------------------------------------------ app
st.set_page_config(page_title=APP_NAME, page_icon=ICON_PATH, layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

api_key = default_key()
if not api_key:
    st.error("No API key found. Add GEMINI_API_KEY to a .env file next to app.py, then restart the app.")
    st.stop()

with st.sidebar:
    st.markdown(f'<div class="brand">{LOGO}<span>{APP_NAME}</span></div>', unsafe_allow_html=True)
    st.subheader("Settings")
    model_label = st.selectbox("Model", list(MODELS))
    lang = st.selectbox("Language", LANGS)
    use_legal = st.checkbox("Include legal guidance", value=True)
    country = st.text_input("Country", placeholder="e.g. Pakistan")
    region = st.text_input("Region or city (optional)", placeholder="e.g. Punjab")
    web = st.checkbox("Search the web when answering questions")
    per_chunk, max_pages, show_raw = 4, 30, False
    if st.button("Start a new report", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.caption("Report content is sent to Google's Gemini API. AI explanation only, not legal or financial advice.")

# ---- welcome message + upload (disappears after analysis)
intro = st.empty()
file, run = None, False
if not st.session_state.get("analysis"):
    with intro.container():
        with st.chat_message("assistant"):
            st.markdown("Hi! Upload an official report and I'll explain it in plain words, check the facts, "
                        "find the deadlines and tell you what to do next. You can then ask me anything about it.")
            file = st.file_uploader("Upload a report (PDF) or a page photo",
                                    type=["pdf", "jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
            run = st.button("Analyze report", type="primary", disabled=not file)

if run:
    with st.chat_message("user"):
        st.markdown(f"📎 {file.name}")
    client = genai.Client(api_key=api_key)
    model = MODELS[model_label]
    log = []
    with st.chat_message("assistant"):
        try:
            with st.status("Reading your report...", expanded=True) as status:
                pages, total = load_pages(file, max_pages)
                st.write(f"Loaded {len(pages)} of {total} pages.")
                chunks = chunk_pages(pages, per_chunk)
                results = []
                for i, ch in enumerate(chunks, 1):
                    st.write(f"Extracting key details: part {i} of {len(chunks)}")
                    results.append(run_agent(client, model, EXTRACTOR_PROMPT, chunk_parts(ch), log))
                facts = merge(results)
                st.write(f"Explaining {len(facts['facts'])} facts")
                analysis = run_agent(client, model, analyst_prompt(lang), [json.dumps(facts, ensure_ascii=False)], log)
                st.write("Checking facts against the original pages")
                verdicts = verify(client, model, facts["facts"], pages, log)
                deadlines = collect_deadlines(facts["facts"], verdicts)

                legal, plan = None, None
                if use_legal:
                    jurisdiction = ", ".join(x for x in (region.strip(), country.strip()) if x)
                    today = date.today().isoformat()
                    st.write("Researching laws and official procedures")
                    legal = research(client, model, facts, analysis.get("summary"),
                                     jurisdiction, lang, today, log)
                    st.write("Building your action plan")
                    dl_days = [{**dl, "days_left": (date.fromisoformat(dl["date"]) - date.today()).days}
                               for dl in deadlines]
                    plan = run_agent(client, model, advisor_prompt(lang, today),
                                     [json.dumps({"facts": facts, "analysis": analysis, "deadlines": dl_days,
                                                  "rules": slim_legal(legal)}, ensure_ascii=False)], log)
                status.update(label="Analysis complete", state="complete")
            save_deadlines(deadlines, facts.get("title"))
            st.session_state.update(pages=pages, facts=facts, analysis=analysis, verdicts=verdicts,
                                    deadlines=deadlines, legal=legal, plan=plan, raw=log, qa=[],
                                    fname=file.name)
            intro.empty()
            st.rerun()
        except ValueError as e:
            st.error(f"{e} Try again or switch model.")
            st.session_state["raw"] = log
        except Exception as e:
            st.error(f"Something went wrong: {e}")

d = st.session_state.get("analysis")

if d:
    facts, verdicts, deadlines = st.session_state["facts"], st.session_state["verdicts"], st.session_state["deadlines"]
    urg = str(d.get("urgency", "")).lower()
    level = "high" if "high" in urg else "medium" if "medium" in urg else "low"
    n_ok = sum(1 for v in verdicts.values() if v.get("status") == "supported")

    with st.chat_message("user"):
        st.markdown(f"📎 {st.session_state.get('fname', 'Report')}")

    with st.chat_message("assistant"):
        st.markdown(f'<span class="pill pill-{level}">Urgency: {h(level.capitalize())}</span>'
                    f'&nbsp; <span class="small">{h(d.get("urgency_reason"))}</span>', unsafe_allow_html=True)
        st.caption(f"{len(facts['facts'])} facts found · {n_ok} verified · {len(deadlines)} deadline(s)")

        t_sum, t_facts, t_dead, t_legal, t_exp = st.tabs(
            ["Summary", "Facts", "Deadlines", "What to do", "Export"])

        with t_sum:
            card("In plain words", d.get("summary"))
            card_list("Key points", d.get("key_points"))
            card_list("What to do", d.get("actions"))
            card("Who is affected", d.get("who_is_affected"))
            card("If ignored", d.get("risks"))
            card("Unclear or missing", d.get("unclear"))

        with t_facts:
            view = st.radio("Show", ["All", "Verified", "Needs review"], horizontal=True)
            for f in facts["facts"]:
                v = verdicts.get(f["id"])
                if v:
                    cls, label = STATUS.get(v.get("status"), ("unk", "Unclear"))
                else:
                    cls, label = "", "Not checked"
                if view == "Verified" and cls != "ok":
                    continue
                if view == "Needs review" and cls == "ok":
                    continue
                note = f" · {h(v['note'])}" if v and v.get("note") else ""
                st.markdown(f'<div class="fact"><b>Page {h(f.get("page"))}</b> · {h(f["text"])}'
                            f'<div class="small">{h(label)}{note}</div></div>'.replace('class="fact"', f'class="fact {cls}"', 1),
                            unsafe_allow_html=True)

        with t_dead:
            if not deadlines:
                st.info("No dated deadlines were found in this report.")
            else:
                rows = []
                for dl in deadlines:
                    left_days = (date.fromisoformat(dl["date"]) - date.today()).days
                    state = "Passed" if left_days < 0 else "Due soon" if left_days <= 7 else "Upcoming"
                    rows.append({"Date": dl["date"], "Days left": left_days, "Status": state,
                                 "Deadline": dl["text"], "Page": dl["page"]})
                urgent = [r for r in rows if r["Status"] == "Due soon"]
                if urgent:
                    st.warning(f"{len(urgent)} deadline(s) due within 7 days.")
                if any(r["Status"] == "Passed" for r in rows):
                    st.error("Some deadlines have already passed.")
                st.dataframe(pd.DataFrame(rows), hide_index=True)

        with t_legal:
            legal, plan = st.session_state.get("legal"), st.session_state.get("plan")
            if not legal or not plan:
                st.info("Turn on 'Include legal guidance' in the sidebar and analyze the report again.")
            else:
                st.caption(f"Based on: {legal.get('jurisdiction_used')}"
                           + ("" if legal.get("jurisdiction_source") == "user"
                              else " (guessed from the report; set your country in the sidebar if wrong)"))
                card("Your situation", plan.get("situation"))

                st.markdown("**Recommended steps**")
                for i, s in enumerate(dicts(plan.get("recommended_path")), 1):
                    by = f" (by {s['by_when']})" if s.get("by_when") else ""
                    st.markdown(f"**{i}. {s.get('step')}**{by}")
                    st.caption(s.get("why"))

                card_list("Documents to prepare", plan.get("documents_to_prepare"))
                card_list("Who to contact", as_list(plan.get("who_to_contact")) + as_list(legal.get("authorities")))
                card("See a lawyer if", plan.get("see_a_lawyer_if"))

                with st.expander("Laws and rules that apply"):
                    for rule in dicts(legal.get("rules")):
                        st.markdown(f"**{rule.get('name') or 'Rule'}**")
                        st.write(rule.get("what_it_says"))
                        st.caption(f"Effect on this case: {rule.get('effect_on_this_case')}")
                with st.expander("Other options"):
                    for o in dicts(plan.get("options")):
                        st.markdown(f"**{o.get('title') or 'Option'}**")
                        st.caption(f"When it fits: {o.get('when_it_fits')}")
                        for step in as_list(o.get("steps")):
                            st.markdown(f"- {step}")
                with st.expander("Sources"):
                    srcs = [s for s in legal.get("_sources", []) if s.get("url")]
                    if srcs and not any(is_official(s) for s in srcs):
                        st.warning("No official government source was found. Treat these rules with extra caution.")
                    for s in srcs:
                        tag = "Official" if is_official(s) else "Other"
                        st.markdown(f"- **{tag}** · [{s['title'] or s['url']}]({s['url']})")
                st.caption("Legal information, not legal advice. Confirm with the official source or a lawyer.")

        with t_exp:
            st.download_button("Download full result (.json)",
                               json.dumps({"facts": facts, "analysis": d, "verification": verdicts,
                                           "legal": st.session_state.get("legal"),
                                           "plan": st.session_state.get("plan")},
                                          ensure_ascii=False, indent=2), "report_summary.json")
            if deadlines:
                st.download_button("Download calendar reminders (.ics)",
                                   make_ics(deadlines, facts.get("title")), "deadlines.ics")

    # ---- chat history
    for item in st.session_state.get("qa", []):
        question, answer = item[0], item[1]
        qsrcs = item[2] if len(item) > 2 else []
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
            for s in qsrcs[:5]:
                st.caption(f"[{s['title'] or s['url']}]({s['url']})")

    # ---- chat input (pinned to the bottom)
    q = st.chat_input("Ask about this report, e.g. How do I challenge this fine, and by when?")
    if q and q.strip():
        client = genai.Client(api_key=api_key)
        corpus = page_corpus(st.session_state["pages"], facts["facts"])
        hits = top_pages(q, corpus)
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    ans, qsrcs = ask_full(client, MODELS[model_label], qa_prompt(lang),
                                          qa_parts(hits, facts, q, st.session_state.get("legal"),
                                                   st.session_state.get("plan")), search=web)
                    st.session_state["qa"].append((q, ans, qsrcs))
                    st.markdown(ans)
                    for s in qsrcs[:5]:
                        st.caption(f"[{s['title'] or s['url']}]({s['url']})")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")