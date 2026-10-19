#!/usr/bin/env python3
"""Astra Q&A pipeline: parse -> retrieve -> synthesize -> cite.

The four stages map directly onto the product requirements:

  1. parse    — a local LLM reads the question, detects its language, and
                rewrites it into an English retrieval query (the docs are en-US,
                so translating acronyms/intent up front sharply improves recall).
  2. retrieve — embed the retrieval query and cosine-rank the release's index.
  3. answer   — a local LLM summarizes the retrieved chunks into an answer,
                written in the *same language the user asked in*.
  4. cite     — list the official ServiceNow doc sources actually used.

Everything runs locally against Ollama; nothing leaves the host.
"""
import os
import re
import sys
import json
import time

import numpy as np

import astra_lib as L


# --------------------------------------------------------------------------
# Index loading + retrieval
# --------------------------------------------------------------------------
def load_matrix(conn):
    rows = conn.execute("SELECT id, path, text, emb FROM chunks").fetchall()
    if not rows:
        raise RuntimeError(
            "Index is empty for this release. Build it with /reindex (or ./astra --index)."
        )
    ids, paths, texts, blobs = zip(*rows)
    M = np.array(
        np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(rows), -1)
    )
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    return list(paths), list(texts), M


def retrieve(query_text, paths, texts, M, k=6):
    q = L.embed([query_text], is_query=True)[0]
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(all="ignore"):
        scores = M @ q
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), paths[i], texts[i]) for i in top]


# --------------------------------------------------------------------------
# Stage 1 — query parsing (language detection + English retrieval query)
# --------------------------------------------------------------------------
PARSE_SYSTEM = (
    "You analyze a user's question about ServiceNow. Respond with ONE JSON object "
    "and nothing else. Keys:\n"
    '  "language": the natural language the question is written in, as an English '
    "name (e.g. English, Japanese, Chinese, Spanish).\n"
    '  "search_query": an English search query capturing the intent, with '
    "ServiceNow acronyms and product names expanded. The docs are English, so "
    "always translate the query to English.\n"
    '  "keywords": 3-8 English keywords/phrases for the topic.\n'
    "Output only the JSON object."
)


def _extract_json(s):
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def parse_query(question):
    """Return {language, search_query, keywords}. Degrades gracefully."""
    fallback = {"language": "", "search_query": question, "keywords": []}
    try:
        raw = L.generate(
            "Question: %s" % question,
            system=PARSE_SYSTEM,
            options={"temperature": 0.0, "num_ctx": 2048},
        )
    except Exception:
        return fallback
    data = _extract_json(raw) or {}
    sq = (data.get("search_query") or "").strip()
    return {
        "language": (data.get("language") or "").strip(),
        "search_query": sq if sq else question,
        "keywords": data.get("keywords") or [],
    }


# --------------------------------------------------------------------------
# Stage 3 — synthesis (answer in the question's language)
# --------------------------------------------------------------------------
def _answer_system(language):
    lang = language or "the same language as the question"
    return (
        "You are a ServiceNow documentation assistant. Answer the user's question "
        "using ONLY the provided context excerpts. If the context does not contain "
        "the answer, say so plainly and suggest what to search for instead — do not "
        "invent facts. Be concise and well-structured.\n"
        "IMPORTANT: Write your entire answer in %s, matching the language the user "
        "asked in. Do not add a sources section; the tool appends sources itself." % lang
    )


def synthesize(question, hits, language):
    context = "\n\n".join(
        "[Source %d: %s]\n%s" % (r + 1, L.rel_markdown_path(p), t)
        for r, (_, p, t) in enumerate(hits)
    )
    prompt = "Context:\n%s\n\nQuestion: %s\n\nAnswer:" % (context, question)
    return L.generate(prompt, system=_answer_system(language)).strip()


# --------------------------------------------------------------------------
# Stage 4 — citations
# --------------------------------------------------------------------------
def build_sources(hits, release):
    """Unique official sources in first-seen (rank) order."""
    seen, out = set(), []
    for _, path, _ in hits:
        u = L.source_urls(path, release)
        if u["path"] in seen:
            continue
        seen.add(u["path"])
        u["title"] = L.doc_title(path)
        out.append(u)
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def ask(question, k=6, retrieve_only=False, release=None):
    """Run the full pipeline. Returns a result dict."""
    rel = release or L.RELEASE()
    conn = L.connect(rel)
    paths, texts, M = load_matrix(conn)

    parsed = parse_query(question) if not retrieve_only else \
        {"language": "", "search_query": question, "keywords": []}

    hits = retrieve(parsed["search_query"], paths, texts, M, k=k)
    result = {
        "release": rel,
        "question": question,
        "parsed": parsed,
        "matches": [(s, p) for s, p, _ in hits],
        "answer": None,
        "sources": [],
    }
    if retrieve_only:
        result["chunks"] = [(p, t) for _, p, t in hits]
        return result

    result["answer"] = synthesize(question, hits, parsed["language"])
    result["sources"] = build_sources(hits, rel)
    return result


# --------------------------------------------------------------------------
# Standalone CLI (plain text; the REPL renders richer output)
# --------------------------------------------------------------------------
def _print_result(res):
    print("Release: %s" % res["release"])
    if res["parsed"]["search_query"] != res["question"]:
        print("Search query: %s" % res["parsed"]["search_query"])
    print("\nTop %d matches:" % len(res["matches"]))
    for rank, (score, path) in enumerate(res["matches"], 1):
        print("  [%d] %.3f  %s" % (rank, score, L.rel_markdown_path(path)))
    if "chunks" in res:
        for rank, (path, text) in enumerate(res["chunks"], 1):
            print("=" * 70)
            print("[%d] %s\n" % (rank, L.rel_markdown_path(path)))
            print(text)
        return
    print("\n" + "-" * 70)
    print(res["answer"])
    print("-" * 70)
    if res["sources"]:
        print("\nOfficial documentation sources:")
        for i, s in enumerate(res["sources"], 1):
            print("  [%d] %s" % (i, s["title"]))
            # OSC 8 hyperlink: clickable in terminals that support it.
            print("      \x1b]8;;%s\x1b\\%s\x1b]8;;\x1b\\" % (s["url"], s["url"]))


def main():
    args = sys.argv[1:]
    k = 6
    retrieve_only = False
    release = None
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "-k":
            k = int(args[i + 1]); i += 2
        elif args[i] == "--retrieve-only":
            retrieve_only = True; i += 1
        elif args[i] == "--release":
            release = args[i + 1]; i += 2
        else:
            rest.append(args[i]); i += 1
    if not rest:
        sys.exit('Usage: astra_ask.py [-k N] [--release NAME] [--retrieve-only] "question"')
    question = " ".join(rest)
    t0 = time.time()
    res = ask(question, k=k, retrieve_only=retrieve_only, release=release)
    _print_result(res)
    if not retrieve_only:
        print("(model=%s, %.1fs)" % (L.GEN_MODEL(), time.time() - t0))


if __name__ == "__main__":
    main()
