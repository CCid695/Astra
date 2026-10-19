"""Shared helpers for Astra — the local ServiceNow docs assistant.

Everything talks to a locally running Ollama server; no network egress beyond
localhost (except `git` operations, which live in astra_corpus.py).

Adapted from the rag-snowdocs / local_model pipeline, extended with:
  - per-release index databases (index-<release>.db)
  - .astrarc config load/save (release, models, docs path)
  - ServiceNow doc path -> official source URL derivation
"""
import os
import re
import json
import sqlite3
import urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Tool version. Bumped by +0.01 on every commit.
VERSION = "0.28"

# --------------------------------------------------------------------------
# Configuration (.astrarc — KEY=VALUE lines, also sourced by the shell launcher)
# --------------------------------------------------------------------------
RC_PATH = os.path.join(HERE, ".astrarc")

DEFAULTS = {
    "ASTRA_RELEASE": "latest",                    # ServiceNow release family: a
                                                   # pinned branch name, or the
                                                   # sentinel "latest" (default)
                                                   # to auto-track whatever the
                                                   # docs repo currently ships
                                                   # as its newest release.
    "ASTRA_GEN_MODEL": "gemma4:12b",              # answer-generation model
    "ASTRA_EMBED_MODEL": "nomic-embed-text",      # embedding model
    "ASTRA_DOCS": os.path.join(HERE, "DOC"),   # docs cloned inside the repo
    "OLLAMA_HOST": "http://127.0.0.1:11434",
    # How long Ollama keeps the gen model resident after a call.
    "ASTRA_KEEP_ALIVE": "30m",
}


def _read_rc():
    vals = {}
    if os.path.exists(RC_PATH):
        with open(RC_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals


def cfg(key):
    """Resolve a config value: environment > .astrarc > built-in default."""
    if key in os.environ:
        return os.environ[key]
    rc = _read_rc()
    if key in rc:
        return rc[key]
    return DEFAULTS.get(key)


def set_cfg(**kwargs):
    """Persist one or more config values to .astrarc (merging with existing)."""
    vals = _read_rc()
    vals.update({k: str(v) for k, v in kwargs.items()})
    with open(RC_PATH, "w", encoding="utf-8") as f:
        f.write("# Astra config — written by the tool. Edit values, keep KEY=VALUE.\n")
        for k in sorted(vals):
            f.write("%s=%s\n" % (k, vals[k]))
    # Reflect the change in the running process immediately. The launcher
    # exports these as env vars, and cfg() reads env before .astrarc — so
    # without this the new value wouldn't take effect until the CLI restarts.
    for k, v in kwargs.items():
        os.environ[k] = str(v)
    return vals


# Convenience accessors ----------------------------------------------------
def OLLAMA():       return cfg("OLLAMA_HOST")
def EMBED_MODEL():  return cfg("ASTRA_EMBED_MODEL")
def GEN_MODEL():    return cfg("ASTRA_GEN_MODEL")
def RELEASE():
    """Current release family. Resolves the "latest" sentinel (the default)
    against the docs repo's advertised default branch, so a newly published
    ServiceNow release is picked up automatically without user action."""
    val = cfg("ASTRA_RELEASE")
    if not val or val.strip().lower() == "latest":
        import astra_corpus as C   # deferred: astra_corpus imports this module
        return C.latest_release() or "australia"
    return val
def DOCS_ROOT():    return os.path.expanduser(cfg("ASTRA_DOCS"))
def KEEP_ALIVE():   return cfg("ASTRA_KEEP_ALIVE")


def db_path(release=None):
    """Per-release vector store. Overridable with ASTRA_DB for testing."""
    if os.environ.get("ASTRA_DB"):
        return os.environ["ASTRA_DB"]
    rel = release or RELEASE()
    return os.path.join(HERE, "index-%s.db" % rel)


# --------------------------------------------------------------------------
# nomic-embed-text task prefixes (required for good retrieval)
# --------------------------------------------------------------------------
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


# --------------------------------------------------------------------------
# Ollama HTTP
# --------------------------------------------------------------------------
def _post(path, payload, timeout=900):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA() + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def server_up():
    try:
        with urllib.request.urlopen(OLLAMA() + "/api/version", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def list_models():
    """Names of models currently pulled into the local Ollama."""
    try:
        with urllib.request.urlopen(OLLAMA() + "/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def embed(texts, is_query=False):
    """Return an (N, D) float32 array of L2-normalized embeddings."""
    prefix = QUERY_PREFIX if is_query else DOC_PREFIX
    inp = [prefix + t for t in texts]
    out = _post("/api/embed", {"model": EMBED_MODEL(), "input": inp})
    arr = np.array(out["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def generate(prompt, system=None, options=None, model=None):
    payload = {
        "model": model or GEN_MODEL(),
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE(),
    }
    if system:
        payload["system"] = system
    payload["options"] = options or {"temperature": 0.2, "num_ctx": 8192}
    out = _post("/api/generate", payload)
    return out.get("response", "")


def preload(model=None):
    """Warm the gen model without generating, so the first query is fast."""
    try:
        _post(
            "/api/generate",
            {"model": model or GEN_MODEL(), "keep_alive": KEEP_ALIVE()},
            timeout=300,
        )
    except Exception:
        pass


# --------------------------------------------------------------------------
# SQLite vector store
# --------------------------------------------------------------------------
def connect(release=None):
    conn = sqlite3.connect(db_path(release), timeout=60)
    # Wait out transient locks (e.g. another writer, or a backup/AV tool
    # briefly holding the file) instead of failing immediately.
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS files(
               path TEXT PRIMARY KEY, mtime REAL, nchunks INTEGER)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chunks(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               path TEXT, ord INTEGER, text TEXT, emb BLOB)"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
    return conn


def chunk_text(text, target=1000, overlap=150):
    """Split on blank lines, then pack paragraphs into ~target-char chunks."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= target:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
            if len(p) > target:
                for i in range(0, len(p), target - overlap):
                    chunks.append(p[i:i + target])
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------
# Doc metadata + official source URL derivation
# --------------------------------------------------------------------------
GITHUB_REPO = "ServiceNow/ServiceNowDocs"


def rel_markdown_path(abs_path, release=None):
    """Path relative to the release's markdown/ root, e.g.
    'healthcare-life-sciences/ctwm-reference.md'. Falls back to basename."""
    p = os.path.abspath(abs_path)
    marker = os.sep + "markdown" + os.sep
    idx = p.rfind(marker)
    if idx != -1:
        return p[idx + len(marker):].replace(os.sep, "/")
    return os.path.basename(p)


def source_urls(abs_path, release=None):
    """Return the official ServiceNow docs link for a doc file.

    Uses the context-sensitive-help (CSH) URL, which deep-links to the rendered
    page on docs.servicenow.com:

        https://docs.servicenow.com/csh?topicname=<file>.html&version=<release>

    where <file> is the markdown file's base name (no directory, no extension)
    and <release> is the release family. Example:
        markdown/servicenow-platform/service-catalog/service-catalog.md  (australia)
        -> https://docs.servicenow.com/csh?topicname=service-catalog.html&version=australia
    """
    rel = release or RELEASE()
    relpath = rel_markdown_path(abs_path, rel)
    topic = os.path.splitext(os.path.basename(relpath))[0]
    return {
        "path": relpath,
        "topic": topic,
        "url": "https://docs.servicenow.com/csh?topicname=%s.html&version=%s" % (topic, rel),
    }


_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def read_frontmatter(abs_path):
    """Return the YAML-ish frontmatter as a flat dict (best-effort, no deps)."""
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(4000)
    except Exception:
        return {}
    m = _FM_RE.match(head)
    if not m:
        return {}
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta


def doc_title(abs_path):
    meta = read_frontmatter(abs_path)
    if meta.get("title"):
        return meta["title"]
    return os.path.basename(abs_path).rsplit(".", 1)[0]
