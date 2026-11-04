"""Shared helpers for Astra — the local ServiceNow docs assistant.

Everything talks to a locally running Ollama server; no network egress beyond
localhost (except `git` operations, which live in astra_corpus.py).

Adapted from the rag-snowdocs / local_model pipeline, extended with:
  - per-release index databases (index-<release>.db)
  - .astrarc config load/save (release, models, docs path)
  - ServiceNow doc path -> official source URL derivation
"""
from __future__ import annotations
import os
import re
import json
import sqlite3
import urllib.request
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Tool version. Bumped by +0.01 on every commit.
VERSION = "0.28"

# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configure logging based on ASTRA_LOG_LEVEL config."""
    log_level = cfg("ASTRA_LOG_LEVEL").upper()
    logger = logging.getLogger("astra")

    if not log_level:
        logger.setLevel(logging.CRITICAL + 1)  # Disable logging
        return logger

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    logger.setLevel(level_map.get(log_level, logging.WARNING))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Initialize logger
_logger = setup_logging()

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
    # Chunking and indexing parameters
    "ASTRA_CHUNK_TARGET": "1000",                 # target chunk size in characters
    "ASTRA_CHUNK_OVERLAP": "150",                 # chunk overlap in characters
    "ASTRA_EMBED_BATCH": "64",                    # chunks per embedding request
    "ASTRA_FILE_BATCH": "200",                    # files per DB transaction
    # Logging: one of "DEBUG", "INFO", "WARNING", "ERROR", or "" (disabled)
    "ASTRA_LOG_LEVEL": "",
}


def _read_rc() -> Dict[str, str]:
    vals: Dict[str, str] = {}
    if os.path.exists(RC_PATH):
        with open(RC_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals


def cfg(key: str) -> str:
    """Resolve a config value: environment > .astrarc > built-in default."""
    if key in os.environ:
        return os.environ[key]
    rc = _read_rc()
    if key in rc:
        return rc[key]
    return DEFAULTS.get(key, "")


def set_cfg(**kwargs: str) -> Dict[str, str]:
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
def OLLAMA() -> str:       return cfg("OLLAMA_HOST")
def EMBED_MODEL() -> str:  return cfg("ASTRA_EMBED_MODEL")
def GEN_MODEL() -> str:    return cfg("ASTRA_GEN_MODEL")
def RELEASE() -> str:
    """Current release family. Resolves the "latest" sentinel (the default)
    against the docs repo's advertised default branch, so a newly published
    ServiceNow release is picked up automatically without user action."""
    val = cfg("ASTRA_RELEASE")
    if not val or val.strip().lower() == "latest":
        import astra_corpus as C   # deferred: astra_corpus imports this module
        return C.latest_release() or "australia"
    return val
def DOCS_ROOT() -> str:    return os.path.expanduser(cfg("ASTRA_DOCS"))
def KEEP_ALIVE() -> str:   return cfg("ASTRA_KEEP_ALIVE")
def CHUNK_TARGET() -> int: return int(cfg("ASTRA_CHUNK_TARGET"))
def CHUNK_OVERLAP() -> int: return int(cfg("ASTRA_CHUNK_OVERLAP"))
def EMBED_BATCH() -> int:  return int(cfg("ASTRA_EMBED_BATCH"))
def FILE_BATCH() -> int:   return int(cfg("ASTRA_FILE_BATCH"))


def db_path(release: Optional[str] = None) -> str:
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
def _post(path: str, payload: Dict[str, Any], timeout: int = 900) -> Dict[str, Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA() + path, data=data, headers={"Content-Type": "application/json"}
    )
    _logger.debug("Ollama request: %s", path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        response = json.loads(r.read().decode())
        _logger.debug("Ollama response status: %d", r.status)
        return response


def server_up() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA() + "/api/version", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def list_models() -> List[str]:
    """Names of models currently pulled into the local Ollama."""
    try:
        with urllib.request.urlopen(OLLAMA() + "/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def embed(texts: List[str], is_query: bool = False) -> np.ndarray:
    """Return an (N, D) float32 array of L2-normalized embeddings."""
    _logger.debug("Embedding %d texts (is_query=%s)", len(texts), is_query)
    prefix = QUERY_PREFIX if is_query else DOC_PREFIX
    inp = [prefix + t for t in texts]
    out = _post("/api/embed", {"model": EMBED_MODEL(), "input": inp})
    arr = np.array(out["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    _logger.debug("Embeddings computed: shape=%s", arr.shape)
    return arr / norms


def generate(prompt: str, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> str:
    _logger.debug("Generating with model: %s", model or GEN_MODEL())
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
    response = out.get("response", "")
    _logger.debug("Generated response length: %d", len(response))
    return response


def preload(model: Optional[str] = None) -> None:
    """Warm the gen model without generating, so the first query is fast."""
    _logger.debug("Preloading model: %s", model or GEN_MODEL())
    try:
        _post(
            "/api/generate",
            {"model": model or GEN_MODEL(), "keep_alive": KEEP_ALIVE()},
            timeout=300,
        )
    except Exception as e:
        _logger.warning("Preload failed: %s", e)


# --------------------------------------------------------------------------
# SQLite vector store
# --------------------------------------------------------------------------
def connect(release: Optional[str] = None) -> sqlite3.Connection:
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


def chunk_text(text: str, target: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
    """Split on blank lines, then pack paragraphs into ~target-char chunks."""
    if target is None:
        target = CHUNK_TARGET()
    if overlap is None:
        overlap = CHUNK_OVERLAP()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur = ""
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


def rel_markdown_path(abs_path: str, release: Optional[str] = None) -> str:
    """Path relative to the release's markdown/ root, e.g.
    'healthcare-life-sciences/ctwm-reference.md'. Falls back to basename."""
    p = os.path.abspath(abs_path)
    marker = os.sep + "markdown" + os.sep
    idx = p.rfind(marker)
    if idx != -1:
        return p[idx + len(marker):].replace(os.sep, "/")
    return os.path.basename(p)


def source_urls(abs_path: str, release: Optional[str] = None) -> Dict[str, str]:
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


def read_frontmatter(abs_path: str) -> Dict[str, str]:
    """Return the YAML-ish frontmatter as a flat dict (best-effort, no deps)."""
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(4000)
    except Exception:
        return {}
    m = _FM_RE.match(head)
    if not m:
        return {}
    meta: Dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta


def doc_title(abs_path: str) -> str:
    meta = read_frontmatter(abs_path)
    if meta.get("title"):
        return meta["title"]
    return os.path.basename(abs_path).rsplit(".", 1)[0]
