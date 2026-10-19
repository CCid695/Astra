#!/usr/bin/env python3
"""Build / refresh the per-release vector index over the ServiceNow docs.

Usage:
  astra_index.py [--release NAME] [--full] [DOCS_DIR]

- Writes to index-<release>.db (one store per release family).
- Incremental by default: files whose mtime is unchanged are skipped.
- --full drops the release's index and rebuilds from scratch.
- DOCS_DIR overrides the markdown directory (default: resolved per release).
"""
import os
import sys
import time
import errno
import sqlite3

import astra_lib as L
import astra_corpus as C

EMBED_BATCH = 64     # chunks per embedding request
FILE_BATCH = 200     # files written (and committed) per transaction
DB_RETRIES = 6       # write attempts before giving up on a transient DB error


# --------------------------------------------------------------------------
# Single-writer lock — only one index build may touch a release's db at a time.
# Prevents two concurrent builds from deadlocking on "database is locked".
# --------------------------------------------------------------------------
def _pid_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM   # exists, just not signalable by us
    return True


def _acquire_lock(release):
    """Take the build lock for a release, or raise if another live build holds it.
    Returns the lock file path (pass to _release_lock)."""
    lock = L.db_path(release) + ".lock"
    if os.path.exists(lock):
        try:
            other = int((open(lock).read().strip() or "0"))
        except Exception:
            other = 0
        if other and other != os.getpid() and _pid_alive(other):
            raise RuntimeError(
                "An index build for '%s' is already running (pid %d). Wait for it "
                "to finish (or stop it), then retry." % (release, other))
        # otherwise the lock is stale (dead pid) — safe to take over.
    with open(lock, "w") as f:
        f.write(str(os.getpid()))
    return lock


def _release_lock(lock):
    try:
        if os.path.exists(lock) and open(lock).read().strip() == str(os.getpid()):
            os.remove(lock)
    except Exception:
        pass


def iter_md(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirpath.split(os.sep):
            continue
        for fn in filenames:
            if fn.lower().endswith(".md"):
                yield os.path.join(dirpath, fn)


def _embed_bytes(texts):
    """Embed a list of chunk texts, returning one emb-blob per text (in order).
    Embeddings are computed in EMBED_BATCH-sized requests."""
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        for v in L.embed(texts[i:i + EMBED_BATCH]):
            out.append(v.tobytes())
    return out


def build_index(release=None, docs_dir=None, full=False, progress=print):
    """Index a release's markdown corpus. Returns (new_files, total_chunks).

    Writes are done per file-batch in a single transaction and are resilient to
    transient SQLite errors (a lock or a security/backup tool briefly grabbing
    the file): on OperationalError the connection is reopened and the batch —
    which is idempotent (delete-by-path then insert) — is retried. Indexing is
    resumable, so anything a hard failure drops is picked up on the next run.
    """
    rel = release or L.RELEASE()
    if docs_dir is None:
        docs_dir = C.markdown_dir(rel, progress=progress)
    if not os.path.isdir(docs_dir):
        raise RuntimeError("Docs dir not found: %s" % docs_dir)

    # Only one build may write a release's db at a time.
    lock = _acquire_lock(rel)
    state = {"conn": None, "cur": None}

    def reconnect():
        if state["conn"] is not None:
            try:
                state["conn"].close()
            except Exception:
                pass
        state["conn"] = L.connect(rel)
        state["cur"] = state["conn"].cursor()

    def retry(op, what):
        delay = 1.0
        for attempt in range(DB_RETRIES):
            try:
                return op()
            except sqlite3.OperationalError as e:
                if attempt == DB_RETRIES - 1:
                    raise
                progress("  DB %s failed: %s — reconnecting, retry %d/%d in %.0fs"
                         % (what, e, attempt + 1, DB_RETRIES - 1, delay))
                time.sleep(delay)
                delay = min(delay * 2, 15)
                reconnect()

    try:
        reconnect()
        if full:
            def _reset():
                state["cur"].execute("DELETE FROM chunks")
                state["cur"].execute("DELETE FROM files")
                state["conn"].commit()
            retry(_reset, "reset")

        seen = {row[0]: row[1]
                for row in state["cur"].execute("SELECT path, mtime FROM files")}
        files = list(iter_md(docs_dir))
        total = len(files)
        progress("Release: %s\nCorpus: %s\nFiles found: %d\nEmbed model: %s\n"
                 % (rel, docs_dir, total, L.EMBED_MODEL()))

        done = skipped = nchunks = 0
        t0 = time.time()
        batch = []  # list of (path, mtime, [chunk_text, ...])

        def write_batch():
            """Embed and persist the current file-batch as one retryable txn."""
            nonlocal done, nchunks
            if not batch:
                return
            texts = [c for (_p, _m, parts) in batch for c in parts]
            embs = _embed_bytes(texts)
            prepared, pos = [], 0
            for path, mtime, parts in batch:
                recs = [(path, o, parts[o], embs[pos + o]) for o in range(len(parts))]
                pos += len(parts)
                prepared.append((path, mtime, recs))

            def _persist():
                cur = state["cur"]
                for path, mtime, recs in prepared:
                    cur.execute("DELETE FROM chunks WHERE path=?", (path,))
                    if recs:
                        cur.executemany(
                            "INSERT INTO chunks(path, ord, text, emb) VALUES (?,?,?,?)",
                            recs)
                    cur.execute(
                        "INSERT OR REPLACE INTO files(path, mtime, nchunks) VALUES (?,?,?)",
                        (path, mtime, len(recs)))
                state["conn"].commit()

            retry(_persist, "write batch")
            done += len(batch)
            nchunks += len(texts)
            batch.clear()

        for fp in files:
            mtime = os.path.getmtime(fp)
            if seen.get(fp) == mtime:
                skipped += 1
                continue
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception as e:
                progress("  skip (read error) %s: %s" % (fp, e))
                continue
            batch.append((fp, mtime, L.chunk_text(text)))
            if len(batch) >= FILE_BATCH:
                write_batch()
                rate = done / (time.time() - t0)
                progress("  indexed %d / %d files (%d new chunks, %.1f files/s)"
                         % (done, total, nchunks, rate))

        write_batch()
        grand = retry(lambda: state["cur"].execute(
            "SELECT COUNT(*) FROM chunks").fetchone()[0], "count")
        progress("\nDone. New/updated files: %d, skipped(unchanged): %d" % (done, skipped))
        progress("Total chunks in index: %d" % grand)
        progress("Elapsed: %.1fs" % (time.time() - t0))
        return done, grand
    finally:
        if state["conn"] is not None:
            try:
                state["conn"].close()
            except Exception:
                pass
        _release_lock(lock)


def main():
    args = sys.argv[1:]
    release = None
    full = False
    docs_dir = None
    i = 0
    while i < len(args):
        if args[i] == "--release":
            release = args[i + 1]; i += 2
        elif args[i] == "--full":
            full = True; i += 1
        else:
            docs_dir = args[i]; i += 1
    build_index(release=release, docs_dir=docs_dir, full=full)


if __name__ == "__main__":
    main()
