"""Corpus management for Astra: clone / update the ServiceNowDocs repo and
expose one markdown directory per release family.

Each ServiceNow release family is a *branch* of the public
`ServiceNow/ServiceNowDocs` repo. The default checkout lives at
`$ASTRA_DOCS` (branch = whatever was cloned, usually the latest family).
Other families are materialized as sibling git *worktrees*
(`<docs>-<release>/`) so several releases can coexist without re-checkouts.
"""
from __future__ import annotations
import os
import subprocess
from typing import Optional, List, Callable

import astra_lib as L

CLONE_URL = "https://github.com/ServiceNow/ServiceNowDocs.git"

# Branches in the repo that are not user-facing release families.
NON_FAMILY = {"main", "HEAD", "mobile", "nofamily", "other", "store"}


def _git(args: List[str], cwd: Optional[str] = None, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git"] + args, cwd=cwd, check=check,
        text=True, capture_output=capture,
    )


def is_cloned() -> bool:
    return os.path.isdir(os.path.join(L.DOCS_ROOT(), ".git"))


def ensure_clone(progress: Callable[[str], None] = print) -> str:
    """Clone the docs repo to $ASTRA_DOCS if it isn't there yet."""
    root = L.DOCS_ROOT()
    if is_cloned():
        return root
    progress("Cloning ServiceNowDocs (~450 MB) into %s ..." % root)
    os.makedirs(os.path.dirname(root), exist_ok=True)
    _git(["clone", CLONE_URL, root], capture=False)
    return root


def default_branch() -> Optional[str]:
    """Branch currently checked out in the main worktree."""
    try:
        r = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=L.DOCS_ROOT())
        return r.stdout.strip()
    except Exception:
        return None


def latest_release() -> Optional[str]:
    """Release family the docs repo currently advertises as its default
    branch (origin/HEAD) — i.e. whatever ServiceNow ships as the newest
    release. Refreshed by `update()` on every /update, so a brand-new
    release family is picked up without any manual /r switch.
    Returns None if the repo isn't cloned yet or HEAD can't be resolved."""
    if not is_cloned():
        return None
    try:
        r = _git(["symbolic-ref", "refs/remotes/origin/HEAD"], cwd=L.DOCS_ROOT())
        name = r.stdout.strip().rsplit("/", 1)[-1]
        return name or None
    except Exception:
        return None


def list_releases() -> List[str]:
    """Available release families (remote branches, minus infra branches)."""
    if not is_cloned():
        return []
    try:
        r = _git(["branch", "-r"], cwd=L.DOCS_ROOT())
    except Exception:
        return []
    names: List[str] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if "->" in line:            # e.g. origin/HEAD -> origin/australia
            continue
        name = line.split("/", 1)[-1] if "/" in line else line
        if name in NON_FAMILY:
            continue
        names.append(name)
    return sorted(set(names))


def worktree_path(release: str) -> str:
    """Filesystem path holding the given release's checkout."""
    if release == default_branch():
        return L.DOCS_ROOT()
    return "%s-%s" % (L.DOCS_ROOT(), release)


def markdown_dir(release: Optional[str] = None, ensure: bool = True, progress: Callable[[str], None] = print) -> str:
    """Return the markdown/ directory for a release, creating a worktree
    on demand for non-default families."""
    rel = release or L.RELEASE()
    if not is_cloned():
        raise RuntimeError(
            "Docs not cloned yet. Run the installer or `/update`."
        )
    wt = worktree_path(rel)
    md = os.path.join(wt, "markdown")
    if os.path.isdir(md) or not ensure:
        return md
    if rel == default_branch():
        return md  # main checkout; markdown should exist already
    # Materialize a worktree for this family.
    progress("Preparing worktree for release '%s' ..." % rel)
    _git(["fetch", "origin", rel], cwd=L.DOCS_ROOT(), check=False)
    _git(["worktree", "add", "--force", wt, "origin/%s" % rel],
         cwd=L.DOCS_ROOT(), capture=False)
    return md


def update(release: Optional[str] = None, progress: Callable[[str], None] = print) -> str:
    """Fetch latest and fast-forward the release's working tree.
    Returns the markdown dir on success."""
    ensure_clone(progress=progress)
    progress("Fetching latest from origin ...")
    _git(["fetch", "origin"], cwd=L.DOCS_ROOT(), check=False, capture=False)
    # Re-sync which branch the remote considers its default *before* resolving
    # "latest", so a release published since our last check (e.g. a brand-new
    # family) is picked up now rather than on some future run.
    _git(["remote", "set-head", "origin", "-a"], cwd=L.DOCS_ROOT(), check=False, capture=False)

    rel = release or L.RELEASE()
    wt = worktree_path(rel)
    if not os.path.isdir(os.path.join(wt, "markdown")):
        # First time we see this release — build the worktree.
        markdown_dir(rel, ensure=True, progress=progress)
    else:
        progress("Updating '%s' checkout ..." % rel)
        # Reset the working tree to the fetched branch tip (docs are read-only
        # for us, so a hard sync is safe and avoids merge prompts).
        _git(["reset", "--hard", "origin/%s" % rel], cwd=wt, check=False, capture=False)
    return os.path.join(wt, "markdown")
