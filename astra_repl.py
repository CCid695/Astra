#!/usr/bin/env python3
"""Astra — interactive CLI for asking questions about the ServiceNow docs,
answered entirely by local models. beacon-style slash-command REPL.

Commands:
  /ask <question>   Ask a question (or just type it — no slash needed).
  /model [name]     Pick the answer-generation model (Gemma 4 12B 4-bit / 8-bit).
  /model_upg        Upgrade installed models to their latest release.
  /r [release]      Show or set the release family. Defaults to auto-tracking
                    the newest one; "/r <name>" pins it, "/r latest" resumes.
  /update           Pull the latest docs for the release and re-index.
  /reindex [--full] Rebuild this release's search index.
  /upgrade          Update Astra itself.
  /clean            Wipe docs, models, and the search index back to a fresh checkout.
  /help             Show this help.
  /exit, /quit      Exit the CLI (or just type 'exit').
"""
import os
import sys
import glob
import time
import shutil
import subprocess

# Raw-terminal key reading for the arrow-key model picker (POSIX only).
try:
    import termios
    import tty
    import select as _select
    _RAWCAP = True
except Exception:  # pragma: no cover
    _RAWCAP = False

import astra_lib as L
import astra_corpus as C
import astra_ask as A
import astra_index as IDX

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.text import Text
except Exception:  # pragma: no cover
    print("The 'rich' package is required. Install it: pip install rich")
    sys.exit(1)

console = Console()

# Curated one-line descriptions for the /model list.
MODEL_NOTES = {
    "gemma4:12b": "Google Gemma 4 12B (4-bit) — default. Fast, lower RAM.",
    "gemma4:12b-it-q8_0": "Google Gemma 4 12B (8-bit) — higher quality, needs ~24 GB RAM.",
    "nomic-embed-text": "Embedding model for retrieval — not a chat model.",
    "nomic-embed-text:latest": "Embedding model for retrieval — not a chat model.",
}

EMBED_ONLY = {"nomic-embed-text", "nomic-embed-text:latest"}

# The only generation models offered in the picker: Gemma 4 12B in 4-bit and
# 8-bit. Both are always shown (installed or not); selecting a missing one
# offers to `ollama pull` it.
RECOMMENDED = [
    "gemma4:12b",
    "gemma4:12b-it-q8_0",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def ensure_server():
    if L.server_up():
        return True
    console.print("[yellow]Ollama server not reachable. Start it with "
                  "`ollama serve` (or open the Ollama app), then retry.[/yellow]")
    return False


def has_index(release=None):
    return os.path.exists(L.db_path(release))


def banner():
    rel = L.RELEASE()
    idx = ("indexed", "green") if has_index(rel) else ("no index — run /reindex", "red")
    tag = (" auto", "dim") if _auto_tracking() else ("", "dim")
    console.print(Panel.fit(
        Text.assemble(
            ("Astra ", "bold cyan"), ("v%s" % L.VERSION, "cyan"),
            "  — local ServiceNow docs assistant\n",
            ("release: ", "dim"), (rel, "green"), tag, ("  (", "dim"), idx, (")\n", "dim"),
            ("model:   ", "dim"), (L.GEN_MODEL(), "green"), "\n\n",
            ("Type a question, or ", "dim"), ("/help", "cyan"), (" for commands.", "dim"),
        ),
        border_style="cyan",
    ))


# --------------------------------------------------------------------------
# command: /ask  (and bare questions)
# --------------------------------------------------------------------------
def cmd_ask(question, k=6):
    question = question.strip()
    if not question:
        console.print("[dim]Usage: /ask <question>[/dim]")
        return
    if not ensure_server():
        return
    if not has_index():
        console.print("[red]No index for release '%s'. Run /reindex first.[/red]"
                      % L.RELEASE())
        return

    t0 = time.time()
    try:
        with console.status("[cyan]Understanding the question…", spinner="dots"):
            res = A.ask(question, k=k)  # runs parse -> retrieve -> answer -> cite
    except RuntimeError as e:
        console.print("[red]%s[/red]" % e)
        return
    except Exception as e:
        console.print("[red]Error: %s[/red]" % e)
        return

    parsed = res["parsed"]
    if parsed["search_query"] and parsed["search_query"] != question:
        console.print("[dim]retrieval query:[/dim] %s   [dim](lang: %s)[/dim]"
                      % (parsed["search_query"], parsed["language"] or "?"))

    # matches
    tbl = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    tbl.add_column("#", justify="right", style="dim", width=3)
    tbl.add_column("score", justify="right", style="cyan", width=6)
    tbl.add_column("source")
    for rank, (score, path) in enumerate(res["matches"], 1):
        tbl.add_row(str(rank), "%.3f" % score, L.rel_markdown_path(path))
    console.print(tbl)

    # answer
    console.print(Panel(Markdown(res["answer"] or "_(no answer)_"),
                        title="Answer", border_style="green"))

    # sources
    if res["sources"]:
        console.print("[bold]Official documentation sources[/bold]")
        for i, s in enumerate(res["sources"], 1):
            # Title is a short, non-wrapping clickable link (most reliable).
            console.print("  [cyan]%d.[/cyan] [link=%s]%s[/link]"
                          % (i, s["url"], s["title"]))
            # URL as an underlined hyperlink; soft_wrap keeps it unbroken so
            # terminals that rely on URL detection still open the whole link.
            console.print("     [link=%s][blue underline]%s[/blue underline][/link]"
                          % (s["url"], s["url"]), soft_wrap=True)
    console.print("[dim](model=%s, %.1fs)[/dim]" % (L.GEN_MODEL(), time.time() - t0))


# --------------------------------------------------------------------------
# command: /model
# --------------------------------------------------------------------------
def _model_candidates(installed):
    """The generation models offered in the picker: just the two Gemma 4 12B
    variants (4-bit and 8-bit)."""
    return [name for name in RECOMMENDED if name not in EMBED_ONLY]


def _read_key():
    """Read a single keypress in raw mode. Returns 'up'/'down'/'enter'/'esc'
    or the literal character.

    Reads the raw fd with os.read (not buffered sys.stdin) so that select()
    on the fd is accurate — a TextIOWrapper could pull the whole escape
    sequence into its buffer, making select miss the trailing bytes.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1)
        if ch == b"\x1b":  # escape — maybe an arrow-key sequence
            r, _, _ = _select.select([fd], [], [], 0.05)
            if not r:
                return "esc"
            seq = os.read(fd, 2)
            if seq[:1] == b"[":
                return {b"A": "up", b"B": "down"}.get(seq[1:2], "other")
            return "esc"
        if ch in (b"\r", b"\n"):
            return "enter"
        if ch == b"\x03":  # Ctrl-C
            raise KeyboardInterrupt
        try:
            return ch.decode()
        except Exception:
            return "other"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _interactive_select(rows, current_idx, title):
    """Arrow-key menu. Returns the chosen index, or None if cancelled."""
    n = len(rows)
    if n == 0:
        return None
    width = max(20, shutil.get_terminal_size((80, 24)).columns - 2)
    rows = [r if len(r) <= width else r[:width - 1] + "…" for r in rows]
    idx = current_idx if 0 <= current_idx < n else 0
    console.print("[bold]%s[/bold]" % title, highlight=False)

    def render(first=False):
        if not first:
            sys.stdout.write("\x1b[%dA" % n)  # move cursor back up over the rows
        for i, row in enumerate(rows):
            sys.stdout.write("\x1b[2K")        # clear the line
            if i == idx:
                sys.stdout.write("\x1b[7m> %s\x1b[0m\n" % row)  # reverse video
            else:
                sys.stdout.write("  %s\n" % row)
        sys.stdout.flush()

    render(first=True)
    while True:
        key = _read_key()
        if key in ("up", "k"):
            idx = (idx - 1) % n
        elif key in ("down", "j"):
            idx = (idx + 1) % n
        elif key == "enter":
            return idx
        elif key in ("q", "esc"):
            return None
        else:
            continue
        render()


def _set_model(name, installed):
    name = name.strip()
    if not name:
        return
    if name in EMBED_ONLY:
        console.print("[red]'%s' is an embedding model, not a chat model.[/red]" % name)
        return
    if name not in installed:
        console.print("[yellow]'%s' is not installed.[/yellow]" % name)
        ans = console.input("Pull it now with ollama? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            console.print("[cyan]Pulling %s …[/cyan]" % name)
            if subprocess.run(["ollama", "pull", name]).returncode != 0:
                console.print("[red]Pull failed — model not changed.[/red]")
                return
        else:
            console.print("[dim]Not changed.[/dim]")
            return
    L.set_cfg(ASTRA_GEN_MODEL=name)
    console.print("[green]Generation model set to %s[/green]" % name)


def cmd_model(arg):
    if not ensure_server():
        return
    installed = set(L.list_models())
    current = L.GEN_MODEL()

    # Explicit name → set directly (scriptable, no menu).
    if arg:
        _set_model(arg.strip(), installed)
        return

    candidates = _model_candidates(installed)

    # Interactive arrow-key picker when we have a real terminal.
    if _RAWCAP and sys.stdin.isatty() and sys.stdout.isatty():
        rows = []
        for name in candidates:
            active = "●" if name == current else " "
            status = "installed" if name in installed else "not installed"
            desc = MODEL_NOTES.get(name, "Ollama model.")
            rows.append("%s %-20s [%-13s] %s" % (active, name, status, desc))
        try:
            sel = _interactive_select(
                rows,
                candidates.index(current) if current in candidates else 0,
                "Select generation model   ↑/↓ move · Enter select · q cancel",
            )
        except KeyboardInterrupt:
            console.print("\n[dim]cancelled[/dim]")
            return
        if sel is None:
            console.print("[dim]cancelled[/dim]")
            return
        if candidates[sel] == current:
            console.print("[dim]Already using %s.[/dim]" % current)
            return
        _set_model(candidates[sel], installed)
        return

    # Non-interactive fallback (piped input): print a table, switch with /model <name>.
    tbl = Table(title="Models", show_header=True, header_style="bold")
    tbl.add_column("", width=2)
    tbl.add_column("model", style="green")
    tbl.add_column("status")
    tbl.add_column("description")
    for name in candidates:
        mark = "►" if name == current else " "
        status = "installed" if name in installed else "not installed"
        tbl.add_row(mark, name, status, MODEL_NOTES.get(name, "Ollama model."))
    console.print(tbl)
    console.print("[dim]Switch with: /model <name>   (current: %s)[/dim]" % current)


# --------------------------------------------------------------------------
# command: /model_upg  (upgrade installed models to the latest release)
# --------------------------------------------------------------------------
def cmd_model_upg():
    if not ensure_server():
        return
    installed = L.list_models()
    if not installed:
        console.print("[yellow]No models installed yet.[/yellow]")
        return
    console.print("[cyan]Upgrading installed models to their latest release…[/cyan]")
    upgraded, failed = [], []
    for m in installed:
        console.print("\n[bold]› %s[/bold]" % m)
        # `ollama pull` re-fetches the tag; if it's already current it's a no-op.
        rc = subprocess.run(["ollama", "pull", m]).returncode
        (upgraded if rc == 0 else failed).append(m)
    console.print()
    if upgraded:
        console.print("[green]Up to date: %s[/green]" % ", ".join(upgraded))
    if failed:
        console.print("[yellow]Could not upgrade: %s[/yellow]" % ", ".join(failed))
        console.print("[dim]These usually need a newer Ollama. Re-run ./install "
                      "to upgrade the Ollama runtime, then try again.[/dim]")


# --------------------------------------------------------------------------
# command: /r  (release)
# --------------------------------------------------------------------------
def _auto_tracking():
    raw = L.cfg("ASTRA_RELEASE")
    return not raw or raw.strip().lower() == "latest"


def cmd_release(arg):
    if not arg:
        cur = L.RELEASE()
        mode = " [dim](auto — tracks latest)[/dim]" if _auto_tracking() else ""
        console.print("Current release: [green]%s[/green]%s  (%s)"
                      % (cur, mode, "indexed" if has_index(cur) else "no index"))
        rels = C.list_releases()
        if rels:
            console.print("Available: %s" % ", ".join(
                "[green]%s[/green]" % r if r == cur else r for r in rels))
            console.print("[dim]Set with: /r <release>  (or /r latest to "
                          "auto-track the newest release again)[/dim]")
        else:
            console.print("[dim]Docs not cloned yet — run /update.[/dim]")
        return

    rel = arg.strip()
    if rel.lower() in ("latest", "auto"):
        L.set_cfg(ASTRA_RELEASE="latest")
        console.print("[green]Now auto-tracking the latest release "
                      "(currently: %s)[/green]" % L.RELEASE())
        return

    rels = C.list_releases()
    if rels and rel not in rels:
        console.print("[yellow]Unknown release '%s'. Available: %s[/yellow]"
                      % (rel, ", ".join(rels)))
        return
    L.set_cfg(ASTRA_RELEASE=rel)
    console.print("[green]Default release pinned to %s[/green]" % rel)
    if not has_index(rel):
        console.print("[yellow]No index for '%s' yet. Run /update (fetch + index) "
                      "or /reindex.[/yellow]" % rel)


# --------------------------------------------------------------------------
# command: /update  (pull docs + re-index)
# --------------------------------------------------------------------------
def cmd_update():
    try:
        # No release passed in: let update() refresh the repo's default-branch
        # pointer first, so an auto-tracking ("latest") config picks up a
        # brand-new release family before we resolve which one to index.
        C.update(progress=console.print)
    except Exception as e:
        console.print("[red]Update failed: %s[/red]" % e)
        return
    rel = L.RELEASE()
    if not ensure_server():
        console.print("[yellow]Docs updated, but Ollama is down — index skipped.[/yellow]")
        return
    console.print("[cyan]Re-indexing release '%s' (incremental)…[/cyan]" % rel)
    try:
        IDX.build_index(release=rel, full=False, progress=console.print)
    except Exception as e:
        console.print("[red]Indexing failed: %s[/red]" % e)


# --------------------------------------------------------------------------
# command: /reindex
# --------------------------------------------------------------------------
def cmd_reindex(arg):
    if not ensure_server():
        return
    full = "--full" in (arg or "")
    rel = L.RELEASE()
    console.print("[cyan]Indexing release '%s' (%s)…[/cyan]"
                  % (rel, "full rebuild" if full else "incremental"))
    try:
        IDX.build_index(release=rel, full=full, progress=console.print)
    except Exception as e:
        console.print("[red]Indexing failed: %s[/red]" % e)


# --------------------------------------------------------------------------
# command: /upgrade  (self-update the tool)
# --------------------------------------------------------------------------
def cmd_upgrade():
    here = L.HERE
    if not os.path.isdir(os.path.join(here, ".git")):
        console.print("[yellow]Astra is not a git checkout, so it can't self-update.\n"
                      "Re-download the latest version manually.[/yellow]")
        return
    console.print("[cyan]Pulling latest Astra…[/cyan]")
    rc = subprocess.run(["git", "-C", here, "pull", "--ff-only"]).returncode
    if rc != 0:
        console.print("[red]git pull failed.[/red]")
        return
    venv_pip = os.path.join(here, ".venv", "bin", "pip")
    pip = venv_pip if os.path.exists(venv_pip) else "pip"
    subprocess.run([pip, "install", "-q", "-r", os.path.join(here, "requirements.txt")])
    console.print("[green]Astra updated. Restart the CLI to load changes.[/green]")


# --------------------------------------------------------------------------
# command: /clean  (wipe docs, models, and the index — back to a fresh checkout)
# --------------------------------------------------------------------------
def cmd_clean():
    here = L.HERE
    docs_root = L.DOCS_ROOT()
    db_files = sorted(glob.glob(os.path.join(here, "index-*.db*")))
    worktrees = sorted(wt for wt in glob.glob(docs_root + "-*") if os.path.isdir(wt))

    wanted_models = set(RECOMMENDED) | {L.GEN_MODEL(), L.EMBED_MODEL()}
    installed = set(L.list_models()) if L.server_up() else set()
    models_present = sorted(m for m in wanted_models if m in installed)

    console.print("[bold]/clean will remove:[/bold]")
    console.print("  index:   %s" % (", ".join(os.path.basename(f) for f in db_files) or "(none)"))
    console.print("  docs:    %s" % (docs_root if os.path.isdir(docs_root) else "(not cloned)"))
    for wt in worktrees:
        console.print("           %s" % wt)
    console.print("  models:  %s" % (", ".join(models_present) or "(none installed)"))
    ans = console.input(r"Proceed? This cannot be undone. \[y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        console.print("[dim]Cancelled.[/dim]")
        return

    for f in db_files:
        try:
            os.remove(f)
        except OSError:
            pass
    console.print("[green]Removed index.[/green]" if db_files else "[dim]No index to remove.[/dim]")

    if os.path.isdir(docs_root):
        subprocess.run(["git", "-C", docs_root, "worktree", "prune"], capture_output=True)
        for wt in worktrees:
            shutil.rmtree(wt, ignore_errors=True)
        shutil.rmtree(docs_root, ignore_errors=True)
        console.print("[green]Removed docs corpus.[/green]")
    else:
        console.print("[dim]Docs corpus already absent.[/dim]")

    if models_present:
        if not ensure_server():
            console.print("[yellow]Ollama not reachable — models left installed.[/yellow]")
        else:
            for m in models_present:
                subprocess.run(["ollama", "rm", m], capture_output=True)
            console.print("[green]Removed models: %s[/green]" % ", ".join(models_present))
    else:
        console.print("[dim]No models to remove.[/dim]")

    console.print("\n[bold green]Environment reset to a fresh checkout.[/bold green] "
                  "Run /update then /reindex (or /model) to rebuild.")


# --------------------------------------------------------------------------
# help / dispatch / loop
# --------------------------------------------------------------------------
HELP = r"""[bold]Commands[/bold]
  [cyan]/ask <question>[/cyan]    Ask a question (or just type it directly).
  [cyan]/model[/cyan]             Pick the generation model with ↑/↓ + Enter (or /model <name>).
  [cyan]/model_upg[/cyan]         Upgrade installed models to their latest release.
  [cyan]/r \[release][/cyan]       Show or set the release. Defaults to auto-tracking
                          the newest one; "/r <name>" pins it, "/r latest" resumes.
  [cyan]/update[/cyan]            Pull the latest docs for the release and re-index.
  [cyan]/reindex \[--full][/cyan]  Rebuild this release's search index.
  [cyan]/upgrade[/cyan]           Update Astra itself.
  [cyan]/clean[/cyan]             Wipe docs, models, and the search index back to a fresh checkout.
  [cyan]/help[/cyan]              Show this help.
  [cyan]/exit[/cyan], [cyan]/quit[/cyan]       Exit the CLI (or just type 'exit')."""


def dispatch(line):
    line = line.strip()
    if not line:
        return True
    if not line.startswith("/"):
        # Bare exit/quit also leaves the CLI (common convention).
        if line.lower() in ("exit", "quit"):
            return False
        cmd_ask(line)
        return True

    parts = line[1:].split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("quit", "exit", "q"):
        return False
    elif cmd in ("help", "h", "?"):
        console.print(HELP)
    elif cmd == "ask":
        cmd_ask(arg)
    elif cmd == "model":
        cmd_model(arg)
    elif cmd in ("model_upg", "model-upg", "upg"):
        cmd_model_upg()
    elif cmd == "r" or cmd == "release":
        cmd_release(arg)
    elif cmd == "update":
        cmd_update()
    elif cmd == "reindex":
        cmd_reindex(arg)
    elif cmd == "upgrade":
        cmd_upgrade()
    elif cmd == "clean":
        cmd_clean()
    else:
        console.print("[red]Unknown command /%s[/red]  (try /help)" % cmd)
    return True


def main():
    # One-shot mode: `astra_repl.py /ask "question"` or a bare question.
    if len(sys.argv) > 1:
        dispatch(" ".join(sys.argv[1:]))
        return
    banner()
    while True:
        try:
            line = console.input("\n[bold cyan]astra[/bold cyan] [dim](%s)[/dim] › "
                                 % L.RELEASE())
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        try:
            if not dispatch(line):
                break
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/dim]")
    console.print("[dim]bye[/dim]")


if __name__ == "__main__":
    main()
