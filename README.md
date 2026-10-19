# Astra

Ask questions about the **ServiceNow product documentation** in natural language —
answered by models running **entirely on your own machine**. Nothing leaves the
host and no cloud tokens are spent.

Astra is an interactive CLI (beacon-style slash commands). Ask in any language:
it understands the question with a local LLM, searches your local copy of the
docs, summarizes an answer **in the language you asked in**, and cites the exact
official documentation pages it used.

```
astra (australia) › How do delegation rules work?
astra (australia) › 委任ルールの作成にはどのロールが必要ですか？
astra (australia) › ¿Qué roles se necesitan para crear reglas de delegación?
```

---

## How it works (the pipeline)

Every `/ask` runs four fully-local stages:

1. **Parse** — a local LLM detects the question's language and rewrites it into an
   English retrieval query (the docs are en-US, so translating intent up front
   sharply improves recall).
2. **Retrieve** — the query is embedded and cosine-ranked against the release's
   local vector index.
3. **Answer** — a local LLM summarizes the retrieved chunks into an answer,
   written back in the **language you asked in**.
4. **Cite** — the official ServiceNow doc pages used are listed with links.

Astra answers strictly from the documentation: if the retrieved context doesn't
contain the answer, it says so rather than guessing.

---

## Quick start

```bash
cd ~/Astra
./install          # one-time setup. Asks: 4-bit or 8-bit model.
./astra            # launch the interactive CLI
./uninstall        # removes everything it installed
```

`install` downloads the runtime + models, sets up Python, clones the docs into
`./DOC` (pulling the latest on re-runs), and builds the index automatically.
The first index of the full corpus can take ~1h. Then just run `./astra`.

---

## The CLI

Launch with `./astra`, then use one of the default skills (slash commands).

**`/ask` is the default skill** — if you type a line without a leading slash, it
is treated as `/ask`. So `/ask How do I create an ACL?` and just
`How do I create an ACL?` do exactly the same thing.

### Default skills

| Skill | What it does |
|---|---|
| `/ask <question>` | **Default.** Ask a ServiceNow documentation question. The slash can be omitted — plain text is sent to `/ask`. Runs parse → retrieve → answer → cite, answering in the language you asked in, strictly from the docs. |
| `/model` | Pick the answer-generation model with an arrow-key menu (↑/↓ + Enter). Offers the two Gemma 4 12B variants — 4-bit and 8-bit; choosing one that isn't installed offers to `ollama pull` it. `/model <name>` sets it directly. |
| `/model_upg` | Upgrade the installed models to their latest release (re-pulls each with `ollama pull`). |
| `/r [release]` | Show, or set, the default ServiceNow release family (e.g. `/r zurich`). With no argument it lists the available releases. Defaults to **auto-tracking the newest release** — pinning to a specific one (via `/r <name>`) disables that; `/r latest` re-enables it. |
| `/update` | Pull the latest docs for the current release and re-index (incremental). When auto-tracking, this is also how a brand-new release family (e.g. the one after `australia`) gets picked up. |
| `/reindex [--full]` | Rebuild this release's search index. `--full` drops and rebuilds from scratch. |
| `/upgrade` | Update Astra itself (git pull + reinstall deps). |
| `/clean` | Wipe the docs corpus, downloaded models, and the search index — resets the environment back to a fresh checkout (`.venv` and config are left intact). Asks for confirmation first. |
| `/help` | Show the command list. |
| `/exit`, `/quit` | Exit the CLI. You can also just type `exit`. |

One-shot mode (no REPL):

```bash
./astra "How do I create an ACL?"      # ask and exit
./astra --index                        # build/refresh the current release's index
./astra --index --full                 # full rebuild
./astra --index --release zurich       # index a specific release
```

---

## Releases (families)

Each ServiceNow release family is a **git branch** of the public
[`ServiceNow/ServiceNowDocs`](https://github.com/ServiceNow/ServiceNowDocs) repo
(`australia`, `zurich`, `yokohama`, `xanadu`, …). Astra keeps **one index per
release** (`index-<release>.db`) so families never mix.

By default Astra doesn't pin to a fixed release — `ASTRA_RELEASE=latest` in
`.astrarc` means it **auto-tracks whichever family the docs repo currently
advertises as newest** (its `origin/HEAD`). Each `/update` re-checks that, so
when ServiceNow ships a new release after `australia`, running `/update` alone
picks it up and re-indexes it — no manual `/r` switch needed. Pin to a specific
release with `/r <name>` (useful if you want to stay on an older family);
`/r latest` returns to auto-tracking.

```
astra › /r                 # show current + available releases
astra › /r zurich          # switch default release
astra › /update            # fetch that release's docs (creates a git worktree
                           #   for non-default families) and index it
```

Switching with `/r` is instant once a release is indexed.

---

## Sources

Every answer ends with the official documentation pages it drew from, as
context-sensitive-help (CSH) links on the ServiceNow docs site:

```
https://docs.servicenow.com/csh?topicname=<file>.html&version=<release>
```

where `<file>` is the markdown file's base name (no directory, no `.md`) and
`<release>` is the current release family. For example the source file
`markdown/servicenow-platform/service-catalog/service-catalog.md` on `australia`
becomes <https://docs.servicenow.com/csh?topicname=service-catalog.html&version=australia>.

---

## Configuration

Settings live in `.astrarc` (written by `install`; edit freely). Environment
variables override it.

```bash
ASTRA_GEN_MODEL=gemma4:12b            # answer-generation model
ASTRA_EMBED_MODEL=nomic-embed-text    # embedding model (must match index time)
ASTRA_RELEASE=australia               # default release family
ASTRA_DOCS=./DOC                      # docs checkout (inside the repo)
OLLAMA_HOST=http://127.0.0.1:11434    # Ollama endpoint
```

Change the generation model at any time with `/model` (it can pull the model for
you). Changing the embedding model requires a `/reindex --full`.

---

## Requirements

- macOS, Python 3.11+
- [Ollama](https://ollama.com) (the installer fetches a compatible build)
- ~8 GB free for the 4-bit model (or ~13 GB for 8-bit), ~450 MB per docs release

---

## Privacy

**A full ask lifecycle — parse, retrieve, answer, cite — spends zero Claude (or
any cloud LLM) tokens and leaks no PII.** Every step runs on local models via
Ollama: your question, the retrieved document chunks, and the generated answer
never leave the host, so nothing you type can be exposed to a third party.

The only network activity is during setup/maintenance: `install` (downloading
Ollama and the models) and `git pull` (updating the public docs). Neither sends
any of your data anywhere — they only download. Day-to-day Q&A is fully offline.
