# Syncing this tutorial to `mcDETECT-tutorial` (Read the Docs)

The live documentation site is built from the separate repository **[mcDETECT-tutorial](https://github.com/chen-yang-yuan/mcDETECT-tutorial)**. This file describes how to add a **GitHub Actions** workflow there so Markdown pages from `**mcDETECT/tutorial/`** are copied into `**docs/**` before each Read the Docs build.

---

## Prerequisites

1. `**mcDETECT-tutorial**` contains a Sphinx + MyST project (e.g. `docs/conf.py`, `docs/index.md` with a `toctree` that lists the copied pages). The Read the Docs project points at that config via `[.readthedocs.yaml](https://docs.readthedocs.com/platform/stable/tutorial/)`.
2. You know the **GitHub owner/repo** name for the **mcDETECT** codebase (e.g. `your-org/mcDETECT`).
3. If **mcDETECT** is **private**, create a **fine-grained personal access token (PAT)** or **GitHub App** token with read access to `mcDETECT`, and add it in `**mcDETECT-tutorial`** as a secret named e.g. `MC_DETECT_READ_TOKEN`. Public source repos usually work without a token for `actions/checkout` of the other repo.

---

## Step 1 — Choose what gets copied

By default, sync **only** published tutorial pages (same as `make sync` in `mcDETECT-tutorial`):

- `README.md` (index / roadmap)
- `01_*.md` … `08_*.md`

Do **not** sync maintainer docs (`readthedocs_github_action.md`, `PUBLISHING_READTHEDOCS.md`) into `docs/tutorial_pages/`; they are not part of the Sphinx `toctree`.

You can adjust paths in the workflow below (e.g. add `09_*.md` when you add a new page).

---

## Step 2 — Create the workflow file in `mcDETECT-tutorial`

In the `**mcDETECT-tutorial`** repo (not in `mcDETECT`), add:

`.github/workflows/sync-tutorial-from-mcdetect.yml`

Paste:

```yaml
# Copies tutorial Markdown from mcDETECT into docs/ for Sphinx + Read the Docs.
# Edit MC_DETECT_REPO if your fork/org name differs.

name: Sync tutorial from mcDETECT

on:
  workflow_dispatch:
  push:
    branches: [main]

env:
  MC_DETECT_REPO: chen-yang-yuan/mcDETECT

jobs:
  sync:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout mcDETECT-tutorial (this repo)
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Checkout mcDETECT (source tutorial)
        uses: actions/checkout@v4
        with:
          repository: ${{ env.MC_DETECT_REPO }}
          path: _upstream_mcdetect
          # token: ${{ secrets.MC_DETECT_READ_TOKEN }}  # if mcDETECT is private

      - name: Copy tutorial markdown into docs
        run: |
          set -eux
          DEST="docs/tutorial_pages"
          mkdir -p "$DEST"
          # Copy split tutorial pages + index; exclude this sync doc from the site if you prefer
          rsync -av \
            --include='README.md' \
            --include='0[1-8]_*.md' \
            --exclude='*' \
            _upstream_mcdetect/tutorial/ \
            "$DEST/"
          # Optional: also copy README as Sphinx index companion
          # cp "$DEST/README.md" docs/tutorial_index.md

      - name: Commit and push if changed
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add docs/tutorial_pages
          if git diff --staged --quiet; then
            echo "No changes."
            exit 0
          fi
          git commit -m "Sync tutorial pages from ${{ env.MC_DETECT_REPO }}"
          git push
```

**Customize:**

- `**MC_DETECT_REPO`** — set to your `**owner/mcDETECT**` repository.
- `**DEST**` — must match where your Sphinx `**conf.py**` / `**index.md**` `toctree` expects the files (here: `docs/tutorial_pages/`).
- `**rsync` patterns** — include `README.md` and `0[1-8]_*.md`; add more `--include` lines if you add files (e.g. `09_*.md`).
- **Private mcDETECT:** create secret `**MC_DETECT_READ_TOKEN`** with repo read scope; the `token:` line already prefers it when set.

---

## Step 3 — Wire Sphinx to the copied paths

In `**mcDETECT-tutorial/docs/**` (example layout):

1. `**index.md**` (or `index.rst`) — use a MyST `**toctree**` with paths like `**tutorial_pages/README**`, `**tutorial_pages/01_what_mcdetect_expects**`, … (no `.md` extension in toctree for Sphinx, or follow MyST conventions for your Sphinx version).
2. Enable `**myst_parser**` in `**conf.py**` so `.md` sources build.

If your `toctree` still points at old filenames, update it once to match `**tutorial_pages/*.md**` after the first sync.

---

## Step 4 — Run the workflow

1. Push the workflow file to `**main**` on `**mcDETECT-tutorial**`.
2. Open **GitHub → Actions → Sync tutorial from mcDETECT → Run workflow** (or push a commit to `**main`** if `on.push` is enabled).
3. Confirm a new commit appears on `**main**` with updated `**docs/tutorial_pages/**`.
4. Trigger a **Read the Docs** build (usually automatic on push) and open your project URL.

---

## Step 5 — Optional: trigger only when mcDETECT changes

The sample workflow runs on `**workflow_dispatch`** and on pushes to `**mcDETECT-tutorial**`. It does **not** auto-run when `**mcDETECT`** changes unless you also:

- add a **repository dispatch** from `**mcDETECT`** (advanced), or  
- use **scheduled** `cron` in `**mcDETECT-tutorial`**, or  
- run **workflow_dispatch** manually after editing the tutorial.

---

## Troubleshooting


| Issue                                         | What to check                                                                                 |
| --------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Checkout of `**mcDETECT`** fails              | Repo name, visibility, `**MC_DETECT_READ_TOKEN**` for private repos                           |
| Sphinx “document not included in any toctree” | Update `index.md` `toctree` after adding new `**0N_*.md**` files                              |
| Read the Docs build fails                     | `.readthedocs.yaml` install path, Python deps (`sphinx`, `myst-parser`), Sphinx build command |


For the official hosting setup, see the Read the Docs tutorial: [https://docs.readthedocs.com/platform/stable/tutorial/](https://docs.readthedocs.com/platform/stable/tutorial/).