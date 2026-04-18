# Publishing the tutorial to Read the Docs (Option A)

Canonical tutorial sources live in **`mcDETECT/tutorial/`** (this directory). The public site is built from the separate repository **`mcDETECT-tutorial`**, which copies Markdown into **`docs/tutorial_pages/`** and runs Sphinx + Read the Docs.

This document describes the **local edit → copy → push** workflow. It matches **Option A**: you edit here, sync into your local **`mcDETECT-tutorial`** clone, then push **`mcDETECT-tutorial`** to GitHub so Read the Docs rebuilds.

---

## What gets published

Only **tutorial content** should be copied into **`mcDETECT-tutorial/docs/tutorial_pages/`**:

| Included | Files |
|----------|--------|
| Yes | `README.md`, `01_*.md` … `08_*.md` (the split tutorial pages) |

**Not** copied (maintainer / meta docs; they stay only in **`mcDETECT/tutorial/`**):

- `readthedocs_github_action.md` — notes for GitHub Actions in **`mcDETECT-tutorial`**
- `PUBLISHING_READTHEDOCS.md` — this file

The **`mcDETECT-tutorial`** `Makefile` `make sync` and the **Sync tutorial from mcDETECT** workflow use the same include rules as above.

---

## Typical workflow

### 1. Edit and commit in `mcDETECT` (recommended)

```bash
cd /path/to/mcDETECT
# edit tutorial/*.md
git add tutorial/
git commit -m "docs(tutorial): ..."
git push origin main
```

### 2. Sync into `mcDETECT-tutorial` and preview

From your **`mcDETECT-tutorial`** clone (often a sibling folder: `../mcDETECT-tutorial`):

```bash
cd /path/to/mcDETECT-tutorial
make sync
```

This runs `rsync` from **`../mcDETECT/tutorial`** (override with `MCDETECT_TUTORIAL_SRC=...` if needed). It **does not** copy maintainer-only files listed above.

Preview the site locally:

```bash
make html
# optional: make open   (macOS)
```

### 3. Commit and push `mcDETECT-tutorial`

```bash
git add docs/tutorial_pages
git status   # confirm only intended files
git commit -m "Sync tutorial pages from mcDETECT"
git push origin main
```

Read the Docs builds from the **`mcDETECT-tutorial`** repo; a push to **`main`** triggers a new build (per your RTD settings).

You can also use **`make push`** in **`mcDETECT-tutorial`** if you use that automation (it stages everything — review before committing).

---

## Adding a new page

1. Add **`09_whatever.md`** (or similar) under **`mcDETECT/tutorial/`**.
2. In **`mcDETECT-tutorial`**, add the docname to **`docs/index.md`** Sphinx `toctree` (no `.md` suffix), e.g. `tutorial_pages/09_whatever`.
3. Extend **`make sync`** and **`.github/workflows/sync-tutorial-from-mcdetect.yml`** `rsync` `--include` patterns so the new file is copied.

---

## Alternative: sync only via GitHub Actions

After you push changes to **`mcDETECT`**, you can run **Actions → Sync tutorial from mcDETECT → Run workflow** on **`mcDETECT-tutorial`**. That updates **`docs/tutorial_pages/`** in the repo without a local `make sync`. You still need **`mcDETECT`** pushed so the workflow can check it out.

---

## Where CI details live

For token scope, workflow YAML, and troubleshooting, see **`readthedocs_github_action.md`** in this directory (not published on Read the Docs).
