# DVC + Google Drive Remote — Setup Guide

How to store large ML artifacts (models, datasets) in **Google Drive** via a
**DVC remote**, so the Git repo stays small and the heavy files are versioned
and shareable.

This guide documents the *exact* working path for a **personal Gmail** account,
including the two walls Google throws up and how to get past them. Follow it top
to bottom.

> **TL;DR of the hard part:** DVC's built-in Google OAuth client is blocked by
> Google. You must create **your own** OAuth client and add yourself as a
> **test user**. Everything else is mechanical.

---

## What goes where

| Lives in Git (small)                         | Lives in the DVC remote (large)          |
| -------------------------------------------- | ---------------------------------------- |
| code, `dvc.yaml`, `dvc.lock`, `.dvc/config`  | `Data/`, `artifacts/`, `models/*.pt`     |
| `params.yaml`, `config/`, `scores*.json`     | (anything listed as a stage `out`)       |

DVC tracks the artifacts by content hash in `dvc.lock`; the actual bytes live in
the remote. A clone gets the code instantly and runs `dvc pull` to fetch bytes.

---

## Part 0 — Prerequisites

- A Google account (this guide assumes **personal Gmail**).
- The project is (or will be) a **Git repo**.
- Artifacts already exist on disk under paths declared as `outs` in `dvc.yaml`.

---

## Part 1 — Install DVC + Google Drive support

```bash
pip install "dvc==3.59.0" dvc-gdrive
```

- `dvc-gdrive` pulls in `pydrive2` (the Drive client).
- It may **downgrade `cryptography`** (e.g. 46 → 43) to satisfy its dependency
  ceiling. Harmless for this project.

> **Gotcha — `_DIR_MARK` ImportError.** If `dvc` errors with
> `cannot import name '_DIR_MARK' from 'pathspec...'`, your `pathspec` is too
> new for DVC 3.59. Pin it back:
> ```bash
> pip install "pathspec==0.12.1"
> ```

---

## Part 2 — Repo / SCM state

DVC refuses to run without Git **unless** you opt out.

- **Normal case (recommended):** make sure you're in a Git repo.
  ```bash
  git init -b main      # if not already a repo
  ```
- **No Git at all:** tell DVC to run standalone.
  ```bash
  dvc config core.no_scm true
  ```
  Undo later with `dvc config --unset core.no_scm` once you add Git. Running
  under Git is preferred — it versions `dvc.lock` so the pipeline is reproducible.

---

## Part 3 — Create the Google Drive folder

1. In Google Drive, create an empty folder (e.g. `pulmoscan-dvc`).
2. Open it and copy the **folder ID** from the URL — the part after `/folders/`:
   ```
   https://drive.google.com/drive/folders/1rsBDKrWwzRGXqT6QmuZm1tDbbnleBkcy
                                           └──────────── this ────────────┘
   ```

---

## Part 4 — Add the DVC remote

```bash
dvc remote add -d gdrive gdrive://<FOLDER_ID>
```

`-d` makes it the default. Verify:

```bash
dvc remote list
cat .dvc/config            # the folder ID is NOT secret — fine to commit
```

`.dvc/config` should now contain:
```ini
[core]
    remote = gdrive
['remote "gdrive"']
    url = gdrive://<FOLDER_ID>
```

> If you `dvc push` now, it fails with **"This app is blocked"** — that's
> expected. DVC's shared OAuth client is dead. Continue to Part 5.

---

## Part 5 — Create your OWN Google OAuth client (the critical part)

**Why:** Google blocks DVC's built-in OAuth client for the sensitive Drive
scope. Your own client + yourself as a test user is the supported workaround.
Free; Drive API has no cost.

### 5.1 Create / select a Cloud project
- Go to **https://console.cloud.google.com/** → create a project (e.g.
  `pulmoscan-dvc`). The $300 free-trial credits are **irrelevant** — nothing
  here costs money.

### 5.2 Enable the Drive API
- **APIs & Services → Library** → search **"Google Drive API"** → **Enable**.
- Direct link: `https://console.cloud.google.com/apis/library/drive.googleapis.com?project=<PROJECT_ID>`

### 5.3 Configure the consent screen + add yourself as a test user
- **APIs & Services → OAuth consent screen** (newer UI: **Google Auth Platform →
  Audience**).
- User type: **External** → Create.
- Fill app name + your email, save through the screens.
- Under **Test users → + Add users**, add **your Gmail address** → **Save**.
  - Direct link: `https://console.cloud.google.com/auth/audience?project=<PROJECT_ID>`
  - ⚠️ **This is the step everyone misses.** Without it, `dvc push` fails with
    **`Error 403: access_denied`** ("can only be accessed by developer-approved
    testers"). Make sure the email matches and you hit Save (allow ~1 min to
    propagate).

### 5.4 Create the OAuth client
- **APIs & Services → Credentials → Create Credentials → OAuth client ID**.
- Application type: **Desktop app** → Create.
- Copy the **Client ID** and **Client secret** (the secret is shown once).

---

## Part 6 — Give DVC your client credentials

Store them in `.dvc/config.local`, which is **gitignored** (never committed):

```bash
dvc remote modify gdrive --local gdrive_client_id "<CLIENT_ID>"
dvc remote modify gdrive --local gdrive_client_secret "<CLIENT_SECRET>"
```

Verify they stay out of Git:

```bash
git check-ignore .dvc/config.local    # must print the path → it's ignored
```

---

## Part 7 — Record artifacts into the DVC cache (creates `dvc.lock`)

If you've never run the pipeline, generate the lock file from the artifacts that
already exist on disk (no need to retrain):

```bash
dvc commit -f
```

This hashes each `out` and copies it into `.dvc/cache` (local disk usage roughly
doubles for the tracked artifacts). It writes/updates `dvc.lock`.

> If you'd rather (re)generate artifacts from scratch instead, run `dvc repro`.

Commit the bookkeeping to Git:

```bash
git add dvc.lock .dvc/config .gitignore
git commit -m "Add DVC lock + Google Drive remote"
git push
```

---

## Part 8 — Push the artifacts to Drive

```bash
dvc push
```

First run triggers OAuth:

1. A browser opens. Choose **your Gmail account**.
2. **"Google hasn't verified this app"** → **Advanced → Go to `<app>` (unsafe) →
   Allow**. Safe — it's *your own* app; the warning is just because it's
   unverified.
3. Upload begins. The token is cached at
   `~/Library/Caches/pydrive2fs/<client_id>/default.json` (gitignored location),
   so future pushes don't re-prompt.

If a transfer stalls or drops, just re-run `dvc push` — it resumes.

---

## Part 9 — Verify

```bash
dvc status -c
```

Expected: **`Cache and remote 'gdrive' are in sync.`** This compares *every*
tracked object against the remote — it's the real confirmation that all files
(not just the first batch) landed. The files also appear in the Drive folder.

---

## Part 10 — Fetching on another machine (or after a fresh clone)

```bash
git clone <repo-url> && cd <repo>
pip install "dvc==3.59.0" dvc-gdrive

# Each collaborator needs their own client creds in config.local (Part 6),
# AND must be added as a test user on the OAuth consent screen (Part 5.3),
# AND must have view access to the Drive folder.
dvc remote modify gdrive --local gdrive_client_id "<CLIENT_ID>"
dvc remote modify gdrive --local gdrive_client_secret "<CLIENT_SECRET>"

dvc pull          # downloads Data/, artifacts/, models from Drive
```

---

## Troubleshooting (the exact errors and fixes)

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `This app is blocked` / `Google blocked this access` | Using DVC's built-in OAuth client | Create your own OAuth client (Part 5) |
| `Error 403: access_denied` — "developer-approved testers" | Your account isn't a test user | Add your Gmail under **Audience → Test users** (Part 5.3), Save, wait ~1 min |
| `cannot import name '_DIR_MARK' from 'pathspec'` | `pathspec` too new for DVC 3.59 | `pip install "pathspec==0.12.1"` |
| `ERROR: ... is not a git repository` | DVC needs Git | `git init -b main` **or** `dvc config core.no_scm true` |
| `dvc push` says everything up to date but remote is empty | No `dvc.lock` / artifacts not committed to cache | `dvc commit -f` (Part 7) |
| Quota / rate-limit errors mid-push | Drive/pydrive2 throttling | Re-run `dvc push` (resumes); large pushes may need a few retries |
| Browser didn't open (headless/SSH) | No local browser | Run on a machine with a browser, or copy the printed auth URL manually |

---

## Security notes

- **`.dvc/config.local` is gitignored** — client ID/secret never reach the repo.
  The committed `.dvc/config` holds only the (non-secret) folder URL.
- A **Desktop-app** OAuth secret is *not* truly confidential per Google's spec,
  but keeping it in `config.local` is still best practice.
- To rotate: delete the client in **APIs & Services → Credentials**, create a
  new one, and re-run the Part 6 commands.

---

## Why not a service account?

Service accounts are non-interactive (great for CI) but **have zero Drive
storage** — they can only own files inside a **Shared Drive**, which requires
**Google Workspace**. On a personal Gmail this path is a dead end; use the OAuth
client flow above. (If you *do* have Workspace + a Shared Drive:
`dvc remote modify gdrive --local gdrive_use_service_account true` and point
`gdrive_service_account_json_file_path` at the JSON key.)

## When Drive isn't worth it

Google Drive is the flakiest DVC backend (OAuth gauntlet, pydrive2 rate limits).
For reliability/CI, an S3-compatible bucket is far less painful — e.g.
**Cloudflare R2** or **Backblaze B2** (both ~10 GB free): `pip install dvc-s3`,
`dvc remote add -d store s3://<bucket>`, then put the access keys in
`config.local`. No browser, no consent screens.

---

## Command cheat-sheet

```bash
# one-time setup
pip install "dvc==3.59.0" dvc-gdrive
dvc remote add -d gdrive gdrive://<FOLDER_ID>
dvc remote modify gdrive --local gdrive_client_id "<CLIENT_ID>"
dvc remote modify gdrive --local gdrive_client_secret "<CLIENT_SECRET>"

# every time you produce/update artifacts
dvc commit -f            # or: dvc repro
git add dvc.lock && git commit -m "update artifacts" && git push
dvc push                 # upload bytes to Drive

# on another machine
dvc pull                 # download bytes from Drive
dvc status -c            # check local cache vs remote
```
