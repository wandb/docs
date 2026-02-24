# Agent prompt: Configure Locadex AI context for W&B docs (Korean, and later Japanese)

## Requirements

- [ ] Access to the [General Translation Dashboard](https://dash.generaltranslation.com/) (Locadex console).
- [ ] The docs repository linked to a Locadex/GT project (GitHub app installed, repository connected).
- [ ] Optional: read access to the [hw-wandb/wandb_docs_translation](https://github.com/hw-wandb/wandb_docs_translation) repo (for configs and language_dicts).
- [ ] Optional: access to `main` branch of wandb/docs with `ko/` (and optionally `ja/`) present, for comparing manual translations when refining glossary or locale context.

## Agent prerequisites

1. **Which locale(s) are you configuring?** (e.g. Korean only now; Japanese later.) Determines which Glossary translations and Locale Context entries to add.
2. **Do you have a Glossary CSV or term list already?** If not, use the runbook to build one from the sources below.
3. **Is the GT project already created and the repo connected?** If not, complete [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify) Steps 1–6 first.

## Task overview

This runbook describes how to capture translation-memory and terminology from (1) the legacy wandb_docs_translation tooling and (2) the manually translated Korean (and later Japanese) content on `main`, and how to configure the Locadex/General Translation platform so that auto-translation uses that context. The goal is consistent terminology and correct “do not translate” behavior for product names and technical terms.

**Where things live:**

| What | Where | Notes |
|------|--------|------|
| **Glossary** (terms, definitions, per-locale translations) | Locadex console → AI Context → Glossary | Drives consistent term usage and “do not translate” for product/feature names. Can bulk-upload via CSV. |
| **Locale Context** (language-specific instructions) | Locadex console → AI Context → Locale Context | e.g. Korean: spacing between alphabets and Hangul, formatting rules. |
| **Style Controls** (tone, audience, project description) | Locadex console → AI Context → Style Controls | Project-wide; applies to all locales. |
| **Which files/locales to translate** | Git → `gt.config.json` | `locales`, `defaultLocale`, `files`. No glossary or prompts in repo. |

So: **steer auto-translation in the Locadex console** (Glossary, Locale Context, Style Controls). **File and locale setup stay in Git** (`gt.config.json`). The optional `dictionary` key in `gt.config.json` is for app UI strings (e.g. gt-next/gt-react), not for docs MDX glossary; docs terminology is managed in the console.

## Context and constraints

### Legacy tooling (wandb_docs_translation)

- **human_prompt.txt**: Lists W&B product/feature names that must **never** be translated (keep in English): Artifacts, Entities, Projects, Runs, Experiments, Datasets, Reports, Sweeps, Weave, Launch, Models, Teams, Users, Workspace, Registered Models. Same for link/list context like `[**word**](link)`.
- **system_prompt.txt**: General rules (valid markdown, translate only comments in code blocks, use the dictionary, do not translate link URLs; for Japanese/Korean: add space when switching between alphabets and CJK characters, and around inline formatting).
- **configs/language_dicts/ko.yaml**: Mixed “translation memory”:
  - **Keep in English** (product/feature name): e.g. `Artifacts`, `Reports`, `Sweeps`, `Experiments`, `run`, `Weave expression`, `Data Visualization`, `Model Management`.
  - **Translate to Korean**: e.g. `artifact` → 아티팩트, `sweep` → 스윕, `project` → 프로젝트, `workspace` → 워크스페이스, `user` → 사용자.

So the convention was: **product/feature names (often capitalized or in UI/list context) stay in English**; **common-noun usage** follows the locale dictionary. Locadex Glossary should reflect both “do not translate” and “translate as X” for each locale.

### Locadex/GT platform behavior

- **Glossary**: Term (as in source) + optional Definition + optional per-locale Translation. For “do not translate,” use the same string as the term for that locale (e.g. Term “W&B”, Translation (ko) “W&B”). For “translate as,” set Translation (ko) to the desired target (e.g. “artifact” → “아티팩트”).
- **Locale Context**: Free-form instructions per target locale (e.g. “Use space between Latin and Korean characters”).
- **Style Controls**: One set for the project (tone, audience, description). Applied to all locales.
- Changes to AI Context do **not** automatically retranslate existing content; use [Retranslate](https://generaltranslation.com/docs/platform/translations/retranslate) to apply new context to already-translated files.

## Step-by-step process

### 1. Gather terminology sources

- **From wandb_docs_translation** (if available):
  - `configs/human_prompt.txt` → list of terms that must never be translated.
  - `configs/language_dicts/ko.yaml` (and later `ja.yaml`) → term → locale translation map.
- **From manual translations on main** (optional): Compare a few EN vs KO (or JA) pages to confirm how product names and common terms were rendered (e.g. “run” vs “실행”, “workspace” vs “워크스페이스”) and add or adjust glossary entries.

**Agent note**: If the agent cannot read the external repo, the runbook can still be followed by a human using the CSV and locale-context text provided in this repo (see runbooks and optional CSV below).

### 2. Build or obtain a Glossary CSV

- Use the pre-built glossary CSV for Korean in this repo: **runbooks/locadex-glossary-ko.csv** (see “Glossary CSV” below), or generate one that includes:
  - **Do-not-translate terms**: One row per term; Definition optional; `ko` (or “Translation (ko)”) = same as Term.
  - **Translated terms**: One row per term; Definition optional; `ko` = desired Korean equivalent.
- Confirm the exact column names expected by the Locadex “Upload Context CSV” (e.g. `Term`, `Definition`, `ko` or `Translation (ko)`). Adjust CSV headers if the console expects different names.
- **CSV format (for valid parsing)**: Use standard CSV quoting so the file parses correctly. The comma is the field separator; any field that contains a comma, double quote, or newline **must** be wrapped in double quotes. Within a quoted field, escape internal double quotes by doubling them (`""`). One term per row (do not put multiple variants like “run, Run” in one cell). When generating or editing the CSV programmatically, use a CSV library or explicitly quote such fields; unquoted commas in Term or Definition will be treated as column boundaries and break the row.

### 3. Configure the Locadex project in the console

1. Sign in to the [General Translation Dashboard](https://dash.generaltranslation.com/).
2. Open the project linked to the wandb/docs repository.
3. Go to **AI Context** (or equivalent: Glossary, Locale Context, Style Controls).

### 4. Upload or add Glossary terms

- **Option A**: Use **Upload Context CSV** to bulk-import the glossary (Term, Definition, and locale column(s)). The platform maps columns to glossary terms and per-locale translations.
- **Option B**: Add terms manually: Term, Definition (helps the model), and for Korean add the translation (same as term for “do not translate,” or the Korean string for “translate as”).

Ensure at least:

- Product/feature names that must stay in English: W&B, Weights & Biases, Artifacts, Runs, Experiments, Sweeps, Weave, Launch, Models, Reports, Datasets, Teams, Users, Workspace, Registered Models, etc., with Korean = same as source.
- Terms that must translate consistently: e.g. artifact → 아티팩트, sweep → 스윕, project → 프로젝트, workspace → 워크스페이스, and other entries from `language_dicts/ko.yaml` (and later `ja.yaml`).

### 5. Set Locale Context for Korean

- Select locale **ko**.
- Add instructions that reflect the legacy system_prompt and good practice for Korean docs, for example:
  - Add a space when switching between Latin letters and Korean characters (including Hangul, Hanja).
  - When using inline formatting (bold, italic, code) around part of a word or phrase in Korean, add spaces before and after the formatted part so the markdown renders correctly.
  - Keep code blocks and link URLs unchanged; translate only surrounding prose and comments in code where appropriate.

Save the locale context.

### 6. Set Style Controls (project-wide)

- **Project description**: e.g. “Documentation for Weights & Biases (W&B): ML experiment tracking, model registry, Weave for LLM ops, and related products.”
- **Target audience**: Developers and ML practitioners.
- **Tone**: Professional, technical, clear. Prefer natural reading over literal translation.

Save.

### 7. Retranslate if needed

- If you already have auto-translated content and you changed Glossary or Locale Context, use the platform’s **Retranslate** flow for the affected files so the new context is applied.

## Verification and testing

- **Glossary**: After upload, spot-check a few terms in the Glossary tab (do-not-translate and translated).
- **Locale Context**: Confirm Korean (and later Japanese) instructions are saved under the correct locale.
- **Quality**: Run or trigger translation on a sample page and check that product names stay in English and that common terms match the glossary (e.g. artifact → 아티팩트 where appropriate).

## Common issues and solutions

### Issue: CSV upload does not map to Glossary

- **Cause**: Column names may not match what the platform expects.
- **Solution**: Check the Locadex/GT docs or in-UI help for “Upload Context CSV” column names (e.g. Term, Definition, locale code). Rename columns in your CSV and re-upload.

### Issue: Terms still translated when they should stay in English

- **Cause**: Term not in Glossary, or “do not translate” not set (missing or wrong locale translation).
- **Solution**: Add the term to the Glossary with the same value for the target locale (e.g. “Artifacts” → ko: “Artifacts”). Add a short Definition so the model understands it is a product/feature name.

### Issue: Japanese (or another locale) needs different rules

- **Cause**: Locale-specific preferences (e.g. polite form, spacing, katakana for product names).
- **Solution**: Add a separate Locale Context for that locale (e.g. ja) and, if needed, additional Glossary entries with a “ja” column or manual entries for Japanese.

## Cleanup instructions

- No temporary branches or files are required in the docs repo for console-only configuration.
- If you generated a one-off script to build the CSV, do not commit it unless the team decides to keep it (see AGENTS.md and user rules on one-off scripts).

## Checklist

- [ ] Gathered terminology from human_prompt, language_dicts/ko.yaml (and ja if applicable).
- [ ] Built or obtained Glossary CSV and confirmed column names for upload.
- [ ] Logged into Locadex console and opened the correct project.
- [ ] Uploaded or added Glossary terms (do-not-translate and translated).
- [ ] Set Locale Context for Korean (and later Japanese if applicable).
- [ ] Set Style Controls (description, audience, tone).
- [ ] Verified with a sample translation and retranslated existing content if needed.

## Glossary CSV

A starter Korean glossary is provided in this repo: **runbooks/locadex-glossary-ko.csv**. Columns:

- **Term**: Source (English) term as it appears in docs.
- **Definition**: Short explanation (helps the AI; optional for upload).
- **ko**: Korean translation. Use the same string as Term for “do not translate”; use the desired Korean string for “translate as.”

To add more terms from `configs/language_dicts/ko.yaml` (or from manual KO pages on main), append rows with the same columns. If the Locadex console expects different column names for locale translations (e.g. “Translation (ko)”), rename the `ko` column when uploading or in the CSV before upload.

### CSV formatting for future generation

When creating or appending to the glossary CSV (by hand or by script), follow these rules so the file remains valid:

- **Delimiter**: Comma (`,`). Do not use comma inside a field unless the field is quoted.
- **Quoting**: Wrap any field in double quotes (`"`) if it contains a comma, a double quote, or a newline. Optionally quote all fields for consistency.
- **Escaping**: Inside a quoted field, represent a literal double quote as two double quotes (`""`).
- **One term per row**: Each row is one term. Do not list multiple variants in one cell (e.g. use separate rows for “run” and “artifact”, not “run, artifact” in the Term column).
- **Tools**: When generating CSV programmatically, use a proper CSV library (e.g. Python `csv` module with `quoting=csv.QUOTE_MINIMAL` or `QUOTE_NONNUMERIC`) so commas and quotes in Term or Definition are handled correctly.

## Notes

- **Japanese later**: When adding Japanese, repeat Locale Context for `ja` (e.g. polite form, spacing between alphabets and Japanese script, inline formatting spaces) and add Glossary entries for `ja` (same approach: do-not-translate = same as source; translate-as = desired Japanese).
- **GT config in Git**: `gt.config.json` already has `locales` and `defaultLocale`. No glossary or AI context is stored there; those live only in the console.
- **References**: [GT Glossary](https://generaltranslation.com/docs/platform/ai-context/glossary), [Locale Context](https://generaltranslation.com/docs/platform/ai-context/locale-context), [Style Controls](https://generaltranslation.com/docs/platform/ai-context/style-controls), [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify).
