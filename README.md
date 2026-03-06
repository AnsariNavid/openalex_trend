# openalex_trend

PyCharm-friendly scripts for OpenAlex collaboration analysis.

## Scripts

- `openalex_collab_report.py` — main report generator.
- `pick_host_institutes.py` — interactive helper to choose host institutes and write them into `config.json`.

## What the report includes

1. Publications for multiple host institutes in a configurable date interval.
2. Filtered output restricted to journals and conferences.
3. Top 10 collaborating institutes in Germany.
4. Named host institutes in the report header.
5. **Collaboration directions per top German partner**, inferred from abstracts of papers co-authored with that partner.
6. Yearly trend section (publication and collaborator-institute changes).
7. Main collaborator individuals with a research-background column and short bios.
8. Theme summaries based on publication abstracts.
9. Smooth narrative + future collaboration section via GPT API (if `OPENAI_API_KEY` is set).

## Configuration

Edit `config.json` (or copy from `config.example.json`):

- `institution_ids`
- `from_date`, `to_date`
- `max_pages`, `per_page`
- `top_collaborators`, `top_individuals`
- `results_dir` (all outputs saved here)
- `output_report`
- `openai_model` (general model)
- `analysis_model` (recommended stronger model for deeper direction analysis, default `gpt-4.1`)

## Run in PyCharm

### Option A: pick host institutes first

1. Open `pick_host_institutes.py` and click **Run**.
2. Search/select institutes interactively.
3. It updates `config.json`.

### Option B: run report directly

1. Open `openalex_collab_report.py` and click **Run**.
2. Report will be written to `results/<output_report>`.

## Optional GPT setup

```bash
export OPENAI_API_KEY="your_key_here"
```

Without this key, the script still runs and generates non-GPT fallback text.

## Progress bars

Both scripts print progress bars for each major operation so it is easy to follow execution.


## Topic-focused script (new)

Use `topic_focused_collab_filter.py` when you want to track only papers relevant to specific topics.

### What it does

1. Uses your host institutes and date range.
2. Keeps only papers co-authored with German collaborator institutes.
3. Reads each paper (title + abstract) and checks relevance to your custom topics one-by-one using a **cheap model** (`cheap_model`).
4. Saves relevant papers with metadata to JSONL.
5. Produces a short topic-level collaboration analysis markdown file.

### Topic config

Copy and edit:

```bash
cp topic_config.example.json topic_config.json
```

Key settings:
- `topics`: list of your topics of interest.
- `cheap_model`: low-cost model for paper-level relevance decisions.
- `analysis_model`: stronger model for final synthesis.
- `output_relevant_jsonl`, `output_analysis_md`: output files under `results_dir`.

### Run

In PyCharm run `topic_focused_collab_filter.py`, or in terminal:

```bash
python topic_focused_collab_filter.py
```
