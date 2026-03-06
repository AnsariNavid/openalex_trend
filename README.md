# openalex_trend

A PyCharm-friendly Python script to build a collaboration report from OpenAlex.

## What it does

1. Extracts publications for **multiple institutions** in a given time interval.
2. Filters publications to keep only **journals** and **conferences**.
3. Lists the **top 10 collaborating institutes in Germany**.
4. Generates a short markdown report including:
   - collaboration statistics,
   - top themes,
   - top collaborator individuals with short bios,
   - a concise GPT-based narrative analysis and future-collaboration section (if `OPENAI_API_KEY` is set).

## Files

- `openalex_collab_report.py` — main script (run directly).
- `config.example.json` — configuration template.
- `config.json` — your active config (edit this file before running).

## Setup

```bash
cp config.example.json config.json
```

Set your GPT API key (optional, only needed for narrative generation):

```bash
export OPENAI_API_KEY="your_key_here"
```

## Run

### In PyCharm

Open `openalex_collab_report.py` and click the **Run** button.

### In terminal

```bash
python openalex_collab_report.py
```

The report path is controlled by `output_report` in `config.json`.
