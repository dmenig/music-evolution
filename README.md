# Music Evolution Graph

Interactive visualization of music genre lineage. The graph (`index.html`) is a self-contained artifact built from `genres.json` via `src/music_evolution/build_html.py`.

## Run locally

```bash
uv run streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Fork or use this repo.
2. Go to https://share.streamlit.io → **New app**.
3. Pick this repo, branch `main`, main file: `streamlit_app.py`.
4. Deploy.

## Rebuild the visualization

```bash
uv run python -m music_evolution.build_html --data genres.json --out index.html
```
