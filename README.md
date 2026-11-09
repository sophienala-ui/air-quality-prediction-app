# Air Quality Prediction — Streamlit

This is a Streamlit version of the uploaded Colab notebook.

## Files

- `app.py` — Streamlit application
- `requirements.txt` — required Python packages

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Upload `app.py` and `requirements.txt` to a GitHub repository, then select:
- Main file: `app.py`
- Branch: `main`

The app downloads the UCI Beijing Multi-Site Air-Quality dataset using `ucimlrepo`.
