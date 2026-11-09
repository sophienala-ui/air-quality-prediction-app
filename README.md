# Air Quality Prediction — SDG 11

This Streamlit app estimates PM2.5 using the Beijing Multi-Site Air Quality
dataset from the UCI Machine Learning Repository.

## Important fix

The original app used `ucimlrepo.fetch_ucirepo(id=501)`. The deployed
environment can report dataset 501 as unavailable for import, so this version
downloads the official UCI dataset archive directly and reads the 12 station
CSV files.

## Files

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies

## Streamlit Cloud

1. Upload `app.py` and `requirements.txt` to your GitHub repository.
2. In Streamlit Community Cloud, choose that repository.
3. Set the main file to `app.py`.
4. Deploy.

The app needs internet access at startup so it can download the official UCI
dataset archive.

## Dataset

Beijing Multi-Site Air Quality, UCI Machine Learning Repository, dataset 501.
The dataset contains 420,768 hourly observations from 12 monitoring stations
covering March 2013 through February 2017.

UCI dataset page:
https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality

License: CC BY 4.0.
