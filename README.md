# Flight Delay Predictor (US DOT data → LightGBM → FastAPI → web page)

Estimate the probability that a US domestic flight **arrives 15+ minutes late**, using Bureau of Transportation Statistics (BTS) on-time data.

```
BTS monthly zips ──download_data.py──▶ Parquet (small, fast)
                                          │  DuckDB (SQL on 100M+ rows)
                                          ▼
                      features.py (history + calendar features, no leakage)
                                          ▼
                      train.py ──▶ artifacts/ (model, lookups, metrics)
                                          ▼
 Browser (web/) ◀── JSON ──▶ FastAPI (api/main.py) ── predictor.py loads artifacts
                                          ▼
                              Docker ▶ Hugging Face Spaces / Cloud Run / Render
```

## 0. Dataset (official source, free, no login)

* Download page (choose **"Reporting Carrier On-Time Performance (1987-present)"** → *Download*):
  **https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD**
* Dataset description: https://transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD
* Direct monthly zip pattern used by `src/download_data.py` (replace `{year}` and `{month}`, month has no leading zero):
  `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip`
  e.g. January 2023 → `..._1987_present_2023_1.zip`

If the script cannot download a month (site slow, certificate or format change), download the CSV by hand from the page above
(select all fields, or at least the ones listed in `src/download_data.py`) and drop it in `data/raw/` with a name ending in `_YEAR_MONTH.csv`
(e.g. `ontime_2023_1.csv`). The script will convert it.

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get data

**Option A: try everything in 5 minutes with fake data (no download):**
```bash
python scripts/make_sample_data.py --years 2021 2022 2023 2024 --rows-per-year 150000
```
**Option B: real data (recommended for your portfolio):** about 7 years ≈ 40M+ rows, ~2 GB as Parquet, takes a while.
```bash
python -m src.download_data --start-year 2018 --end-year 2024
# smoke test first:  python -m src.download_data --start-year 2023 --end-year 2023 --months 1 2 3
```
You need **at least 4 distinct years** (1 for history, ≥1 train, 1 validation, 1 test).
Do not mix fake and real files in `data/parquet/`.

## 3. Learn the data (notebook)

```bash
jupyter notebook notebooks/flight_delay_end_to_end.ipynb
```

## 4. Train

```bash
python -m src.train --quick                               # ~10 seconds, checks the pipeline
python -m src.train --max-train-rows 5000000 --memory-limit 8GB   # real run
```
This writes `artifacts/model.joblib`, `artifacts/lookups.joblib`, `artifacts/metrics.json`.
Read `metrics.json`: compare the model to the baselines. Real data with schedule + history features typically gives a modest ROC-AUC
(roughly 0.65 to 0.72). That is normal: weather and the inbound aircraft's delay are the missing signals. Report your own numbers honestly.

## 5. Run the app locally (backend + frontend together)

```bash
uvicorn api.main:app --reload --port 8000
```
Open http://localhost:8000 (web page) and http://localhost:8000/docs (interactive API docs).
Tests: `pytest -q`.

### How the frontend and backend are connected

* `api/main.py` creates the FastAPI app and defines JSON routes under `/api/...`.
* The last line, `app.mount("/", StaticFiles(directory="web", html=True))`, makes the **same server** also serve `web/index.html`, `style.css` and `app.js`.
* `web/app.js` calls the API with relative URLs, e.g. `fetch("/api/predict", {method: "POST", body: JSON.stringify({...})})`.
  Same origin → no CORS configuration needed, one deployment, one URL.
* Flow: page loads → `GET /api/meta` fills the airport list → choosing an airport calls `GET /api/destinations?origin=ORD` →
  choosing a destination calls `GET /api/carriers?origin=ORD&dest=LAX` → pressing the button calls `POST /api/predict` → the response
  (`probability`, `risk`, `reasons`, `context`) is drawn on the boarding-pass result and the chart.
* **If you host the web page on a different domain** (e.g. Netlify/Vercel) and the API elsewhere: in `web/app.js` prefix every
  path with your API URL (`const API = "https://your-api.example.com"`), and on the server set the env var
  `ALLOWED_ORIGINS=https://your-site.netlify.app` (this turns on CORS).

## 6. Deploy

The image only contains code + `artifacts/` (not the data), so it is small. **Commit `artifacts/`** (it is not git-ignored).

### Option 1: Hugging Face Spaces (free, easiest)
1. Create an account → **New Space** → SDK: **Docker** → blank template → public.
2. In your project folder:
   ```bash
   git lfs install
   git lfs track "*.joblib"          # model files are > 10 MB
   git add .gitattributes
   git init && git add . && git commit -m "flight delay app"
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push --force space HEAD:main
   ```
3. Add this to the top of `README.md` (Spaces reads it) if the Space asks for it:
   ```
   ---
   title: Flight Delay Predictor
   sdk: docker
   app_port: 7860
   ---
   ```
4. Wait for the build; your app is at `https://<your-username>-<space-name>.hf.space`.

### Option 2: Google Cloud Run (real cloud skills, generous free tier)
```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud run deploy flight-delay --source . --region us-central1 --allow-unauthenticated --memory 1Gi
```
Cloud Run builds the Dockerfile, injects `$PORT`, and prints a public URL.

### Option 3: Render / Railway / Fly.io
Push the repo to GitHub → create a **Web Service** from the repo → environment *Docker* → deploy. Nothing else to configure.

### Test the Docker image locally first
```bash
docker build -t flight-delay .
docker run --rm -p 7860:7860 flight-delay      # http://localhost:7860
```

### CI
`.github/workflows/ci.yml` runs on every push: builds sample data, trains a tiny model, runs the tests.

## 7. Project layout

```
src/config.py          paths, feature lists, constants
src/download_data.py   BTS zip → Parquet
src/features.py        SQL feature engineering + calendar features + time split
src/train.py           baselines, LightGBM, calibration, saves artifacts
src/lookups.py         small lookup tables for serving
src/predictor.py       inputs → features → probability + reasons
api/main.py            FastAPI routes + serves web/
web/                   index.html, style.css, app.js
notebooks/             step-by-step learning notebook
scripts/               fake data generator, local run helper
tests/test_api.py      API tests
Dockerfile             production image
```

## 8. Concepts to be able to explain in an interview

* **Leakage**: only pre-departure information is used; history features use the previous year only.
* **Time-based split**: train → validation → test are consecutive years, never random.
* **Metrics**: ROC-AUC, PR-AUC, Brier score, calibration. Accuracy is misleading with ~20% positives.
* **Training/serving parity**: `features.py` and `predictor.py` share the same calendar and encoding code.
* **Honest limits**: no live weather, no inbound-aircraft status; a "30%" is a probability, not a promise.

## 9. Upgrade ideas
Weather features (Meteostat/NOAA), aircraft-rotation delay (previous flight of the same tail number), Optuna tuning, MLflow tracking,
PySpark/Dask feature pipeline, monthly retraining (Airflow/Prefect), a drift-monitoring page, a second page with airline/airport dashboards.
