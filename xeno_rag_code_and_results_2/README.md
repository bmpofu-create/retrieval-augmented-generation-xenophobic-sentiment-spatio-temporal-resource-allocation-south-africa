# Xeno-RAG pipeline
Run from `src/` in order: `s01_features.py` -> `s03_sentiment_models.py` -> `s04_eda.py` -> `HORIZON=1 s05_zone_model.py` and `HORIZON=0 s05_zone_model.py` -> `s06_dashboard_data.py` (+ `dashboard_template.html`) -> `s07_report.py`.
Needs pandas, numpy, scipy, scikit-learn, matplotlib, python-docx. Edit `DATA` in `common.py`.
`rag_llm.py` builds the RAG prompt for any LLM callable (not run in reported results). `train_distilbert.py` is an untested Colab script.

Run the scripts one after another from the `src` folder. The whole run takes about 15 to 20 minutes, and almost all of that is step 2.

**1. Put the model-ready CSV next to this folder's `src` directory.** `src/common.py` reads `SA_XenoDataset_model_ready.csv` from the pipeline root. That file is not part of the git repo.

**2. Run the scripts in order.**

```
cd src
python s01_features.py
python s03_sentiment_models.py
python s04_eda.py
```

- `s01_features.py` builds the time-series, virality and spatial features. It takes seconds.
- `s03_sentiment_models.py` runs the sentiment benchmark and builds the vector index. It takes about 10 minutes.
- `s04_eda.py` makes the KDE maps and the lead-lag analysis, and saves the figures to `figures/`.

**3. Run the zone model twice**, once for next-week forecasting and once for same-week classification.

Windows (Command Prompt):
```
set HORIZON=1
python s05_zone_model.py
set HORIZON=0
python s05_zone_model.py
```

Windows (PowerShell):
```
$env:HORIZON=1; python s05_zone_model.py
$env:HORIZON=0; python s05_zone_model.py
```

Mac/Linux:
```
HORIZON=1 python s05_zone_model.py
HORIZON=0 python s05_zone_model.py
```

**4. Build the dashboard data and the Word report.**

```
python s06_dashboard_data.py
python s07_report.py
```

`s07_report.py` writes `Xeno_RAG_Implementation_Report.docx` into `outputs/`.

**5. Build the dashboard page.** `README.md` doesn't mention this step, and no script does it. After `s06`, run this from `src`:

```
python -c "t=open('dashboard_template.html').read(); d=open('../outputs/dashboard_data.json').read(); open('../outputs/dashboard.html','w').write(t.replace('/*DATA*/null',d))"
```

Then double-click `outputs/dashboard.html` to open it in your browser. A ready-made copy of that file is already in the zip.

**Where results go.** Everything is written to `outputs/` (tables, predictions, report) and `figures/` (PNG charts). The main files to look at are:
- `sentiment_model_comparison.csv`
- `zone_model_comparison_nowcast.csv`
- `eda_results.json`

**Improvements.** These leave the published benchmark files untouched. From `src`:

```
python s02_label_audit.py
python s03b_binary_sentiment.py
python s05b_zone_rates.py
```

`s02` writes a 200-tweet review sheet and does not relabel anything. `s03b` trains a binary hostile model, dropping texts that are labelled both Negative and not. `s05b` rescores each town against its own past weeks and keeps mobilisation language as its own feature. Put an independent incident file at `incidents_independent.csv` (columns `date`, `town`, `attacks`) if you have one; otherwise the zone target is still the attack count on the tweets. Read `outputs/improvement_summary.json`.

**Things to know:**
- Run the scripts in the order above, because each one reads the previous one's output.
- Run both `HORIZON=1` and `HORIZON=0` before `s06`, or it will fail.
- If you get `ModuleNotFoundError: common`, you're not in the `src` folder.
- The code uses a fixed random seed (42), so you should get the same numbers I did.