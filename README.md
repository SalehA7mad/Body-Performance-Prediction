# 🏋️ Body Performance ML

Predict a person's **physical performance class (A–D)** and **broad-jump distance** from body measurements, using classic machine-learning models, plus an interactive Streamlit app.

![CI](https://github.com/<your-username>/body-performance-ml/actions/workflows/ci.yml/badge.svg)

## What's inside

| Task | Target | Models compared | Best (selected by cross-validation) |
|---|---|---|---|
| Classification | `class` (A = best … D = lowest) | KNN, Decision Tree, SVM, Neural Network | Neural Network |
| Regression | `broad jump_cm` | Linear Regression, KNN, SVR, Neural Network | Neural Network |

### Results (held-out 20% test set)

**Classification** (balanced 4-class problem, random guessing = 25%)

| Model | CV accuracy | Test accuracy | Test F1 |
|---|---|---|---|
| KNN | 0.614 | 0.622 | 0.623 |
| Decision Tree | 0.678 | 0.664 | 0.666 |
| SVM (RBF) | 0.704 | 0.704 | 0.702 |
| **Neural Network** | **0.722** | **0.735** | **0.733** |

**Regression** (broad jump, cm)

| Model | CV R² | Test R² | Test RMSE (cm) | Test MAE (cm) |
|---|---|---|---|---|
| Linear Regression | 0.793 | 0.790 | 18.2 | 13.8 |
| KNN | 0.784 | 0.783 | 18.5 | 14.0 |
| SVR | 0.791 | 0.793 | 18.0 | 13.7 |
| **Neural Network** | **0.796** | **0.798** | **17.8** | **13.6** |

> Linear regression is almost as good as the neural net on the regression task, so a simple model is a reasonable choice there.
> Exact figures are written to `reports/metrics.json` every time you train.

## Project structure

```
body-performance-ml/
├── app/streamlit_app.py          # interactive demo
├── data/bodyPerformance.csv      # dataset
├── models/                       # trained models (classifier / regressor)
├── notebooks/
│   └── Body_Performance_Analysis.ipynb   # EDA + modelling narrative (with outputs)
├── reports/
│   ├── metrics.json              # all model scores & best hyper-parameters
│   └── figures/                  # confusion matrix, actual-vs-predicted, comparisons
├── src/bodyperf/
│   ├── config.py                 # paths, column names, feature lists
│   ├── data.py                   # load / validate / clean / feature engineering
│   ├── models.py                 # model zoo, grids, metrics
│   ├── plots.py                  # plotting helpers
│   └── train.py                  # train → compare → save best models
├── tests/test_pipeline.py
├── .github/workflows/ci.yml
├── requirements.txt · requirements-dev.txt · pyproject.toml · Makefile
└── LICENSE
```

## Quick start

```bash
git clone https://github.com/<your-username>/body-performance-ml.git
cd body-performance-ml

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .

python -m bodyperf.train          # retrain & compare models (~5 min on 1 CPU core)
streamlit run app/streamlit_app.py
pytest -q
```

> Saved `.joblib` models are tied to the scikit-learn version they were created with. If the app reports it could not load them, simply run `python -m bodyperf.train` again.

## Methodology

1. **Cleaning** – drop duplicates; clip body/strength outliers to 1.5×IQR fences; replace blood-pressure outliers with the in-range mean; drop impossible `sit and bend forward_cm ≥ 150` rows.
2. **Features** – add BMI, encode gender (M=0, F=1).
3. **Modelling** – every model sits in a `Pipeline(StandardScaler → model)`; hyper-parameters are tuned with 5-fold `GridSearchCV` **on the training set only**; the test set is used once for reporting; the "best" model is chosen by CV score.
4. **Regression features** exclude the `class` label (it is derived from the physical tests and would leak the answer).

## Dataset

[Body performance data](https://www.kaggle.com/datasets/kukuroo3/body-performance-data) (Korea Sports Promotion Foundation), ~13.4k participants aged 21–64 with 11 measurements and a performance grade A–D. Check the original source for its licence before redistributing the data.

## Disclaimer

This is an educational project and is **not** medical or fitness advice.

## License

MIT, see [LICENSE](LICENSE).
