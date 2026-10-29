# Amalgam Finale: Invariant Polymer Property Prediction

Streamlit app and modular Python backend for predicting seven polymer properties from repeat-unit SMILES: `tg`, `egc`, `egb`, `nc`, `eps`, `ei`, `eea`.

Official metric: unweighted mean R² across all seven properties.\
Dataset Link : <https://drive.google.com/file/d/1lXC54EnmtSXV5jeyact_ILtd3fCMdoOt/view?usp=sharing>

## Core Pipeline

- RDKit canonicalization, oligomer-periodicity reduction, wildcard carbon capping.
- 1,027 invariant features: backbone ratios, RDKit descriptors, MACCS keys, Morgan/FCFP fingerprints, fragments, polymer motifs, ring/bond statistics, physical descriptors.
- Five learners: LightGBM, CatBoost, XGBoost, ExtraTrees, standardized Ridge.
- Ten-fold cross-validation, NNLS non-negative blend weights.
- Cascaded transfer: Level 1 tg/egc; Level 2 egb, nc, eea, ei; Level 3 eps with Maxwell `nc²`.
- Physical linkers, exact train-target overrides, and property bounds.

## Project Layout

```text
app.py
pages/     # seven dashboard pages
src/       # preprocessing, features, inference, validation
training/  # offline artifact generation
artifacts/ # trained model files
```

Default datasets: `D:\Amalgam\train.csv` and `D:\Amalgam\test.csv`. `PI1M.csv` is never loaded.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run Locally

```powershell
streamlit run app.py
```

Generate model artifacts before prediction:

```powershell
python training/train_models.py
```

Artifacts are saved under `artifacts/`. Cross-validation metrics and SHAP pages require generated artifacts. Training can be memory-intensive.

## Features

- Single SMILES validation, RDKit previews, all-property prediction, CSV export.
- Dataset exploration, target distributions, missing-value summaries, filtered downloads.
- Batch CSV prediction with invalid-row reporting.
- Pipeline tests and compile-safe modular backend.

## Validation

```powershell
python -m compileall app.py src pages training
```

Run additional quality checks with pytest, mypy, ruff, and black when maintaining the backend. Missing model artifacts are reported honestly in the UI.