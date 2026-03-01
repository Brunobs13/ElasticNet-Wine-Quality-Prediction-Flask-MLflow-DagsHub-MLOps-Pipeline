# Data Science Project - Wine Quality Prediction

This project predicts wine quality using a full machine learning pipeline and a small Flask app for online predictions.

## Description
End-to-end wine quality prediction project with a structured ML pipeline, MLflow tracking on DagsHub, and a Flask interface for real-time predictions.

It was built to show practical MLOps flow in a simple way:
- data ingestion
- data validation
- data transformation
- model training
- model evaluation with MLflow (DagsHub)
- web prediction endpoint

## Project Goal
The goal is to estimate wine quality (`quality`) from physicochemical features such as acidity, pH, sulphates, alcohol, and sulfur dioxide levels.

## End-to-End Flow
1. Download source data from URL (`config/config.yaml`)
2. Validate schema (`schema.yaml`)
3. Split train/test data
4. Train ElasticNet model (`params.yaml`)
5. Evaluate and log metrics to MLflow (DagsHub URI)
6. Serve predictions via Flask UI

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- Model used in training: **ElasticNet Regressor** (`sklearn.linear_model.ElasticNet`)
- Flask
- MLflow
- DagsHub (tracking backend)
- joblib / YAML config system

## Model Details
- Algorithm: ElasticNet (regression)
- Hyperparameters source: `params.yaml`
  - `alpha`
  - `l1_ratio`
- Evaluation metrics logged:
  - RMSE
  - MAE
  - R2

## Project Structure
```text
datascienceproject/
├── app.py
├── main.py
├── config/
│   └── config.yaml
├── params.yaml
├── schema.yaml
├── src/datascience/
│   ├── components/
│   ├── config/
│   ├── constants/
│   ├── entity/
│   ├── pipeline/
│   └── utils/
├── templates/
├── research/
└── artifacts/   (generated at runtime)
```

## Run Locally
### 1) Create environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run full training pipeline
```bash
python main.py
```
This executes all stages in order and generates artifacts.

### 3) Run the web app
```bash
python app.py
```
App runs on:
- `http://localhost:8080`

Useful routes:
- `GET /` -> input form
- `GET /train` -> trigger training
- `GET/POST /predict` -> online inference

## Configuration
Main config files:
- `config/config.yaml` -> paths and artifact locations
- `schema.yaml` -> expected columns and target
- `params.yaml` -> model hyperparameters (`alpha`, `l1_ratio`)

## MLflow / DagsHub
The evaluation stage logs metrics and model info to MLflow using DagsHub tracking URI.

In this codebase, the URI is set in configuration manager:
- `src/datascience/config/configuration.py`

If you need authenticated tracking, set your MLflow/DagsHub credentials in your local environment before running the pipeline.

## Artifacts Generated
After training, you should see outputs under `artifacts/`, including:
- transformed train/test files
- trained model (`model.joblib`)
- evaluation metrics JSON

## Notebooks
The `research/` folder contains notebook-based experiments for each stage and fast iteration.

## Current Notes
- `Dockerfile` exists but is currently empty in this repository state.
- `.gthub/workflows` is a placeholder path and not an active GitHub Actions folder.

## What this project demonstrates
- Clean pipeline-by-stage architecture
- Config-driven ML workflow
- Basic reproducibility and experiment tracking
- Practical path from training to serving
