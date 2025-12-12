# Week 11 Wildfire Prediction

## Overview
This repository explores how historical weather patterns relate to daily wildfire activity across the United States. The workflow combines station metadata, multi-decade weather archives, and the US Forest Service wildfire incident database to:

- curate analysis-ready parquet and CSV extracts for the 1992-2015 period
- engineer seasonal and lagged weather features aligned with ignition records
- train LightGBM-based models that predict (1) whether any fire occurs, (2) the number of fires, (3) log-area burned, and (4) the probability distribution over leading cause codes
- visualize model calibration, residuals, and feature importances so stakeholders can see which conditions drive extreme fire days

## Repository Layout
- `scripts/filter_data.py` – standalone data-prep script that trims raw weather data to US stations, converts Julian discovery/containment dates to calendar dates, and exports parquet/CSV artifacts
  - Inputs: `data/cities.csv`, `data/countries.csv`, `data/daily_weather.parquet`, `data/wildfires.sqlite`
   - Outputs: `data/us_daily_weather_1992_2015.parquet`, `data/wildfires_fires.{csv,parquet}`
- `shared_data/` – light-weight lookup tables (cities, countries, regional clusters) checked into source control for quick reference
- `scripts/wildfires.ipynb` – end-to-end notebook for feature engineering, model training, and diagnostic visualizations (counts, area, and cause-specific classifiers), uses the light-weight tables
- `scripts/geopandas_demo.py` – self-contained GeoPandas sanity check that loads station metadata, performs state joins/buffers, overlays station scatter atop the per-state choropleth, plots wildfire ignition points (including a 100 km buffer overlay that highlights fires within coverage), and drops sample plots under `artifacts/geopandas_demo`
- `requirements.txt` – Python dependencies for both the script and notebook environments
- `notes.txt` – running scratchpad with open analysis questions and visualization requests

> Large raw inputs (the parquet weather archive and `wildfires.sqlite` database) are not tracked in Git. Place them under `data/` to match the paths expected by the scripts.

## Getting Started
1. **Create a Python environment** (Windows PowerShell example):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. **Install the project requirements**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Stage raw data**:
   - Download / copy the weather parquet, wildfire SQLite database, and supporting city metadata into `data/`
   - Optional reference tables already live in `shared_data/`
4. **Run the preprocessing script**:
   ```powershell
   python scripts/filter_data.py
   ```
   This filters the weather history to US stations between 1992-01-01 and 2015-12-31, casts date fields, and materializes compact parquet/CSV files used downstream.
5. **Open the modeling notebook**:
   - Launch VS Code or JupyterLab
   - Run `scripts/wildfires.ipynb` sequentially to build features, fit LightGBM models, evaluate metrics (AUC/RMSE), and render plots such as predicted-vs-actual charts and feature importance bars for area and cause models.

## Modeling Highlights
- **Feature Engineering**: Encodes state, station, and day-of-year signals; rolls 30-day means for temperature, precipitation, wind, and pressure; merges lagged fire statistics within varying radius cutoffs using BallTree queries.
- **Targets**: Binary "any fire" indicator, daily fire counts, log-transformed area burned, and the share of top ignition causes (e.g., lightning, campfire).
- **Models**: Gradient boosted trees via LightGBM with tuned training/validation splits and multi-output regressors for cause probabilities.
- **Evaluation & Visualization**: Notebook cells output scatter plots (predicted vs. actual), feature importance rankings, and diagnostic tables answering questions like "Which conditions increase the odds of cause code 1 (Lightning) versus 4 (Campfire)?" and "What drives large burned areas?"

## Extending the Project
- Compare additional algorithms (e.g., XGBoost, temporal CNNs) against the LightGBM baselines.
- Integrate drought indices or satellite-derived vegetation layers to capture fuel conditions.
- Automate daily model retraining with scheduled jobs plus lightweight dashboards for feature alerts.
- Contribute new exploratory notebooks or visualizations that address the open questions captured in `notes.txt`.

Feel free to open issues or PRs with enhancements, bug fixes, or new analyses.
