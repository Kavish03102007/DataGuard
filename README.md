# DataGuard — Intelligent Data Quality & Anomaly Detection
## Features
- CSV and XLSX upload (10 MB limit)
- Profiling: dimensions, data types, missing values, unique values, duplicates
- IQR and Z-score statistical outlier flags
- Isolation Forest on suitable numeric columns
- Transparent project-defined quality score
- User-approved cleaning: median/mode imputation, duplicate removal, text trimming
- Cleaned CSV download

## Windows / PowerShell setup
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000.
If activation is blocked in this terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
Use `sample_data/sample_sales.csv` for the demo.

## Notes
- Score formula: 40% completeness + 25% uniqueness + 35% anomaly health. It is a project-defined indicator, not an industry standard.
- An anomaly is a review flag, not proof of an error or fraud.
- Cleaning operates on a copy; the original upload is not overwritten.
- Demo state is held in memory and is cleared when Flask restarts.
- Chart.js and Google Fonts load from CDNs; charts/fonts require internet access.
