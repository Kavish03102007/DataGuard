import traceback
from pathlib import Path
from uuid import uuid4
from io import BytesIO
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, abort
from werkzeug.utils import secure_filename
from analyzer import analyze_dataframe
from cleaner import apply_cleaning

ROOT = Path(__file__).resolve().parent
UPLOADS, OUTPUTS = ROOT/"uploads", ROOT/"outputs"
UPLOADS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
app = Flask(__name__)
app.secret_key = os.environ.get("DATAGUARD_SECRET", "demo-secret-change-before-deployment")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
DATASETS = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("dataset")
    if not f or not f.filename:
        flash("Choose a file first."); return redirect(url_for("index"))
    name = secure_filename(f.filename)
    ext = Path(name).suffix.lower()
    if ext not in {".csv", ".xlsx"}:
        flash("Upload a .csv or .xlsx file."); return redirect(url_for("index"))
    key = uuid4().hex
    path = UPLOADS / f"{key}_{name}"
    f.save(path)
    try:
        df = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path, engine="openpyxl")
        if df.empty: raise ValueError("The file contains no data rows.")
        if len(df) > 100000 or len(df.columns) > 200:
            raise ValueError("Demo limit: 100,000 rows and 200 columns.")
        report = analyze_dataframe(df)
        DATASETS[key] = {"df": df, "name": name, "cleaned": None, "report": report}
        preview_columns = [str(c) for c in df.columns]
        preview_rows = []
        for row in df.head(15).itertuples(index=False, name=None):
            preview_rows.append([
                "Missing" if pd.isna(value) else str(value.item() if hasattr(value, "item") else value)
                for value in row
            ])
        return render_template(
            "results.html", key=key, filename=name, r=report,
            preview_columns=preview_columns, preview_rows=preview_rows,
            preview_total=len(df)
        )
    except Exception as e:
        traceback.print_exc()
        path.unlink(missing_ok=True)
        flash(f"Could not analyse file: {e}")
        return redirect(url_for("index"))

@app.route("/clean/<key>", methods=["POST"])
def clean(key):
    item = DATASETS.get(key)
    if not item: abort(404)
    selected = {k: request.form.get(k) == "on" for k in
                ["numeric_missing", "categorical_missing", "duplicates", "trim_text"]}
    cleaned, actions = apply_cleaning(item["df"], selected)
    out = OUTPUTS / f"cleaned_{key}.csv"
    cleaned.to_csv(out, index=False)
    item["cleaned"] = out
    return render_template("cleaned.html", key=key, actions=actions,
                           before=len(item["df"]), after=len(cleaned), columns=len(cleaned.columns))

@app.route("/download-anomalies/<key>")
def download_anomalies(key):
    item = DATASETS.get(key)
    if not item:
        abort(404)

    export_rows = []
    for record in item["report"].get("anomaly_records", []):
        row = {
            "row_number": record.get("row_number", ""),
            "severity": record.get("severity", ""),
            "severity_reason": record.get("severity_reason", ""),
            "detection_methods": ", ".join(record.get("methods", [])),
            "reasons": " | ".join(record.get("reasons", [])),
        }
        for pair in record.get("values", []):
            row[str(pair.get("column", "column"))] = pair.get("value", "")
        export_rows.append(row)

    columns = ["row_number", "severity", "severity_reason", "detection_methods", "reasons"]
    for column in item["df"].columns:
        if str(column) not in columns:
            columns.append(str(column))
    csv_data = pd.DataFrame(export_rows, columns=columns).to_csv(index=False)
    memory_file = BytesIO(csv_data.encode("utf-8-sig"))
    memory_file.seek(0)
    return send_file(
        memory_file, as_attachment=True,
        download_name=f"dataguard_anomaly_report_{Path(item['name']).stem}.csv",
        mimetype="text/csv"
    )


@app.route("/download/<key>")
def download(key):
    item = DATASETS.get(key)
    if not item or not item["cleaned"]: abort(404)
    return send_file(item["cleaned"], as_attachment=True,
                     download_name=f"dataguard_cleaned_{Path(item['name']).stem}.csv",
                     mimetype="text/csv")

@app.errorhandler(413)
def too_large(_):
    flash("File exceeds the 10 MB upload limit.")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
