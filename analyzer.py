
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def analyze_dataframe(df):
    rows, cols = df.shape

    missing = df.isna().sum()

    profile = [
        {
            "name": str(c),
            "dtype": str(df[c].dtype),
            "missing": int(missing[c]),
            "missing_pct": round(100 * missing[c] / max(rows, 1), 1),
            "unique": int(df[c].nunique(dropna=True))
        }
        for c in df.columns
    ]

    duplicates = int(df.duplicated().sum())
    total_cells = max(rows * cols, 1)

    completeness = 100 * (
        1 - int(missing.sum()) / total_cells
    )
    uniqueness = 100 * (
        1 - duplicates / max(rows, 1)
    )

    numeric = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    def is_identifier(column):
        name = str(column).lower().replace("-", "_")
        tokens = name.split("_")

        return (
            name in {"id", "uuid", "key"}
            or any(t in {"id", "uuid"} for t in tokens)
        )

    numeric = [
        c for c in numeric if not is_identifier(c)
    ]

    statistical_rows = set()
    statistical_reasons = {}
    stats = []

    for c in numeric:
        vals = pd.to_numeric(
            df[c], errors="coerce"
        ).dropna()

        if vals.empty:
            continue

        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1

        series = pd.to_numeric(
            df[c], errors="coerce"
        )

        iqr_mask = (
            (series < q1 - 1.5 * iqr)
            | (series > q3 + 1.5 * iqr)
        ).fillna(False)

        sd = vals.std()

        if pd.notna(sd) and sd > 0:
            z_scores = (series - vals.mean()) / sd
            z_mask = (z_scores.abs() > 3).fillna(False)
        else:
            z_scores = pd.Series(
                0, index=df.index, dtype=float
            )
            z_mask = pd.Series(
                False, index=df.index
            )

        for idx in df.index:
            reasons = []

            if iqr_mask.loc[idx]:
                reasons.append(
                    f"IQR outlier in {c}: {series.loc[idx]}"
                )

            if z_mask.loc[idx]:
                reasons.append(
                    f"Z-score outlier in {c}: {z_scores.loc[idx]:.2f}"
                )

            if reasons:
                statistical_rows.add(idx)
                statistical_reasons.setdefault(
                    idx, []
                ).extend(reasons)

        stats.append({
            "name": str(c),
            "min": round(float(vals.min()), 3),
            "mean": round(float(vals.mean()), 3),
            "median": round(float(vals.median()), 3),
            "max": round(float(vals.max()), 3),
            "iqr": int(iqr_mask.sum()),
            "z": int(z_mask.sum())
        })

    usable = [
        c for c in numeric
        if df[c].notna().sum() >= 5
        and df[c].nunique(dropna=True) > 1
    ]

    ml_rows = set()
    ml_status = (
        "Skipped: no suitable numeric columns "
        "or fewer than 10 rows."
    )

    if usable and rows >= 10:
        matrix = df[usable].replace(
            [np.inf, -np.inf], np.nan
        )

        matrix = matrix.fillna(
            matrix.median()
        ).fillna(0)

        try:
            model = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42
            )

            predictions = model.fit_predict(matrix)

            ml_rows = set(
                df.index[predictions == -1]
            )

            ml_status = (
                f"Isolation Forest evaluated "
                f"{len(usable)} numeric column(s)."
            )

        except Exception as e:
            ml_status = f"Isolation Forest skipped: {e}"

    combined = statistical_rows | ml_rows

    anomaly_records = []

    for idx in sorted(
        combined,
        key=lambda x: df.index.get_loc(x)
    ):
        position = df.index.get_loc(idx)

        methods = []
        reasons = []

        if idx in statistical_rows:
            methods.append("Statistical")
            reasons.extend(
                statistical_reasons.get(idx, [])
            )

        if idx in ml_rows:
            methods.append("Isolation Forest")
            reasons.append(
                "Machine-learning model classified "
                "this record as an anomaly."
            )

        values = []

        for column, value in df.loc[idx].items():
            if pd.isna(value):
                display_value = "Missing"
            elif isinstance(value, np.generic):
                display_value = str(value.item())
            else:
                display_value = str(value)

            values.append({
                "column": str(column),
                "value": display_value
            })

        statistical_signal_count = sum(
            1 for reason in reasons
            if reason.startswith("IQR outlier") or reason.startswith("Z-score outlier")
        )

        # Severity is a project-defined review priority, not a probability of error.
        if ("Statistical" in methods and "Isolation Forest" in methods) or statistical_signal_count >= 2:
            severity = "High"
            severity_reason = "Multiple independent detection signals were found."
        elif "Statistical" in methods:
            severity = "Medium"
            severity_reason = "At least one statistical outlier rule was triggered."
        else:
            severity = "Low"
            severity_reason = "Flagged by the machine-learning model only; inspect in context."

        anomaly_records.append({
            "row_number": position + 1,
            "methods": methods,
            "reasons": reasons,
            "values": values,
            "severity": severity,
            "severity_reason": severity_reason
        })

    anomaly_health = max(
        0,
        100 - 100 * len(combined) / max(rows, 1)
    )

    score = round(
        0.40 * completeness
        + 0.25 * uniqueness
        + 0.35 * anomaly_health
    )

    return {
        "rows": rows,
        "cols": cols,
        "missing_total": int(missing.sum()),
        "duplicates": duplicates,
        "completeness": round(completeness, 1),
        "uniqueness": round(uniqueness, 1),
        "stat_count": len(statistical_rows),
        "ml_count": len(ml_rows),
        "anomaly_count": len(combined),
        "score": max(0, min(100, score)),
        "profile": profile,
        "stats": stats,
        "labels": [str(c) for c in df.columns],
        "missing": [int(missing[c]) for c in df.columns],
        "ml_status": ml_status,
        "anomaly_records": anomaly_records
    }
