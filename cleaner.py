import pandas as pd

def apply_cleaning(source, selected):
    df = source.copy(deep=True)
    actions = []
    if selected.get("trim_text"):
        changed = 0
        for c in df.select_dtypes(include=["object","string"]).columns:
            before = df[c].copy()
            df[c] = df[c].map(lambda x: x.strip() if isinstance(x,str) else x)
            changed += int((before != df[c]).fillna(False).sum())
        actions.append(f"Trimmed whitespace in text fields ({changed} cell(s) changed).")
    if selected.get("numeric_missing"):
        count = 0
        for c in df.select_dtypes(include=["number"]).columns:
            n = int(df[c].isna().sum()); med = df[c].median()
            if n and pd.notna(med): df[c] = df[c].fillna(med); count += n
        actions.append(f"Filled {count} missing numeric cell(s) with column medians.")
    if selected.get("categorical_missing"):
        count = 0
        for c in df.select_dtypes(include=["object","string","category"]).columns:
            n = int(df[c].isna().sum()); mode = df[c].mode(dropna=True)
            if n and not mode.empty: df[c] = df[c].fillna(mode.iloc[0]); count += n
        actions.append(f"Filled {count} missing categorical cell(s) with column modes.")
    if selected.get("duplicates"):
        before = len(df); df = df.drop_duplicates()
        actions.append(f"Removed {before-len(df)} exact duplicate row(s).")
    if not actions: actions.append("No cleaning action selected; exported data is unchanged.")
    return df, actions
