import os
import re
import pandas as pd
import numpy as np

BASE_DIR = "."

FILES = {
    "RF_GA": "Random_Forest/GA/GA.shap_importance.csv",
    "RF_PTB": "Random_Forest/PTB/PTB.shap_importance.csv",
    "GB_GA": "Gradient_Boosting/GA/GA.shap_importance.csv",
    "GB_PTB": "Gradient_Boosting/PTB/PTB.shap_importance.csv",
    "NN_GA": "NeuralNetworks/GA/Top20SHAPfeatures.txt",
    "NN_PTB": "NeuralNetworks/PTB/Top20SHAPfeatures.txt",
    "ElasticNet_GA": "Linear-Logistic_Regression/GA/ElasticNetSHAP.txt",
    "RidgeLogit_PTB": "Linear-Logistic_Regression/PTB/RidgeSHAP.txt",
}

def clean_feature_name(feature):
    feature = re.sub(r"^(num__|bin__|cat__)", "", str(feature))
    feature = re.sub(r"_[0-9]+$", "", feature)
    return feature

def read_csv_shap(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"Feature": "feature", "MeanAbsSHAP": "importance"})
    df["variable"] = df["feature"].apply(clean_feature_name)

    # aggregate one-hot categories to parent variable for consensus only
    out = (
        df.groupby("variable", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["variable", "rank", "importance"]]

def read_txt_shap(path):
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if "mean|SHAP|" not in line and ":" not in line:
                continue

            # Handles:
            # 1. num__PC3 mean|SHAP| = 0.123
            # num__PC3: 0.123
            m1 = re.search(r"\d+\.\s+(.+?)\s+mean\|SHAP\|\s*=\s*([0-9.eE+-]+)", line)
            m2 = re.search(r"(.+?):\s*([0-9.eE+-]+)", line)

            if m1:
                feature = m1.group(1).strip()
                importance = float(m1.group(2))
            elif m2:
                feature = m2.group(1).strip()
                importance = float(m2.group(2))
            else:
                continue

            rows.append({
                "feature": feature,
                "variable": clean_feature_name(feature),
                "importance": importance
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["variable", "rank", "importance"])

    out = (
        df.groupby("variable", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["variable", "rank", "importance"]]

def read_importance_file(path):
    if path.endswith(".csv"):
        return read_csv_shap(path)
    return read_txt_shap(path)

all_tables = []

for model_name, rel_path in FILES.items():
    path = os.path.join(BASE_DIR, rel_path)

    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        continue

    tab = read_importance_file(path)
    tab = tab.rename(columns={
        "rank": f"{model_name}_rank",
        "importance": f"{model_name}_importance"
    })

    all_tables.append(tab)

if not all_tables:
    raise RuntimeError("No input files found.")

consensus = all_tables[0]

for tab in all_tables[1:]:
    consensus = consensus.merge(tab, on="variable", how="outer")

rank_cols = [c for c in consensus.columns if c.endswith("_rank")]

consensus["N_models_top20"] = consensus[rank_cols].le(20).sum(axis=1)
consensus["N_models_top10"] = consensus[rank_cols].le(10).sum(axis=1)
consensus["Mean_rank"] = consensus[rank_cols].mean(axis=1, skipna=True)
consensus["Median_rank"] = consensus[rank_cols].median(axis=1, skipna=True)

consensus = consensus.sort_values(
    ["N_models_top10", "N_models_top20", "Mean_rank"],
    ascending=[False, False, True]
)

consensus.to_csv("ConsensusFeatureTable.tsv", sep="\t", index=False)
consensus.to_csv("ConsensusFeatureTable.csv", index=False)

print(consensus.head(30))
print("\nWrote:")
print("ConsensusFeatureTable.tsv")
print("ConsensusFeatureTable.csv")
