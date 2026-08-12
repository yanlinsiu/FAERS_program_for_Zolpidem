from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REBUILD_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REBUILD_DIR / "outputs"
PROJECT_DIR = REBUILD_DIR.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=OUTPUT_DIR / "13_corrected_main_dataset.parquet")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "14_analysis_ready_dataset.parquet")
    args = parser.parse_args()
    source = args.input.resolve()
    target = args.output.resolve()
    master = pd.read_csv(PROJECT_DIR / "configs" / "sedative_hypnotic_drug_master.csv")
    data = pd.read_parquet(source)
    data["primaryid_example"] = data["primaryid"]

    drug_keys = master["drug_key"].tolist()
    z_drugs = master.loc[master["drug_group"].eq("z_drug"), "drug_key"].tolist()
    groups = ["z_drug", "benzodiazepine", "orexin_antagonist", "other_insomnia_related"]
    for suffix in ["ps_ss", "ps_only"]:
        data[f"exposure_other_z_drug_{suffix}"] = data[
            [
                f"exposure_eszopiclone_{suffix}",
                f"exposure_zaleplon_{suffix}",
                f"exposure_zopiclone_{suffix}",
            ]
        ].any(axis=1)
        drug_columns = [f"exposure_{drug}_{suffix}" for drug in drug_keys]
        group_columns = [f"exposure_{group}_{suffix}" for group in groups]
        z_columns = [f"exposure_{drug}_{suffix}" for drug in z_drugs]
        data[f"n_sedative_hypnotic_drugs_{suffix}"] = data[drug_columns].sum(axis=1).astype("int16")
        data[f"n_sedative_hypnotic_groups_{suffix}"] = data[group_columns].sum(axis=1).astype("int8")
        data[f"mixed_z_drug_{suffix}"] = data[z_columns].sum(axis=1).gt(1)
        data[f"mixed_sedative_hypnotic_group_{suffix}"] = data[f"n_sedative_hypnotic_groups_{suffix}"].gt(1)
        data[f"z_drug_plus_benzo_{suffix}"] = (
            data[f"exposure_z_drug_{suffix}"] & data[f"exposure_benzodiazepine_{suffix}"]
        )

    serious = pd.read_parquet(source.parent / "09_serious_outcome_matrix.parquet")
    missing_serious = [column for column in serious.columns if column != "caseid" and column not in data.columns]
    if missing_serious:
        data = data.merge(serious[["caseid", *missing_serious]], on="caseid", how="left", validate="one_to_one")
        for column in missing_serious:
            data[column] = data[column].fillna(False).astype(bool)

    data["analysis_eligible_main"] = (
        data["caseid"].notna()
        & data["year"].notna()
        & data["age_group_3"].isin(["65-74", "75-84", ">=85"])
        & data["sex_clean"].isin(["M", "F"])
    )
    data["analysis_eligible_insomnia"] = data["analysis_eligible_main"] & data["indi_insomnia"]
    data["analysis_eligible_clean_ps_ss"] = (
        data["analysis_eligible_main"] & ~data["mixed_sedative_hypnotic_group_ps_ss"]
    )
    data["analysis_eligible_clean_ps_only"] = (
        data["analysis_eligible_main"] & ~data["mixed_sedative_hypnotic_group_ps_only"]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(target, index=False, compression="zstd")
    print(f"Wrote {target}")
    print(f"Rows: {len(data):,}; eligible: {int(data['analysis_eligible_main'].sum()):,}")


if __name__ == "__main__":
    main()
