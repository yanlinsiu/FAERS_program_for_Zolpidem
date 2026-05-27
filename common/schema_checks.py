from __future__ import annotations

from pathlib import Path

import pandas as pd


SIGNAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "caseid",
    "is_fall",
    "is_zolpidem_any",
    "is_zolpidem_suspect",
    "is_zolpidem_suspect_ps",
    "is_other_zdrug_suspect",
    "is_other_zdrug_suspect_ps",
    "suspect_role_any",
    "suspect_role_any_ps",
    "target_drug_group",
    "target_drug_group_ps",
    "age_group",
    "sex_clean",
    "year",
    "quarter",
)

FEATURE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "caseid",
    "is_zolpidem",
    "is_zolpidem_any",
    "is_zaleplon",
    "is_zopiclone",
    "is_eszopiclone",
    "is_benzo",
    "is_antidepressant",
    "is_antipsychotic",
    "is_opioid",
    "is_antiepileptic",
    "drug_n",
    "distinct_drug_n",
    "polypharmacy_5",
)

CASE_INDEX_REQUIRED_COLUMNS: tuple[str, ...] = (
    "caseid",
    "primaryid",
    "year",
    "quarter",
)

FORBIDDEN_LEGACY_COLUMNS: tuple[str, ...] = (
    "target_drug",
    "drug_group",
    "zolpidem_as_ps",
    "zolpidem_as_suspect",
    "other_zdrug_as_suspect",
)


def require_columns(df: pd.DataFrame, required: tuple[str, ...] | list[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required official columns: {missing}")


def forbid_legacy_columns(df: pd.DataFrame, table_name: str) -> None:
    present = [col for col in FORBIDDEN_LEGACY_COLUMNS if col in df.columns]
    if present:
        raise ValueError(f"{table_name} contains forbidden legacy columns: {present}")


def validate_signal_schema(df: pd.DataFrame, table_name: str = "signal dataset") -> None:
    require_columns(df, SIGNAL_REQUIRED_COLUMNS, table_name)
    forbid_legacy_columns(df, table_name)


def validate_feature_schema(df: pd.DataFrame, table_name: str = "feature dataset") -> None:
    require_columns(df, FEATURE_REQUIRED_COLUMNS, table_name)
    forbid_legacy_columns(df, table_name)


def validate_case_index_schema(df: pd.DataFrame, table_name: str = "case index dataset") -> None:
    require_columns(df, CASE_INDEX_REQUIRED_COLUMNS, table_name)
    forbid_legacy_columns(df, table_name)


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna(False).astype(bool)


def normalized_caseids(df: pd.DataFrame) -> pd.Series:
    return df["caseid"].where(df["caseid"].notna(), "").astype(str).str.strip()


def expected_target_group(df: pd.DataFrame, suffix: str = "") -> pd.Series:
    suspect = bool_series(df, f"suspect_role_any{suffix}")
    zolpidem = bool_series(df, f"is_zolpidem_suspect{suffix}")
    other = bool_series(df, f"is_other_zdrug_suspect{suffix}")
    expected = pd.Series("no_suspect_drug", index=df.index)
    expected.loc[suspect] = "no_target_zdrug_suspect"
    expected.loc[other] = "other_zdrug_only"
    expected.loc[zolpidem] = "zolpidem_only"
    expected.loc[zolpidem & other] = "both_zolpidem_and_other_zdrug"
    return expected


def _audit_row(check: str, status: str, value: object, detail: str = "") -> dict[str, object]:
    return {
        "check": check,
        "status": status,
        "value": value,
        "detail": detail,
    }


def _missing_columns(df: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    return [col for col in required if col not in df.columns]


def _present_forbidden_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in FORBIDDEN_LEGACY_COLUMNS if col in df.columns]


def audit_core_analysis_tables(
    case_index: pd.DataFrame,
    signal: pd.DataFrame,
    feature: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    table_specs = (
        ("case_index", case_index, CASE_INDEX_REQUIRED_COLUMNS),
        ("signal", signal, SIGNAL_REQUIRED_COLUMNS),
        ("feature", feature, FEATURE_REQUIRED_COLUMNS),
    )

    for table_name, df, required_cols in table_specs:
        missing = _missing_columns(df, required_cols)
        rows.append(
            _audit_row(
                f"{table_name}_required_columns",
                "fail" if missing else "pass",
                len(missing),
                "|".join(missing),
            )
        )
        forbidden = _present_forbidden_columns(df)
        rows.append(
            _audit_row(
                f"{table_name}_forbidden_legacy_columns",
                "fail" if forbidden else "pass",
                len(forbidden),
                "|".join(forbidden),
            )
        )

    if any(row["status"] == "fail" for row in rows):
        return pd.DataFrame(rows)

    case_ids = normalized_caseids(case_index)
    signal_ids = normalized_caseids(signal)
    feature_ids = normalized_caseids(feature)

    for table_name, ids in (
        ("case_index", case_ids),
        ("signal", signal_ids),
        ("feature", feature_ids),
    ):
        duplicate_count = int(ids.duplicated().sum())
        empty_count = int(ids.eq("").sum())
        rows.append(
            _audit_row(
                f"{table_name}_unique_caseid",
                "fail" if duplicate_count else "pass",
                duplicate_count,
            )
        )
        rows.append(
            _audit_row(
                f"{table_name}_nonempty_caseid",
                "fail" if empty_count else "pass",
                empty_count,
            )
        )

    case_index_set = pd.Index(case_ids)
    signal_set = pd.Index(signal_ids)
    feature_set = pd.Index(feature_ids)
    signal_missing_feature = len(signal_set.difference(feature_set))
    signal_missing_case_index = len(signal_set.difference(case_index_set))
    feature_missing_signal = len(feature_set.difference(signal_set))
    rows.extend(
        [
            _audit_row(
                "signal_rows_without_feature",
                "fail" if signal_missing_feature else "pass",
                signal_missing_feature,
            ),
            _audit_row(
                "signal_rows_without_case_index",
                "fail" if signal_missing_case_index else "pass",
                signal_missing_case_index,
            ),
            _audit_row(
                "feature_rows_without_signal",
                "fail" if feature_missing_signal else "pass",
                feature_missing_signal,
            ),
        ]
    )

    implication_checks = {
        "ps_zolpidem_implies_ps_ss_zolpidem": bool_series(signal, "is_zolpidem_suspect_ps")
        & ~bool_series(signal, "is_zolpidem_suspect"),
        "ps_other_zdrug_implies_ps_ss_other_zdrug": bool_series(signal, "is_other_zdrug_suspect_ps")
        & ~bool_series(signal, "is_other_zdrug_suspect"),
        "ps_suspect_implies_ps_ss_suspect": bool_series(signal, "suspect_role_any_ps")
        & ~bool_series(signal, "suspect_role_any"),
        "zolpidem_suspect_implies_zolpidem_any": bool_series(signal, "is_zolpidem_suspect")
        & ~bool_series(signal, "is_zolpidem_any"),
    }
    for check, mask in implication_checks.items():
        error_count = int(mask.sum())
        rows.append(_audit_row(check, "fail" if error_count else "pass", error_count))

    for suffix, group_col in (("", "target_drug_group"), ("_ps", "target_drug_group_ps")):
        expected = expected_target_group(signal, suffix=suffix)
        actual = signal[group_col].where(signal[group_col].notna(), "").astype(str).str.strip()
        mismatch_count = int(actual.ne(expected).sum())
        rows.append(
            _audit_row(
                f"{group_col}_matches_boolean_flags",
                "fail" if mismatch_count else "pass",
                mismatch_count,
            )
        )

    if "fall_pt_count" in signal.columns:
        fall_count = pd.to_numeric(signal["fall_pt_count"], errors="coerce").fillna(0)
        fall_flag = bool_series(signal, "is_fall")
        count_without_flag = int((fall_count.gt(0) & ~fall_flag).sum())
        flag_without_count = int((fall_flag & fall_count.eq(0)).sum())
        rows.append(
            _audit_row(
                "fall_count_positive_but_flag_false",
                "fail" if count_without_flag else "pass",
                count_without_flag,
            )
        )
        rows.append(
            _audit_row(
                "fall_flag_true_but_count_zero",
                "fail" if flag_without_count else "pass",
                flag_without_count,
            )
        )

    if "polypharmacy" in feature.columns:
        mismatch = int(
            (
                bool_series(feature, "polypharmacy_5")
                != feature["polypharmacy"].fillna(False).astype(bool)
            ).sum()
        )
        rows.append(
            _audit_row(
                "polypharmacy_alias_matches_polypharmacy_5",
                "fail" if mismatch else "pass",
                mismatch,
            )
        )

    return pd.DataFrame(rows)


def write_audit_report(report: pd.DataFrame, output_file: str | Path) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path

