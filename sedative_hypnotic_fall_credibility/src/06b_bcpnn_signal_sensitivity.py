from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SIGNAL_TABLE = PROJECT_DIR / "outputs" / "tables" / "table_1_signal_landscape.csv"
DEFAULT_TABLE_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s16_bcpnn_signal_sensitivity.csv"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "06b_bcpnn_signal_sensitivity_qc.csv"

COUNT_COLUMNS = ["a", "b", "c", "d"]


def read_signal_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Signal landscape table not found: {path}")
    df = pd.read_csv(path)
    missing = [column for column in COUNT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Signal landscape table is missing count columns: {missing}")
    return df


def posterior_ic_samples(
    a: int,
    b: int,
    c: int,
    d: int,
    prior_alpha: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    posterior_alpha = np.array([a, b, c, d], dtype=float) + prior_alpha
    draws = rng.gamma(shape=posterior_alpha, scale=1.0, size=(n_samples, 4))
    probs = draws / draws.sum(axis=1, keepdims=True)
    p11 = probs[:, 0]
    p10 = probs[:, 1]
    p01 = probs[:, 2]
    p_drug = p11 + p10
    p_event = p11 + p01
    return np.log2(p11 / (p_drug * p_event))


def summarize_target(
    row: pd.Series,
    prior_alpha: float,
    n_samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    a, b, c, d = [int(row[column]) for column in COUNT_COLUMNS]
    total = a + b + c + d
    exposed_total = a + b
    event_total = a + c
    expected = exposed_total * event_total / total if total else np.nan

    ic_samples = posterior_ic_samples(a, b, c, d, prior_alpha, n_samples, rng)
    return {
        "analysis_level": row["analysis_level"],
        "target_key": row["target_key"],
        "target_label": row["target_label"],
        "drug_group": row["drug_group"],
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "observed_count": a,
        "expected_count": expected,
        "BCPNN_IC_mean": float(np.mean(ic_samples)),
        "BCPNN_IC_median": float(np.median(ic_samples)),
        "BCPNN_IC025": float(np.quantile(ic_samples, 0.025)),
        "BCPNN_IC975": float(np.quantile(ic_samples, 0.975)),
        "BCPNN_Pr_IC_gt_0": float(np.mean(ic_samples > 0)),
        "BCPNN_signal_positive": bool(np.quantile(ic_samples, 0.025) > 0),
        "prior": f"Dirichlet({prior_alpha}, {prior_alpha}, {prior_alpha}, {prior_alpha})",
        "posterior_samples": n_samples,
        "random_seed_note": "Monte Carlo posterior sampling; rerun with the same seed for reproducible summaries.",
        "method_note": (
            "BCPNN-style IC sensitivity using a Dirichlet-multinomial posterior "
            "for the 2x2 drug-by-event table. This is separate from the OE/OE05 main signal table."
        ),
    }


def build_bcpnn_table(signal: pd.DataFrame, prior_alpha: float, n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    available = signal[signal["column_present"].astype(bool)].copy()
    for _, row in available.iterrows():
        rows.append(summarize_target(row, prior_alpha, n_samples, rng))
    return pd.DataFrame(rows)


def build_qc(signal: pd.DataFrame, bcpnn: pd.DataFrame, prior_alpha: float, n_samples: int, seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "qc_domain": "bcpnn_signal_sensitivity",
                "metric": "input_rows",
                "value": len(signal),
                "note": "Rows read from table_1_signal_landscape.csv.",
            },
            {
                "qc_domain": "bcpnn_signal_sensitivity",
                "metric": "analyzed_rows",
                "value": len(bcpnn),
                "note": "Rows with available exposure columns.",
            },
            {
                "qc_domain": "bcpnn_signal_sensitivity",
                "metric": "prior_alpha",
                "value": prior_alpha,
                "note": "Symmetric Dirichlet prior alpha per 2x2 cell.",
            },
            {
                "qc_domain": "bcpnn_signal_sensitivity",
                "metric": "posterior_samples",
                "value": n_samples,
                "note": "Monte Carlo samples per target.",
            },
            {
                "qc_domain": "bcpnn_signal_sensitivity",
                "metric": "random_seed",
                "value": seed,
                "note": "Used for reproducible posterior summaries.",
            },
        ]
    )


def validate_results(bcpnn: pd.DataFrame) -> None:
    if bcpnn.empty:
        raise ValueError("BCPNN sensitivity table is empty.")
    numeric_columns = [
        "expected_count",
        "BCPNN_IC_mean",
        "BCPNN_IC_median",
        "BCPNN_IC025",
        "BCPNN_IC975",
        "BCPNN_Pr_IC_gt_0",
    ]
    if not np.isfinite(bcpnn[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("BCPNN sensitivity table contains non-finite numeric results.")
    if not ((bcpnn["BCPNN_Pr_IC_gt_0"] >= 0) & (bcpnn["BCPNN_Pr_IC_gt_0"] <= 1)).all():
        raise ValueError("BCPNN posterior probabilities fall outside [0, 1].")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-table", type=Path, default=DEFAULT_SIGNAL_TABLE)
    parser.add_argument("--table-out", type=Path, default=DEFAULT_TABLE_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    parser.add_argument("--prior-alpha", type=float, default=0.5)
    parser.add_argument("--posterior-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260622)
    args = parser.parse_args()

    if args.prior_alpha <= 0:
        raise ValueError("--prior-alpha must be positive.")
    if args.posterior_samples < 10000:
        raise ValueError("--posterior-samples should be at least 10,000 for stable quantiles.")

    signal = read_signal_table(args.signal_table)
    bcpnn = build_bcpnn_table(signal, args.prior_alpha, args.posterior_samples, args.seed)
    validate_results(bcpnn)
    qc = build_qc(signal, bcpnn, args.prior_alpha, args.posterior_samples, args.seed)

    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)
    bcpnn.to_csv(args.table_out, index=False, encoding="utf-8-sig")
    qc.to_csv(args.qc_out, index=False, encoding="utf-8-sig")

    print(f"Wrote {args.table_out}")
    print(f"Wrote {args.qc_out}")
    print(f"Targets analyzed: {len(bcpnn):,}")
    print(f"BCPNN-positive targets: {int(bcpnn['BCPNN_signal_positive'].sum()):,}")


if __name__ == "__main__":
    main()
