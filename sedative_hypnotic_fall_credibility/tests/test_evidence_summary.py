from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "12_evidence_summary.py"
SPEC = importlib.util.spec_from_file_location("evidence_summary", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def estimates(values: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"comparison_id": f"cmp_{index}", "ROR": value, "ROR_95CI_low": low, "ROR_95CI_high": high}
        for index, (value, low, high) in enumerate(values)
    ])


def target(comparison_count: int = 2):
    return module.EvidenceTarget("test", "Test", "test", "group", None,
                                 tuple(f"cmp_{index}" for index in range(comparison_count)))


class EvidenceSummaryTests(unittest.TestCase):
    def test_missing_comparison_is_not_available_not_zero(self):
        result = module.consistency_component(pd.DataFrame(columns=["comparison_id"]), target(), "ROR", "ROR_95CI_low", "ROR_95CI_high")
        self.assertIsNone(result["score"])
        self.assertEqual(result["available_points"], 0)
        self.assertEqual(result["status"], "not_available")

    def test_one_positive_and_one_neutral_comparison_cannot_receive_full_points(self):
        result = module.consistency_component(estimates([(1.5, 1.2, 1.8), (0.95, 0.8, 1.1)]), target(), "ROR", "ROR_95CI_low", "ROR_95CI_high")
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["status"], "neutral")

    def test_precise_reverse_result_is_flagged_as_contradictory(self):
        result = module.consistency_component(estimates([(3.0, 2.0, 4.0), (0.85, 0.76, 0.95)]), target(), "ROR", "ROR_95CI_low", "ROR_95CI_high")
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["status"], "contradictory")

    def test_missing_prespecified_comparison_makes_domain_unavailable(self):
        result = module.consistency_component(estimates([(1.5, 1.2, 1.8)]), target(2), "ROR", "ROR_95CI_low", "ROR_95CI_high")
        self.assertIsNone(result["score"])
        self.assertEqual(result["status"], "not_available")
        self.assertIn("cmp_1", result["detail"])

    def test_low_orexin_signal_is_not_interpreted_as_protective(self):
        components = {name: module.component(None, 1, "not_available", "") for name in module.CORE_DOMAINS}
        components["traditional_signal_strength"] = module.component(0, 2, "contradictory", "ROR upper CI below 1")
        self.assertEqual(module.support_class(0, 2, components), "insufficient evidence")

    def test_available_score_below_six_is_insufficient_even_with_high_fraction(self):
        components = {name: module.component(None, 1, "not_available", "") for name in module.CORE_DOMAINS}
        self.assertEqual(module.support_class(5, 5, components), "insufficient evidence")

    def test_spec_core_domains_sum_to_nine(self):
        spec = pd.read_csv(Path(__file__).resolve().parents[1] / "configs" / "credibility_score_spec.csv")
        module.validate_spec(spec)
        self.assertEqual(spec.loc[spec["domain_type"] == "core", "max_points"].sum(), 9)

    def test_spec_threshold_drift_is_rejected(self):
        spec = pd.read_csv(Path(__file__).resolve().parents[1] / "configs" / "credibility_score_spec.csv")
        spec.loc[spec["domain"] == "traditional_signal_strength", "threshold"] = "strong_n=99;weak_n=5"
        with self.assertRaisesRegex(ValueError, "thresholds"):
            module.validate_spec(spec)

    def test_leave_one_domain_out_recomputes_class_and_change_flag(self):
        summary = pd.DataFrame([{"target_key": "test", "target_label": "Test", "obtained_score": 8,
                                 "available_score": 8, "support_class": "high support"}])
        details = pd.DataFrame([
            {"target_key": "test", "target_label": "Test", "domain_type": "core", "evidence_domain": name,
             "score": score, "available_points": maximum, "status": "supportive"}
            for name, score, maximum in (
                ("traditional_signal_strength", 2, 2), ("active_comparator_consistency", 2, 2),
                ("adjusted_model_stability", 2, 2), ("exposure_definition_sensitivity", 2, 2),
                ("reporting_structure_stability", None, 1))
        ])
        details.loc[details["score"].isna(), ["available_points", "status"]] = [0, "not_available"]
        result = module.leave_one_domain_out(summary, details)
        omitted = result[result["omitted_domain"] == "traditional_signal_strength"].iloc[0]
        self.assertEqual(omitted["loo_support_class"], "moderate support")
        self.assertTrue(omitted["class_changed"])

    def test_threshold_sensitivity_emits_all_scenarios(self):
        root = Path(__file__).resolve().parents[1]
        table_dir = root / "outputs" / "tables"
        inputs = tuple(pd.read_csv(path) if path is not None else pd.DataFrame() for path in (
            table_dir / "table_1_signal_landscape.csv", table_dir / "table_2_active_comparator_results.csv",
            table_dir / "table_3_adjusted_ror.csv", table_dir / "table_s3_ps_only_sensitivity.csv",
            table_dir / "table_s4_excluding_mixed_exposure_sensitivity.csv",
            table_dir / "table_s5_reporting_source_stratified_sensitivity.csv",
            table_dir / "table_4_phenotype_fingerprint_by_drug_group.csv",
            root / "configs" / "credibility_external_evidence.csv"))
        baseline, _ = module.build_summary(*inputs)
        result = module.threshold_sensitivity(inputs, baseline)
        self.assertEqual(set(result["scenario"]), {"baseline", "lenient_case_count", "conservative_case_count",
                                                   "point_estimate_consistency"})
        self.assertFalse(result["class_changed_from_baseline"].any())


if __name__ == "__main__":
    unittest.main()
