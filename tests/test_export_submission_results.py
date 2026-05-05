from scripts.analysis.export_submission_results import _variant_label


def test_variant_label_prefers_algorithm_specific_label() -> None:
    assert _variant_label("central_baseline", "td3") == "Centralized TD3 baseline"
    assert _variant_label("central_baseline", "sac") == "Centralized SAC baseline"

