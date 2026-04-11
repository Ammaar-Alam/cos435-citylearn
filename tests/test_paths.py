from cos435_citylearn.paths import CONFIGS_DIR, DATA_DIR, REPO_ROOT, RESULTS_DIR, repo_path


def test_repo_paths_point_to_expected_roots() -> None:
    assert REPO_ROOT.name == "cos435-citylearn"
    assert CONFIGS_DIR == repo_path("configs")
    assert DATA_DIR == repo_path("data")
    assert RESULTS_DIR == repo_path("results")
