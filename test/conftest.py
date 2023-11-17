import pytest


def pytest_addoption(parser):
    """
    https://docs.pytest.org/en/7.1.x/example/markers.html
    """
    parser.addoption(
        "--include_pymc_tests",
        action="store_true",
        default=False,
        help="Run Bayesian tests that require PyMC, which are skipped by default. ",
    )


def pytest_collection_modifyitems(config, items):
    """
    https://docs.pytest.org/en/7.1.x/example/markers.html
    """
    if not config.getoption("--include_pymc_tests"):
        skip_slow_tests = pytest.mark.skip(
            reason="Skipping Bayesian tests that require PyMC - use --include_pymc_tests to run"
        )
        for item in items:
            if "pymc_test" in item.keywords:
                item.add_marker(skip_slow_tests)
