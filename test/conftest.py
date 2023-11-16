import pytest
import os
import tempfile
import shutil

TEST_ENV_VARS = {"SPEARMINT_USER": "test", "SPEARMINT_HOME": tempfile.mkdtemp()}


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


@pytest.fixture(scope="session", autouse=True)
def test_env_vars():
    yield TEST_ENV_VARS


@pytest.fixture(scope="session", autouse=True)
def test_config():
    current_env = dict(os.environ)

    # Set up test environment
    os.environ.update(TEST_ENV_VARS)
    from spearmint import config

    yield config

    # Reset the env
    os.environ.clear()
    os.environ.update(current_env)


@pytest.fixture()
def testdir():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture()
def test_observations():
    from spearmint.utils import generate_fake_observations

    return generate_fake_observations(distribution="bernoulli")


@pytest.fixture()
def test_samples():
    from spearmint.utils import generate_fake_observations
    from spearmint.stats import Samples

    observations = generate_fake_observations(distribution="bernoulli")["metric"].values
    return Samples(observations=observations, name="test")


@pytest.fixture()
def proportions_data_small():
    from spearmint.utils import generate_fake_observations

    return generate_fake_observations(
        distribution="bernoulli", n_treatments=6, n_observations=6 * 50
    )
