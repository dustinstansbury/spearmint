import pytest
import os
import tempfile
import shutil

TEST_ENV_VARS = {"SPEARMINT_USER": "test", "SPEARMINT_HOME": tempfile.mkdtemp()}


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
