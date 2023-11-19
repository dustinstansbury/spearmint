import logging
import os
import tempfile

import pytest

TEST_ENV_VARS = {"SPEARMINT_USER": "test", "SPEARMINT_HOME": tempfile.mkdtemp()}


@pytest.fixture(scope="session", autouse=True)
def test_config():
    original_env = dict(os.environ)

    # Set update test env
    os.environ.update(TEST_ENV_VARS)

    import importlib

    from spearmint import config

    # Reload the config with updated env vars
    importlib.reload(config)

    yield config

    # Reset the env to original state
    os.environ.clear()
    os.environ.update(original_env)


def test_env_vars(test_config):
    assert TEST_ENV_VARS["SPEARMINT_HOME"] == test_config.SPEARMINT_HOME
    assert TEST_ENV_VARS["SPEARMINT_USER"] == test_config.SPEARMINT_USER


def test_config_file(test_config):
    assert os.path.isfile(test_config.SPEARMINT_CONFIG)


def test_logger_level(test_config):
    assert test_config.logger.level == getattr(
        logging, (test_config.get("core", "logging_level"))
    )


def test_expand_env_var(test_config):
    assert "blammo" == test_config.expand_env_var("blammo")


def test_render_config_template(test_config):
    template = "{SPEARMINT_USER}"
    assert test_config.render_config_template(template) == test_config.SPEARMINT_USER


def test_get_set(test_config):
    test_config.set("core", "test_value", "test")
    assert test_config.get("core", "test_value") == "test"
    assert test_config.CONFIG.get("core", "test_value") == "test"
