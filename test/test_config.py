import logging
import os


def test_logger_level(test_config):
    assert test_config.logger.level == getattr(
        logging, (test_config.get("core", "logging_level"))
    )


def test_expand_env_var(test_config):
    assert "blammo" == test_config.expand_env_var("blammo")


def test_render_config_template(test_config):
    template = "{SPEARMINT_USER}"
    assert test_config.render_config_template(template) == test_config.SPEARMINT_USER


def test_home(test_config, test_env_vars):
    assert test_env_vars["SPEARMINT_HOME"] == test_config.SPEARMINT_HOME


def test_user(test_config, test_env_vars):
    assert test_env_vars["SPEARMINT_USER"] == test_config.SPEARMINT_USER


def test_config_file(test_config):
    assert os.path.isfile(test_config.SPEARMINT_CONFIG)


def test_get_set(test_config):
    test_config.set("core", "test_value", "test")
    assert test_config.get("core", "test_value") == "test"
    assert test_config.CONFIG.get("core", "test_value") == "test"
