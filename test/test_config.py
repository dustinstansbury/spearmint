import os
import logging
from spearmint import config


def test_logger_level():
    assert config.logger.level == getattr(
        logging, (config.get("core", "logging_level"))
    )


def test_expand_env_var():
    assert "blammo" == config.expand_env_var("blammo")


def test_render_config_template():
    template = "{SPEARMINT_USER}"
    assert config.render_config_template(template) == config.SPEARMINT_USER


def test_home():
    assert os.environ["SPEARMINT_HOME"] == config.SPEARMINT_HOME


def test_user():
    assert os.environ["SPEARMINT_USER"] == config.SPEARMINT_USER


def test_config_file():
    assert os.path.isfile(config.SPEARMINT_CONFIG)


def test_coerce_value():
    assert config.coerce_value("true") is True
    assert config.coerce_value("false") is False
    assert isinstance(config.coerce_value("1.0"), float)
    assert isinstance(config.coerce_value("1"), int)
    assert isinstance(config.coerce_value("a,b"), list)


def test_get_set():
    config.set("core", "test_value", "test")
    assert config.get("core", "test_value") == "test"
    assert config.CONFIG.get("core", "test_value") == "test"
