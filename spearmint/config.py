import os
import logging
import getpass

from configparser import ConfigParser

from spearmint.utils import coerce_value, mkdir
from spearmint.typing import Any, Union, Iterable


CONFIG_TEMPLATE = """
# Define default configuration for environment, including any common/standard
# treatments, enrollments, measures, and attributes.

# ----------------------- BEGIN TEMPLATE -----------------------

[core]
spearmint_home={SPEARMINT_HOME}

# Logging level
logging_level=INFO

# Default experiment values, used to reduce parameter
# footprint for common Experiment instantiations
[experiment]
control=control
treatment=treatment
enrollment=enrollment
measures=metric
attributes=attr_0,attr_1

[constants]:
default_alpha=.05
min_obs_for_z=30

[pymc]:
default_bayesian_inference_method=sample
"""

TEMPLATE_BEGIN_PATTERN = (
    "# ----------------------- BEGIN TEMPLATE -----------------------"
)


def expand_env_var(env_var: str) -> None:
    """
    Expands (potentially nested) env vars by repeatedly applying
    `expandvars` and `expanduser` until interpolation stops having
    any effect.
    """
    if not env_var:
        return env_var
    while True:
        interpolated = os.path.expanduser(os.path.expandvars(str(env_var)))
        if interpolated == env_var:
            return interpolated
        else:
            env_var = interpolated


def render_config_template(template: str) -> str:
    """
    Generates a configuration from the provided template + variables defined in
    current scope

    Parameters
    ----------
    template : str
        A configuration template with interpolations `{{variables}}`

    Returns
    -------
    rendered_template : str
        The template with variables interpolated from environment, global, and
        local scope.
    """
    all_vars = {k: v for d in [globals(), locals()] for k, v in d.items()}
    return template.format(**all_vars)


# ConfigParser is "old-style" class which is a classobj, not a type.
# We thus use multiple inheritance with object to fix
class SpearmintConfigParser(ConfigParser, object):
    """
    Custom config parser, with some validations
    """

    def __init__(self, *args, **kwargs) -> None:
        super(SpearmintConfigParser, self).__init__(*args, **kwargs)
        self.is_validated = False

    def _validate(self) -> None:
        self.is_validated = True

    def read(self, filenames: Union[str, Iterable[str]]) -> None:
        ConfigParser.read(self, filenames)
        self._validate()


# Home directory and configuration locations.
# We default to ~/.spearmint and ~/spearmint/spearmint.cfg if not provided
if "SPEARMINT_HOME" not in os.environ:
    SPEARMINT_HOME = expand_env_var("~/.spearmint")
    os.environ["SPEARMINT_HOME"] = SPEARMINT_HOME
else:
    SPEARMINT_HOME = expand_env_var(os.environ["SPEARMINT_HOME"])

mkdir(SPEARMINT_HOME)

if "SPEARMINT_CONFIG" not in os.environ:
    if os.path.isfile(expand_env_var("~/spearmint.cfg")):
        SPEARMINT_CONFIG = expand_env_var("~/spearmint.cfg")
    else:
        SPEARMINT_CONFIG = SPEARMINT_HOME + "/spearmint.cfg"
else:
    SPEARMINT_CONFIG = expand_env_var(os.environ["SPEARMINT_CONFIG"])


if "SPEARMINT_USER" not in os.environ:
    SPEARMINT_USER = getpass.getuser()
    os.environ["SPEARMINT_USER"] = SPEARMINT_USER
else:
    SPEARMINT_USER = os.environ["SPEARMINT_USER"]

# If needed, write the config file
if not os.path.isfile(SPEARMINT_CONFIG):
    logging.info(f"Creating new spearment config file in: {SPEARMINT_CONFIG}")
    with open(SPEARMINT_CONFIG, "w") as f:
        cfg = render_config_template(CONFIG_TEMPLATE)
        f.write(cfg.split(TEMPLATE_BEGIN_PATTERN)[-1].strip())


CONFIG = SpearmintConfigParser()
CONFIG.read(SPEARMINT_CONFIG)


# Give the entire module get/set methods
def get(section: str, key: str, **kwargs) -> Any:
    """
    Retrieve properly-typed variables from config.

    Parameters
    ----------
    section : str
        The top-level section of the config to get
    key : str
        The specific config value to get

    Returns
    -------
    value : Any
        The propertly-typed configuration value

    Example
    -------
    from spearmint import config
    # print the currently-configure SPEARMINT_HOME directory
    print(config.get('core', 'spearmint_home'))
    """
    return coerce_value(CONFIG.get(section, key, **kwargs))


def set(section: str, option: str, value: Any) -> None:
    CONFIG.set(section, option, value)


DEFAULT_ALPHA = get("constants", "default_alpha")
MIN_OBS_FOR_Z = get("constants", "min_obs_for_z")
DEFAULT_BAYESIAN_INFERENCE_METHOD = get("pymc", "default_bayesian_inference_method")

logger = logging.getLogger(__name__)
logger.setLevel(get("core", "logging_level"))
