import os
import logging
from configparser import ConfigParser
import getpass


CONFIG_TEMPLATE = """
# Define default configuration for environment, including any common/standard
# treatments, enrollments, measures, and attributes.

# ----------------------- BEGIN TEMPLATE -----------------------

[core]
spearmint_home={SPEARMINT._HOME}

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

[stan]:
model_cache={SPEARMINT._HOME}/compiled_stan_models/
default_bayesian_inference_method=sample
"""

TEMPLATE_BEGIN_PATTERN = (
    "# ----------------------- BEGIN TEMPLATE -----------------------"
)


def expand_env_var(env_var):
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


def render_config_template(template):
    """
    Generates a configuration from the provided template + variables defined in
    current scope
    :param template: a config content templated with {{variables}}
    """
    all_vars = {k: v for d in [globals(), locals()] for k, v in d.items()}
    return template.format(**all_vars)


# ConfigParser is "old-style" class which is a classobj, not a type.
# We thus use multiple inheritance with object to fix
class AbracadabraConfigParser(ConfigParser, object):
    """
    Custom config parser, with some validations
    """

    def __init__(self, *args, **kwargs):
        super(AbracadabraConfigParser, self).__init__(*args, **kwargs)
        self.is_validated = False

    def _validate(self):
        self.is_validated = True

    def read(self, filenames):
        ConfigParser.read(self, filenames)
        self._validate()


def mk_dir(dirname):
    if not os.path.isdir(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            raise Exception("Could not create directory {}:\n{}".format(dirname, e))


# Home directory and configuration locations.
# We default to ~/spearmint and ~/spearmint/abracadspearmint.cfg if not provided
if "SPEARMINT._HOME" not in os.environ:
    SPEARMINT._HOME = expand_env_var("~/spearmint")
    os.environ["SPEARMINT._HOME"] = SPEARMINT._HOME
else:
    SPEARMINT._HOME = expand_env_var(os.environ["SPEARMINT._HOME"])

mk_dir(SPEARMINT._HOME)

if "SPEARMINT._CONFIG" not in os.environ:
    if os.path.isfile(expand_env_var("~/abracadspearmint.cfg")):
        SPEARMINT._CONFIG = expand_env_var("~/abracadspearmint.cfg")
    else:
        SPEARMINT._CONFIG = SPEARMINT._HOME + "/abracadspearmint.cfg"
else:
    SPEARMINT._CONFIG = expand_env_var(os.environ["SPEARMINT._CONFIG"])


if "SPEARMINT._USER" not in os.environ:
    SPEARMINT._USER = getpass.getuser()
    os.environ["SPEARMINT._USER"] = SPEARMINT._USER
else:
    SPEARMINT._USER = os.environ["SPEARMINT._USER"]

# Write the config file, if needed
if not os.path.isfile(SPEARMINT._CONFIG):
    logging.info(f"Creating new Abracadabra config file in: {SPEARMINT._CONFIG}")
    with open(SPEARMINT._CONFIG, "w") as f:
        cfg = render_config_template(CONFIG_TEMPLATE)
        f.write(cfg.split(TEMPLATE_BEGIN_PATTERN)[-1].strip())


def coerce_value(val):
    """
    Coerce config variables to proper types
    """

    def isnumeric(val):
        try:
            float(val)
            return True
        except ValueError:
            return False

    if isnumeric(val):
        try:
            return int(val)
        except ValueError:
            return float(val)

    lower_val = str(val.lower())
    if lower_val in ("true", "false"):
        if "f" in lower_val:
            return False
        else:
            return True

    if "," in val:
        return [coerce_value(v.strip()) for v in val.split(",")]
    return val


CONFIG = AbracadabraConfigParser()
CONFIG.read(SPEARMINT._CONFIG)


# Give the entire module get/set methods
def get(section, key, **kwargs):
    """
    Retrieve typed variables from config.

    Example
    -------
    from abra import config
    # print the currently-configure SPEARMINT._HOME directory
    print(config.get('core', 'spearmint_home'))
    """
    return coerce_value(CONFIG.get(section, key, **kwargs))


def set(section, option, value, update=False):
    CONFIG.set(section, option, value)


def search_config(df, section, key):
    """
    Search a dataframe `df` for parameters defined in the global configuration.

    Parameters
    ----------
    df: dataframe
        raw data to search
    param_name: str
        type of parameter to search ('entities', 'metrics', or 'attributes')
    """
    available = get(section, key)
    available = [available] if not isinstance(available, list) else available
    columns = df.columns
    return [c for c in columns if c in available]


DEFAULT_ALPHA = get("constants", "default_alpha")
MIN_OBS_FOR_Z = get("constants", "min_obs_for_z")

STAN_MODEL_CACHE = get("stan", "model_cache")
DEFAULT_BAYESIAN_INFERENCE_METHOD = get("stan", "default_bayesian_inference_method")

logger = logging.getLogger(__name__)
logger.setLevel(get("core", "logging_level"))
