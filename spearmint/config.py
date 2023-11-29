import getpass
import logging
import os
from configparser import ConfigParser
from dataclasses import dataclass

import holoviews as hv

from spearmint.typing import Any, Iterable, Union
from spearmint.utils import coerce_value, mkdir

CONFIG_TEMPLATE = """
# Define default configuration for environment, including any common/standard
# treatments, enrollments, measures, and attributes.

# ----------------------- BEGIN TEMPLATE -----------------------

[core]
spearmint_home={SPEARMINT_HOME}

# Logging level
logging_level=INFO

# Defaults
[experiment]
default_measure_names=metric
enrollment=enrollment

[hypothesis_test]:
default_control_name=control
default_variation_name=variation
default_treatment_name=treatment
default_metric_name=metric
default_hypothesis=larger
default_inference_method=frequentist
default_alpha=.05
min_obs_for_z_test=30
default_attribute_names=attr_0,attr_1

[vis]
vis_backend=bokeh
figure_width_pixels=400
figure_height_pixels=400

[bayesian_inference]:
default_parameter_estimation_method=analytic
default_mcmc_sampler=nuts
n_posterior_samples=1000
"""

TEMPLATE_BEGIN_PATTERN = (
    "# ----------------------- BEGIN TEMPLATE -----------------------"
)


def expand_env_var(env_var: str) -> Any:
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

    def read(self, filenames: Union[str, Iterable[str]]) -> None:  # type: ignore
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


# Give this module get/set methods
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


# Configure Logging
logger = logging.getLogger(__name__)
logger.setLevel(get("core", "logging_level"))

# Configure defaults
DEFAULT_METRIC_NAME = get("hypothesis_test", "default_metric_name")
DEFAULT_TREATMENT_NAME = get("hypothesis_test", "default_treatment_name")
DEFAULT_CONTROL_NAME = get("hypothesis_test", "default_control_name")
DEFAULT_VARIATION_NAME = get("hypothesis_test", "default_variation_name")
DEFAULT_INFERENCE_METHOD = get("hypothesis_test", "default_inference_method")
DEFAULT_HYPOTHESIS = get("hypothesis_test", "default_hypothesis")
DEFAULT_ALPHA = get("hypothesis_test", "default_alpha")


MIN_OBS_FOR_Z_TEST = get("hypothesis_test", "min_obs_for_z_test")
# DEFAULT_BAYESIAN_INFERENCE_METHOD = get("pymc", "bayesian_inference_method")

# Configure Bayesian Inference
N_POSTERIOR_SAMPLES = get("bayesian_inference", "n_posterior_samples")
DEFAULT_PARAMETER_ESTIMATION_METHOD = get(
    "bayesian_inference", "default_parameter_estimation_method"
)


# Configure Visualization
# TODO: add these to default config
_DEFAULT_VIS_BACKEND = get("vis", "vis_backend")
FIGURE_WIDTH_PIXELS = get("vis", "figure_width_pixels")
FIGURE_HEIGHT_PIXELS = get("vis", "figure_height_pixels")


def _get_vis_backend():
    if _DEFAULT_VIS_BACKEND == "bokeh":
        try:
            import bokeh

            return bokeh.__name__
        except ImportError:
            logger.warning(
                "Bokeh not available, falling back to matplotlib visualizaiton backend"
            )
            return "matplotlib"

    return _DEFAULT_VIS_BACKEND


# Set the visualization backend
VIS_BACKEND = _get_vis_backend()
hv.extension(VIS_BACKEND)


def _get_figure_params():
    """Return backend-specific figure params"""

    def _get_figsize_kwargs():
        if VIS_BACKEND == "bokeh":
            return dict(width=FIGURE_WIDTH_PIXELS, height=FIGURE_HEIGHT_PIXELS)
        elif VIS_BACKEND == "matplotlib":
            import matplotlib

            pix_per_inch = matplotlib.rcParams["figure.dpi"]
            aspect_ratio = FIGURE_WIDTH_PIXELS / FIGURE_HEIGHT_PIXELS
            fig_inches = max([FIGURE_WIDTH_PIXELS, FIGURE_HEIGHT_PIXELS]) / pix_per_inch
            return {"aspect": aspect_ratio, "fig_inches": fig_inches}

    def _get_hover_kwargs():
        if VIS_BACKEND == "matplotlib":
            # No hover in matplotlib
            return {}
        return {"tools": ["hover"]}

    figure_kwargs = {}
    figure_kwargs.update(_get_figsize_kwargs())
    figure_kwargs.update(_get_hover_kwargs())

    return figure_kwargs


def _get_points_plot_params():
    if VIS_BACKEND == "matplotlib":
        return {"s": 50}
    return {"size": 5}


FIGURE_PARAMS = _get_figure_params()
POINTS_PLOT_PARAMS = _get_points_plot_params()


@dataclass
class COLORS:
    blue = "#4257B2"
    light_blue = "#A1C4FD"
    cyan = "#3CCFCF"
    green = "#388E34"
    light_green = "#28CC7D"
    dark_green = "#006060"
    yellow = "#FFCD1F"
    salmon = "#FF725B"
    red = "#FB3640"
    dark_red = "#AE2024"
    purple = "#8842C0"
    gray = "#687174"
    dark_gray = "#455357"
    light_gray = "#C0CACE"
    brown = "#665000"
    black = "#000000"
