from spearmint.experiment import Experiment
from spearmint.hypothesis_test import HypothesisTest, CustomMetric
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spearmint")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["Experiment", "HypothesisTest", "CustomMetric"]
