from spearmint.utils import set_backend
from spearmint.experiment import Experiment
from spearmint.dataset import Dataset
from spearmint.hypothesis_test import HypothesisTest, HypothesisTestSuite, CustomMetric
from spearmint.stats import Samples, MultipleComparisonCorrection
from spearmint.inference.bayesian.delta import BayesianDelta
from spearmint.inference.frequentist.means import MeansDelta
from spearmint.inference.frequentist.proportions import ProportionsDelta
from spearmint.inference.frequentist.rates import RatesRatio
from spearmint.inference.frequentist.bootstrap import BootstrapDelta

VISUALIZATION_BACKEND = set_backend()  # set backend for any visualization support

__all__ = [
    "Experiment",
    "Dataset",
    "Samples",
    "HypothesisTest",
    "HypothesisTestSuite",
    "MultipleComparisonCorrection",
    "BayesianDelta",
    "MeansDelta",
    "ProportionsDelta",
    "RatesRatio",
    "BootstrapDelta",
    "CustomMetric",
    "VISUALIZATION_BACKEND",
]
