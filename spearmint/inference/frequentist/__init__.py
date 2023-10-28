import numpy as np
from scipy.stats import norm

from spearmint.inference import InferenceProcedure
from spearmint.stats import CompareMeans
from spearmint.typing import Tuple


class FrequentistTestStatisticalDistribution:
    """A distrubiton that implements scipy.stats API"""

    def ppf(self, *args, **kwargs):
        raise NotImplemented


class FrequentistInferenceProcedure(InferenceProcedure):
    def __init__(
        self,
        test_statistic_distribution: FrequentistTestStatisticalDistribution = norm,
        *args,
        **kwargs,
    ):
        """
        Base class for frequentist hypothesis test inference procedures, including
        tests for Means and Proportion deltas. Uses statsmodels.CompareMeans
        objects under the hood

        Parameters
        ----------
        test_statistic_distribution : FrequentistTestStatisticalDistribution, optional
            The probability distribution that captures the shape of the test
            statistic, by default scipy.stats.norm.
        """
        super().__init__(*args, **kwargs)

        self.test_statistic_distribution = test_statistic_distribution
        self.comparison: CompareMeans = None

    @property
    def control_name(self):
        return self.comparison.d2.name

    @property
    def variation_name(self):
        return self.comparison.d1.name

    @property
    def hypothesis_sm(self):
        """
        Statsmodels-compatible hypothesis string
        """
        return self.hypothesis if self.hypothesis != "unequal" else "two-sided"

    def accept_hypothesis(self, statistic_value: float) -> bool:
        """
        Accept the null hypothesis based on the calculated statistic and statitic
        distribution.
        """
        if self.hypothesis == "larger":
            return statistic_value > self.test_statistic_distribution.ppf(
                1 - self.alpha
            )
        elif self.hypothesis == "smaller":
            return statistic_value < self.test_statistic_distribution.ppf(self.alpha)
        elif self.hypothesis == "unequal":
            return abs(statistic_value) > self.test_statistic_distribution.ppf(
                1 - self.alpha / 2.0
            )
        else:
            raise ValueError(f"Unknown hypothesis: {self.hypothesis}")

    @property
    def delta_ci_percentiles(self) -> Tuple[float, float]:
        """Percentiles for the confidence interval on delta, based on `alpha`"""
        if self.hypothesis == "larger":
            return (self.alpha, np.inf)
        elif self.hypothesis == "smaller":
            return (-np.inf, 1 - self.alpha)
        elif self.hypothesis == "unequal":
            return ((self.alpha / 2.0), 1 - (self.alpha / 2.0))
        else:
            raise ValueError("Unknown hypothesis: {!r}".format(self.hypothesis))

    @property
    def test_statistic_name(self) -> str:
        return self.comparison.test_statistic_name
