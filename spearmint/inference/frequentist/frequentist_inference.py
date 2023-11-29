from collections import OrderedDict

import numpy as np
from scipy.stats import norm

from spearmint.inference.inference_base import InferenceProcedure, InferenceResults
from spearmint.stats import CompareMeans
from spearmint.table import SpearmintTable
from spearmint.typing import FilePath, List, Protocol, Tuple, Union, Optional
from spearmint.utils import format_value, process_warnings


class FrequentistInferenceResults(InferenceResults):
    """
    Class for storing, displaying, visualizing, and exporting frequentist
    hypothesis test results.

    delta_confidence_interval : Tuple[float, float]
        The values and percentiles associated with the lower and upper bounds of
        the confidence interval around delta, based on `alpha`
    delta_confidence_interval_percentiles : Tuple[float, float]
        The pdf percentiles associated with `delta_confidence_interval`
    test_statistic_name : str
        The name of the test statistic used to calculate the p-value for the test
    test_statistic_value : float
        The value of the test statistic used to calculate the p-value
    p_value : float in (0, 1)
        The p-value, based on the test value of `statistic_value`
    power : float in (0, 1)
        The statistical power of the experiment, or the probability of correctly
        rejecting the null hypothesis.
    degrees_freedom : int
        The degrees of freedom for the experiment, if applicable (e.g. t-tests)
    correction_method : str
        The name of the multiple comparison correction method used, if any
    """

    def __init__(
        self,
        delta_confidence_interval: Tuple[float, float],
        delta_confidence_interval_percentiles: Tuple[float, float],
        test_statistic_name: str,
        p_value: float,
        power: float = np.nan,
        degrees_freedom: Optional[int] = None,
        test_statistic_value: float = np.nan,
        correction_method: Optional[str] = None,
        warnings: Optional[Union[str, List[str]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.power = power
        self.delta_confidence_interval = delta_confidence_interval
        self.delta_confidence_interval_percents = delta_confidence_interval_percentiles
        self.test_statistic_name = test_statistic_name
        self.test_statistic_value = test_statistic_value
        self.p_value = p_value
        self.degrees_freedom = degrees_freedom
        self.correction_method = correction_method
        self.warnings = warnings  # type: ignore
        self.estimate_relative_delta_confidence_interval()

    def estimate_relative_delta_confidence_interval(self):
        """
        Estimate the confidence on the relative difference using interpolation.
        """
        self.delta_ci_relative = tuple(
            100
            * (
                (np.array(self.delta_confidence_interval) + np.array(self.control.mean))
                / self.control.mean
                - 1.0
            )
        )

    @property
    def _specific_properties(self):
        """
        Properties specific to the current type of test
        """
        return OrderedDict(
            [
                ("test_type", "frequentist"),
                ("p", self.p_value),
                ("p_interpretation", "p-value"),
                ("delta_confidence_interval", self.delta_confidence_interval),
                (
                    "delta_confidence_interval_percentiles",
                    self.delta_confidence_interval_percents,
                ),
                ("relative_delta_confidence_interval", self.delta_ci_relative),
                ("ci_interpretation", "Confidence Interval"),
                ("p_value", self.p_value),
                ("power", self.power),
                ("statistic_name", self.test_statistic_name),
                ("statistic_value", self.test_statistic_value),
                ("degrees_freedom", self.degrees_freedom),
                ("mc_correction", self.correction_method),
            ]
        )

    def _render_stats_table(self):
        self._stats_table = FrequentistInferenceResultsTable(self)
        return self._stats_table

    def visualize(self, outfile: Optional[FilePath] = None, *args, **kwargs):
        return self.visualization_function(self, outfile=outfile)


class FrequentistInferenceResultsTable(SpearmintTable):
    def __init__(self, results: FrequentistInferenceResults):
        super().__init__(title=f"{results.comparison_type} Results", show_header=False)

        # Add results rows
        self.add_row(
            "Delta",
            format_value(results.delta, precision=4),
        )
        self.add_row(
            "Delta CI",
            format_value(results.delta_confidence_interval, precision=4),
        )
        self.add_row(
            "Delta-relative",
            format_value(results.delta_relative, precision=4) + " %",
        )
        self.add_row(
            "Delta-relative CI",
            format_value(results.delta_ci_relative, precision=4) + " %",
        )
        self.add_row(
            "Delta CI %-tiles",
            format_value(results.delta_confidence_interval_percents, precision=4),
        )
        self.add_row(
            "Effect Size",
            format_value(results.effect_size, precision=4),
        )
        alpha_corrected = (
            " (corrected)" if results.correction_method is not None else ""
        )
        self.add_row(
            "alpha" + alpha_corrected,
            format_value(results.alpha, precision=3),
        )
        self.add_row(
            "Power",
            format_value(results.power, precision=3),
        )
        self.add_row(
            "Variable Type",
            results.variable_type,
        )
        self.add_row(
            "Inference Method",
            results.inference_method,
        )
        self.add_row(
            f"Test statistic ({results.test_statistic_name})",
            format_value(results.test_statistic_value, precision=2),
        )
        self.add_row(
            "p-value",
            format_value(results.p_value, precision=4),
        )
        if results.degrees_freedom is not None:
            self.add_row(
                "Degrees of Freedom",
                format_value(results.degrees_freedom, precision=0),
            )
        self.add_row(
            "Hypothesis",
            results.hypothesis_text,
        )
        self.add_row(
            "Accept Hypothesis",
            str(results.accept_hypothesis),
        )
        if results.correction_method is not None:
            self.add_row(
                "MC Correction",
                results.correction_method,
            )
        if results.warnings:
            self.add_row(
                "Warnings",
                process_warnings(results.warnings),
            )


class FrequentistTestStatisticalDistribution(Protocol):
    """A distrubiton that implements scipy.stats API"""

    def ppf(self, *args, **kwargs):
        pass


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
