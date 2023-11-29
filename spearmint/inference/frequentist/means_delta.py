from numpy import inf

from spearmint.config import MIN_OBS_FOR_Z_TEST
from spearmint.stats import MeanComparison, Samples
from spearmint.typing import FilePath, Tuple, Optional

from .frequentist_inference import (
    FrequentistInferenceProcedure,
    FrequentistInferenceResults,
)


def visualize_means_delta_results(
    results: FrequentistInferenceResults, outfile: Optional[FilePath] = None
):  # pragma: no cover
    # Lazy import
    import holoviews as hv

    from spearmint import vis

    # Sample distributinos
    control_dist = vis.plot_gaussian(
        mean=results.control.mean,
        std=results.control.std,
        label=results.control.name,
        color=vis.CONTROL_COLOR,
    )

    # Variation components
    variation_dist = vis.plot_gaussian(
        mean=results.variation.mean,
        std=results.variation.std,
        label=results.variation.name,
        color=vis.VARIATION_COLOR,
    )

    # Mean confidence intervals
    control_ci = vis.plot_interval(
        *results.control.confidence_interval(1 - results.alpha),
        middle=results.control.mean,
        label=results.control.name,
        color=vis.CONTROL_COLOR,
        show_interval_text=True,
    )

    variation_ci = vis.plot_interval(
        *results.variation.confidence_interval(1 - results.alpha),
        middle=results.variation.mean,
        label=results.variation.name,
        color=vis.VARIATION_COLOR,
        show_interval_text=True,
    )

    distribution_plot = control_dist * variation_dist * control_ci * variation_ci
    distribution_plot = distribution_plot.relabel(
        "Sample Distribution and\nCentral Tendency Estimates"
    ).opts(legend_position="top_right", xlabel="Value", ylabel="pdf")

    # Delta distribution
    mean_delta = results.variation.mean - results.control.mean
    std_delta = (
        (results.control.var / results.control.nobs)
        + (results.variation.var / results.control.nobs)
    ) ** 0.5

    delta_dist = vis.plot_gaussian(
        mean=mean_delta,
        std=std_delta,
        label="Delta Distribution",
        color=vis.DELTA_COLOR,
    )

    if results.hypothesis == "larger":
        left_bound = results.delta_confidence_interval[0]
        right_bound = inf
    elif results.hypothesis == "smaller":
        right_bound = results.delta_confidence_interval[1]
        left_bound = inf
    else:
        left_bound = results.delta_confidence_interval[0]
        right_bound = results.delta_confidence_interval[1]

    max_pdf_height = delta_dist.data["pdf"].max()
    ci_percent = round(100 * (1 - results.alpha))
    delta_ci = vis.plot_interval(
        left_bound,
        right_bound,
        mean_delta,
        color=vis.NEUTRAL_COLOR,
        label=f"{ci_percent}% Confidence Interval",
        show_interval_text=True,
        vertical_offset=-(max_pdf_height * 0.01),
    )

    zero_delta_vline = hv.Spikes(
        ([0.0], [max_pdf_height]), vdims="pdf", label="Null Delta"
    ).opts(color=vis.COLORS.red)

    delta_plot = delta_dist * delta_ci * zero_delta_vline
    delta_plot = (
        delta_plot.relabel("Means Delta")
        .opts(xlabel="delta", ylabel="pdf")
        .opts(legend_position="top_right")
    )

    visualization = distribution_plot + delta_plot
    visualization.opts(shared_axes=False)

    if outfile is not None:
        vis.save_visualization(visualization, outfile)

    return visualization


class MeansDelta(FrequentistInferenceProcedure):
    """
    Frequentist inference procedure to test for the difference in two sample
    means. Assumes normality for large sample sizes, or t-distribution for small
    sample sizes.
    """

    def __init__(self, variance_assumption: str = "unequal", *args, **kwargs):
        """
        Parameters
        ----------
        variance_assumption : str
            whether to use pooled or unequal variance assumptions
            - 'pooled': assume the same variance
            - 'unequal': use Smith-Satterthwait dof when calculating t-stat
        """
        super().__init__(*args, **kwargs)
        self.variance_assumption = variance_assumption

    @property
    def test_stats(self) -> dict:
        return getattr(self.comparison, f"{self.test_statistic_name}_test_stats")

    @property
    def delta_ci(self) -> Tuple[float, float]:
        """
        Calculate the frequentist confidence interval percentiles and values.
        """
        ci_function = getattr(
            self.comparison, f"{self.test_statistic_name}confint_diff"
        )
        return ci_function(self.alpha, self.hypothesis_sm, self.variance_assumption)

    # @abstractmethod
    def _run_inference(
        self, control_samples: Samples, variation_samples: Samples, **inference_kwargs
    ) -> None:
        nobs = min(control_samples.nobs, variation_samples.nobs)
        test_statistic_name = "z" if nobs > MIN_OBS_FOR_Z_TEST else "t"
        self.comparison = MeanComparison(
            samples_a=variation_samples,
            samples_b=control_samples,
            test_statistic_name=test_statistic_name,
            alpha=self.alpha,
            hypothesis=self.hypothesis,
        )

    # @abstractmethod
    def _make_results(self) -> FrequentistInferenceResults:
        test_stats = self.test_stats
        degrees_freedom = test_stats.get("degrees_freedom", None)
        accept_hypothesis = self.accept_hypothesis(test_stats["statistic_value"])

        return FrequentistInferenceResults(
            control=self.comparison.d2,
            variation=self.comparison.d1,
            metric_name=self.metric_name,
            comparison_type="Means Delta",
            delta=self.comparison.delta,
            delta_relative=self.comparison.delta_relative,
            effect_size=self.comparison.effect_size,
            alpha=self.comparison.alpha,
            power=self.comparison.power,
            delta_confidence_interval=self.delta_ci,
            delta_confidence_interval_percentiles=self.delta_ci_percentiles,
            hypothesis=self.hypothesis,
            inference_method=self.inference_method,
            variable_type=self.variable_type,
            warnings=self.comparison.warnings,
            test_statistic_name=test_stats["statistic_name"],
            test_statistic_value=test_stats["statistic_value"],
            p_value=test_stats["p_value"],
            degrees_freedom=degrees_freedom,
            accept_hypothesis=accept_hypothesis,
            visualization_function=visualize_means_delta_results,
        )
