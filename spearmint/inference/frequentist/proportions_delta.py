import numpy as np

from spearmint.stats import ProportionComparison, Samples
from spearmint.typing import FilePath, Tuple, Optional

from .frequentist_inference import (
    FrequentistInferenceProcedure,
    FrequentistInferenceResults,
)


def visualize_proportions_delta_results(
    results: FrequentistInferenceResults, outfile: Optional[FilePath] = None
):  # pragma: no cover
    # Lazy import
    import holoviews as hv

    from spearmint import vis

    # Sample distribution comparison plot
    control_dist = vis.plot_binomial(
        p=results.control.mean,
        n=results.control.nobs,
        label=results.control.name,
        color=vis.CONTROL_COLOR,
        alpha=0.5,
    ).opts(axiswise=True)

    variation_dist = vis.plot_binomial(
        p=results.variation.mean,
        n=results.variation.nobs,
        label=results.variation.name,
        color=vis.VARIATION_COLOR,
        alpha=0.5,
    ).opts(axiswise=True)

    def get_binomial_cis(samples, alpha):
        """Convert proportionality to successful # trials"""
        confidence = 1 - alpha
        cis = np.round(
            np.array(samples.confidence_interval(confidence=confidence)) * samples.nobs
        ).astype(int)

        mean = np.round(samples.mean * samples.nobs).astype(int)
        return cis[0], cis[1], mean

    control_ci = vis.plot_interval(
        *get_binomial_cis(results.control, results.alpha),
        label=results.control.name,
        color=vis.CONTROL_COLOR,
        show_interval_text=True,
    )  # type: ignore # (mypy bug, see #6799)

    variation_ci = vis.plot_interval(
        *get_binomial_cis(results.variation, results.alpha),
        label=results.variation.name,
        color=vis.VARIATION_COLOR,
        show_interval_text=True,
    )  # type: ignore # (mypy bug, see #6799)

    distribution_plot = control_dist * variation_dist * control_ci * variation_ci
    distribution_plot = distribution_plot.relabel(
        "Sample Distribution and\nCentral Tendency Estimates"
    ).opts(legend_position="top_right", xlabel="# Successful Trials", ylabel="pdf")

    # Delta distribution plot
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
        right_bound = np.inf
    elif results.hypothesis == "smaller":
        right_bound = results.delta_confidence_interval[1]
        left_bound = np.inf
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
        delta_plot.relabel("Proportionality Delta")
        .opts(xlabel="delta", ylabel="pdf")
        .opts(legend_position="top_right")
    )

    visualization = distribution_plot + delta_plot
    visualization.opts(shared_axes=False)
    if outfile is not None:
        vis.save_visualization(visualization, outfile)

    return visualization


class ProportionsDelta(FrequentistInferenceProcedure):
    """
    Frequentist inference procedure to test for the difference in two sample
    proportions (i.e. conversion rates). Uses Gaussian approximation for sample
    distributions, and thus, assumes adequate sample sizes (i.e. N > 30). If
    this condition is violated, you may receive warnings during inference.
    """

    def __init__(self, variance_assumption: str = "pooled", *args, **kwargs):
        """
        Parameters
        ----------
        variance_assumption : False or float in (0, 1)
            whether to calculate variance based on sample or use control or
            another variance
        """
        super().__init__(*args, **kwargs)
        self.variance_assumption = variance_assumption

    @property
    def test_stats(self) -> dict:
        return self.comparison.z_test_stats

    @property
    def delta_ci(self) -> Tuple[float, float]:
        """
        Calculate confidence interval percentiles and values.
        """
        variance_assumption = (
            self.variance_assumption
            if self.variance_assumption == "pooled"
            else "unequal"
        )
        return self.comparison.zconfint_diff(
            self.alpha, self.hypothesis_sm, variance_assumption
        )

    # @abstractmethod
    def _run_inference(
        self, control_samples: Samples, variation_samples: Samples, **inference_kwargs
    ) -> None:
        """
        Run the inference procedure over the samples with a selected alpha
        value
        """

        self.comparison = ProportionComparison(
            samples_a=variation_samples,
            samples_b=control_samples,
            alpha=self.alpha,
            hypothesis=self.hypothesis,
        )

    # @abstractmethod
    def _make_results(self) -> FrequentistInferenceResults:
        """
        Package up inference results
        """
        test_stats = self.test_stats
        accept_hypothesis = self.accept_hypothesis(test_stats["statistic_value"])

        return FrequentistInferenceResults(
            control=self.comparison.d2,
            variation=self.comparison.d1,
            metric_name=self.metric_name,
            comparison_type="Proportions Delta",
            delta=self.comparison.delta,
            delta_relative=self.comparison.delta_relative,
            effect_size=self.comparison.effect_size,
            alpha=self.comparison.alpha,
            power=self.comparison.power,
            delta_confidence_interval=self.delta_ci,
            delta_confidence_interval_percentiles=self.delta_ci_percentiles,
            inference_method=self.inference_method,
            variable_type=self.variable_type,
            warnings=self.comparison.warnings,
            test_statistic_name=test_stats["statistic_name"],
            test_statistic_value=test_stats["statistic_value"],
            p_value=test_stats["p_value"],
            degrees_freedom=None,
            hypothesis=self.hypothesis,
            accept_hypothesis=accept_hypothesis,
            visualization_function=visualize_proportions_delta_results,
        )
