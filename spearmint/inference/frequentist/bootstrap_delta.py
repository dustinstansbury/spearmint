import numpy as np

from spearmint.stats import BootstrapStatisticComparison, Samples
from spearmint.typing import Callable, FilePath, Optional

from .frequentist_inference import (
    FrequentistInferenceProcedure,
    FrequentistInferenceResults,
)


def visualize_bootstrap_delta_results(
    results: FrequentistInferenceResults, outfile: Optional[FilePath] = None
):  # pragma: no cover
    # Lazy import
    import holoviews as hv

    from spearmint import vis

    test_statistic_label = results.test_statistic_name.replace("_", " ")
    test_statistic_title = test_statistic_label.title()
    control_samples = results.aux["control_bootstrap_samples"]
    variation_samples = results.aux["variation_bootstrap_samples"]
    delta_samples = results.aux["delta_bootstrap_samples"]

    ci_percent = round(100 * (1 - results.alpha))
    ci_bounds = np.round(100 * np.array((results.alpha / 2, 1 - results.alpha / 2)))

    control_dist = vis.plot_kde(
        samples=control_samples.data,
        label=results.control.name,
        color=vis.CONTROL_COLOR,
    )

    variation_dist = vis.plot_kde(
        samples=variation_samples.data,
        label=results.variation.name,
        color=vis.VARIATION_COLOR,
    )

    # Confidence intervals
    control_ci = vis.plot_interval(
        *control_samples.percentiles(ci_bounds),
        middle=control_samples.mean,
        label=results.control.name,
        color=vis.CONTROL_COLOR,
        show_interval_text=True,
    )  # type: ignore # (mypy bug, see #6799)

    variation_ci = vis.plot_interval(
        *variation_samples.percentiles(ci_bounds),
        middle=variation_samples.mean,
        label=results.variation.name,
        color=vis.VARIATION_COLOR,
        show_interval_text=True,
    )  # type: ignore # (mypy bug, see #6799)

    distribution_plot = control_dist * variation_dist * control_ci * variation_ci
    distribution_plot = distribution_plot.relabel(
        f"{test_statistic_title} Comparison"
    ).opts(legend_position="top_right", xlabel=test_statistic_label, ylabel="pdf")

    delta_dist = vis.plot_kde(
        samples=delta_samples.data,
        label="Delta Distribution",
        color=vis.DELTA_COLOR,
    )

    max_pdf_height = delta_dist.data["pdf"].max()

    mean_delta = results.aux["delta_bootstrap_samples"].mean

    delta_ci = vis.plot_interval(
        *delta_samples.percentiles(ci_bounds),
        mean_delta,
        color=vis.NEUTRAL_COLOR,
        label=f"{ci_percent}% Confidence Interval",
        show_interval_text=True,
        vertical_offset=-(max_pdf_height * 0.01),
    )  # type: ignore # (mypy bug, see #6799)

    zero_delta_vline = hv.Spikes(
        ([0.0], [max_pdf_height]), vdims="pdf", label="Null Delta"
    ).opts(color=vis.COLORS.red)

    delta_plot = delta_dist * delta_ci * zero_delta_vline
    delta_plot_title = f"{test_statistic_title} Delta"
    delta_plot = (
        delta_plot.relabel(delta_plot_title)
        .opts(xlabel="delta", ylabel="pdf")
        .opts(legend_position="top_right")
    )

    visualization = distribution_plot + delta_plot
    visualization.opts(shared_axes=False)
    # visualization.opts(shared_axes=False).cols(1)

    if outfile is not None:
        vis.save_visualization(visualization, outfile)

    return visualization


class BootstrapDelta(FrequentistInferenceProcedure):
    """
    Runs frequentist inference procedure to test for the difference in a bootstrapped
    test statistic estimate
    """

    def __init__(self, statistic_function: Callable = np.mean, *args, **kwargs):
        """
        Parameters
        ----------
        statistic_function : Callable
            Function that returns a scalar test statistic when provided a sequence
            of samples.

        """
        super(BootstrapDelta, self).__init__(*args, **kwargs)
        self.statistic_function = statistic_function

    @property
    def test_stats(self):
        return self.comparison.bootstrap_test_stats

    @property
    def delta_ci(self):
        """
        Calculate confidence interval around deltas with percentiles and values.
        """
        return self.comparison.confidence_interval(1 - self.alpha)

    # @abstractmethod
    def _run_inference(
        self, control_samples: Samples, variation_samples: Samples, **inference_kwargs
    ) -> None:
        self.comparison = BootstrapStatisticComparison(
            samples_a=variation_samples,
            samples_b=control_samples,
            alpha=self.alpha,
            hypothesis=self.hypothesis,
            statistic_function=self.statistic_function,
        )

    def accept_hypothesis(self, statistic_value: float) -> bool:
        """
        Overloads `FrequentistInferenceProcedure.accept_hypothesis()` method
        to use the boostrapped estimate of the sampling distribition to test the
        Null hypothesis
        """
        if self.hypothesis == "larger":
            return self.alpha > self.comparison.null_dist.prob_greater_than(
                statistic_value
            )
        elif self.hypothesis == "smaller":
            return self.alpha > 1 - self.comparison.null_dist.prob_greater_than(
                statistic_value
            )
        elif self.hypothesis == "unequal":
            return abs(statistic_value) > abs(
                self.comparison.null_dist.percentiles(100 * (1 - self.alpha / 2.0))
            )
        else:
            raise ValueError("Unknown hypothesis: {!r}".format(self.hypothesis))

    # @abstractmethod
    def _make_results(self) -> FrequentistInferenceResults:
        test_stats = self.test_stats
        accept_hypothesis = self.accept_hypothesis(test_stats["statistic_value"])

        aux = {
            "control_bootstrap_samples": self.comparison.control_bootstrap,
            "variation_bootstrap_samples": self.comparison.variation_bootstrap,
            "delta_bootstrap_samples": self.comparison.deltas_dist,
        }

        return FrequentistInferenceResults(
            control=self.comparison.d2,
            variation=self.comparison.d1,
            metric_name=self.metric_name,
            comparison_type="Bootstrap Delta",
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
            degrees_freedom=None,
            accept_hypothesis=accept_hypothesis,
            visualization_function=visualize_bootstrap_delta_results,
            aux=aux,
        )
