import numpy as np
from scipy.stats import norm

from spearmint.stats import RateComparison, Samples
from spearmint.typing import FilePath, Tuple, Optional

from .frequentist_inference import (
    FrequentistInferenceProcedure,
    FrequentistInferenceResults,
)


def visualize_rates_ratio_results(
    results: FrequentistInferenceResults, outfile: Optional[FilePath] = None
):  # pragma: no cover
    # Lazy import
    import holoviews as hv

    from spearmint import vis

    # Sample distribution comparison plot
    control_dist = vis.plot_poisson(
        mu=results.control.mean,
        color=vis.CONTROL_COLOR,
        label=results.control.name,
        alpha=0.5,
    )

    variation_dist = vis.plot_poisson(
        mu=results.variation.mean,
        color=vis.VARIATION_COLOR,
        label=results.variation.name,
        alpha=0.5,
    )

    # Proportion/conversion rate confidence intervals plot
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
    ).opts(legend_position="top_right", xlabel="N Events", ylabel="pdf")

    # Delta distribution plot
    mean_ratio = results.variation.mean / results.control.mean
    std_delta = (
        (results.control.var / results.control.nobs)
        + (results.variation.var / results.control.nobs)
    ) ** 0.5

    delta_dist = vis.plot_gaussian(
        mean=mean_ratio,
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
        left=left_bound,
        right=right_bound,
        middle=mean_ratio,
        color=vis.NEUTRAL_COLOR,
        label=f"{ci_percent}% Confidence Interval",
        show_interval_text=True,
        vertical_offset=-(max_pdf_height * 0.01),
    ).opts(ylabel="", xlabel="rate")

    one_delta_vline = hv.Spikes(
        ([1.0], [max_pdf_height]), vdims="pdf", label="Null Ratio"
    ).opts(color=vis.COLORS.red)

    delta_plot = delta_dist * delta_ci * one_delta_vline
    delta_plot = (
        delta_plot.relabel("Rates Ratio")
        .opts(xlabel="ratio", ylabel="pdf")
        .opts(legend_position="top_right")
    )

    visualization = distribution_plot + delta_plot
    visualization.opts(shared_axes=False)

    if outfile is not None:
        vis.save_visualization(visualization, outfile)

    return visualization


class RatesRatio(FrequentistInferenceProcedure):
    """
    Frequentist inference procedure to test for the difference in two sample
    event rates. Use to compare the number of events measured in a standard
    block of time (e.g. clicks in first minute of each user session).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def test_stats(self) -> dict:
        return self.comparison.rates_test_stats

    @property
    def delta_ci(self) -> Tuple[float, float]:
        """
        Calculate confidence interval for  rates ratio. Intervals outside of 1
        support the alternative hypothesis.

        Calculation follows Li, Tang, & Wong, 2008, using the MOVER-R method,
        and Rao Score intervals (Altman et al, 2000) for individual rate interval
        estimates (aka "FRSI")

        Returns
        -------
        CIs : list
            [(CI_lower, CI_upper), (CI_lower_percentile, CI_upper_percentile)]

        References
        ----------
        Li H.Q, Tang ML, Wong WK (2008) Confidence intervals for ratio of two
            Poisson rates using the method of variance estimates recovery.
            Biometrical Journal 50 (2008)
        Altman D., Machin D., Bryant TN. et al. (2000) "Statistics with confidence"
            (2nd). BMJ Books: Bristol.
        """

        def rao_score_interval(X, z, t):
            # individual rate interval method 2 (Altman et al., 2000)
            a = X + 0.5 * z**2.0
            b = z * np.sqrt(X + 0.25 * z**2.0)
            return (a - b) / t, (a + b) / t

        if self.hypothesis == "larger":
            z = norm.ppf(1 - self.alpha)
        elif self.hypothesis == "smaller":
            z = norm.ppf(self.alpha)
        elif self.hypothesis == "unequal":
            z = np.abs(norm.ppf(1 - self.alpha / 2.0))

        control = self.comparison.d2
        variation = self.comparison.d1

        X1, t1 = control.data.sum(), control.nobs
        X2, t2 = variation.data.sum(), variation.nobs

        lam_1 = X1 / t1
        lam_2 = X2 / t2

        lam_2_lam_1 = lam_2 * lam_1

        l2, u2 = rao_score_interval(X1, z, t1)
        l1, u1 = rao_score_interval(X2, z, t2)

        # Gu et al, 2008; Eq 3
        L = (
            lam_2_lam_1
            - np.sqrt(
                lam_2_lam_1**2 - l1 * (2 * lam_2 - l1) * (u2 * (2 * lam_1 - u2))
            )
        ) / (u2 * (2 * lam_1 - u2))

        # Gu et al, 2008; Eq 4
        U = (
            lam_2_lam_1
            + np.sqrt(
                lam_2_lam_1**2 - u1 * (2 * lam_2 - u1) * (l2 * (2 * lam_1 - l2))
            )
        ) / (l2 * (2 * lam_1 - l2))

        return (L, U)

    # @abstractmethod
    def _run_inference(
        self, control_samples: Samples, variation_samples: Samples, **inference_kwargs
    ) -> None:
        self.comparison = RateComparison(
            samples_a=variation_samples,
            samples_b=control_samples,
            alpha=self.alpha,
            hypothesis=self.hypothesis,
        )

    # @abstractmethod
    def _make_results(self) -> FrequentistInferenceResults:
        test_stats = self.test_stats
        accept_hypothesis = self.accept_hypothesis(test_stats["statistic_value"])

        return FrequentistInferenceResults(
            control=self.comparison.d2,
            variation=self.comparison.d1,
            metric_name=self.metric_name,
            comparison_type="Rates Ratio",
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
            visualization_function=visualize_rates_ratio_results,
        )
