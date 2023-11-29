import numpy as np
from scipy import optimize
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.api import CompareMeans, DescrStatsW

from spearmint.config import (
    DEFAULT_ALPHA,
    DEFAULT_HYPOTHESIS,
    MIN_OBS_FOR_Z_TEST,
    logger,
)
from spearmint.table import SpearmintTable
from spearmint.typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
    Union,
    Optional,
)
from spearmint.utils import format_value

SUPPORTED_POWER_STATISTICS = ("t", "z")


def _get_solve_power_function(statistic: str) -> Callable:
    if statistic not in SUPPORTED_POWER_STATISTICS:
        raise ValueError(f"Statistic '{statistic}' not supported for power calculation")

    if statistic == "t":
        from statsmodels.stats.power import tt_ind_solve_power as solve_power
    else:
        from statsmodels.stats.power import zt_ind_solve_power as solve_power
    return solve_power  # type: ignore


def bonferroni_correction(alpha_orig: float, p_values: Tuple[float]) -> float:
    """
    Calculate the correcteed alpha value for a frequentist hypothesis test using
    Bonferroni's method.

    en.wikipedia.org/wiki/Bonferroni_correction

    Parameters
    ----------
    alpha_orig : float
        alpha value before correction
    p_values: Tuple[float]
        The p-values associated with concurrent hypothesis tests being run

    Returns
    -------
    alpha_corrected: float
        new critical value (i.e. the corrected alpha)
    """
    n_tests = len(p_values)
    return alpha_orig / n_tests


def sidak_correction(alpha_orig: float, p_values: Tuple[float]) -> float:
    """
    Calculate the correcteed alpha value for a frequentist hypothesis test using
    Sidak's method.

    en.wikipedia.org/wiki/%C5%A0id%C3%A1k_correction

    Parameters
    ----------
    alpha_orig : float
        alpha value before correction
    p_values: Tuple[float]
        The p-values associated with concurrent hypothesis tests being run

    Returns
    -------
    alpha_corrected: float
        new critical value (i.e. the corrected alpha)
    """
    n_tests = len(p_values)
    return 1.0 - (1.0 - alpha_orig) ** (1.0 / n_tests)


def fdr_bh_correction(fdr: float, p_values: Tuple[float]) -> float:
    """
    Benjamini-Hochberg false-discovery rate adjustment procedure.

    pdfs.semanticscholar.org/af6e/9cd1652b40e219b45402313ec6f4b5b3d96b.pdf

    Parameters
    ----------
    fdr : float
        False Discovery Rate (q*), proportion of significant results that are
        actually false positives
    p_values: Iterable[float]
        The p-values associated with concurrent hypothesis tests being run

    Returns
    -------
    alpha_corrected: float
        The corrected alpha value
    """
    n_tests = len(p_values)

    def p_i(i):
        return i * fdr / n_tests

    p_sorted = np.sort(np.asarray(p_values))

    significant_idx = [i for i, val in enumerate(p_sorted, 1) if val <= p_i(i)]
    rank = np.max(significant_idx) if significant_idx else 1
    return p_i(rank)


MULTIPLE_COMPARISON_CORRECTION_METHODS = {
    "b": bonferroni_correction,
    "bonferroni": bonferroni_correction,
    "s": sidak_correction,
    "sidak": sidak_correction,
    "bh": fdr_bh_correction,
    "fdr_bh": fdr_bh_correction,
}


def _get_multiple_comparison_correction_function(
    method: Union[str, Callable]
) -> Callable:
    if isinstance(method, Callable):  # type: ignore
        if MULTIPLE_COMPARISON_CORRECTION_METHODS.values():
            return method  # type: ignore
    elif isinstance(method, str):
        if method in MULTIPLE_COMPARISON_CORRECTION_METHODS.keys():
            return MULTIPLE_COMPARISON_CORRECTION_METHODS[method]

    raise ValueError(f"Multiple correction method {method} not supported")


class MultipleComparisonCorrection:
    """
    Perform multiple comparison adjustment of alpha based on a sequence of
    p_values that result from two or more hypothesis tests inference procedures.

    p_values : Tuple[float]
        A list of p_values resulting from two or more hypothesis tests.
    method :  str
        One of the following correction methods:
            - 'bonferroni', 'b' : one-step Bonferroni correction
            - 'sidak', 's' : one-step Sidak correction
            - 'fdr_bh', 'bh; : Benjamini/Hochberg (non-negative)
    alpha : float in (0, 1)
        The desired probability of Type I error
    reject_null: list[bool]
        For each probablity, whether or not to reject the null hypothsis given
        the updated values for alpha.
    """

    __ATTRS__ = ["ntests", "method", "alpha_orig", "alpha_corrected"]

    def __init__(self, p_values, method="sidak", alpha=DEFAULT_ALPHA):
        self.mc_correction_method = _get_multiple_comparison_correction_function(method)
        self.ntests = len(p_values)
        self.alpha_orig = alpha
        self.alpha_corrected = self.mc_correction_method(alpha, p_values)
        self.accept_hypothesis = [p < self.alpha_corrected for p in p_values]

    @property
    def mc_correction_method_name(self) -> str:
        return self.mc_correction_method.__name__.split("_correction")[0]


def estimate_experiment_sample_sizes(
    delta: float,
    statistic: str = "z",
    alpha: float = 0.05,
    power: float = 0.8,
    *args,
    **kwargs,
) -> Tuple[int, ...]:
    """
    Calculate the sample size required for each treatement in order to observe a
    difference of `delta` between control and variation groups, given a
    particular `alpha` (Type I error rate) and `power` (1 - Type II error rate).

    Parameters
    ----------
    delta : float
        The absolute difference in means between control and variation groups
    statistic : string
        Either:
            - 'z' or 't' if interpreting effect size as scaled difference of means
            - 'rates_ratio' if interpereting effect size as the ratio of means
    alpha : float [0, 1)
            The assumed Type I error of the test
    power : float [0, 1)
        The desired statistical power of the test
    *args, **kwargs
        Model-specific arguments

    Returns
    -------
    sample_sizes : List[int]
        The estimated sample sizes for the control and variation treatments

    Example 1: Continuous Variables
    ----

    Estimate the sample size required to observe significant difference between
    two binomial distributions that differ by .01 in mean probability with
    Type I error = 0.05 (default) and Power = 0.8 (default)

    ```python
    >>> prob_control = .49
    >>> std_control = (prob_control * (1 - prob_control))**.5  # Binomial std
    >>> prob_variation = std_variation = .50
    >>> delta = prob_variation - prob_control
    >>> estimate_experiment_sample_sizes(
        delta=delta,
        statistic='z',
        std_control=std_control,
        std_variation=std_variation
    )
    (39236, 39236)
    ```python

    Example 2: Count Variables
    ---
    Replicate Example 1 from Gu et al, 2008

    ```python
    >>> R = 4  # ratio under alternative hypothesis
    >>> control_rate = .0005
    >>> variation_rate = R * control_rate
    >>> delta = variation_rate - control_rate
    >>> estimate_experiment_sample_sizes(
        delta,
        statistic='rates_ratio',
        control_rate=control_rate,
        alpha=.05,
        power=.9,
        control_exposure_time=2.,
        sample_size_ratio=.5
    )
    (8590, 4295)
    ```
    """
    if statistic in ("t", "z"):
        # std_control and/or std_variation are in *args, or **kwargs
        return cohens_d_sample_size(delta, alpha, power, statistic, *args, **kwargs)
    elif statistic == "rates_ratio":
        return ratio_sample_size(alpha, power, delta, *args, **kwargs)
    else:
        raise ValueError("Unknown statistic")


def cohens_d(delta: float, std_control: float, std_variation: Optional[float]) -> float:
    """Calculate the Cohen's d effect size comparing two samples. For details
    see https://en.wikiversity.org/wiki/Cohen%27s_d

    Parameters
    ----------
    delta : float
        The measured difference in means between the two samples
    std_control : float
        The standard deviation for the control sample
    std_variation : float, optional
        If provided, the variation for the treatment sample. If None provided,
        we assume the treatment has the same standard deviation as the control.

    Returns
    -------
    float
        Cohen'd d metric for effect size.
    """
    std_variation = std_variation if std_variation else std_control
    std_pooled = np.sqrt((std_control**2 + std_variation**2) / 2.0)
    return delta / std_pooled


def cohens_d_sample_size(
    delta: float,
    alpha: float,
    power: float,
    statistic: str,
    std_control: float,
    std_variation: Optional[float] = None,
    sample_size_ratio: float = 1.0,
) -> Tuple[int, int]:
    """
    Calculate sample size required to observe a significantly reliable difference
    between groups a and b. Assumes Cohen's d definition of effect size and an
    enrollment ratio of 1.0 between groups a and b by default.

    Parameters
    ----------
    std_control : float
        An estiamte of the expected sample standard deviation of control
        group
    nobs_control : int
        The number of control observations.
    std_variation : float
        An estimate of the expected sample standard deviation of variation
        group. If not provided, we assume homogenous variances for the
        two groups.

    Returns
    -------
    sample_sizes : list[int]
        The estimated sample sizes for the control and variation treatments

    Example
    -------
    Estimate of sample sizes required to observe a significant difference between
    two binomial distributions that have the same variance, but differ by .01
    in mean probability.

    ```python
    >>> mean_prob_control = .49
    >>> mean_prob_variation = std_variation = .5
    >>> std_control = (mean_prob_control * (1 - mean_prob_control))**.5  # Bernoulli stddev
    >>> std_variation = std_control
    >>> delta = mean_prob_variation - mean_prob_control
    >>> cohens_d_sample_size(
        delta=delta,
        alpha=.05,
        power=.8,
        statistic='z',
        std_control=std_control,
        std_variation=std_variation
    )
    (39228, 39228)
    ```

    References
    ----------
    Cohen, J. (1988). Statistical power analysis for the behavioral sciences
        (2nd ed.). Hillsdale, NJ: Lawrence Earlbaum Associates.
    """

    effect_size = cohens_d(delta, std_control, std_variation)

    solve_power_function = _get_solve_power_function(statistic)
    N1 = int(
        solve_power_function(
            effect_size, alpha=alpha, power=power, ratio=sample_size_ratio
        )
    )
    N2 = int(N1 * sample_size_ratio)
    return N1, N2


def ratio_sample_size(
    alpha: float,
    power: float,
    delta: float,
    control_rate: float,
    control_exposure_time: float = 1.0,
    null_ratio: float = 1.0,
    sample_size_ratio: float = 1.0,
    exposure_time_ratio: float = 1.0,
) -> Tuple[int, ...]:
    """
    Calculate sample size required to observe a significantly reliable ratio of
    rates between variation and control groups. Follows power calculation outlined
    in Gu et al, 2008.

    Parameters
    ----------
    control_rate : float
        The poisson rate of the control group
    control_exposure_time : float
        The number of time units of the control exposure. Default is 1.0
    null_ratio : float
        The ratio of variation to control rates under the null hypothesis.
        Default is 1.
    sample_size_ratio : float
        The ratio of sample sizes of the variation to the control groups. Default is
        1, thus assuming equal sample sizes.
    exposure_time_ratio : float
        The ratio of the variation exposure time to the control. Default is 1.0,
        thus assuming equal exposure times

    Returns
    -------
    sample_sizes : Tuple[int, int]
        Sample sizes for each group

    Example
    -------
    Replicate Example 1 from Gu et al, 2008

    ```python
    >>> R = 4  # ratio under alternative hypothesis
    >>> control_rate = .0005
    >>> variation_rate = R * control_rate
    >>> delta = variation_rate - control_rate
    >>> ratio_sample_size(
        alpha=.05,
        power=.9,
        delta=delta,
        control_rate=control_rate,
        control_exposure_time=2.,
        sample_size_ratio=.5
    )
    (8590, 4295)
    ```

    Returns (8590, 4295), which have been validated to be more accurate than
    the result reported in Gu et al, due to rounding precision. For details
    see "Example 2 – Validation using Gu (2008)" section of
    http://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Tests_for_the_Ratio_of_Two_Poisson_Rates.pdf

    References
    ----------
    Gu, K., Ng, H.K.T., Tang, M.L., and Schucany, W. 2008. 'Testing the Ratio of
        Two Poisson Rates.' Biometrical Journal, 50, 2, 283-298.
    Huffman, M. 1984. 'An Improved Approximate Two-Sample Poisson Test.'
        Applied Statistics, 33, 2, 224-226.
    """

    # convert absolute difference to ratio
    alternative_ratio = float(control_rate + delta) / control_rate
    variation_exposure_time = exposure_time_ratio * control_exposure_time

    z_alpha = norm.ppf(1 - alpha)
    z_power = norm.ppf(power)

    def objective(x):
        ratio_proposed = (x[1] * variation_exposure_time) / (
            x[0] * control_exposure_time
        )
        loss = np.abs(null_ratio - (alternative_ratio / ratio_proposed))
        return loss

    def con1(x):
        """General sample size ratio constraint"""
        return (float(x[1]) / x[0]) - sample_size_ratio

    def con2(x):
        """Control sample size constraint, outlined in Gu et al, 2008, Equation 10"""

        N1, N2 = x
        d = (control_exposure_time * N1) / (variation_exposure_time * N2)
        A = 2 * (1.0 - np.sqrt(null_ratio / alternative_ratio))
        C = np.sqrt((null_ratio + d) / alternative_ratio)
        D = np.sqrt((alternative_ratio + d) / alternative_ratio)
        return x[0] - (((z_alpha * C + z_power * D) / A) ** 2.0 - (3.0 / 8)) / (
            control_exposure_time * control_rate
        )

    constraint1 = {"type": "eq", "fun": con1}
    constraint2 = {"type": "eq", "fun": con2}
    constraints = [constraint1, constraint2]

    results = optimize.minimize(
        objective,
        (10, 10),
        bounds=((1, None), (1, None)),
        constraints=constraints,
        method="SLSQP",
        tol=1e-10,
    )
    return tuple(int(np.ceil(n)) for n in results.x)


def highest_density_interval(
    samples: Iterable[float], credible_mass: float = 0.95
) -> np.ndarray:
    """
    Calculate the bounds of the highest density interval (HDI) with width
    `credible_mass` under the distribution of samples.

    Parameters
    ----------
    samples: Iterable[float]
        The samples to compute the interval over
    credible_mass: float in (0, 1)
        The credible mass under the empricial distribution

    Returns
    -------
    hdi: Tuple[float, float]
        The lower and upper bounds of the highest density interval

    Example
    -------
    >>> samples = np.random.randn(100000)  # standard normal samples
    >>> highest_density_interval(samples, credible_mass=0.95)  # should be approx +/- 1.96
    (-1.9600067573529922, 1.9562495685489785)
    """
    _samples = np.asarray(sorted(samples))
    n = len(_samples)

    interval_idx_inc = int(np.floor(credible_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = _samples[interval_idx_inc:] - _samples[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError("Too few elements for interval calculation")

    min_idx = np.argmin(interval_width)
    hdi_min = _samples[min_idx]
    hdi_max = _samples[min_idx + interval_idx_inc]
    return np.array((hdi_min, hdi_max))


class EmpiricalCdf:
    """
    Class that calculates the empirical cumulative distribution function for a
    set of samples.
    """

    def __init__(self, samples: Iterable[float]):
        self.samples = samples
        self._cdf = ECDF(samples)

    @property
    def samples_cdf(self) -> ECDF:
        """
        Return the cdf evaluated at those samples used to estimate the cdf
        parameters.
        """
        if not hasattr(self, "_samples_cdf"):
            self._samples_cdf = self.evaluate(sorted(self.samples))
        return self._samples_cdf

    def __call__(self, values: Optional[Iterable[float]] = None) -> ECDF:
        """
        Callable interface for the EmpiricalCdf object

        Parameters
        ----------
        values: Iterable[float]
            Observations at which to evaluate the CDF.

        Returns
        -------
        ECDF: statsmodels.distributions.empirical_distribution.ECDF
            An empricial cumulative distribution funcion object. For details,
            see https://www.statsmodels.org/stable/generated/statsmodels.distributions.empirical_distribution.ECDF.html
        """
        return self.evaluate(values)

    def evaluate(self, values: Optional[Iterable[float]] = None) -> ECDF:
        """
        Evaluate the cdf for a sequence of values. If None provided, evaluate
        at all the `samples` used to estimate the CDF

        Parameters
        ----------
        values: Iterable[float]
            Observations at which to evaluate the CDF.

        Returns
        -------
        ECDF: statsmodels.distributions.empirical_distribution.ECDF
            An empricial cumulative distribution funcion object. For details,
            see https://www.statsmodels.org/stable/generated/statsmodels.distributions.empirical_distribution.ECDF.html
        """
        if values is None:
            return self.samples_cdf
        return self._cdf(values)


class Samples(DescrStatsW):
    """
    Class for holding samples and calculating various statistics on those
    samples. Any invalid observations (None, nan, or inf) are ignored.
    """

    def __init__(self, observations: Iterable[float], name: str = "Samples"):
        """
        Parameters
        ----------
        observations : Iterable[float]
            An array-like of observations
        name : str, optional
            A name used when displaying the sample statistics.

        Raises
        ------
        ValueError if all observations are None, nan, or Inf.
        """
        self.name = name
        self._raw_observations = observations
        self._summary_table = None
        valid_observations = self._valid_observations(observations)
        super(Samples, self).__init__(valid_observations)

    def _valid_observations(self, observations: Iterable[float]) -> np.ndarray:
        def valid(o):
            if o is None:
                return False
            if np.isnan(o):
                return False
            if np.isinf(0):
                return False
            return True

        observations = list(filter(valid, observations))
        if self.name:
            name_string = f"{self.name}"
        else:
            name_string = ""
        if not observations:
            raise ValueError(f"All {name_string} observations are nan or None")
        else:
            return np.array(observations)

    @property
    def summary(self) -> None:
        """Print a summary table for the valid observations in the sample"""
        if self._summary_table is None:
            self._summary_table = SamplesSummaryTable(self)  # type: ignore
        self._summary_table.print()  # type: ignore

    @property
    def max(self) -> float:
        return np.max(self.data)

    def permute(self) -> np.ndarray:
        """Shuffle the samples"""
        return np.random.choice(self.data, int(self.nobs))

    def sort(self) -> np.ndarray:
        """Return the observations in ascending order"""
        if not hasattr(self, "_sorted"):
            self._sorted = np.array(sorted(self.data))
        return self._sorted

    def percentiles(
        self, prct: Tuple[float, ...] = (2.5, 25.0, 50.0, 75.0, 97.5)
    ) -> np.ndarray:
        """Calculate the provided percentiles of the sample

        Parameters
        ----------
        prct : Iterable[float]
            Values in the range (0, 100) defining the percentiles to calcualte.

        Returns
        -------
        percentiles : np.ndarray
            The associated values in samples for the provided `prct` percentiles.
        """
        return np.percentile(self.data, prct)

    @property
    def cdf(self) -> EmpiricalCdf:
        """Return the empricial cumulative distribution function for the samples"""
        if not hasattr(self, "_cdf"):
            self._cdf = EmpiricalCdf(self.data)
        return self._cdf

    def prob_greater_than(
        self, values: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Return the cumulative probability of `values` under the current samples'
        empirical CDF.
        """
        return 1.0 - self.cdf(np.asarray(values, dtype=float))

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate the `confidence`-% confidence interval around the mean estimate,
        assuming a Gaussian approximation for the sample distribution.

        Parameters
        ----------
        confidence : float
            Confidence level for the interval.

        Returns
        -------
        ci : Tuple[float, float]
            The upper and lower bounds of the the `confidence`-% confidence interval
            around the mean estimate.
        """

        alpha = (1 - confidence) / 2.0  # symmetric about mean
        z = norm.ppf(1 - alpha)
        ci = z * self.std_err

        return self.mean - ci, self.mean + ci

    @property
    def std_err(self) -> Tuple[float, float]:
        return (self.var / self.nobs) ** 0.5

    def hdi(
        self, credible_mass: float = 0.95
    ) -> Union[Tuple[float, float], Tuple[None, None]]:
        """
        Calculate the bounds of the highest density interval (HDI) with width
        `credible_mass` under the distribution of samples.

        Parameters
        ----------
        credible_mass: float in (0, 1)
            The amount of probability under the returned HDI

        Returns
        -------
        hdi: Tuple[float, float]
            The boundary of the highest density interval for the sample
            distribution, (HDI_lower, HDI_upper)
        """
        try:
            hdi = highest_density_interval(self.data, credible_mass)
            # unpack
            return hdi[0], hdi[1]
        except Exception as e:
            logger.warning(e)
            return (None, None)


class SamplesSummaryTable(SpearmintTable):
    def __init__(self, samples: Samples):
        """Summary statistics comparing control and variation samples"""
        sample_name = samples.name if samples.name is not None else ""
        super().__init__(title="Samples Summary")

        self.add_column("", justify="right")
        self.add_column(sample_name)

        self.add_row(
            "Samples",
            format_value(samples.nobs),
        )
        self.add_row(
            "Mean",
            format_value(samples.mean),
        )
        self.add_row(
            "Standard Error",
            format_value(samples.std_err),
        )
        self.add_row(
            "Variance",
            format_value(samples.var),
        )


class SamplesComparisonTable(SpearmintTable):
    def __init__(
        self,
        control_samples: Samples,
        variation_samples: Union[Samples, List[Samples]],
        metric_name: str = "",
    ):
        """Summary statistics comparing control and variation samples"""
        super().__init__(title="Samples Comparison")

        if isinstance(variation_samples, Samples):
            variation_samples = [variation_samples]

        self.add_column(metric_name, justify="right")
        self.add_column(control_samples.name)

        for vs in variation_samples:
            self.add_column(vs.name)

        def get_variation_row_values(value, precision=4):
            "handle case when there are more than one variation conditions"
            return [
                format_value(getattr(vs, value), precision=precision)
                for vs in variation_samples
            ]

        self.add_row(
            "Samples",
            format_value(control_samples.nobs, precision=0),
            *get_variation_row_values("nobs", precision=0),
        )
        self.add_row(
            "Mean",
            format_value(control_samples.mean),
            *get_variation_row_values("mean", precision=4),
        )
        self.add_row(
            "Standard Error",
            format_value(control_samples.std_err),
            *get_variation_row_values("std_err", precision=4),
        )
        self.add_row(
            "Variance",
            format_value(control_samples.var),
            *get_variation_row_values("var", precision=4),
        )
        deltas = [None] + [
            format_value(vs.mean - control_samples.mean, precision=4)
            for vs in variation_samples
        ]
        self.add_row("Delta", *deltas)


class MeanComparison(CompareMeans):
    """
    Class for comparing the means of two continuous sample distributions,
    provides a number of helpful summary statistics about the comparison. Assumes
    the distributions can be approximated with a Gaussian distrubution.
    """

    def __init__(
        self,
        samples_a: Samples,
        samples_b: Samples,
        alpha: float = DEFAULT_ALPHA,
        test_statistic_name: str = "t",
        hypothesis: str = DEFAULT_HYPOTHESIS,
    ):
        """
        Parameters
        ----------
        samples_a : `Samples`
            Samples from group A (e.g. control group)
        samples_b : `Samples`
            Samples from group B (e.g. treatment group)
        alpha : float in (0, 1)
            The acceptable Type I error rate in the comparison
        test_statistic_name: str
            The name of the hypothesis test statistic used in the comparison.
                -   't': to use a t-test (small sample size, N <= 30)
                -   'z': to use a z-test (large samples size, N > 30)
        hypothesis : str
            Defines the assumed alternative hypothesis. Can be either:
                -   'larger': indicating we hypothesize that group B's mean is
                    larger than group A's
                -   'smaller': indicating we hypothesize that group B's mean is
                    smaller than group A's
                -   'unequal': indicating we hypothesize that group B's mean is
                    not equal to group A's (i.e. two-tailed test).
        """
        super().__init__(samples_a, samples_b)

        self.alpha = alpha
        self._test_statistic_name = test_statistic_name
        self.hypothesis = hypothesis
        self.warnings: List = []
        self._comparison_table = None

    @property  # property allows different test statistic name definitions
    def test_statistic_name(self) -> str:
        return self._test_statistic_name

    @property
    def pooled_variance(self) -> float:
        """The pooled variance of the the two groups"""
        return ((self.d2.nobs - 1) * self.d2.var + (self.d1.nobs - 1) * self.d1.var) / (
            self.d2.nobs + self.d1.nobs - 2
        )

    @property
    def delta(self) -> float:
        """The absolute difference between the group means"""
        return self.d1.mean - self.d2.mean

    @property
    def delta_relative(self) -> float:
        """The relative difference between the group means"""
        return (self.d1.mean - self.d2.mean) / np.abs(self.d2.mean)

    @property
    def effect_size(self) -> float:
        """The Cohen's d effect size of the comparison"""
        return self.delta / np.sqrt(self.pooled_variance)

    @property
    def test_direction(self) -> str:
        """
        The directionality of the hyopthesis test. Will either be 'larger',
        'smaller', or 'two-sided'
        """
        return self.hypothesis if self.hypothesis != "unequal" else "two-sided"

    @property
    def power(self) -> float:
        """
        The statistical power of the comparison (i.e. 1-beta of the comparison,
        where beta is the Type II error rate)
        """
        ratio = self.d1.nobs / self.d2.nobs

        solve_power_function = _get_solve_power_function(self.test_statistic_name)

        return solve_power_function(
            effect_size=self.effect_size,
            nobs1=self.d2.nobs,
            alpha=self.alpha,
            ratio=ratio,
            alternative=self.test_direction,
        )

    @property
    def comparison(self) -> None:
        """Displays a table summarizing the sample comparison comparison."""
        if self._comparison_table is None:
            self._comparison_table = SamplesComparisonTable(self.d1, self.d2)  # type: ignore
        self._comparison_table.print()  # type: ignore

    @property
    def t_test_stats(self) -> Dict[str, Any]:
        """Results for a t-test (small sample sizes)

        Returns
        -------
        test_stats : Dict[str, Any]
            The resulting test statistics, with the following structure:
            ```
            {
                "statistic_name": "t",
                "statistic_value": float,
                "p_value": float,
                "df": int,
                "hypothesis": str
            }
            ```
        """

        tstat, pval, degrees_freedom = self.ttest_ind(alternative=self.test_direction)
        return {
            "statistic_name": "t",
            "statistic_value": tstat,
            "degrees_freedom": degrees_freedom,
            "hypothesis": self.hypothesis,
            "p_value": pval,
            "alpha": self.alpha,
            "power": self.power,
        }

    @property
    def z_test_stats(self) -> Dict[str, Any]:
        """Results for a z-test (large sample sizes)

        Returns
        -------
        test_stats : Dict[str, Any]
            The resulting test statistics, with the following structure:
            ```
            {
                "statistic_name": "t",
                "statistic_value": float,
                "p_value": float,
                "df": int,
                "hypothesis": str
            }
            ```
        """
        tstat, pval = self.ztest_ind(alternative=self.test_direction)
        return {
            "statistic_name": "z",
            "statistic_value": tstat,
            "hypothesis": self.hypothesis,
            "p_value": pval,
            "alpha": self.alpha,
            "power": self.power,
        }


class ProportionComparison(MeanComparison):
    """
    Class for comparing the proportions of two sample distributions, provides a
    number of helpful summary statistics about the comparison. In order to use the
    z-distribution, we assume normality oF proportions and thus, by proxy, adequate
    sample sizes (e.g. N > 30).
    """

    def __init__(self, variance_assumption: str = "pooled", *args, **kwargs):
        """
        Parameters
        ----------
        samples_a : `Samples`
            Samples from group A (e.g. control group)
        samples_b : `Samples`
            Samples from group B (e.g. treatment group)
        alpha : float in (0, 1)
            The acceptable Type I error rate in the comparison
        hypothesis : str
            Defines the assumed alternative hypothesis. Can be either:
                -   'larger': indicating we hypothesize that group B's mean is
                    larger than group A's
                -   'smaller': indicating we hypothesize that group B's mean is
                    smaller than group A's
                -   'unequal': indicating we hypothesize that group B's mean is
                    not equal to group A's (i.e. two-tailed test).
        variance_assumption : str, optional
            The assumed variance assumption. If "pooled" (default), we calculate
            the pooled variance, otherwise we simply calculate the global
            variance across both samples
        """

        super().__init__(*args, **kwargs)
        self._test_statistic_name = "z"
        nobs = min(self.d1.nobs, self.d2.nobs)

        # to use Normal approx, must have "large" N
        if nobs < MIN_OBS_FOR_Z_TEST:
            warning = (
                "Normality assumption violated, > ",
                f"{MIN_OBS_FOR_Z_TEST} observations required.",
            )

            logger.warn(warning)
            self.warnings.append(warning)

        self.variance_assumption = variance_assumption

    @property
    def pooled_variance(self) -> float:
        if self.variance_assumption == "pooled":
            p1 = self.d1.mean
            p2 = self.d2.mean
            var1 = p1 * (1 - p1)
            var2 = p2 * (1 - p2)
            return ((self.d1.nobs - 1) * var1 + (self.d2.nobs - 1) * var2) / (
                self.d1.nobs + self.d2.nobs - 2
            )
        else:  # global variance
            p = np.mean(np.r_[self.d1.data, self.d2.data])
            return p * (1 - p)

    @property
    def z_test_stats(self) -> Dict[str, Any]:
        """Test statistics for proportions comparison on normal (z) test

        Returns
        -------
        test_stats: Dict[str, float]
            The resulting test statistics, with the following structure:
            ```
            {
                "statistic_name": str = "z",
                "statistic_value": float,
                "p_value": float,
                "hypothesis": str
            }
            ```
        """
        from statsmodels.stats.proportion import proportions_ztest

        prop_var = self.pooled_variance
        n_1 = self.d1.nobs
        s_1 = sum(self.d1.data)
        n_2 = self.d2.nobs
        s_2 = sum(self.d2.data)
        zstat, pval = proportions_ztest(
            [s_1, s_2], [n_1, n_2], alternative=self.test_direction, prop_var=prop_var
        )
        return {
            "statistic_name": "z",
            "statistic_value": zstat,
            "hypothesis": self.hypothesis,
            "p_value": pval,
            "alpha": self.alpha,
            "power": self.power,
        }


class RateComparison(MeanComparison):
    """
    Class for comparing the rates of two sample distributions, provides a number
    of helpful summary statistics about the comparison. Uses the exact conditional
    test based on binomial distribution, as described in Gu et al (2008)

    References
    ----------
    Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
    Biometrical Journal 50 (2008) 2, 2008
    """

    def __init__(self, null_ratio: float = 1.0, *args, **kwargs):
        """
        Parameters
        ----------
        samples_a : `Samples`
                Samples from group A (e.g. control group)
        samples_b : `Samples`
            Samples from group B (e.g. treatment group)
        alpha : float in (0, 1)
        hypothesis : str
                Defines the assumed alternative hypothesis. Can be either:
                    -   'larger': indicating we hypothesize that group B's mean is
                        larger than group A's
                    -   'smaller': indicating we hypothesize that group B's mean is
                        smaller than group A's
                    -   'unequal': indicating we hypothesize that group B's mean is
                        not equal to group A's (i.e. two-tailed test).

        null_ratio : float, optional
            The ratio of the two samples underneath the null hypothesis. Default
            is 1.0, i.e. that the two samples are equivalent, and thus their
            ratio is unity.
        """
        super().__init__(*args, **kwargs)
        self._test_statistic_name = "W"
        self.null_ratio = null_ratio

    @property
    def rates_ratio(self) -> float:
        """
        Return the comparison ratio of the null rates ratio and the observed
        rates ratio.
        """
        actual_ratio = (self.d1.sum * self.d1.nobs) / (self.d2.sum * self.d2.nobs)
        return self.null_ratio / actual_ratio

    @property
    def delta(self) -> float:
        """
        Delta is the ratio of the variation to the control rates
        """
        return self.d1.mean / self.d2.mean

    @property
    def delta_relative(self) -> float:
        """Return the athe ratio of the variation to the control rates"""
        return self.delta

    @property
    def rates_test_stats(self) -> Dict[str, Any]:
        """
        Run the rates comparison hyptothesis test. Uses the W5 statistic defined
        in Gu et al., 2008

        Returns
        -------
        test_stats : Dict[str, Any]
            The resulting test statistics, with the following structure:
            ```
            {
                "statistic_name": "W",
                "statistic_value": float,
                "p_value": float,
                "hypothesis": str
            }
            ```
            The (W-statistic, p-value) of the test, where W-statistic is the W5
            statistic from Gu et al., 2008.
        """
        X1, X2 = self.d2.sum, self.d1.sum
        t1, t2 = self.d2.nobs, self.d1.nobs
        d = t1 / t2
        W = (
            2
            * (
                np.sqrt(X2 + (3.0 / 8))
                - np.sqrt((self.null_ratio / d) * (X1 + (3.0 / 8)))
            )
            / np.sqrt(1 + (self.null_ratio / d))
        )

        if self.hypothesis == "larger":
            pval = 1 - norm.cdf(W)
        elif self.hypothesis == "smaller":
            pval = norm.cdf(W)
        elif self.hypothesis == "unequal":
            pval = 1 - norm.cdf(abs(W))

        return {
            "statistic_name": "W",
            "statistic_value": W,
            "hypothesis": self.hypothesis,
            "p_value": pval,
            "alpha": self.alpha,
            "power": self.power,
        }

    @property
    def effect_size(self) -> float:
        """
        Effect size ranges from 0-1
        """
        return 1 - self.rates_ratio

    @property
    def power(self) -> float:
        """
        Return the statistical power of the current test. Follows the calculation
        from W statistic 5 in Gu et al., 2008
        """
        N2, t2 = self.d1.sum, self.d1.nobs
        N1, t1 = self.d2.sum, self.d2.nobs

        lambda_2, lambda_1 = np.abs(self.d1.mean), np.abs(self.d2.mean)
        alternative_ratio = np.abs(lambda_2 / lambda_1)
        z = norm.ppf(1 - self.alpha)

        d = float(t1 * N1) / (t2 * N2)

        A = np.abs(2.0 * (1.0 - np.sqrt(self.null_ratio / alternative_ratio)))
        B = np.sqrt(lambda_1 * t1 * N1 + (3.0 / 8))
        C = np.sqrt((self.null_ratio + d) / alternative_ratio)
        D = np.sqrt((alternative_ratio + d) / alternative_ratio)
        W = (A * B - z * C) / D

        return round(norm.cdf(W), 4)


class BootstrapStatisticComparison(MeanComparison):
    """
    Class for comparing a bootstrapped test statistic for two samples. Provides
    a number of helpful summary statistics about the comparison.

    References
    ----------
    Efron, B. (1981). "Nonparametric estimates of standard error: The jackknife,
    the bootstrap and other methods". Biometrika. 68 (3): 589-599
    """

    def __init__(
        self,
        n_bootstraps: int = 1000,
        statistic_function: Callable = np.mean,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        samples_a : `Samples`
            Samples from group A (e.g. control group)
        samples_b : `Samples`
            Samples from group B (e.g. treatment group)
        alpha : float in (0, 1)
            The acceptable Type I error rate in the comparison
        hypothesis : str
            Defines the assumed alternative hypothesis. Can be either:
                -   'larger': indicating we hypothesize that group B's mean is
                    larger than group A's
                -   'smaller': indicating we hypothesize that group B's mean is
                    smaller than group A's
                -   'unequal': indicating we hypothesize that group B's mean is
                    not equal to group A's (i.e. two-tailed test).

        n_bootstraps : int
            The number of bootstrap samples to draw use for estimates.
        statistic_function : Callable, optional
            Function that returns a scalar test statistic when provided a sequence
            of samples. Defaults to the mean.
        *args, **kwargs:
            Valid arguments to `MeanComparison`
        """

        super().__init__(*args, **kwargs)

        self.n_bootstraps = n_bootstraps
        self.statistic_function = statistic_function

    @property
    def test_statistic_name(self) -> str:
        return f"bootstrap_{self.statistic_function.__name__}"

    @property
    def bootstrap_test_stats(self) -> Dict[str, Any]:
        """
        Run the sample comparison hyptothesis test. Uses the bootstrapped sample statistics

        Returns
        -------
        test_stats: Dict[str, Any]
            The resulting test statistics, with the following structure:
            ```
            {
                "statistic_name": "bootstrap_delta",
                "statistic_value": float,
                "statistic_function_name": str,
                "n_bootstraps": int,
                "p_value": float,
                "hypothesis": str
            }
            ```
        """
        all_samples = np.concatenate([self.d1.data, self.d2.data]).astype(float)

        d1_samples = np.random.choice(
            all_samples, (int(self.d1.nobs), self.n_bootstraps), replace=True
        )
        d1_statistics = np.apply_along_axis(
            self.statistic_function, axis=0, arr=d1_samples
        )

        d2_samples = np.random.choice(
            all_samples, (int(self.d2.nobs), self.n_bootstraps), replace=True
        )
        d2_statistics = np.apply_along_axis(
            self.statistic_function, axis=0, arr=d2_samples
        )

        control_bs_samples = np.random.choice(
            self.d2.data, (int(self.d2.nobs), self.n_bootstraps), replace=True
        )
        control_statistics = np.apply_along_axis(
            self.statistic_function, axis=0, arr=control_bs_samples
        )
        self.control_bootstrap = Samples(control_statistics, name="control")

        variation_bs_samples = np.random.choice(
            self.d1.data, (int(self.d1.nobs), self.n_bootstraps), replace=True
        )
        variation_statistics = np.apply_along_axis(
            self.statistic_function, axis=0, arr=variation_bs_samples
        )
        self.variation_bootstrap = Samples(variation_statistics, name="variation")

        # The null sampling distribution of test_statistic deltas
        self.null_dist = Samples(
            d2_statistics - d1_statistics, name=f"{self.test_statistic_name}-null"
        )

        if self.hypothesis == "larger":
            pval = 1 - self.null_dist.cdf(self.delta)  # type: ignore
        elif self.hypothesis == "smaller":
            pval = self.null_dist.cdf(self.delta)  # type: ignore
        elif self.hypothesis == "unequal":
            pval = 1 - self.null_dist.cdf(abs(self.delta))  # type: ignore

        return {
            "statistic_name": self.test_statistic_name,
            "statistic_value": self.delta,
            "statistic_function_name": self.statistic_function.__name__,
            "n_bootstraps": self.n_bootstraps,
            "hypothesis": self.hypothesis,
            "p_value": pval,
            "alpha": self.alpha,
            "power": self.power,
        }

    def delta_ci(self) -> Tuple:
        return self.confidence_interval(1 - self.alpha)

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate the `confidence`-% confidence interval around the statistic delta.
        Uses bootstrapped approximation the statistic sampling distribution.

        Returns
        -------
        ci : Tuple[float, float]
            the `confidence`-% confidence interval around the statistic estimate.
        """
        alpha = 1 - confidence

        return self.deltas_dist.percentiles([100 * alpha, 100 * (1 - alpha)])  # type: ignore

    @property
    def deltas_dist(self) -> Samples:
        if not hasattr(self, "_deltas_dist"):
            d1_samples = np.random.choice(
                self.d1.data, (int(self.d1.nobs), self.n_bootstraps), replace=True
            )
            d1_statistics = np.apply_along_axis(
                self.statistic_function, axis=0, arr=d1_samples
            )

            d2_samples = np.random.choice(
                self.d2.data, (int(self.d2.nobs), self.n_bootstraps), replace=True
            )
            d2_statistics = np.apply_along_axis(
                self.statistic_function, axis=0, arr=d2_samples
            )

            self._deltas_dist = Samples(
                d1_statistics - d2_statistics, name=f"{self.test_statistic_name}-deltas"
            )

        return self._deltas_dist

    @property
    def delta(self) -> float:
        """
        Return the average difference in test statistic distributions
        """
        return self.deltas_dist.mean

    @property
    def delta_relative(self) -> float:
        """Return the average difference in test statistic distributions, as
        a percent change.
        """
        return self.delta / np.abs(self.statistic_function(self.d2.data))

    @property
    def power(self) -> float:
        """
        Return the statistical power of the current test.
        """

        # Need to run the bootstrap in order to obtain the power
        if not hasattr(self, "null_dist"):
            _ = self.bootstrap_test_stats

        critical_value = self.null_dist.percentiles(100.0 * (1 - self.alpha))  # type: ignore
        return self.deltas_dist.prob_greater_than(critical_value)  # type: ignore
