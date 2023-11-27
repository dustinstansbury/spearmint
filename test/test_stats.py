import numpy as np
import pytest

from spearmint import stats

TEST_SAMPLES_NAME = "TestSamples"
TEST_CONTINUOUS_SAMPLE_VARIANCE = 1.0
TEST_BINARY_SAMPLE_VARIANCE = 0.5**2
TEST_COUNT_SAMPLE_RATE = 10


@pytest.fixture
def continuous_test_samples():
    np.random.seed(123)
    return stats.Samples(
        np.random.randn(1000) * TEST_CONTINUOUS_SAMPLE_VARIANCE**0.5,
        name=TEST_SAMPLES_NAME,
    )


@pytest.fixture
def binary_test_samples():
    np.random.seed(123)
    return stats.Samples(
        np.random.rand(1000),
        name=TEST_SAMPLES_NAME,
    )


@pytest.fixture
def count_test_samples():
    np.random.seed(123)
    return stats.Samples(
        np.random.poisson(TEST_COUNT_SAMPLE_RATE, 1000),
        name=TEST_SAMPLES_NAME,
    )


def test_bonferroni_correction():
    p_vals_orig = [0.05, 0.05]
    alpha_orig = 0.05
    corrected = stats.bonferroni_correction(alpha_orig, p_vals_orig)
    assert corrected < alpha_orig


def test_sidak_correction():
    p_vals_orig = [0.05, 0.05]
    alpha_orig = 0.05
    corrected = stats.sidak_correction(alpha_orig, p_vals_orig)
    assert corrected < alpha_orig


def test_fdr_bh_correction():
    p_vals_orig = [0.5, 0.5]
    fdr_orig = 0.05
    corrected = stats.fdr_bh_correction(fdr_orig, p_vals_orig)
    assert corrected < fdr_orig

    p_vals_orig = [0.05, 0.05]
    corrected = stats.fdr_bh_correction(fdr_orig, p_vals_orig)
    assert corrected == fdr_orig


def test_multiple_comparison_correction():
    p_values = np.arange(0.001, 0.1, 0.01)
    mc = stats.MultipleComparisonCorrection(p_values, method="b")

    assert mc.alpha_corrected < mc.alpha_orig
    assert mc.accept_hypothesis[0]
    assert not mc.accept_hypothesis[-1]

    with pytest.raises(ValueError):
        stats.MultipleComparisonCorrection(p_values, method="unknown")


def test_estimate_experiment_sample_sizes_z():
    prob_control = 0.49
    std_control = (prob_control * (1 - prob_control)) ** 0.5  # Binomial std
    prob_variation = std_variation = 0.50
    delta = prob_variation - prob_control
    assert stats.estimate_experiment_sample_sizes(
        delta=delta, statistic="z", std_control=std_control, std_variation=std_variation
    ) == (39236, 39236)


def test_estimate_experiment_sample_sizes_t():
    prob_control = 0.49
    std_control = (prob_control * (1 - prob_control)) ** 0.5  # Binomial std
    prob_variation = std_variation = 0.50
    delta = prob_variation - prob_control
    assert stats.estimate_experiment_sample_sizes(
        delta=delta, statistic="t", std_control=std_control, std_variation=std_variation
    ) == (39237, 39237)


def test_estimate_experiment_sample_sizes_ratio():
    # Replicate Example 1 from Gu et al, 2008
    R = 4  # ratio under alternative hypothesis
    control_rate = 0.0005
    variation_rate = R * control_rate
    delta = variation_rate - control_rate
    assert stats.estimate_experiment_sample_sizes(
        delta,
        statistic="rates_ratio",
        control_rate=control_rate,
        alpha=0.05,
        power=0.9,
        control_exposure_time=2.0,
        sample_size_ratio=0.5,
    ) == (8590, 4295)


def test_estimate_experiment_sample_sizes_unknown():
    with pytest.raises(ValueError):
        stats.estimate_experiment_sample_sizes(delta=None, statistic="unknown")


def test_cohens_d_sample_size_unknown_statistic():
    with pytest.raises(ValueError):
        stats.cohens_d_sample_size(
            delta=1, alpha=0.05, power=0.8, std_control=1, statistic="unknown"
        )


def test_highest_density_interval():
    samples = np.random.randn(1000)
    hdi = stats.highest_density_interval(samples, credible_mass=0.9)
    assert hdi[0] >= -1.96  # test lower HDI for 95% hdi
    assert hdi[1] <= 1.96  # test upper HDI for 95% hdi
    with pytest.raises(ValueError):
        # Not enough samples
        stats.highest_density_interval([1], 1.0)


def test_empirical_cdf_class(continuous_test_samples):
    observations = continuous_test_samples._raw_observations
    ecdf = stats.EmpiricalCdf(observations)
    assert ecdf.samples_cdf[-1] == 1.0

    # __call__
    assert np.all(ecdf([-100, 100]) == np.array([0.0, 1.0]))

    # evaluate
    assert np.all(ecdf.evaluate([-100, 100]) == np.array([0.0, 1.0]))
    assert len(ecdf.evaluate()) == len(observations)


def test_samples(continuous_test_samples):
    assert continuous_test_samples.mean <= 0.01
    assert continuous_test_samples.std - 1 <= 0.01
    assert not np.all(
        continuous_test_samples.permute() == continuous_test_samples.permute()
    )
    assert np.all(
        continuous_test_samples.sort()
        == sorted(continuous_test_samples._raw_observations)
    )
    continuous_test_samples.summary  # print the table
    assert len(continuous_test_samples._summary_table._print_history) > 0


def test_samples_summary_table(continuous_test_samples):
    samples_table = stats.SamplesSummaryTable(samples=continuous_test_samples)
    samples_table.print()
    printed_lines = samples_table._print_history[-1].split("\n")

    assert "Samples Summary" in printed_lines[0]
    assert TEST_SAMPLES_NAME in printed_lines[2]
    assert "Samples" in printed_lines[4]
    assert "Mean" in printed_lines[5]
    assert "Standard Error" in printed_lines[6]
    assert "Variance" in printed_lines[7]


def test_continuous_samples_comparison_table(continuous_test_samples):
    control_samples = continuous_test_samples
    variation_samples = continuous_test_samples

    # Single variation instance
    samples_table = stats.SamplesComparisonTable(
        control_samples=control_samples, variation_samples=variation_samples
    )
    samples_table.print()
    printed_lines = samples_table._print_history[-1].split("\n")

    assert "Samples Comparison" in printed_lines[0]
    assert TEST_SAMPLES_NAME in printed_lines[2]
    assert "Mean" in printed_lines[5]
    assert "Standard Error" in printed_lines[6]
    assert "Variance" in printed_lines[7]
    assert "Delta" in printed_lines[8]

    # Multiple variation instance
    samples_table = stats.SamplesComparisonTable(
        control_samples=control_samples,
        variation_samples=[variation_samples, variation_samples],
    )
    samples_table.print()
    printed = samples_table._print_history[-1]
    assert "Samples Comparison" in printed
    assert TEST_SAMPLES_NAME in printed


def test_mean_comparison(continuous_test_samples):
    # Same samples, so
    means_comparison = stats.MeanComparison(
        samples_a=continuous_test_samples, samples_b=continuous_test_samples
    )
    means_comparison.comparison  # test printed summary table
    assert len(means_comparison._comparison_table._print_history) > 0
    assert means_comparison.delta == 0
    assert means_comparison.alpha == stats.DEFAULT_ALPHA
    assert means_comparison.test_direction == stats.DEFAULT_HYPOTHESIS
    assert np.isclose(
        means_comparison.pooled_variance, TEST_CONTINUOUS_SAMPLE_VARIANCE, atol=0.01
    )
    assert means_comparison.delta_relative == 0
    assert means_comparison.effect_size == 0
    # overlapping distributions, power should be similar to false positive rate
    assert np.isclose(means_comparison.power, means_comparison.alpha, atol=0.001)
    t_test_stats = means_comparison.t_test_stats
    assert t_test_stats["statistic_name"] == "t"
    assert t_test_stats["statistic_value"] == 0
    assert t_test_stats["p_value"] == 0.5
    assert t_test_stats["alpha"] == stats.DEFAULT_ALPHA
    assert np.isclose(t_test_stats["power"], t_test_stats["alpha"], atol=0.001)

    assert (
        t_test_stats["degrees_freedom"]
        == (continuous_test_samples.nobs + continuous_test_samples.nobs) - 2
    )


def test_proportion_comparison(binary_test_samples):
    # Same samples
    proportion_comparison = stats.ProportionComparison(
        samples_a=binary_test_samples, samples_b=binary_test_samples
    )
    assert proportion_comparison.delta == 0
    assert np.isclose(
        proportion_comparison.pooled_variance, TEST_BINARY_SAMPLE_VARIANCE, atol=0.01
    )
    assert proportion_comparison.delta_relative == 0
    assert proportion_comparison.effect_size == 0
    # overlapping distributions, power should be similar to false positive rate
    assert np.isclose(
        proportion_comparison.power, proportion_comparison.alpha, atol=0.001
    )

    z_test_stats = proportion_comparison.z_test_stats
    assert z_test_stats["statistic_name"] == "z"
    assert z_test_stats["statistic_value"] == 0
    assert z_test_stats["p_value"] == 0.5
    assert z_test_stats["alpha"] == stats.DEFAULT_ALPHA
    assert np.isclose(z_test_stats["power"], z_test_stats["alpha"], atol=0.001)


def test_rates_comparison(count_test_samples):
    # Same samples
    rate_comparison = stats.RateComparison(
        samples_a=count_test_samples, samples_b=count_test_samples
    )

    assert rate_comparison.delta == 1.0
    assert rate_comparison.delta_relative == 1.0
    assert np.isclose(rate_comparison.pooled_variance, TEST_COUNT_SAMPLE_RATE, atol=0.1)
    assert rate_comparison.effect_size == 0
    # overlapping distributions, power should be similar to false positive rate
    assert np.isclose(rate_comparison.power, rate_comparison.alpha, atol=0.001)

    rates_test_stats = rate_comparison.rates_test_stats
    assert rates_test_stats["statistic_name"] == "W"
    assert rates_test_stats["statistic_value"] == 0
    assert rates_test_stats["p_value"] == 0.5
    assert rates_test_stats["alpha"] == stats.DEFAULT_ALPHA
    assert np.isclose(rates_test_stats["power"], rates_test_stats["alpha"], atol=0.001)


def test_bootstrap_comparison(continuous_test_samples):
    # Same samples
    bs_comparison = stats.BootstrapStatisticComparison(
        samples_a=continuous_test_samples,
        samples_b=continuous_test_samples,
        n_bootstraps=1000,
    )

    assert np.isclose(bs_comparison.delta, 0.0, atol=0.001)
    assert np.isclose(bs_comparison.delta_relative, 0.0, atol=0.05)
    assert np.isclose(
        bs_comparison.pooled_variance, TEST_CONTINUOUS_SAMPLE_VARIANCE, atol=0.1
    )
    assert np.isclose(bs_comparison.effect_size, 0.0, atol=0.001)

    # overlapping distributions, power should be similar to false positive rate
    assert np.isclose(bs_comparison.power, bs_comparison.alpha, atol=0.05)

    bootstrap_test_stats = bs_comparison.bootstrap_test_stats
    assert bootstrap_test_stats["statistic_function_name"] == "mean"
    assert bootstrap_test_stats["statistic_name"] == "bootstrap_mean"
    assert np.isclose(bootstrap_test_stats["statistic_value"], 0, atol=0.001)
    assert np.isclose(bootstrap_test_stats["p_value"], 0.5, atol=0.05)
    assert bootstrap_test_stats["alpha"] == stats.DEFAULT_ALPHA
    assert np.isclose(
        bootstrap_test_stats["power"], bootstrap_test_stats["alpha"], atol=0.01
    )
