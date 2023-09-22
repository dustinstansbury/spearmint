import pytest
import numpy as np
from spearmint.stats import (
    Samples,
    MultipleComparisonCorrection,
    EmpiricalCdf,
    ProportionComparison,
)
from abra import stats


def test_multiple_comparison():
    p_values = np.arange(0.001, 0.1, 0.01)
    mc = MultipleComparisonCorrection(p_values, method="b")

    assert mc.alpha_corrected < mc.alpha_orig
    assert mc.accept_hypothesis[0]
    assert not mc.accept_hypothesis[-1]

    with pytest.raises(ValueError):
        MultipleComparisonCorrection(p_values, method="unknown")


def test_highest_density_interval():
    samples = np.random.randn(1000)
    hdi = stats.highest_density_interval(samples)
    with pytest.raises(ValueError):
        hdi = stats.highest_density_interval([1], 1.0)


def test_bonferroni_correction():
    p_vals_orig = [0.05, 0.05]
    alpha_orig = 0.05
    corrected = stats.bonferroni(alpha_orig, p_vals_orig)
    assert corrected < alpha_orig


def test_sidak_correction():
    p_vals_orig = [0.05, 0.05]
    alpha_orig = 0.05
    corrected = stats.sidak(alpha_orig, p_vals_orig)
    assert corrected < alpha_orig


def test_fdr_bh_correction():
    p_vals_orig = [0.5, 0.5]
    fdr_orig = 0.05
    corrected = stats.fdr_bh(fdr_orig, p_vals_orig)
    assert corrected < fdr_orig

    p_vals_orig = [0.05, 0.05]
    corrected = stats.fdr_bh(fdr_orig, p_vals_orig)
    assert corrected == fdr_orig


def test_estimate_experiment_sample_sizes_z():
    prob_control = 0.49
    std_control = (prob_control * (1 - prob_control)) ** 0.5  # Binomial std
    prob_variation = std_variation = 0.50
    delta = prob_variation - prob_control
    assert stats.estimate_experiment_sample_sizes(
        delta=delta, statistic="z", std_control=std_control, std_variation=std_variation
    ) == [39236, 39236]


def test_estimate_experiment_sample_sizes_t():
    prob_control = 0.49
    std_control = (prob_control * (1 - prob_control)) ** 0.5  # Binomial std
    prob_variation = std_variation = 0.50
    delta = prob_variation - prob_control
    assert stats.estimate_experiment_sample_sizes(
        delta=delta, statistic="t", std_control=std_control, std_variation=std_variation
    ) == [39237, 39237]


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
    ) == [8590, 4295]


def test_estimate_experiment_sample_sizes_unknown():
    with pytest.raises(ValueError):
        stats.estimate_experiment_sample_sizes(delta=None, statistic="unknown")


def test_cohens_d_sample_size_unknown_statistic():
    with pytest.raises(ValueError):
        stats.cohens_d_sample_size(
            delta=1, alpha=0.05, power=0.8, std_control=1, statistic="unknown"
        )


def test_empirical_cdf():
    # Standard Normal samples
    samples = np.random.randn(1000)
    ecdf = EmpiricalCdf(samples)

    assert ecdf.samples_cdf[-1] == 1.0

    # __call__
    assert np.all(ecdf([-100, 100]) == np.array([0.0, 1.0]))

    # evaluate
    assert np.all(ecdf.evaluate([-100, 100]) == np.array([0.0, 1.0]))
    assert len(ecdf.evaluate()) == len(samples)


def test_samples():
    np.random.seed(123)
    samples_ = np.random.randn(10000)
    samples = Samples(samples_)
    assert samples.mean <= 0.01
    assert samples.std - 1 <= 0.01
    assert not np.all(samples.permute() == samples.permute())
    assert np.all(samples.sort() == sorted(samples_))
