import numpy as np
import pytest

from spearmint import Experiment, HypothesisTest
from spearmint.utils import generate_fake_observations


def custom_statistic(values) -> float:
    """Silly statistic functino for testing custom boostraps"""
    return np.mean(values) ** 2


@pytest.fixture()
def proportions_data():
    return generate_fake_observations(
        distribution="bernoulli",
        n_treatments=6,
        n_attributes=1,
        n_observations=6 * 1000,
        random_seed=123,
    )


def test_small_default_bootstrap_unequal_ab_test(proportions_data):
    exp = Experiment(proportions_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="B",
        hypothesis="unequal",
        inference_method="bootstrap",
    )
    results_ab = exp.run_test(test_ab)

    assert results_ab.test_statistic_name == "bootstrap_mean"
    assert results_ab.accept_hypothesis


def test_small_default_bootstrap_unequal_aa_test(proportions_data):
    exp = Experiment(proportions_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        hypothesis="unequal",
        inference_method="bootstrap",
    )

    results_ab = exp.run_test(test_ab)

    assert results_ab.test_statistic_name == "bootstrap_mean"
    assert not results_ab.accept_hypothesis


def test_small_default_bootstrap_smaller_ab_test(proportions_data):
    exp = Experiment(proportions_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="D",
        hypothesis="smaller",
        inference_method="bootstrap",
    )
    results_ab = exp.run_test(test_ab)

    assert not results_ab.accept_hypothesis


def test_small_bootstrap_larger_ab_test(proportions_data):
    exp = Experiment(proportions_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="D",
        hypothesis="smaller",
        inference_method="bootstrap",
    )
    results_ab = exp.run_test(test_ab)

    assert not results_ab.accept_hypothesis


def test_small_custom_statistic_bootstrap_ab_test(proportions_data):
    exp = Experiment(proportions_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="D",
        hypothesis="larger",
        inference_method="bootstrap",
        statistic_function=custom_statistic,
    )
    results_ab = exp.run_test(test_ab)

    assert results_ab.test_statistic_name == "bootstrap_custom_statistic"
    assert results_ab.accept_hypothesis


def test_small_custom_statistic_bootstrap_smaller_ab_test(proportions_data):
    exp = Experiment(proportions_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="D",
        hypothesis="smaller",
        inference_method="bootstrap",
        statistic_function=custom_statistic,
    )
    results_ab = exp.run_test(test_ab)

    assert not results_ab.accept_hypothesis


@pytest.mark.filterwarnings("ignore")
def test_small_custom_statistic_bootstrap_aa_test(proportions_data):
    exp = Experiment(proportions_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        hypothesis="unequal",
        inference_method="bootstrap",
        statistic_function=custom_statistic,
    )
    # AA test results in division by zero
    results_ab = exp.run_test(test_ab)

    assert results_ab.test_statistic_name == "bootstrap_custom_statistic"
    assert not results_ab.accept_hypothesis
