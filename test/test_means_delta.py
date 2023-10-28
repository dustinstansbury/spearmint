import pytest

from spearmint import Experiment, HypothesisTest
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def means_data():
    return generate_fake_observations(
        distribution="gaussian", n_treatments=6, n_observations=6 * 50, random_seed=123
    )


def test_means_delta_experiment_t(means_data):
    """Small sample sizes defautl to t-tests"""
    exp = Experiment(means_data.sample(29))

    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        hypothesis="unequal",
        inference_method="means_delta",
    )
    results_ab = exp.run_test(test_ab)
    assert results_ab.test_statistic_name == "t"


def test_means_delta_experiment_unequal_ab(means_data):
    exp = Experiment(means_data)

    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="F",
        hypothesis="unequal",
        inference_method="means_delta",
    )
    results_ab = exp.run_test(test_ab)
    assert results_ab.test_statistic_name == "z"
    assert results_ab.accept_hypothesis


def test_means_delta_experiment_larger_ab(means_data):
    exp = Experiment(means_data)

    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="F",
        hypothesis="larger",
        inference_method="means_delta",
    )
    results_ab = exp.run_test(test_ab)
    assert results_ab.test_statistic_name == "z"
    assert results_ab.accept_hypothesis


def test_means_delta_experiment_smaller_ab(means_data):
    exp = Experiment(means_data)

    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="F",
        hypothesis="smaller",
        inference_method="means_delta",
    )
    results_ab = exp.run_test(test_ab)
    assert results_ab.test_statistic_name == "z"
    assert not results_ab.accept_hypothesis


def test_means_delta_experiment_aa(means_data):
    exp = Experiment(means_data)

    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        hypothesis="larger",
        inference_method="means_delta",
    )
    results_ab = exp.run_test(test_ab)
    assert results_ab.test_statistic_name == "z"
    assert not results_ab.accept_hypothesis
