import pytest

from spearmint import Experiment, HypothesisTest
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def continuous_data():
    return generate_fake_observations(
        distribution="gaussian", n_treatments=6, n_observations=6 * 50, random_seed=123
    )


def test_means_delta_experiment_t(continuous_data):
    """Small sample sizes defautl to t-tests"""
    exp = Experiment(continuous_data.sample(29))

    test_aa = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        hypothesis="unequal",
        inference_method="frequentist",
        variable_type="continuous",
    )
    results_aa = exp.run_test(test_aa)
    results_aa.display()
    assert results_aa.test_statistic_name == "t"


def test_means_delta_experiment_unequal_ab(continuous_data):
    exp = Experiment(continuous_data)

    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="F",
        hypothesis="unequal",
        inference_method="frequentist",
        variable_type="continuous",
    )
    results_ab = exp.run_test(test_ab)
    results_ab.display()
    assert results_ab.test_statistic_name == "z"
    assert results_ab.accept_hypothesis


def test_means_delta_experiment_larger_ab(continuous_data):
    exp = Experiment(continuous_data)

    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="F",
        hypothesis="larger",
        inference_method="frequentist",
        variable_type="continuous",
    )
    results_ab = exp.run_test(test_ab)
    results_ab.display()
    assert results_ab.test_statistic_name == "z"
    assert results_ab.accept_hypothesis


def test_means_delta_experiment_smaller_ab(continuous_data):
    exp = Experiment(continuous_data)

    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="F",
        hypothesis="smaller",
        inference_method="frequentist",
        variable_type="continuous",
    )
    results_ab = exp.run_test(test_ab)
    results_ab.display()
    assert results_ab.test_statistic_name == "z"
    assert not results_ab.accept_hypothesis


def test_means_delta_experiment_aa(continuous_data):
    exp = Experiment(continuous_data)

    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        hypothesis="larger",
        inference_method="frequentist",
        variable_type="continuous",
    )
    results_ab = exp.run_test(test_ab)
    results_ab.display()
    assert results_ab.test_statistic_name == "z"
    assert not results_ab.accept_hypothesis


def test_means_delta_default(continuous_data):
    exp = Experiment(continuous_data)

    test_ab = HypothesisTest(metric="metric", control="A", variation="B")
    results_ab = exp.run_test(test_ab)
    results_ab.display()
    assert results_ab.accept_hypothesis
    assert test_ab.inference_method == "frequentist"
    assert test_ab.variable_type == "continuous"
    assert results_ab.accept_hypothesis
