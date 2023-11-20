import pytest

from spearmint import Experiment, HypothesisTest
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def binary_data():
    return generate_fake_observations(
        distribution="bernoulli",
        n_treatments=6,
        n_attributes=1,
        n_observations=6 * 1000,
        random_seed=123,
    )


def test_large_proportions_delta_expermiment(binary_data):
    exp = Experiment(binary_data)

    # run 'A/A' test
    test_aa = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        hypothesis="larger",
        inference_method="frequentist",
        variable_type="binary",
    )
    results_aa = exp.run_test(test_aa)

    results_aa.display()
    assert results_aa.test_statistic_name == "z"
    assert not results_aa.accept_hypothesis

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="B",
        hypothesis="larger",
        inference_method="frequentist",
        variable_type="binary",
    )
    results_ab = exp.run_test(test_ab)
    results_ab.display()

    assert results_ab.test_statistic_name == "z"
    assert results_ab.accept_hypothesis


def test_proportions_delta_ab_unequal(binary_data):
    exp = Experiment(binary_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="F",
        hypothesis="unequal",
        inference_method="frequentist",
        variable_type="binary",
    )
    results_ab = exp.run_test(test_ab)

    results_ab.display()
    assert results_ab.test_statistic_name == "z"
    assert results_ab.accept_hypothesis


def test_proportions_delta_ab_larger(binary_data):
    exp = Experiment(binary_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="F",
        hypothesis="larger",
        inference_method="frequentist",
        variable_type="binary",
    )
    results_ab = exp.run_test(test_ab)
    results_ab.display()
    assert results_ab.accept_hypothesis


def test_proportions_delta_ab_smaller(binary_data):
    exp = Experiment(binary_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="F",
        hypothesis="smaller",
        inference_method="frequentist",
        variable_type="binary",
    )
    results_ab = exp.run_test(test_ab)
    results_ab.display()
    assert not results_ab.accept_hypothesis


def test_proportions_delta_aa(binary_data):
    exp = Experiment(binary_data)

    # run A/A test
    test_aa = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        hypothesis="larger",
        inference_method="frequentist",
        variable_type="binary",
    )
    results_aa = exp.run_test(test_aa)
    results_aa.display()
    assert not results_aa.accept_hypothesis


def test_proportions_delta_default(binary_data):
    exp = Experiment(binary_data)

    # run A/B test
    test_ab = HypothesisTest(metric="metric", control="A", variation="F")
    results_ab = exp.run_test(test_ab)
    results_ab.display()
    assert results_ab.accept_hypothesis
    assert test_ab.inference_method == "frequentist"
    assert test_ab.variable_type == "binary"
    assert results_ab.accept_hypothesis
