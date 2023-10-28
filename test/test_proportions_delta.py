import pytest

from spearmint import Experiment, HypothesisTest
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def proportions_data():
    return generate_fake_observations(
        distribution="bernoulli",
        n_treatments=6,
        n_attributes=1,
        n_observations=6 * 1000,
        random_seed=123,
    )


def test_large_proportions_delta_expermiment(proportions_data):
    exp = Experiment(proportions_data)

    # run 'A/A' test
    test_aa = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        hypothesis="larger",
        inference_method="proportions_delta",
    )
    results_aa = exp.run_test(test_aa)

    assert results_aa.test_statistic_name == "z"
    assert not results_aa.accept_hypothesis

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="B",
        hypothesis="larger",
        inference_method="proportions_delta",
    )
    results_ab = exp.run_test(test_ab)

    assert results_ab.test_statistic_name == "z"
    assert results_ab.accept_hypothesis


def test_proportions_delta_ab_unequal(proportions_data):
    exp = Experiment(proportions_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="F",
        hypothesis="unequal",
        inference_method="proportions_delta",
    )
    results_ab = exp.run_test(test_ab)

    assert results_ab.test_statistic_name == "z"
    assert results_ab.accept_hypothesis


def test_proportions_delta_ab_larger(proportions_data):
    exp = Experiment(proportions_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="F",
        hypothesis="larger",
        inference_method="proportions_delta",
    )
    results_ab = exp.run_test(test_ab)
    assert results_ab.accept_hypothesis


def test_proportions_delta_ab_smaller(proportions_data):
    exp = Experiment(proportions_data)

    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="F",
        hypothesis="smaller",
        inference_method="proportions_delta",
    )
    results_ab = exp.run_test(test_ab)
    assert not results_ab.accept_hypothesis


def test_proportions_delta_aa(proportions_data):
    exp = Experiment(proportions_data)

    # run A/A test
    test_aa = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        hypothesis="larger",
        inference_method="proportions_delta",
    )
    results_aa = exp.run_test(test_aa)
    assert not results_aa.accept_hypothesis
