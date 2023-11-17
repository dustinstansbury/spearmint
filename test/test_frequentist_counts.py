import pytest

from spearmint import Experiment, HypothesisTest
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def rates_data():
    return generate_fake_observations(
        distribution="poisson", n_treatments=6, n_observations=6 * 50, random_seed=123
    )


def test_rates_ratio_larger(rates_data):
    exp = Experiment(data=rates_data)
    ab_test = HypothesisTest(
        metric="metric",
        hypothesis="larger",
        control="A",
        variation="C",
        inference_method="frequentist",
        variable_type="counts",
    )
    ab_results = exp.run_test(ab_test)
    assert ab_results.accept_hypothesis


def test_rates_ratio_smaller(rates_data):
    exp = Experiment(data=rates_data)
    ab_test = HypothesisTest(
        metric="metric",
        hypothesis="smaller",
        control="A",
        variation="C",
        inference_method="frequentist",
        variable_type="counts",
    )
    ab_results = exp.run_test(ab_test)
    assert not ab_results.accept_hypothesis


def test_rates_ratio_unequal(rates_data):
    exp = Experiment(data=rates_data)
    ab_test = HypothesisTest(
        metric="metric",
        hypothesis="unequal",
        control="A",
        variation="C",
        inference_method="frequentist",
        variable_type="counts",
    )
    ab_results = exp.run_test(ab_test)
    assert ab_results.accept_hypothesis


def test_rates_ratio_aa(rates_data):
    exp = Experiment(data=rates_data)
    aa_test = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        inference_method="frequentist",
        variable_type="counts",
    )
    aa_results = exp.run_test(aa_test)
    assert not aa_results.accept_hypothesis
