import pytest

from spearmint import Experiment, HypothesisTest
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def counts_data():
    return generate_fake_observations(
        distribution="poisson", n_treatments=6, n_observations=6 * 50, random_seed=123
    )


def test_rates_ratio_larger(counts_data):
    exp = Experiment(data=counts_data)
    ab_test = HypothesisTest(
        metric="metric",
        hypothesis="larger",
        control="A",
        variation="C",
        inference_method="frequentist",
        variable_type="counts",
    )
    results_ab = exp.run_test(ab_test)
    results_ab.display()
    assert results_ab.accept_hypothesis


def test_rates_ratio_smaller(counts_data):
    exp = Experiment(data=counts_data)
    ab_test = HypothesisTest(
        metric="metric",
        hypothesis="smaller",
        control="A",
        variation="C",
        inference_method="frequentist",
        variable_type="counts",
    )
    results_ab = exp.run_test(ab_test)
    results_ab.display()
    assert not results_ab.accept_hypothesis


def test_rates_ratio_unequal(counts_data):
    exp = Experiment(data=counts_data)
    ab_test = HypothesisTest(
        metric="metric",
        hypothesis="unequal",
        control="A",
        variation="C",
        inference_method="frequentist",
        variable_type="counts",
    )
    results_ab = exp.run_test(ab_test)
    results_ab.display()
    assert results_ab.accept_hypothesis


def test_rates_ratio_aa(counts_data):
    exp = Experiment(data=counts_data)
    aa_test = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        inference_method="frequentist",
        variable_type="counts",
    )
    results_aa = exp.run_test(aa_test)
    results_aa.display()
    assert not results_aa.accept_hypothesis


def test_rates_ratio_default(counts_data):
    exp = Experiment(data=counts_data)
    ab_test = HypothesisTest(metric="metric", control="A", variation="C")
    results_ab = exp.run_test(ab_test)
    results_ab.display()
    assert results_ab.accept_hypothesis
    assert ab_test.inference_method == "frequentist"
    assert ab_test.variable_type == "counts"
