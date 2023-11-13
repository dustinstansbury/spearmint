import pytest
from spearmint import Experiment, HypothesisTest
from spearmint.inference.bayesian.bayesian_inference import (
    UnsupportedParameterEstimationMethod,
)
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def counts_data():
    return generate_fake_observations(
        distribution="poisson", n_treatments=3, n_observations=3 * 100
    )


def test_bayesian_poisson_ab_mcmc(counts_data):
    exp = Experiment(data=counts_data)

    test = HypothesisTest(
        inference_method="poisson",
        metric="metric",
        control="A",
        variation="C",
        # parameter_estimation_method="mcmc"  # MCMC is default
    )
    test_results = exp.run_test(test)

    test_results.display()

    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


def test_bayesian_poisson_aa_mcmc(counts_data):
    exp = Experiment(data=counts_data)

    test = HypothesisTest(
        inference_method="poisson",
        metric="metric",
        control="A",
        variation="A",
        # parameter_estimation_method="mcmc"  # MCMC is default
    )
    test_results = exp.run_test(test)

    test_results.display()

    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


def test_poisson_advi(counts_data):
    """
    ADVI parameter estimation not supported for discrete PDFs like the
    Poisson.
    """
    exp = Experiment(data=counts_data)

    test = HypothesisTest(
        inference_method="poisson",
        metric="metric",
        control="A",
        variation="A",
        parameter_estimation_method="advi",
    )
    with pytest.raises(UnsupportedParameterEstimationMethod):
        exp.run_test(test)
