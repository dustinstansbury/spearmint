import pytest

from spearmint import config
from spearmint import Experiment, HypothesisTest
from spearmint.inference.bayesian.bayesian_inference import (
    UnsupportedParameterEstimationMethodException,
)
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def counts_data():
    return generate_fake_observations(
        distribution="poisson", n_treatments=3, n_observations=3 * 100
    )


@pytest.mark.pymc_test
def test_bayesian_counts_default(counts_data):
    exp = Experiment(data=counts_data)

    test = HypothesisTest(
        metric="metric", control="A", variation="C", inference_method="bayesian"
    )
    test_results = exp.run_test(test)

    test_results.display()

    assert test_results.model_name == "poisson"  # Default model for counts data
    assert (
        test_results.bayesian_parameter_estimation_method
        == config.DEFAULT_PARAMETER_ESTIMATION_METHOD
    )  # MCMC is default
    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


@pytest.mark.pymc_test
def test_bayesian_poisson_ab_mcmc(counts_data):
    exp = Experiment(data=counts_data)

    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="C",
        inference_method="bayesian",
        bayesian_model_name="poisson",
        bayesian_parameter_estimation_method="mcmc",
    )
    test_results = exp.run_test(test)

    test_results.display()

    assert test_results.model_name == "poisson"  # Default model for counts data
    assert (
        test_results.bayesian_parameter_estimation_method == "mcmc"
    )  # MCMC is default
    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


@pytest.mark.pymc_test
def test_bayesian_poisson_aa_mcmc(counts_data):
    exp = Experiment(data=counts_data)

    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        inference_method="bayesian",
        # bayesian_parameter_estimation_method="mcmc"  # MCMC is default
    )
    test_results = exp.run_test(test)

    test_results.display()

    assert test_results.model_name == "poisson"  # Default model for counts data
    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


@pytest.mark.pymc_test
def test_poisson_aa_advi(counts_data):
    """
    ADVI parameter estimation not supported for discrete PDFs like the
    Poisson.
    """
    exp = Experiment(data=counts_data)

    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        variable_type="counts",
        inference_method="bayesian",
        bayesian_parameter_estimation_method="advi",
    )
    with pytest.raises(UnsupportedParameterEstimationMethodException):
        exp.run_test(test)


def test_poisson_ab_analytic(counts_data):
    """
    ADVI parameter estimation not supported for discrete PDFs like the
    Poisson.
    """
    exp = Experiment(data=counts_data)

    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="B",
        inference_method="bayesian",
        bayesian_parameter_estimation_method="analytic",
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert test_results.model_name == "poisson"  # Default model for counts data
    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


def test_poisson_aa_analytic(counts_data):
    """
    ADVI parameter estimation not supported for discrete PDFs like the
    Poisson.
    """
    exp = Experiment(data=counts_data)

    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        inference_method="bayesian",
        bayesian_parameter_estimation_method="analytic",
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert test_results.model_name == "poisson"  # Default model for counts data
    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )
