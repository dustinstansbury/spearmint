import pytest
from spearmint import Experiment, HypothesisTest
from spearmint.inference.bayesian.bayesian_inference import (
    UnsupportedParameterEstimationMethodException,
)
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def binary_data():
    return generate_fake_observations(
        distribution="bernoulli",
        n_treatments=3,
        n_observations=3 * 100,
        random_seed=123,
    )


@pytest.mark.pymc_test
def test_bernoulli_mcmc_model_params(binary_data):
    exp = Experiment(data=binary_data)

    test = HypothesisTest(
        inference_method="bernoulli",
        metric="metric",
        control="A",
        variation="C",
        parameter_estimation_method="mcmc",
        model_params=dict(prior_alpha=2.0),
    )
    test_results = exp.run_test(test)
    assert test_results.model_hyperparams["prior_alpha"] == 2
    assert test_results.model_hyperparams["prior_beta"] == 1  # default


@pytest.mark.pymc_test
def test_bernoulli_ab_mcmc(binary_data):
    exp = Experiment(data=binary_data)

    test = HypothesisTest(
        inference_method="bernoulli",
        metric="metric",
        control="A",
        variation="C",
        # parameter_estimation_method="mcmc"  # Default
    )
    test_results = exp.run_test(test)

    test_results.display()
    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


@pytest.mark.pymc_test
def test_bernoulli_aa_mcmc(binary_data):
    exp = Experiment(data=binary_data)

    test = HypothesisTest(
        inference_method="bernoulli",
        metric="metric",
        control="A",
        variation="A",
        # parameter_estimation_method="mcmc"  # Default
    )
    test_results = exp.run_test(test)

    test_results.display()
    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


@pytest.mark.pymc_test
def test_bernoulli_ab_advi(binary_data):
    exp = Experiment(data=binary_data)
    test = HypothesisTest(
        inference_method="bernoulli",
        metric="metric",
        control="A",
        variation="C",
        parameter_estimation_method="advi",
    )
    test_results = exp.run_test(test)

    test_results.display()
    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


@pytest.mark.pymc_test
def test_bernoulli_aa_advi(binary_data):
    exp = Experiment(data=binary_data)
    test = HypothesisTest(
        inference_method="bernoulli",
        metric="metric",
        control="A",
        variation="A",
        parameter_estimation_method="advi",
    )
    test_results = exp.run_test(test)

    test_results.display()
    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


def test_bernoulli_analytic_model_params(binary_data):
    exp = Experiment(data=binary_data)

    test = HypothesisTest(
        inference_method="bernoulli",
        metric="metric",
        control="A",
        variation="C",
        parameter_estimation_method="analytic",
        model_params=dict(prior_alpha=2.0),
    )
    test_results = exp.run_test(test)
    assert test_results.model_hyperparams["prior_alpha"] == 2
    assert test_results.model_hyperparams["prior_beta"] == 1  # default


def test_bernoulli_aa_analytic(binary_data):
    exp = Experiment(data=binary_data)
    test = HypothesisTest(
        inference_method="bernoulli",
        metric="metric",
        control="A",
        variation="A",
        parameter_estimation_method="analytic",
    )
    test_results = exp.run_test(test)

    test_results.display()
    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


def test_bernoulli_ab_analytic(binary_data):
    exp = Experiment(data=binary_data)
    test = HypothesisTest(
        inference_method="bernoulli",
        metric="metric",
        control="A",
        variation="B",
        parameter_estimation_method="analytic",
    )
    test_results = exp.run_test(test)

    test_results.display()
    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


@pytest.mark.pymc_test
def test_binomial_mcmc_model_params(binary_data):
    exp = Experiment(data=binary_data)

    test = HypothesisTest(
        inference_method="binomial",
        metric="metric",
        control="A",
        variation="C",
        parameter_estimation_method="mcmc",
        model_params=dict(prior_alpha=2.0, possible_outcomes=3.0),
    )
    test_results = exp.run_test(test)
    assert test_results.model_hyperparams["prior_alpha"] == 2
    assert test_results.model_hyperparams["possible_outcomes"] == 3
    assert test_results.model_hyperparams["prior_beta"] == 1  # default


@pytest.mark.pymc_test
def test_binomial_ab_mcmc(binary_data):
    exp = Experiment(data=binary_data)

    test = HypothesisTest(
        inference_method="binomial",
        metric="metric",
        control="A",
        variation="C",
        # parameter_estimation_method="mcmc"  # Default
    )
    test_results = exp.run_test(test)

    test_results.display()

    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


@pytest.mark.pymc_test
def test_binomial_aa_mcmc(binary_data):
    exp = Experiment(data=binary_data)

    test = HypothesisTest(
        inference_method="binomial",
        metric="metric",
        control="A",
        variation="A",
        # parameter_estimation_method="mcmc"  # Default
    )
    test_results = exp.run_test(test)

    test_results.display()

    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


@pytest.mark.pymc_test
def test_binomial_advi(binary_data):
    """
    ADVI parameter estimation not supported for discrete PDFs like the
    binomial.
    """
    exp = Experiment(data=binary_data)

    test = HypothesisTest(
        inference_method="binomial",
        metric="metric",
        control="A",
        variation="C",
        parameter_estimation_method="advi",
    )

    with pytest.raises(UnsupportedParameterEstimationMethodException):
        exp.run_test(test)


def test_binomial_analytic_model_params(binary_data):
    exp = Experiment(data=binary_data)

    test = HypothesisTest(
        inference_method="binomial",
        metric="metric",
        control="A",
        variation="C",
        parameter_estimation_method="analytic",
        model_params=dict(prior_alpha=2.0),
    )
    test_results = exp.run_test(test)
    assert test_results.model_hyperparams["prior_alpha"] == 2
    assert test_results.model_hyperparams["prior_beta"] == 1  # default


def test_binomial_aa_analytic(binary_data):
    exp = Experiment(data=binary_data)
    test = HypothesisTest(
        inference_method="binomial",
        metric="metric",
        control="A",
        variation="A",
        parameter_estimation_method="analytic",
    )
    test_results = exp.run_test(test)

    test_results.display()
    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


def test_binommial_ab_analytic(binary_data):
    exp = Experiment(data=binary_data)
    test = HypothesisTest(
        inference_method="binomial",
        metric="metric",
        control="A",
        variation="B",
        parameter_estimation_method="analytic",
    )
    test_results = exp.run_test(test)

    test_results.display()
    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
