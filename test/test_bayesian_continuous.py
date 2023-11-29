import pytest

from spearmint import config
from spearmint import Experiment, HypothesisTest
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def continuous_data():
    return generate_fake_observations(
        distribution="gaussian", n_treatments=2, n_observations=2 * 100, random_seed=123
    )


@pytest.mark.pymc_test
def test_bayesian_continuous_default(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        metric="metric", control="A", variation="B", inference_method="bayesian"
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert test_results.model_name == "gaussian"
    assert (
        test_results.bayesian_parameter_estimation_method
        == config.DEFAULT_PARAMETER_ESTIMATION_METHOD
    )
    assert test_results.model_hyperparams["prior_mean_mu"] == 0.0
    assert test_results.model_hyperparams["prior_var_mu"] == 5.0

    if config.DEFAULT_PARAMETER_ESTIMATION_METHOD in ("mcmc", "advi"):
        assert test_results.model_hyperparams["prior_mean_sigma"] == 1.0
        assert test_results.model_hyperparams["prior_var_sigma"] == 5.0

    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


@pytest.mark.pymc_test
def test_bayesian_gaussian_ab_mcmc(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="B",
        inference_method="bayesian",
        # bayesian_parameter_estimation_method="mcmc",  # MCMC is default
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert (
        test_results.model_name == "gaussian"
    )  # Gaussian is default for continuous data
    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


@pytest.mark.pymc_test
def test_bayesian_gaussian_aa_mcmc(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        inference_method="bayesian",
        # bayesian_parameter_estimation_method="mcmc",  # MCMC is default
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert (
        test_results.model_name == "gaussian"
    )  # Gaussian is default for continuous data
    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


@pytest.mark.pymc_test
def test_bayesian_gaussian_ab_advi(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="B",
        inference_method="bayesian",
        bayesian_parameter_estimation_method="advi",  # use ADVI parameter estimation
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert (
        test_results.model_name == "gaussian"
    )  # Gaussian is default for continuous data
    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


@pytest.mark.pymc_test
def test_bayesian_gaussian_aa_advi(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        inference_method="bayesian",
        bayesian_parameter_estimation_method="advi",  # use ADVI parameter estimation
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert (
        test_results.model_name == "gaussian"
    )  # Gaussian is default for continuous data
    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


def test_bayesian_gaussian_ab_analytic(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="B",
        inference_method="bayesian",
        bayesian_parameter_estimation_method="analytic",  # use Analytic parameter estimation
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert (
        test_results.model_name == "gaussian"
    )  # Gaussian is default for continuous data
    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


def test_bayesian_gaussian_aa_analytic(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        inference_method="bayesian",
        bayesian_parameter_estimation_method="analytic",  # use Analytic parameter estimation
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert (
        test_results.model_name == "gaussian"
    )  # Gaussian is default for continuous data
    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


@pytest.mark.pymc_test
def test_bayesian_student_t_ab_mcmc(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="B",
        inference_method="bayesian",
        bayesian_model_name="student_t",
        bayesian_parameter_estimation_method="mcmc",  # MCMC is default
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert test_results.model_name == "student_t"
    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


@pytest.mark.pymc_test
def test_bayesian_student_t_aa_mcmc(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        inference_method="bayesian",
        bayesian_model_name="student_t",
        bayesian_parameter_estimation_method="mcmc",
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert test_results.model_name == "student_t"
    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


@pytest.mark.pymc_test
def test_bayesian_student_t_ab_advi(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="B",
        inference_method="bayesian",
        bayesian_model_name="student_t",
        bayesian_parameter_estimation_method="advi",  # use ADVI parameter estimation
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert test_results.model_name == "student_t"
    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


@pytest.mark.pymc_test
def test_bayesian_student_t_aa_advi(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        inference_method="bayesian",
        bayesian_model_name="student_t",
        bayesian_parameter_estimation_method="advi",  # use ADVI parameter estimation
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert test_results.model_name == "student_t"
    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )
