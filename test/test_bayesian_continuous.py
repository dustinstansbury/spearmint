import pytest
from spearmint import Experiment, HypothesisTest
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def continuous_data():
    return generate_fake_observations(
        distribution="gaussian", n_treatments=2, n_observations=2 * 100, random_seed=123
    )


# @pytest.mark.mcmc_test
def test_bayesian_gaussian_ab_mcmc(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        inference_method="gaussian",
        metric="metric",
        control="A",
        variation="B",
        # parameter_estimation_method="mcmc",  # MCMC is default
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


def test_bayesian_gaussian_aa_mcmc(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        inference_method="gaussian",
        metric="metric",
        control="A",
        variation="A",
        # parameter_estimation_method="mcmc",  # MCMC is default
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


def test_bayesian_gaussian_ab_advi(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        inference_method="gaussian",
        metric="metric",
        control="A",
        variation="B",
        parameter_estimation_method="advi",  # use ADVI parameter estimation
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


def test_bayesian_gaussian_aa_advi(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        inference_method="gaussian",
        metric="metric",
        control="A",
        variation="A",
        parameter_estimation_method="advi",  # use ADVI parameter estimation
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


def test_bayesian_gaussian_ab_analytic(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        inference_method="gaussian",
        metric="metric",
        control="A",
        variation="B",
        parameter_estimation_method="analytic",  # use Analytic parameter estimation
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


def test_bayesian_gaussian_aa_analytic(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        inference_method="gaussian",
        metric="metric",
        control="A",
        variation="A",
        parameter_estimation_method="analytic",  # use Analytic parameter estimation
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


def test_bayesian_student_t_ab_mcmc(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        inference_method="student_t",
        metric="metric",
        control="A",
        variation="B",
        # parameter_estimation_method="mcmc",  # MCMC is default
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


def test_bayesian_student_t_aa_mcmc(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        inference_method="student_t",
        metric="metric",
        control="A",
        variation="A",  # parameter_estimation_method="mcmc",  # MCMC is default
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )


def test_bayesian_student_t_ab_advi(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        inference_method="student_t",
        metric="metric",
        control="A",
        variation="B",
        parameter_estimation_method="advi",  # use ADVI parameter estimation
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert test_results.accept_hypothesis
    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0


def test_bayesian_student_t_aa_advi(continuous_data):
    exp = Experiment(data=continuous_data)
    test = HypothesisTest(
        inference_method="student_t",
        metric="metric",
        control="A",
        variation="A",
        parameter_estimation_method="advi",  # use ADVI parameter estimation
    )

    test_results = exp.run_test(test)
    test_results.display()

    assert not test_results.accept_hypothesis
    assert (
        not pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
    )
