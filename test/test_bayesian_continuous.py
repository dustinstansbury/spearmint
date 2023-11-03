import pytest
from spearmint import Experiment, HypothesisTest
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def continuous_data():
    return generate_fake_observations(
        distribution="gaussian", n_treatments=2, n_observations=2 * 100
    )


# @pytest.mark.mcmc_test
def test_bayesian_gaussian_ab_mcmc(continuous_data):
    exp = Experiment(data=continuous_data)
    inference_procedure_init_params = dict(inference_method="mcmc")
    test = HypothesisTest(
        inference_method="gaussian",
        metric="metric",
        control="A",
        variation="B",
        inference_procedure_init_params=inference_procedure_init_params,
    )

    test_results = exp.run_test(test)
    test_results.display()
    test_results_df = test_results.to_dataframe()

    # import ipdb

    # ipdb.set_trace()

    # assert pytest.approx(test_results.prob_greater, rel=0.1, abs=0.01) == 1.0


def test_bayesian_student_t_ab_mcmc(continuous_data):
    exp = Experiment(data=continuous_data)
    inference_procedure_init_params = dict(inference_method="mcmc")
    test = HypothesisTest(
        inference_method="student_t",
        metric="metric",
        control="A",
        variation="B",
        inference_procedure_init_params=inference_procedure_init_params,
    )

    test_results = exp.run_test(test)
    test_results.display()
    test_results_df = test_results.to_dataframe()
