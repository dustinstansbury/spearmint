import pytest
from spearmint import Experiment, HypothesisTest
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def continuous_data():
    return generate_fake_observations(
        distribution="bernoulli", n_treatments=3, n_observations=3 * 100
    )


def test_bayesian_bernoulli_ab_mcmc(continuous_data):
    exp = Experiment(data=continuous_data)

    inference_procedure_init_params = dict(inference_method="mcmc")
    test = HypothesisTest(
        inference_method="bernoulli",
        metric="metric",
        control="A",
        variation="C",
        inference_procedure_init_params=inference_procedure_init_params,
    )
    test_results = exp.run_test(test)

    test_results.display()
    test_results_df = test_results.to_dataframe()

    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0
