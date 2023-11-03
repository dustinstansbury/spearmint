import pytest
from spearmint import Experiment, HypothesisTest
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def counts_data():
    return generate_fake_observations(
        distribution="poisson", n_treatments=3, n_observations=3 * 100
    )


def test_bayesian_poisson_ab_mcmc(counts_data):
    exp = Experiment(data=counts_data)

    inference_procedure_init_params = dict(inference_method="mcmc")
    test = HypothesisTest(
        inference_method="poisson",
        metric="metric",
        control="A",
        variation="C",
        inference_procedure_init_params=inference_procedure_init_params,
    )
    test_results = exp.run_test(test)

    test_results.display()
    test_results_df = test_results.to_dataframe()

    assert pytest.approx(test_results.prob_greater_than_zero, rel=0.1, abs=0.01) == 1.0

    import ipdb

    ipdb.set_trace()
