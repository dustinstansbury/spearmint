import pytest
import numpy as np
from spearmint.utils import generate_fake_observations
from spearmint.experiment import Experiment
from spearmint.hypothesis_test import HypothesisTest, HypothesisTestGroup, CustomMetric


@pytest.fixture()
def test_data():
    return generate_fake_observations(
        distribution="gaussian", n_treatments=6, n_observations=6 * 50, random_seed=123
    )


def test_hypothesis_test_group(test_data):
    exp = Experiment(data=test_data)

    # run 'A/A' test (should never reject null)
    test_aa = HypothesisTest(
        metric="metric",
        control="A",
        variation="A",
        hypothesis="larger",
        inference_method="means_delta",
    )
    # run A/B test
    test_ab = HypothesisTest(
        metric="metric",
        control="A",
        variation="B",
        hypothesis="larger",
        inference_method="means_delta",
    )
    correction_method = "b"  # shorthand for Bonferonni
    test_group = HypothesisTestGroup(
        tests=[test_aa, test_ab], correction_method=correction_method
    )
    test_group_results = exp.run_test_group(test_group)

    test_group_results.display()

    assert (
        test_group_results.corrected_results[0].correction_method.__name__
        == "bonferroni_correction"
    )
    assert test_group_results.original_results[0].correction_method is None

    # corrected alpha should be smaller
    alpha_orig = test_group_results.original_results[0].alpha
    alpha_corrected = test_group_results.corrected_results[0].alpha

    assert alpha_orig > alpha_corrected
    assert not test_group_results.corrected_results[0].accept_hypothesis
    assert test_group_results.corrected_results[1].accept_hypothesis


def test_custom_metric(test_data):
    exp = Experiment(test_data)

    def custom_metric(row):
        return 4 + np.random.rand() if row["treatment"] != "A" else np.random.rand()

    test_ab = HypothesisTest(
        metric=CustomMetric(custom_metric),
        control="A",
        variation="B",
        hypothesis="larger",
        inference_method="means_delta",
    )
    results_ab = exp.run_test(test_ab)
    results_ab.to_dataframe()

    results_ab.display()

    assert results_ab.test_statistic_name == "z"
    assert results_ab.accept_hypothesis
