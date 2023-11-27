import numpy as np
import pytest

from spearmint.experiment import Experiment
from spearmint.hypothesis_test import CustomMetric, HypothesisTest, HypothesisTestGroup
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def test_data():
    return generate_fake_observations(
        distribution="gaussian", n_treatments=6, n_observations=6 * 50, random_seed=123
    )


def test_hypothesis_test(test_data):
    exp = Experiment(data=test_data)
    default_test = HypothesisTest(
        metric="metric",
        control="A",
        variation="B",
    )
    # Defaults
    assert default_test.inference_method == "frequentist"  # inference method
    assert default_test.variable_type is None  # variable type not inferred
    assert default_test.hypothesis == "larger"

    # Infer datatype from `metric`
    exp.run_test(default_test)

    assert (
        default_test.variable_type == "continuous"
    )  # datatype inferred during `run_test``

    # Copying
    copy_test = default_test.copy(
        variation="C", inference_method="bootstrap", variable_type="binary"
    )
    assert copy_test.variation == "C"
    assert copy_test.inference_method == "bootstrap"
    assert copy_test.variable_type == "binary"

    custom_test = HypothesisTest(
        metric="metric",
        control="A",
        variation="B",
        variable_type="counts",
        inference_method="frequentist",
    )

    assert custom_test.inference_method == "frequentist"
    assert custom_test.variable_type == "counts"  # explicit variable type

    exp.run_test(custom_test)

    assert (
        custom_test.variable_type == "counts"
    )  # explicit variable type doesn't change

    with pytest.raises(ValueError):
        invalid_inference_method_test = HypothesisTest(
            metric="metric",
            control="A",
            variation="B",
            inference_method="invalid_inference_method",
        )
        exp.run_test(invalid_inference_method_test)

    with pytest.raises(ValueError):
        invalid_variable_type_test = HypothesisTest(
            metric="metric",
            control="A",
            variation="B",
            inference_method="frequentist",
            variable_type="invalid_variable_type",
        )
        exp.run_test(invalid_variable_type_test)


def test_hypothesis_test_group(test_data):
    exp = Experiment(data=test_data)

    # run 'A/A' test (should not reject null)
    test_aa = HypothesisTest(
        metric="metric", control="A", variation="A", hypothesis="larger"
    )
    # run A/B test
    test_ab = HypothesisTest(
        metric="metric", control="A", variation="B", hypothesis="larger"
    )
    correction_method = "b"  # shorthand for Bonferonni
    test_group = HypothesisTestGroup(
        tests=[test_aa, test_ab], correction_method=correction_method
    )
    test_group_results = exp.run_test_group(test_group)

    test_group_results.display()

    assert (
        test_group_results.corrected_results[0].correction_method
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
    )
    results_ab = exp.run_test(test_ab)
    assert test_ab.metric_column == "custom_metric"

    results_ab.display()

    assert results_ab.accept_hypothesis
    assert results_ab.test_statistic_name == "z"
    assert results_ab.accept_hypothesis


def test_filter_segments(test_data):
    unsegmented_test = HypothesisTest(metric="metric", control="A", variation="B")
    unsegmented_data = unsegmented_test.filter_segments(test_data)
    assert unsegmented_data.equals(test_data)

    segmented_test = HypothesisTest(
        metric="metric", control="A", variation="B", segmentation="attr_0 == 'A0b'"
    )
    segmented_data = segmented_test.filter_segments(test_data)
    assert np.all(segmented_data.attr_0 == "A0b")


def test_filter_variation(test_data):
    test = HypothesisTest(
        metric="metric", control="A", variation="B", treatment="treatment"
    )

    control_data = test.filter_variations(test_data, test.control)
    variation_data = test.filter_variations(test_data, test.variation)

    assert np.all(control_data.treatment == test.control)
    assert np.all(variation_data.treatment == test.variation)
