import pytest

from spearmint.dataset import DataFrame, Dataset, DatasetException, search_config
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def test_observations():
    return generate_fake_observations(distribution="bernoulli")


def test_search_config():
    df = generate_fake_observations(n_observations=1)

    # Test against default config template
    assert (
        "treatment" in search_config(df, "hypothesis_test", "default_treatment_name")[0]
    )
    assert "metric" in search_config(df, "hypothesis_test", "default_metric_name")
    assert "attr_0" in search_config(df, "hypothesis_test", "default_attribute_names")
    assert "attr_1" in search_config(df, "hypothesis_test", "default_attribute_names")


def test_default_init(test_observations):
    dataset = Dataset(test_observations)
    # default configuration template
    assert dataset.treatment == "treatment"
    assert "metric" in dataset.measures
    assert (
        dataset.__repr__()
        == "Dataset(treatment='treatment', measures=['metric'], attributes=['attr_0', 'attr_1'])"
    )


def test_properties(test_observations):
    dataset = Dataset(test_observations)

    selected_treatments = ["A", "B"]
    mask = test_observations.treatment.isin(selected_treatments)
    dataset = Dataset(test_observations[mask])
    assert dataset.cohorts == selected_treatments

    cohort_measures = dataset.cohort_measures
    assert isinstance(cohort_measures, DataFrame)
    assert cohort_measures.shape[0] == len(dataset.cohorts)


def test_segments(test_observations):
    dataset = Dataset(df=test_observations, attributes=["attr_0", "attr_1"])

    segments = dataset.segments("attr_0")
    assert isinstance(segments, list)
    assert ("A", "A0a") in segments
    assert ("B", "A0b") in segments

    segment_samples = dataset.segment_samples("attr_0")
    assert isinstance(segment_samples, DataFrame)
    assert ("A", "A0a") in segment_samples.index
    assert ("B", "A0b") in segment_samples.index

    # Replace column names with spaced column names, ensure that
    # segmentation based off pandas.query still works as exptected
    test_observations.columns = [c.replace("_", " ") for c in test_observations.columns]
    dataset = Dataset(df=test_observations, attributes=["attr 0", "attr 1"])

    segments = dataset.segments("attr 0")
    assert isinstance(segments, list)
    assert ("A", "A0a") in segments
    assert ("B", "A0b") in segments

    segment_samples = dataset.segment_samples("attr 0")
    assert isinstance(segment_samples, DataFrame)
    assert ("A", "A0a") in segment_samples.index
    assert ("B", "A0b") in segment_samples.index


def test_exceptions(test_observations):
    with pytest.raises(DatasetException):
        Dataset(test_observations, treatment="unknown treatment")

    with pytest.raises(DatasetException):
        Dataset(test_observations, measures="unknown measures string")

    with pytest.raises(DatasetException):
        Dataset(test_observations, measures=["unknown measures", "list"])

    with pytest.raises(DatasetException):
        Dataset(test_observations, attributes="unknown attribures string")

    with pytest.raises(DatasetException):
        Dataset(test_observations, attributes=["unknown attributes", "list"])

    with pytest.raises(DatasetException):
        Dataset(test_observations, metadata="unknown metadata string")

    with pytest.raises(DatasetException):
        Dataset(test_observations, metadata=["unknown metadata", "list"])
