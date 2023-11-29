from collections import OrderedDict

import pytest

from spearmint.inference import InferenceProcedure, InferenceResults
from spearmint.stats import Samples
from spearmint.typing import DataFrame
from spearmint.utils import generate_fake_observations


@pytest.fixture()
def test_samples():
    observations = generate_fake_observations(distribution="bernoulli")["metric"].values
    return Samples(observations=observations, name="test")


class ExtendedInferenceResults(InferenceResults):
    def __init__(self, alpha=0.05, *args, **kwargs):
        super().__init__(
            metric_name="test",
            delta=0.0,
            delta_relative=0.0,
            effect_size=0.0,
            hypothesis="larger",
            alpha=alpha,
            accept_hypothesis=False,
            inference_method="test_inference",
            variable_type="test_variable_type",
            warnings=["a warning"],
            visualization_function=lambda x: x,  # placeholder visualization function
            segmentation="test_segmentation",
            *args,
            **kwargs
        )

    @property
    def _specific_properties(self) -> OrderedDict:
        return OrderedDict([("specific_property", "blah")])

    def _render_stats_table(self):
        class DummyStatsTable:
            """placeholder stats table with print method"""

            def print(self):
                pass

        return DummyStatsTable()


class ExtendedInferenceProcedure(InferenceProcedure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_run = False

    def _run_inference(
        self, control_samples, variation_samples, **inference_kwargs
    ) -> None:
        self.inference_run = True
        self.control_samples = control_samples
        self.variation_samples = variation_samples

    def _make_results(self) -> InferenceResults:
        return ExtendedInferenceResults(
            control=self.control_samples,
            variation=self.variation_samples,
            alpha=self.alpha,
        )


def validate_results(results, alpha: float = 0.05):
    assert results.dict["control_name"] == results.dict["variation_name"]
    assert results.dict["control_nobs"] == results.dict["variation_nobs"]
    assert results.dict["control_mean"] == results.dict["variation_mean"]
    assert results.dict["control_ci"] == results.dict["variation_ci"]
    assert results.dict["control_var"] == results.dict["variation_var"]
    assert results.dict["metric"] == "test"
    assert results.dict["delta"] == 0
    assert results.dict["delta_relative"] == 0
    assert results.dict["effect_size"] == 0
    assert results.dict["alpha"] == alpha
    assert results.dict["hypothesis"] == "larger"
    assert results.dict["accept_hypothesis"] is False
    assert results.dict["inference_method"] == "test_inference"
    assert results.dict["variable_type"] == "test_variable_type"
    assert results.dict["warnings"] == "a warning"
    assert results.dict["specific_property"] == "blah"


def test_inference_results(test_samples):
    results = ExtendedInferenceResults(control=test_samples, variation=test_samples)
    validate_results(results)
    results.summary

    # Check datarame export support
    results_df = results.to_dataframe()
    assert isinstance(results_df, DataFrame)

    # We added one _specific property
    assert len(results_df.columns) == len(results._base_properties) + 1


def test_inference_procedure(test_samples):
    inference_procedure = ExtendedInferenceProcedure(inference_method="test", alpha=0.1)
    inference_results = inference_procedure.run(
        control_samples=test_samples, variation_samples=test_samples
    )
    assert inference_procedure.inference_run
    assert inference_procedure.alpha == 0.1
    assert inference_results == inference_procedure.results
    validate_results(inference_procedure.results, inference_procedure.alpha)

    inference_procedure.results.summary
