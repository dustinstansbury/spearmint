from copy import deepcopy
from datetime import datetime

from spearmint.config import (
    DEFAULT_ALPHA,
    DEFAULT_INFERENCE_METHOD,
    DEFAULT_HYPOTHESIS,
    DEFAULT_METRIC_NAME,
    DEFAULT_TREATMENT_NAME,
)
from spearmint.inference import InferenceResults, get_inference_procedure
from spearmint.stats import MultipleComparisonCorrection, Samples
from spearmint.typing import Callable, DataFrame, FilePath, Optional, List, Union
from spearmint.utils import ensure_dataframe, infer_variable_type


class CohortFilter:
    """
    Filtering interface for selecting cohorts of a dataframe. Powered pandas
    `query` method.

    Parameters
    ----------
    treatment_column : str
        The column associated with various treatments
    treatment_name : str
        When applied, selects out rows with `treatment_column` == `treatment_name`
    """

    def __init__(self, treatment_column: str, treatment_name: str):
        self.treatment_column = treatment_column
        self.treatment_name = treatment_name

    def apply(self, data: DataFrame) -> DataFrame:
        """
        Parameters
        ----------
        data : DataFrame

        Returns
        -------
        cohort_data : DataFrame
            cohort data, filtered from `data`
        """
        return data.query(f"`{self.treatment_column}` == '{self.treatment_name}'")


class SegmentFilter:
    """
    Simple filtering interface for selecting subsets of a dataframe. Powered
    pandas `query` interface.
    """

    def __init__(self, segment_query: Optional[str] = None):
        self.segment_query = segment_query

    def apply(self, data: DataFrame) -> DataFrame:
        if self.segment_query is not None:
            return data.query(self.segment_query)

        return data


class CustomMetric:
    """
    Metric definition that combines one or more measure columns in dataframe.

    Parameters
    ----------
    metric_function: Callable
        the function that defines an operation performed on each row of the
        dataframe, resulting in a derived value.
    """

    def __init__(self, metric_function: Callable):
        self.metric_function = metric_function

    def apply(self, data: DataFrame) -> DataFrame:
        return data.apply(self.metric_function, axis=1)


class HypothesisTest:
    """
    Executes directives for running inference a procudure to compare the central
    tendencies of two groups of random samples.

    Parameters
    ----------

    """

    def __init__(
        self,
        control: str,
        variation: str,
        metric: Optional[Union[str, CustomMetric]] = DEFAULT_METRIC_NAME,
        treatment: Optional[Union[str, CustomMetric]] = DEFAULT_TREATMENT_NAME,
        inference_method: str = DEFAULT_INFERENCE_METHOD,
        hypothesis: Optional[str] = DEFAULT_HYPOTHESIS,
        segmentation: Optional[Union[str, List[str]]] = None,
        variable_type: Optional[str] = None,
        **inference_procedure_params,
    ):
        """
        Parameters
        ----------
        control : str
            The name of the control treatment
        variation : str
            The name of the experimental treatment
        metric: str | CustomMetric (optional)
            a performance indicator over which statistical analyses
            will be performed. Each can either be a measurment in the experiment's
            dataset (i.e. a column name), or an instance of a CustomMetric. If
            None provided, we use the setting in
            spearmint.cfg::hypothesis_test.default_metric_name
        inference_method : str (optional)
            The name of the inference method used to perform the hypothesis test.
            Can be one of "frequentist", "bayesian", "bootstrap". If None provided,
            we use the setting in
            spearmint.cfg::hypothesis_test.default_inference_method
        hypothesis: str (optional)
            One of "larger", "smaller", "unqual". The hypothesis to test when
            comparing the `control` and `variation` groups. If
            None provided, we use the setting in
            spearmint.cfg::hypothesis_test.default_hypothesis
        treatment: str (optional)
            The name of the column in the experiement observations associated
            with treatment naems. If None provided, we use the setting in
            spearmint.cfg::hypothesis_test.default_treatment_name
        segmentation : str | List[str] (optional)
            Defines a list of logical filter operations that follows the conventions
            used by Panda's dataframe query api and segments the treatments into subgroups.
            If a list provided, all operations are combined using logical AND
        variable_type : optional str, None
            One of 'continuous', 'binary', 'counts'. Explicitly declares the variable
            type. If None provided, we infer the variable type from the values
            of `metric`.
        **inference_procedure_params : dict
            Any additional parameters used to initialize the inference procedure

        """
        self.metric = metric
        self.treatment = treatment
        self.control = control
        self.variation = variation
        self.inference_method = (
            inference_method.lower().replace("-", "_").replace(" ", "_")
        )
        self.variable_type = variable_type
        self.hypothesis = hypothesis

        # TODO: add back in segmentation to InferenceProcedure
        if isinstance(segmentation, list):
            segmentation = " & ".join(segmentation)
        self.segmentation = segmentation

        if isinstance(self.metric, CustomMetric):
            self.metric_column = self.metric.metric_function.__name__
        else:
            self.metric_column = str(metric)

        self.inference_procedure_params = inference_procedure_params

    def _add_custom_metric_column(self, _data):
        data = _data.copy()
        self.metric_name = self.metric.metric_function.__name__
        data[self.metric_name] = self.metric.apply(data)
        return data

    def filter_variations(self, data: DataFrame, variation_name: str) -> DataFrame:
        """
        Helper function used to pull out the data series observations for a
        variation.
        """
        data = ensure_dataframe(data)
        if isinstance(self.metric, CustomMetric):
            data = self._add_custom_metric_column(data)
        else:
            self.metric_name = self.metric

        cohort_filter = CohortFilter(self.treatment, variation_name)  # type: ignore
        return cohort_filter.apply(data)

    def filter_segments(self, data: DataFrame) -> DataFrame:
        """
        Helper function used to filter observations for all segments.
        """
        data = ensure_dataframe(data)
        segment_filter = SegmentFilter(self.segmentation)
        return segment_filter.apply(data)

    def filter_metrics(self, data: DataFrame) -> DataFrame:
        """
        Helper function used to filter out observations that have invalid metric
        values.
        """
        return data[data.notna()[self.metric_name]][self.metric_name]

    def run(
        self,
        control_samples: Samples,
        variation_samples: Samples,
        alpha: float = DEFAULT_ALPHA,
    ) -> InferenceResults:
        """
        Run the statistical test inference procedure comparing two groups of
        samples.

        Parameters
        ----------
        control_samples : an instance of `Samples` class
            The samples for the control treatment group
        variation_samples : an instance of `Samples` class
            The samples for the experimental treatment group
        alpha : float in (0, 1) (optional)
            Is either:
                -   the 'significance level' for frequentist tests
                -   the one minus the credible mass of the posterior over differences
                    between the two groups.
            If None provided, we assume an alpha = 0.05
        inference_kwargs : dict
            Any additional keyword args to be provided to the inference procedure

        Returns
        -------
        results : InferenceResults
            The object class holding the summary of the experiment

        """
        self.variable_type = (
            self.variable_type
            if self.variable_type is not None
            else infer_variable_type(control_samples.data)
        )

        self.inference_procedure = get_inference_procedure(
            variable_type=self.variable_type,
            inference_method=self.inference_method,
            metric_name=self.metric_column,
            hypothesis=self.hypothesis,
            **self.inference_procedure_params,
        )

        # Update alphas
        self.alpha = alpha
        self.inference_procedure.alpha = alpha

        results = self.inference_procedure.run(
            control_samples,
            variation_samples,
        )

        results.metric_name = self.metric_column
        # results.segmentation = self.segmentation
        return results

    def copy(self, **update_kwargs) -> "HypothesisTest":
        """
        Make a copy of the current hypothesis test, substituting any provided
        values in `**update_kwargs`
        """
        copy = deepcopy(self)
        inference_kwargs = update_kwargs.get("infer_kwargs", {})
        for k, v in update_kwargs.items():
            setattr(copy, k, v)

        copy.inference_procedure = get_inference_procedure(
            copy.variable_type, copy.inference_method, **inference_kwargs  # type: ignore  # `.variable_type` inherited, or determined during inference
        )
        return copy


class HypothesisTestGroup:
    """
    A group of simultaneous hypothesis test. Performs multiple correction.

    Parameters
    ----------
    tests : Iterable[HypothesisTest]
        The tests to run
    correction_method : str
        One of the following correction methods:
            'bonferroni', 'b' : one-step Bonferroni correction
            'sidak', 's' : one-step Sidak correction
            'fdr_bh', 'bh; : Benjamini/Hochberg (non-negative)
    """

    def __init__(self, tests: List[HypothesisTest], correction_method: str = "sidak"):
        self.tests = tests
        self.correction_method = correction_method


class GroupInferenceResults:
    """
    Store, display, visualize, and export the results of statistical test
    suite.
    """

    def __init__(
        self,
        test_group: HypothesisTestGroup,
        correction: MultipleComparisonCorrection,
        original_results: List[InferenceResults],
        corrected_results: List[InferenceResults],
    ) -> None:
        self.tests = test_group.tests
        self.ntests = len(self.tests)
        self.original_results = original_results
        self.corrected_results = corrected_results
        self.correction = correction
        self.run_at: Optional[datetime] = None

    def display(self) -> None:
        for ii, res in enumerate(self.corrected_results):
            print("-" * 60 + f"\nTest {ii + 1} of {self.ntests}")
            res.display()

    def visualize(self, outfile: Optional[FilePath] = None, *args, **kwargs):
        for _, res in enumerate(self.corrected_results):
            res.visualize(outfile=outfile)
