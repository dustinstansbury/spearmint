from spearmint.typing import List, DataFrame, Union, Callable, Iterable, FilePath
from spearmint.utils import ensure_dataframe
from spearmint.config import DEFAULT_ALPHA
from spearmint.stats import (
    MultipleComparisonCorrection,
    Samples,
)
from spearmint.inference import (
    InferenceProcedure,
    InferenceResults,
    get_inference_procedure,
)


# from datetime import datetime
from copy import deepcopy


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

    def __init__(self, segment_query: str = None):
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
    f: function
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
    """

    def __init__(
        self,
        inference_method: str,
        metric: Union[str, CustomMetric] = None,
        control: str = None,
        variation: str = None,
        segmentation: Union[str, List[str]] = None,
        # date=None,
        **inference_procedure_init_params,
    ):
        """
        Parameters
        ----------
        inference_method : str
            The name of the inference method used to perform the hypothesis test.
            Can be one of the following:

                Frequentist Inference:
                    - 'means_delta'         Continuous
                    - 'proprortions_delta'  Proportions
                    - 'rates_ratio'         Counts / rates
                Bayesian Inference:
                    - 'gaussian'            Continuous
                    - 'exp_student_t'       Continuous
                    - 'bernoulli'           Proportions / binary
                    - 'beta_binomial'       Proportions
                    - 'binomial'            Proportions
                    - 'gamma_poisson'       Counts / rates

        metric: str or CustomMetric instance (optional)
            a performance indicator over which statistical analyses
            will be performed. Each can either be a measurment in the experiment's
            dataset, or an instance of a CustomMetric. If None provided, the
        control : str
            The name of the control treatment
        variation : str
            The name of the experimental treatment
        segmentation : str | List[str] (optional)
            Defines a list of logical filter operations that follows the conventions
            used by Panda's dataframe query api and segments the treatments into subgroups.
            If a list provided, all operations are combined using logical AND
        **inference_procedure_init_params : dict
            Any additional parameters used to initialize the inference procedure

        """
        self.metric = metric
        self.control = control
        self.variation = variation
        self.inference_method = inference_method

        # TODO: add back in segmentation to InferenceProcedure
        if isinstance(segmentation, list):
            segmentation = " & ".join(segmentation)
        self.segmentation = segmentation

        # if date is None:
        #     self.date = datetime.now().date()

        if isinstance(self.metric, CustomMetric):
            self.metric_column = self.metric.metric_function.__name__
        else:
            self.metric_column = metric

        self.inference_procedure = get_inference_procedure(
            inference_method=self.inference_method,
            metric_name=self.metric_column,
            **inference_procedure_init_params,
        )

    def _add_custom_metric_column(self, _data):
        data = _data.copy()
        self.metric_name = self.metric.metric_function.__name__
        data[self.metric_name] = self.metric.apply(data)
        return data

    def filter_variations(self, data, variation_name, treatment=None):
        """
        Helper function used to pull out the data series observations for a
        variation.
        """
        data = ensure_dataframe(data)
        if isinstance(self.metric, CustomMetric):
            data = self._add_custom_metric_column(data)
        else:
            self.metric_name = self.metric

        if treatment is None:
            if hasattr(self, "treatment"):
                treatment = self.treatment
            else:
                raise ValueError(
                    "Cant't determine the treatment column, ",
                    "please provide as `treatment` argument",
                )

        cohort_filter = CohortFilter(treatment, variation_name)
        return cohort_filter.apply(data)

    def filter_segments(self, data):
        """
        Helper function used to filter observations for all segments.
        """
        data = ensure_dataframe(data)
        segment_filter = SegmentFilter(self.segmentation)
        return segment_filter.apply(data)

    def filter_metrics(self, data):
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
        # **inference_kwargs,
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
        results : subclass of `TestResultsBase`
            The container class holding the summary of the experiment

        """

        # Update alphas
        self.alpha = alpha
        self.inference_procedure.alpha = alpha

        results = self.inference_procedure.run(
            control_samples,
            variation_samples,
            # **inference_kwargs,
        )

        results.metric_name = self.metric_column
        results.segmentation = self.segmentation
        return results

    def copy(self, **update_kwargs) -> "HypothesisTest":
        """
        Make a copy of the current hypothesis test, substituting any provided
        values in `**update_kwargs`
        """
        copy = deepcopy(self)
        inference_kwargs = update_kwargs.get("infer_kwargs", {})
        for k, v in update_kwargs.items():
            if hasattr(copy, k):
                setattr(copy, k, v)

        copy.inference_procedure = get_inference_procedure(
            copy.inference_method, **inference_kwargs
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

    def __init__(
        self, tests: Iterable[HypothesisTest], correction_method: str = "sidak"
    ):
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ntests={self.ntests}, correction_method='{self.correction.method}')"

    def display(self) -> None:
        for ii, res in enumerate(self.corrected_results):
            print("-" * 60 + f"\nTest {ii + 1} of {self.ntests}")
            res.display()

    def visualize(self, outfile: FilePath = None, *args, **kwargs):
        for _, res in enumerate(self.corrected_results):
            res.visualize(outfile=outfile)
