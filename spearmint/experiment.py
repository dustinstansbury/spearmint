import copy
from datetime import datetime

from pandas import DataFrame

from spearmint.config import DEFAULT_ALPHA
from spearmint.dataset import Dataset
from spearmint.hypothesis_test import (
    GroupInferenceResults,
    HypothesisTest,
    HypothesisTestGroup,
    InferenceResults,
)
from spearmint.stats import MultipleComparisonCorrection, Samples
from spearmint.typing import List, Optional


class Experiment:
    """
    Container for data and metadata, used to run various hypothesis tests on
    subsets of the data.
    """

    def __init__(
        self,
        data: DataFrame,
        treatment: Optional[str] = None,
        measures: Optional[List[str]] = None,
        attributes: Optional[List[str]] = None,
        metadata: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        data : DataFrame
            the tabular data to analyze, must have columns that correspond
            with `treatment`, `measures`, `attributes`, and `enrollment` if any
            of those are defined.
        treatment : str
            The column in `data` that identifies the association of each
            enrollment in the experiment with one of the experiment conditions. If
            None provided, the global config is searched to identify potential
            treatments in `data`.
        measures : List[str]
            Columns in `data` that are associated with indicator measurements.
            If None provided, the global config is searched to identify potential
            measures in `data`.
        attributes : List[str]
            the columns in `data` that define segmenting attributes
            associated with each enrollment in the experiment. If None provided, the
            global config is searched to identify potential attributes in `data`.
        metadata : List[str]
            Any additional columns in `data` that should be included in the
            experiment dataset. For examples, these columns can be used for
            custom metrics or segmentations.

        name : str
            the name of the experiment
        """

        # metadata
        # if date is None:
        #     self.date = datetime.now().date()
        # elif isinstance(date, datetime):
        #     self.date = date
        # else:
        #     raise ValueError("date parameter must be a datetime.datetime object")

        # self.name = name

        self.created_at = datetime.now()
        self.run_on = None

        # TODO: update `df`` to`data`` in Dataset class for consistency
        self.dataset = Dataset(
            df=data,
            treatment=treatment,
            measures=measures,
            attributes=attributes,
            metadata=metadata,
        )

    @property
    def ds(self):
        """
        Shorthand for the experiment dataset instance
        """
        return self.dataset

    @property
    def measures(self):
        return self.dataset.measures

    @property
    def attributes(self):
        return self.dataset.attributes

    @property
    def observations(self):
        return self.ds.data

    def run_test(
        self,
        test: HypothesisTest,
        alpha: float = DEFAULT_ALPHA,
        correction_method: Optional[str] = None,
        display_results: Optional[bool] = False,
        visualize_results: Optional[bool] = False,
    ) -> InferenceResults:
        """
        Given a HypothesisTest, run the test and return the results.

        Parameters
        ----------
        test : HypothesisTest
            A hypothesis test object
        alpha : float
            The Type I error assumed by the experimenter when running the test
        correction_method: str
            Correction method used when running test (if any)
        display_results : boolean
            Whether to print the test results to stdout
        visualize_results : boolean
            Whether to render a visual representation of the results

        Returns
        -------
        test_results: an instance of a InferenceResults or sublass
            The results of the statistical test.
        """

        control_obs = test.filter_variations(self.dataset, test.control)
        control_obs = test.filter_segments(control_obs)
        control_obs = test.filter_metrics(control_obs)
        control_samples = Samples(control_obs, name=test.control)

        variation_obs = test.filter_variations(self.dataset, test.variation)
        variation_obs = test.filter_segments(variation_obs)
        variation_obs = test.filter_metrics(variation_obs)
        variation_samples = Samples(variation_obs, name=test.variation)

        test_results = test.run(
            control_samples,
            variation_samples,
            alpha=alpha,
            # inference_kwargs=inference_kwargs,
        )
        test_results.correction_method = correction_method

        if display_results:
            test_results.display()

        if visualize_results:
            test_results.visualize()

        test_results.run_at = datetime.now()
        return test_results

    def run_test_group(
        self,
        test_group: HypothesisTestGroup,
        alpha=DEFAULT_ALPHA,
        display_results: bool = False,
        visualize_results: bool = False,
        **inference_kwargs,
    ) -> GroupInferenceResults:
        """
        Run all tests, in a group, performing alpha correction, and adjusting
        the inference results to mitigate multiple comparison error.

        Parameters
        ----------
        test_group : HypothesisTestGroup
            A hypothesis test object
        alpha : float
            The Type I error assumed by the experimenter when running the test
        display_results : boolean
            Whether to print the test results to stdout
        visualize_results : boolean
            Whether to render a visual representation of the results

        Returns
        -------
        test_results: GroupInferenceResults
            The corrected results of the statistical inferences, when applied
            within the context of the group.
        """

        corrected_tests = [copy.deepcopy(t) for t in test_group.tests]

        # run original tests
        original_results = [
            self.run_test(test, alpha, **inference_kwargs) for test in test_group.tests
        ]

        # Run grouped test with correction
        # get p_values for multiple comparison procedure
        p_values = [t.p_value for t in original_results]

        correction_method = test_group.correction_method
        correction = MultipleComparisonCorrection(
            p_values=p_values, alpha=alpha, method=correction_method
        )

        # Rerun hypothesis tests with updated alpha
        corrected_results = [
            self.run_test(
                test,
                alpha=correction.alpha_corrected,
                correction_method=correction.mc_correction_method.__name__,
            )
            for test in corrected_tests
        ]

        test_group_results = GroupInferenceResults(
            test_group=test_group,
            correction=correction,
            original_results=original_results,
            corrected_results=corrected_results,
        )

        if display_results:
            test_group_results.display()

        if visualize_results:
            test_group_results.visualize()

        test_group_results.run_at = datetime.now()

        return test_group_results
