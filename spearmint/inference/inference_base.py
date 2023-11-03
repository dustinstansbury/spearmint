from abc import ABC, abstractmethod
from collections import OrderedDict

from datetime import datetime

from holoviews import Element

from spearmint.config import logger
from spearmint.typing import FilePath, List, Callable
from spearmint.mixin import DataframeableMixin
from spearmint.stats import Samples, SamplesComparisonTable, DEFAULT_ALPHA
from spearmint.utils import process_warnings


class InferenceResultsDisplayError(Exception):
    """Raised when there is an error displaying inference procedure results"""

    pass


class InferenceResultsVisualizationError(Exception):
    """Raised when there is an error visualizaing inference procedure results"""

    pass


class InferenceResultsMissingError(Exception):
    """Raised when there are no results associated with an inference procedure."""

    pass


class InferenceResults(DataframeableMixin):
    """
    Base class for displaying and saving statistical inference results

    Parameters
    ----------
    control : Samples
        the control samples
    variation : Samples
        the variation samples
    metric_name : str
        The name of the metric that is being compared
    delta: float
        the absolute difference between the variation and control sample means
    delta_relative: float
        the percent difference between the variation and control sample means
    hypothesis : str
        Human-readable message for interpreting the experiment
    alpha : float in (0, 1)
        The "significance" level, or type I error rate for the experiment
    accept_hypothesis : boolean
        Whether or not to accept the alternative hypothesis, based on `p_value`
        and `alpha` and any correction method.
    warnings : list[str]
        A list of any warning messages accumultaed during the test
    aux : dict
        Auxillary variables used for displaying or visualizing specific types
        of tests.
    """

    def __init__(
        self,
        control: Samples,
        variation: Samples,
        metric_name: str,
        delta: float,
        delta_relative: float,
        effect_size: float,
        hypothesis: str,
        alpha: float,
        accept_hypothesis: bool,
        inference_method: str = None,
        comparison_type: str = None,
        warnings: List[str] = [],
        aux: dict = {},
        visualization_function: Callable = None,
    ):
        self.control = control
        self.variation = variation
        self.metric_name = metric_name
        self.delta = delta
        self.delta_relative = delta_relative
        self.effect_size = effect_size
        self.hypothesis = hypothesis
        self.alpha = alpha
        self.accept_hypothesis = accept_hypothesis
        self.inference_method = inference_method
        self.comparison_type = comparison_type
        self.warnings = process_warnings(warnings)
        self.aux = aux
        self.visualization_function = visualization_function

        self.created_at: datetime.now()
        self.run_at: datetime.timestamp = None

        # These properties should be updated by the running the inference
        # self.segmentation = None

    def display(self) -> None:
        """Display the inference procedure results to the console"""
        try:
            self.summary
        except Exception as e:
            raise InferenceResultsDisplayError(e)

    def visualize(self, outfile: FilePath = None, *args, **kwargs) -> Element:
        """Visualize the inference procedure results

        Parameters
        ----------
        outfile : FilePath, optional
            If provided, use to save the output visualization, by default None

        Returns
        -------
        Element : holoviews.Element
            A constructed visualization object

        Raises
        ------
        InferenceResultsVisualizationError
            If the `_render_visualization` is not implemented. Note: we do not
            require `_render_visualization` via abstractmethod, leaving the
            option for std-out-only reporting
        """
        try:
            return self.visualization_function(self, outfile)

        except Exception as e:
            raise InferenceResultsVisualizationError(e)

    def _render_stats_table(self):
        raise NotImplemented(
            "No implementation of `_render_stats_table` for ",
            f"class {self.__name__}, cannot execute `.summary`",
        )

    def to_csv(self, outfile: FilePath, delimiter=","):
        """
        Export result to delimiter-separated value file
        """
        results_df = self.to_dataframe()
        results_df.to_csv(outfile, sep=delimiter, encoding="utf8", index=False)

    @property
    def hypothesis_text(self):
        if self.hypothesis == "larger":
            return f"{self.variation.name} is larger"
        elif self.hypothesis == "smaller":
            return f"{self.variation.name} is smaller"
        elif self.hypothesis == "unequal":
            return f"{self.variation.name} != {self.control.name}"
        else:
            raise ValueError("Unknown hypothesis: {self.hypothesis}")

    @property
    def samples_comparison(self):
        if not hasattr(self, "_samples_comparison_table"):
            self._samples_comparison_table = SamplesComparisonTable(
                self.control, self.variation
            )
        self._samples_comparison_table.print()

    @property
    def stats(self):
        if not hasattr(self, "_stats_table"):
            self._stats_table = self._render_stats_table()
        self._stats_table.print()

    @property
    def summary(self):
        self.samples_comparison
        self.stats

    @property
    def _base_properties(self):
        """
        Properties that belong to all inference procedure results
        """
        return OrderedDict(
            [
                ("control_name", self.control.name),
                ("control_nobs", self.control.nobs),
                ("control_mean", self.control.mean),
                ("control_ci", self.control.confidence_interval(self.alpha)),
                ("control_var", self.control.var),
                ("variation_name", self.variation.name),
                ("variation_nobs", self.variation.nobs),
                ("variation_mean", self.variation.mean),
                ("variation_ci", self.variation.confidence_interval(self.alpha)),
                ("variation_var", self.variation.var),
                ("metric", self.metric_name),
                ("delta", self.delta),
                ("delta_relative", 100 * self.delta_relative),
                ("effect_size", self.effect_size),
                ("alpha", self.alpha),
                ("hypothesis", self.hypothesis),
                ("accept_hypothesis", self.accept_hypothesis),
                ("inference_method", self.inference_method),
                # ("segmentation", self.segmentation),
                ("warnings", self.warnings),
            ]
        )

    @property
    def _specific_properties(self) -> OrderedDict:
        """
        Properties for specific inference procedures
        """
        return OrderedDict()

    def to_dict(self) -> OrderedDict:
        _dict = self._base_properties
        _dict.update(self._specific_properties)
        return _dict

    @property
    def dict(self) -> OrderedDict:
        return self.to_dict()


class InferenceProcedure(ABC):
    """
    Base class for all inference procedures. Must implement the following methods:

        -   `_run_inference()` : runs the inference procedure, updating any internal
            state needed for displaying and/or visualizing results
        -   `_make_results()` : generate `InferenceResults` based on the outcome
            of `_run_inference()`
    """

    def __init__(
        self,
        inference_method: str,
        metric_name: str = None,
        hypothesis: str = "larger",
        alpha: float = DEFAULT_ALPHA,
        **inference_procedure_init_params,
    ):
        self.inference_method = inference_method
        self.metric_name = metric_name
        self.hypothesis = hypothesis
        self.alpha = alpha

        # These are updated after runnint the inference procedure
        self._results = None

    @abstractmethod
    def _run_inference(
        self, control_samples: Samples, variation_samples: Samples, **inference_kwargs
    ) -> None:
        raise NotImplementedError("Implement me")

    @abstractmethod
    def _make_results(self) -> InferenceResults:
        """
        Generate and return a `InferenceResults` object
        """
        raise NotImplementedError("Implement me")

    def run(
        self,
        control_samples: Samples,
        variation_samples: Samples,
    ) -> InferenceResults:
        """
        Run inference procedure on control / variation samples, and report results

        Parameters
        ----------
        control : Samples
            the control samples
        variation : Samples
            the variation samples
        **inference_kwargs: kwargs
            Any inference-specific arguments to include in the call signature.

        Returns
        -------
        results : InferenceResults
            The results of running the inference procedure

        """
        self._run_inference(control_samples, variation_samples)
        self._results = self._make_results()
        return self.results

    @property
    def results(self) -> InferenceResults:
        """
        Raises
        ------
        InferenceResultsMissingError
            if the `run` method hasn't been executed
        """
        if self._results is None:
            raise InferenceResultsMissingError(
                "You likely need to execute the `run` method.",
                " Or check the implementation of `_make_results`",
            )
        return self._results


def get_inference_procedure(
    inference_method: str, **inference_procedure_init_params
) -> InferenceProcedure:
    _method = inference_method.lower().replace("-", "_").replace(" ", "_")
    if _method in ("means_delta"):
        from .frequentist.means_delta import MeansDelta as IP

    elif _method in ("proportions_delta"):
        from .frequentist.proportions_delta import ProportionsDelta as IP

    elif _method in ("rates_ratio"):
        from .frequentist.rates_ratio import RatesRatio as IP

    elif _method in ("bootstrap"):
        from .frequentist.bootstrap_delta import BootstrapDelta as IP

    elif _method in (
        "gaussian",
        "bernoulli",
        "binomial",
        "beta_binomial",
        "gamma_poisson",
        "student_t",
        "exp_student_t",
    ):
        from .bayesian.bayesian_inference import BayesianInferenceProcedure as IP
    else:
        raise ValueError(f"Unknown inference method {inference_method}")
    return IP(inference_method=inference_method, **inference_procedure_init_params)
