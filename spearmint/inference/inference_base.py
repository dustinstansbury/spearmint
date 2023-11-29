from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from datetime import datetime

from holoviews import Element

from spearmint.typing import FilePath, List, Callable, Optional, Any, Union
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


class VariableType(str, Enum):
    continuous = "continuous"
    binary = "binary"
    counts = "counts"


class InferenceMethod(str, Enum):
    frequentist = "frequentist"
    bayesian = "bayesian"
    bootstrap = "bootstrap"


class Hypothesis(str, Enum):
    larger = "larger"
    smaller = "smaller"
    unequal = "unequal"


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
        hypothesis: Hypothesis,
        alpha: float,
        accept_hypothesis: bool,
        inference_method: InferenceMethod,
        variable_type: VariableType,
        visualization_function: Callable,
        comparison_type: Optional[str] = None,
        warnings: Optional[Union[str, List[str]]] = None,
        correction_method: Optional[str] = None,
        segmentation: Optional[str] = None,
        aux: dict = {},
    ):
        self.control = control
        self.variation = variation
        self.metric_name = metric_name
        self.delta = delta
        self.delta_relative = delta_relative
        self.effect_size = effect_size
        self.hypothesis = hypothesis
        self.variable_type = variable_type
        self.alpha = alpha
        self.accept_hypothesis = accept_hypothesis
        self.inference_method = inference_method
        self.comparison_type = comparison_type
        self.warnings = process_warnings(warnings)
        self.aux = aux
        self.visualization_function = visualization_function
        self.correction_method = correction_method
        self.segmentation = segmentation

        self.created_at: datetime = datetime.now()

        # These properties should be updated by the running inference proc.
        self.run_at: Optional[datetime] = None
        self.p_value: Optional[float] = None  # only Frequentist/Bootstrap tests

    def display(self) -> None:
        """Display the inference procedure results to the console"""
        try:
            self.summary
        except Exception as e:
            raise InferenceResultsDisplayError(e)

    def visualize(
        self, outfile: Optional[FilePath] = None, *args, **kwargs
    ) -> Element:  # pragma: no cover
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
            return self.visualization_function(self, outfile, *args, **kwargs)

        except Exception as e:
            raise InferenceResultsVisualizationError(e)

    def _render_stats_table(self):
        raise NotImplementedError(
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
                ("variable_type", self.variable_type),
                ("segmentation", self.segmentation),
                ("warnings", self.warnings),
            ]
        )

    @property
    def _specific_properties(self) -> OrderedDict:
        """
        Properties for specific inference procedures
        """
        return OrderedDict()

    def to_dict(self) -> OrderedDict[Any, Any]:
        _dict = self._base_properties
        _dict.update(self._specific_properties)
        assert isinstance(_dict, OrderedDict)
        return _dict

    @property
    def dict(self) -> OrderedDict:
        return self.to_dict()


class InferenceProcedure(ABC):
    """
    Base class for all inference procedures. Must implement the following
    private abstracmethods:

        -   `_run_inference()` : runs the inference procedure, updating any internal
            state needed for displaying and/or visualizing results
        -   `_make_results()` : generate `InferenceResults` based on the outcome
            of `_run_inference()`
    """

    def __init__(
        self,
        variable_type: VariableType = VariableType.continuous,
        inference_method: InferenceMethod = InferenceMethod.frequentist,
        hypothesis: Hypothesis = Hypothesis.larger,
        alpha: float = DEFAULT_ALPHA,
        metric_name: Optional[str] = None,
        **inference_procedure_params,
    ):
        """_summary_

        Parameters
        ----------
        variable_type : VariableType, optional
            The variable type we run inference on. One of "continuous", "binary",
            or "counts"; by default "continuous".
        inference_method : InferenceMethod, optional
            The inference method to use. One of "frequentist", "bootstrap",
            or "bayesian"; by default "frequentist"
        hypothesis : Hypothesis, optional
            The directionality of the hypothesis, namely whether the variation
            is "larger", "smaller", or "unequal" compared to the control;
            default "larger". If None provided we use the value configured in
            `spearmint.cfg::hypothesis_test::default_hypothesis`
        alpha : float, optional
            The acceptable Type I error for the inference procedure. If None
            provided, we use the value configured in
            `spearmint.cfg::hypothesis_test::default_alpha`
        metric_name : str, optional
            The name of the metric used to compare groups during the inference
            procedure, mostly used for reporting. By default None
        **inference_procedure_params
            Any additional parameters used to customize a specific inference
            procedure on init.
        """
        self.variable_type = variable_type
        self.inference_method = inference_method
        self.metric_name = metric_name
        self.hypothesis = hypothesis
        self.alpha = alpha

        # Updated after running the inference procedure
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
        pass

    def run(
        self,
        control_samples: Samples,
        variation_samples: Samples,
    ) -> InferenceResults:
        """
        Run the current inference procedure on control / variation samples.

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
        self._results = self._make_results()  # type: ignore
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
    variable_type: str, inference_method: str, **inference_procedure_params
) -> InferenceProcedure:
    if inference_method == InferenceMethod.frequentist:
        if variable_type == VariableType.continuous:
            from .frequentist.means_delta import MeansDelta as IP

        elif variable_type == VariableType.binary:
            from .frequentist.proportions_delta import ProportionsDelta as IP  # type: ignore

        elif variable_type == VariableType.counts:
            from .frequentist.rates_ratio import RatesRatio as IP  # type: ignore
        else:
            raise ValueError(f"Unknown variable type `{variable_type}`")

    elif inference_method == InferenceMethod.bootstrap:
        from .frequentist.bootstrap_delta import BootstrapDelta as IP  # type: ignore

    elif inference_method == InferenceMethod.bayesian:
        from .bayesian.bayesian_inference import BayesianInferenceProcedure as IP  # type: ignore
    else:
        raise ValueError(f"Unknown inference method `{inference_method}`")

    return IP(
        variable_type=variable_type,
        inference_method=inference_method,
        **inference_procedure_params,
    )
