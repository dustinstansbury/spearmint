from dataclasses import dataclass

import numpy as np
import pymc as pm
from arviz import InferenceData

from spearmint.config import N_POSTERIOR_SAMPLES, DEFAULT_PARAMETER_ESTIMATION_METHOD
from spearmint.inference.bayesian.models.analytic_base import BayesianAnalyticModel
from spearmint.inference.inference_base import (
    InferenceProcedure,
    InferenceResults,
    InferenceResultsMissingError,
    OrderedDict,
)
from spearmint.stats import Samples
from spearmint.table import SpearmintTable
from spearmint.typing import Any, Dict, FilePath, Tuple, Union, Optional
from spearmint.utils import format_value, process_warnings

# TODO: Enums for all these
CONTINUOUS_MODEL_NAMES = ["gaussian", "student_t"]

BINARY_MODEL_NAMES = [
    "bernoulli",
    "binomial",
]

COUNTS_MODEL_NAMES = ["poisson"]
SUPPORTED_BAYESIAN_MODEL_NAMES = (
    CONTINUOUS_MODEL_NAMES + BINARY_MODEL_NAMES + COUNTS_MODEL_NAMES
)

DEFAULT_VARIABLE_TYPE_MODELS = {
    "continuous": "gaussian",
    "binary": "binomial",
    "counts": "poisson",
}

SUPPORTED_PARAMETER_ESTIMATION_METHODS = ("mcmc", "advi", "analytic")


class UnsupportedParameterEstimationMethodException(Exception):
    pass


def _get_model_name(
    variable_type: str,
    model_name: Optional[str],
) -> str:
    if model_name is None:
        return DEFAULT_VARIABLE_TYPE_MODELS[variable_type]

    clean_model_name = model_name.replace("-", "_").replace(" ", "_")
    if clean_model_name not in SUPPORTED_BAYESIAN_MODEL_NAMES:
        raise ValueError(f"Unsupported model: {clean_model_name}")
    return clean_model_name


def _get_model_datatype(model_name: str) -> type:
    if model_name in CONTINUOUS_MODEL_NAMES:
        return float
    return int


def _get_delta_param(model_name):
    """Bayesian mnodel param we calculate delta on"""
    if model_name in CONTINUOUS_MODEL_NAMES:
        return "mu"
    elif model_name in BINARY_MODEL_NAMES:
        return "p"
    elif model_name in COUNTS_MODEL_NAMES:
        return "lambda"


class BayesianInferenceResults(InferenceResults):
    """
    Class for storing, displaying, visualizing, and exporting Bayesian
    hypothesis test results.
    """

    def __init__(
        self,
        prior: Samples,
        delta_posterior: Samples,
        delta_relative_posterior: Samples,
        control_posterior: Samples,
        variation_posterior: Samples,
        effect_size_posterior: Samples,
        delta_hdi: Tuple[float, float],
        delta_relative_hdi: Tuple[float, float],
        effect_size_hdi: Tuple[float, float],
        hdi_percentiles: Tuple[float, float],
        prob_greater_than_zero: float,
        bayesian_parameter_estimation_method: str,
        model_name: str,
        data_type: type,
        bayesian_model_params: Optional[Dict[str, Any]] = None,
        model_hyperparams: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior = prior
        self.control_posterior = control_posterior
        self.variation_posterior = variation_posterior
        self.delta_posterior = delta_posterior
        self.delta_relative_posterior = delta_relative_posterior
        self.effect_size_posterior = effect_size_posterior
        self.delta_hdi = delta_hdi
        self.delta_relative_hdi = delta_relative_hdi
        self.effect_size_hdi = effect_size_hdi
        self.prob_greater_than_zero = prob_greater_than_zero
        self.bayesian_parameter_estimation_method = bayesian_parameter_estimation_method
        self.data_type = data_type
        self.model_name = model_name
        self.bayesian_model_params = (
            bayesian_model_params if bayesian_model_params else {}
        )
        self.model_hyperparams = model_hyperparams if model_hyperparams else {}
        self.hdi_percentiles = hdi_percentiles

    @property
    def _specific_properties(self):
        """
        Properties specific to the current type of test
        """
        return OrderedDict(
            [
                ("test_type", "bayesian"),
                (
                    f"p({self.variation.name} > {self.control.name})",
                    self.prob_greater_than_zero,
                ),
                ("delta_hdi", self.delta_hdi),
                ("hdi_percentiles", self.hdi_percentiles),
                ("relative_delta_hdi", self.delta_relative_hdi),
                ("effect_size_hdi", self.effect_size_hdi),
                ("credible_mass", 1 - self.alpha),
            ]
        )

    def _render_stats_table(self):
        self._stats_table = BayesianInferenceResultsTable(self)
        return self._stats_table


class BayesianInferenceResultsTable(SpearmintTable):
    def __init__(self, results: BayesianInferenceResults):
        super().__init__(title="Bayesian Delta Results", show_header=False)

        # Add results rows
        self.add_row(
            "Delta",
            format_value(results.delta, precision=4),
        )
        self.add_row(
            "Delta HDI",
            format_value(results.delta_hdi, precision=4),
        )
        self.add_row(
            "Delta Relative",
            format_value(100 * results.delta_relative, precision=2) + " %",
        )
        self.add_row(
            "Delta-relative HDI",
            format_value(100 * np.array(results.delta_relative_hdi), precision=2)
            + " %",
        )
        self.add_row(
            "Effect Size",
            format_value(results.effect_size, precision=4),
        )
        self.add_row(
            "Effect Size HDI",
            format_value(results.effect_size_hdi, precision=4),
        )
        self.add_row(
            "HDI %-tiles",
            format_value(results.hdi_percentiles, precision=4),
        )
        self.add_row(
            "Credible Mass",
            format_value(1 - results.alpha, precision=2),
        )
        self.add_row(
            "Variable Type",
            results.variable_type,
        )
        self.add_row(
            "Inference Method",
            "Bayesian",
        )
        self.add_row(
            "Model Name",
            results.model_name,
        )
        self.add_row(
            "Estimation Method",
            results.bayesian_parameter_estimation_method,
        )
        self.add_row(
            f"p({results.variation.name} > {results.control.name})",
            format_value(results.prob_greater_than_zero, precision=4),
        )
        self.add_row(
            "Hypothesis",
            results.hypothesis_text,
        )
        self.add_row(
            "Accept Hypothesis",
            str(results.accept_hypothesis),
        )
        if results.warnings:
            self.add_row(
                "Warnings",
                process_warnings(results.warnings),
            )


def visualize_bayesian_delta_results(
    results: BayesianInferenceResults,
    outfile: Optional[FilePath] = None,
    include_prior: bool = False,
):  # pragma: no cover
    # Lazy import
    import holoviews as hv

    from spearmint import vis

    control_posterior = results.control_posterior
    variation_posterior = results.variation_posterior
    delta_samples = results.delta_posterior
    credible_mass = 1 - results.alpha
    comparison_param = _get_delta_param(results.model_name)

    control_dist = vis.plot_kde(
        samples=control_posterior.data,
        label=results.control.name,
        color=vis.CONTROL_COLOR,
    )

    variation_dist = vis.plot_kde(
        samples=variation_posterior.data,
        label=results.variation.name,
        color=vis.VARIATION_COLOR,
    )

    # Confidence intervals
    control_ci = vis.plot_interval(
        *control_posterior.hdi(credible_mass),
        middle=control_posterior.mean,
        label=results.control.name,
        color=vis.CONTROL_COLOR,
        show_interval_text=True,
    )  # type: ignore # (mypy bug, see #6799)

    variation_ci = vis.plot_interval(
        *variation_posterior.hdi(credible_mass),
        middle=variation_posterior.mean,
        label=results.variation.name,
        color=vis.VARIATION_COLOR,
        show_interval_text=True,
    )  # type: ignore # (mypy bug, see #6799)

    distribution_plot = control_dist * variation_dist * control_ci * variation_ci

    if include_prior:
        prior_dist = vis.plot_kde(
            samples=results.prior.data, label="prior", color=vis.PRIOR_COLOR
        )
        distribution_plot *= prior_dist

    distribution_plot = distribution_plot.relabel(
        f"Posterior {comparison_param} Comparison"
    ).opts(
        legend_position="top_right",
        xlabel=f"Posterior {comparison_param}",
        ylabel="pdf",
    )

    delta_dist = vis.plot_kde(
        samples=delta_samples.data,
        label="Delta Distribution",
        color=vis.DELTA_COLOR,
    )

    max_pdf_height = delta_dist.data["pdf"].max()
    mean_delta = delta_samples.mean

    delta_ci = vis.plot_interval(
        *delta_samples.hdi(credible_mass),
        mean_delta,
        color=vis.NEUTRAL_COLOR,
        label=f"{credible_mass}% HDI",
        show_interval_text=True,
        vertical_offset=-(max_pdf_height * 0.01),
    )  # type: ignore # (mypy bug, see #6799)

    vline = hv.Spikes(([0.0], [max_pdf_height]), vdims="pdf", label="Null Delta").opts(
        color=vis.COLORS.red
    )

    delta_plot = delta_dist * delta_ci * vline
    delta_plot_title = f"Posterior {comparison_param} Delta"
    delta_plot = (
        delta_plot.relabel(delta_plot_title)
        .opts(xlabel="delta", ylabel="pdf")
        .opts(legend_position="top_right")
    )

    visualization = distribution_plot + delta_plot
    visualization.opts(shared_axes=False)

    if outfile is not None:
        vis.save_visualization(visualization, outfile)

    return visualization


@dataclass
class _BayesianModel:
    """Container for Bayesian model object and available fitting options"""

    model_name: str
    model_object: Union[pm.Model, BayesianAnalyticModel]
    mcmc_estimation_supported: bool
    advi_estimation_supported: bool
    analytic_estimation_supported: bool
    hyperparams: Dict[str, Any]


def _build_bayesian_inference_model(
    model_name: str,
    control_samples: Samples,
    variation_samples: Samples,
    bayesian_parameter_estimation_method: str,
    **bayesian_model_params,
) -> _BayesianModel:
    mcmc_estimation_supported = True  # MCMC works for all models
    advi_estimation_supported = False  # ADVI not supported for all models
    analytic_estimation_supported = False  # No Analytic solution for all model

    model_type = (
        "analytic" if bayesian_parameter_estimation_method == "analytic" else "pymc"
    )

    if model_name in CONTINUOUS_MODEL_NAMES:
        from .models import continuous

        advi_estimation_supported = True  # ADVI suppored for all continuous models
        analytic_estimation_supported = True if model_name == "gaussian" else False
        model_builder = getattr(continuous, f"build_{model_name}_{model_type}_model")

    if model_name in BINARY_MODEL_NAMES:
        from .models import binary

        advi_estimation_supported = True if model_name == "bernoulli" else False
        analytic_estimation_supported = True
        model_builder = getattr(binary, f"build_{model_name}_{model_type}_model")

    if model_name in COUNTS_MODEL_NAMES:
        from .models import counts

        advi_estimation_supported = False  # Discrete distributions not supported
        analytic_estimation_supported = True
        model_builder = getattr(counts, f"build_{model_name}_{model_type}_model")

    model_object, hyperparams = model_builder(
        control_samples, variation_samples, **bayesian_model_params
    )

    return _BayesianModel(
        model_name=model_name,
        model_object=model_object,
        mcmc_estimation_supported=mcmc_estimation_supported,
        advi_estimation_supported=advi_estimation_supported,
        analytic_estimation_supported=analytic_estimation_supported,
        hyperparams=hyperparams,
    )


def _fit_model_mcmc(model: _BayesianModel, **inference_kwargs) -> InferenceData:
    if model.mcmc_estimation_supported:
        with model.model_object:  # type: ignore  # context manager only on PyMC model_objects
            return pm.sample(**inference_kwargs)
    raise UnsupportedParameterEstimationMethodException(
        f"MCMC not supported for {model.model_name} model"
    )


def _fit_model_advi(model: _BayesianModel, **inference_kwargs) -> InferenceData:
    if model.advi_estimation_supported:
        with model.model_object:  # type: ignore  # context manager only on PyMC model_objects
            mean_field = pm.fit(method="advi", **inference_kwargs)

            return mean_field.sample(N_POSTERIOR_SAMPLES)

    raise UnsupportedParameterEstimationMethodException(
        f"ADVI parameter estimation not supported for {model.model_name} model"
    )


def _fit_model_analytic(model: _BayesianModel, **inference_kwargs) -> InferenceData:
    if model.analytic_estimation_supported:
        return model.model_object.sample(N_POSTERIOR_SAMPLES)

    raise UnsupportedParameterEstimationMethodException(
        f"Analytic parameter estimation not supported for {model.model_name} model"
    )


class BayesianInferenceProcedure(InferenceProcedure):
    """
    Bayesian inference procedure to test for the difference between two
    samples.
    """

    def __init__(
        self,
        bayesian_model_name: Optional[str] = None,
        bayesian_model_params: Optional[dict] = None,
        bayesian_parameter_estimation_method: str = DEFAULT_PARAMETER_ESTIMATION_METHOD,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model_name: str
            The name of the inference model to use:
                -   "gaussian"       : Hierarchical Gaussian model (continuous)
                -   "student-t"      : Hierarchical Student's t model (continuous)
                -   "binomial"       : Beta-Binomial hierarchical model (binary)
                -   "bernoulli"      : Beta-Bernoulli hierarchical model (binary)
                -   "poisson"        : Gamma-Poisson hierarichcal model (counts)
        bayesian_parameter_estimation_method: str
            The method used estimate the posterior model parameters. One of:
                -   'mcmc' : use Markov Chain Monte Carlo via PyMC. All models,
                    supported, but may not scale well with large datasets.
                -   'advi' : use Autodiff Variational Inference; scales better than
                    'mcmc', but can only accurately estimate smooth, single-mode
                    posteriors.
                -   'analytic' : uses prior-likelihood conjugacy to estimate the
                    parameters using an analytic solution. Scales best, but only
                    supported by `gaussian`, `beta-binomial`, and `gamma-poisson`
                    models.
            Default is 'mcmc'.
        *args, **kwargs
            Arguments specific for initializing each type of Bayesian model.
            See bayesian inference README for details on specifications for each
            model.
        """
        super().__init__(*args, **kwargs)
        self.model_name = _get_model_name(self.variable_type, bayesian_model_name)
        self.data_type = _get_model_datatype(self.model_name)
        assert (
            bayesian_parameter_estimation_method
            in SUPPORTED_PARAMETER_ESTIMATION_METHODS
        )
        self.bayesian_parameter_estimation_method = bayesian_parameter_estimation_method
        self.bayesian_model_params = (
            bayesian_model_params if bayesian_model_params else {}
        )
        self.inference_data: Optional[InferenceData] = None

    def _process_samples(self, samples: Samples) -> Samples:
        return Samples(np.array(samples.data, dtype=self.data_type), name=samples.name)

    # @abstractmethod
    def _run_inference(
        self, control_samples: Samples, variation_samples: Samples, **inference_kwargs
    ) -> None:
        """
        Run the inference procedure over the samples

        Parameters
        ----------
        control_samples: Samples
            the samples for the control condition
        variation_samples: Samples
            the samples for the varitation condition
        """
        control_samples = self._process_samples(control_samples)
        variation_samples = self._process_samples(variation_samples)

        _bayesian_model = _build_bayesian_inference_model(
            model_name=self.model_name,
            control_samples=control_samples,
            variation_samples=variation_samples,
            bayesian_parameter_estimation_method=self.bayesian_parameter_estimation_method,
            **self.bayesian_model_params,
        )

        if self.bayesian_parameter_estimation_method == "mcmc":
            self.inference_data = _fit_model_mcmc(_bayesian_model, **inference_kwargs)

        elif self.bayesian_parameter_estimation_method == "advi":
            self.inference_data = _fit_model_advi(_bayesian_model, **inference_kwargs)

        elif self.bayesian_parameter_estimation_method == "analytic":
            self.inference_data = _fit_model_analytic(
                _bayesian_model, **inference_kwargs
            )

        # Set attributes
        comparison_param = _get_delta_param(self.model_name)
        self.control = control_samples
        self.variation = variation_samples
        self.prior = self.posterior_samples("prior")
        self.control_posterior = self.posterior_samples(f"{comparison_param}_control")
        self.variation_posterior = self.posterior_samples(
            f"{comparison_param}_variation"
        )
        self.delta_posterior = self.posterior_samples("delta")
        self.delta_relative_posterior = self.posterior_samples("delta_relative")
        self.effect_size_posterior = self.posterior_samples("effect_size")
        self.model_hyperparams = _bayesian_model.hyperparams

    def posterior_samples(self, parameter_name: str) -> Samples:
        """
        Return samples from the model posterior for the requested parameter

        Parameters
        ----------
        parameter_name : str
            The name of the model parameter to sample

        Returns
        -------
        posterior_samples : Samples
            Samples from the inference procedure's Bayesian model posterior
            distribution.

        Raises
        ------
        InferenceResultsMissingError
            If the inference procedure has not been run, there is no posterior
            to sample.
        """
        if self.inference_data is not None:
            return Samples(
                observations=self.inference_data.posterior[  # type: ignore  # posterior attribute not initialized by default
                    parameter_name
                ].values.flatten(),
                name=parameter_name,
            )
        raise InferenceResultsMissingError("You may need to execute the .run() method")

    @property
    def test_stats(self) -> Dict[str, Any]:
        prob_greater_than_zero = self.delta_posterior.prob_greater_than(0.0)
        return {
            "model_name": self.model_name,
            "delta": self.delta_posterior.mean,
            "delta_relative": self.delta_relative_posterior.mean,
            "delta_hdi": self.delta_posterior.hdi(1 - self.alpha),
            "delta_relative_hdi": self.delta_relative_posterior.hdi(1 - self.alpha),
            "effect_size": self.effect_size_posterior.mean,
            "effect_size_hdi": self.effect_size_posterior.hdi(1 - self.alpha),
            "hdi_percentiles": (self.alpha / 2, 1 - self.alpha / 2),
            "hypothesis": self.hypothesis,
            "prob_greater_than_zero": prob_greater_than_zero,
            "accept_hypothesis": prob_greater_than_zero >= 1 - self.alpha,
        }

    # @abstractmethod
    def _make_results(self) -> BayesianInferenceResults:
        """
        Package up inference results
        """
        test_stats = self.test_stats

        results = BayesianInferenceResults(
            control=self.control,
            variation=self.variation,
            prior=self.prior,
            control_posterior=self.control_posterior,
            variation_posterior=self.variation_posterior,
            delta_posterior=self.delta_posterior,
            delta_relative_posterior=self.delta_relative_posterior,
            effect_size_posterior=self.effect_size_posterior,
            bayesian_parameter_estimation_method=self.bayesian_parameter_estimation_method,
            model_name=self.model_name,
            alpha=self.alpha,
            hypothesis=self.hypothesis,
            inference_method=self.inference_method,
            variable_type=self.variable_type,
            metric_name="posterior_delta",
            data_type=self.data_type,
            bayesian_model_params=self.bayesian_model_params,
            model_hyperparams=self.model_hyperparams,
            delta=test_stats["delta"],
            delta_relative=test_stats["delta_relative"],
            delta_hdi=test_stats["delta_hdi"],
            delta_relative_hdi=test_stats["delta_relative_hdi"],
            effect_size=test_stats["effect_size"],
            effect_size_hdi=test_stats["effect_size_hdi"],
            hdi_percentiles=test_stats["hdi_percentiles"],
            prob_greater_than_zero=test_stats["prob_greater_than_zero"],
            accept_hypothesis=test_stats["accept_hypothesis"],
            visualization_function=visualize_bayesian_delta_results,
        )

        return results
