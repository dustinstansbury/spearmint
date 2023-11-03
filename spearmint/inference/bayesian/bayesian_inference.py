import numpy as np
import pymc as pm

from dataclasses import dataclass

from spearmint.typing import Union, Dict, Any, Tuple, FilePath
from spearmint.table import SpearmintTable
from spearmint.stats import Samples
from spearmint.utils import format_value, process_warnings
from spearmint.inference.inference_base import (
    InferenceProcedure,
    InferenceResultsMissingError,
)


N_MEAN_FIELD_SAMPLES = 1000

CONTINUOUS_MODEL_NAMES = ["gaussian", "student_t"]

BINARY_MODEL_NAMES = [
    "bernoulli",
    "binomial",
]

COUNTS_MODEL_NAMES = ["poisson"]
SUPPORTED_BAYESIAN_MODEL_NAMES = (
    CONTINUOUS_MODEL_NAMES + BINARY_MODEL_NAMES + COUNTS_MODEL_NAMES
)


def _get_model_name(model_name: str) -> str:
    clean_model_name = model_name.replace("-", "_").replace(" ", "_")
    if clean_model_name not in SUPPORTED_BAYESIAN_MODEL_NAMES:
        raise ValueError(f"Unsupported model: {clean_model_name}")
    return clean_model_name


def _get_model_datatype(model_name: str) -> Union[float, int]:
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


from spearmint.inference.inference_base import InferenceResults, OrderedDict


class BayesianInferenceResults(InferenceResults):
    """
    Class for storing, displaying, visualizing, and exporting Bayesian
    hypothesis test results.
    """

    def __init__(
        self,
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
        model_name: str,
        data_type: Union[float, int],
        model_params: Dict[str, Any] = None,
        model_hyperparams: Dict[str, Any] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.control_posterior = control_posterior
        self.variation_posterior = variation_posterior
        self.delta_posterior = delta_posterior
        self.delta_relative_posterior = delta_relative_posterior
        self.effect_size_posterior = effect_size_posterior
        self.delta_hdi = delta_hdi
        self.delta_relative_hdi = delta_relative_hdi
        self.effect_size_hdi = effect_size_hdi
        self.prob_greater_than_zero = prob_greater_than_zero
        self.model_name = model_name
        self.data_type = data_type
        self.model_params = model_params if model_params else {}
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
        self._stats_table = BayesianTestResultsTable(self)
        return self._stats_table


class BayesianTestResultsTable(SpearmintTable):
    def __init__(self, results: BayesianInferenceResults):
        super().__init__(title="Bayesian Delta Results", show_header=False)

        # Add results rows
        self.add_row(
            "Delta",
            format_value(results.delta, precision=4),
        )
        self.add_row(
            f"Delta HDI",
            format_value(results.delta_hdi, precision=4),
        )
        self.add_row(
            "Delta Relative",
            format_value(100 * results.delta_relative, precision=2) + " %",
        )
        self.add_row(
            f"Delta-relative HDI",
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
            "Inference Method",
            "Bayesian",
        )
        self.add_row(
            "Model Name",
            results.model_name,
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
    results: BayesianInferenceResults, outfile: FilePath = None
):
    # Lazy import
    from spearmint import vis
    import holoviews as hv

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
    )

    variation_ci = vis.plot_interval(
        *variation_posterior.hdi(credible_mass),
        middle=variation_posterior.mean,
        label=results.variation.name,
        color=vis.VARIATION_COLOR,
        show_interval_text=True,
    )

    distribution_plot = control_dist * variation_dist * control_ci * variation_ci
    distribution_plot = distribution_plot.relabel(
        f"Posterior {comparison_param} Comparison"
    ).opts(
        legend_position="right", xlabel=f"Posterior {comparison_param}", ylabel="pdf"
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
    )

    vline = hv.Spikes(([0.0], [max_pdf_height]), vdims="pdf", label="Null Delta").opts(
        color=vis.COLORS.red
    )

    delta_plot = delta_dist * delta_ci * vline
    delta_plot_title = f"Posterior {comparison_param} Delta"
    delta_plot = (
        delta_plot.relabel(delta_plot_title)
        .opts(xlabel="delta", ylabel="pdf")
        .opts(legend_position="right")
    )

    visualization = distribution_plot + delta_plot
    visualization.opts(shared_axes=False).cols(1)

    if outfile is not None:
        vis.save_visualization(visualization, outfile)

    return visualization


@dataclass
class _BayesianModel:
    """Container for PyMC model and available fitting options"""

    model_name: str
    pymc_model: pm.Model
    mcmc_estimation_supported: bool
    advi_estimation_supported: bool
    analytic_estimation_supported: bool
    hyperparams: Dict[str, Any]


def _get_bayesian_inference_model(
    model_name: str,
    control_samples: Samples,
    variation_samples: Samples,
    **model_params,
) -> Tuple[_BayesianModel, Dict[str, Any]]:
    # Defaults
    mcmc_estimation_supported = True  # MCMC supported by all models
    advi_estimation_supported = False
    analytic_estimation_supported = False  # Currently no Analytic support

    if model_name == "gaussian":
        from .models.continuous import build_gaussian_model as model_builder

    if model_name == "student_t":
        from .models.continuous import build_student_t_model as model_builder

    if model_name == "bernoulli":
        from .models.binary import build_bernoulli_model as model_builder

    if model_name == "binomial":
        from .models.binary import build_binomial_model as model_builder

    if model_name == "poisson":
        from .models.counts import build_poisson_model as model_builder

    pymc_model, hyperparams = model_builder(
        control_samples, variation_samples, **model_params
    )

    return _BayesianModel(
        model_name=model_name,
        pymc_model=pymc_model,
        mcmc_estimation_supported=mcmc_estimation_supported,
        advi_estimation_supported=advi_estimation_supported,
        analytic_estimation_supported=analytic_estimation_supported,
        hyperparams=hyperparams,
    )


def _fit_model_mcmc(model: _BayesianModel, **inference_kwargs):
    with model.pymc_model:
        return pm.sample(**inference_kwargs)


def _fit_model_advi(model: _BayesianModel, **inference_kwargs):
    with model.pymc_model:
        mean_field = pm.fit(method="advi", **inference_kwargs)
        return mean_field.sample(N_MEAN_FIELD_SAMPLES)


def _fit_model_analytic(model: _BayesianModel, **inference_kwargs):
    raise NotImplemented("Analytic parameter estimation API still a WIP")


class BayesianInferenceProcedure(InferenceProcedure):
    """
    Bayesian inference procedure to test for the difference between two
    samples.
    """

    def __init__(
        self,
        parameter_estimation_method: str = "mcmc",
        model_params: dict = None,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        inference_procedure: str
            The name of the inference model to use:
                -   "gaussian"       : Hierarchical Gaussian model (continuous)
                -   "student-t"      : Hierarchical Student's t model (continuous)
                -   "binomial"       : Beta-Binomial hierarchical model (binary)
                -   "bernoulli"      : Beta-Bernoulli hierarchical model (binary)
                -   "poisson"        : Gamma-Poisson hierarichcal model (counts)
        parameter_estimation_method: str
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
        self.model_name = _get_model_name(self.inference_method)
        self.data_type = _get_model_datatype(self.model_name)
        self.parameter_estimation_method = parameter_estimation_method
        self.model_params = model_params if model_params else {}
        self.inference_results = None

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

        self._bayesian_model = _get_bayesian_inference_model(
            model_name=self.model_name,
            control_samples=control_samples,
            variation_samples=variation_samples,
            **self.model_params,
        )

        if self.parameter_estimation_method == "mcmc":
            if self._bayesian_model.mcmc_estimation_supported:
                self.inference_results = _fit_model_mcmc(
                    self._bayesian_model, **inference_kwargs
                )

        elif self.parameter_estimation_method == "advi":
            if self._bayesian_model.advi_estimation_supported:
                self.inference_results = _fit_model_advi(
                    self._bayesian_model, **inference_kwargs
                )

        elif self.parameter_estimation_method == "analytic":
            self.inference_results = _fit_model_analytic(
                self._bayesian_model, **inference_kwargs
            )

        self.control = control_samples
        self.variation = variation_samples

        comparison_param = _get_delta_param(self.model_name)
        self.control_posterior = self.posterior_samples(f"{comparison_param}_control")
        self.variation_posterior = self.posterior_samples(
            f"{comparison_param}_variation"
        )

        self.delta_posterior = self.posterior_samples("delta")
        self.delta_relative_posterior = self.posterior_samples("delta_relative")
        self.effect_size_posterior = self.posterior_samples("effect_size")
        self.model_hyperparams = self._bayesian_model.hyperparams

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
        if self.inference_results is not None:
            return Samples(
                observations=self.inference_results.posterior[
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
            control_posterior=self.control_posterior,
            variation_posterior=self.variation_posterior,
            delta_posterior=self.delta_posterior,
            delta_relative_posterior=self.delta_relative_posterior,
            effect_size_posterior=self.effect_size_posterior,
            model_name=self.model_name,
            alpha=self.alpha,
            hypothesis=self.hypothesis,
            inference_method=self.inference_method,
            metric_name="posterior_delta",
            data_type=self.data_type,
            model_params=self.model_params,
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
