import pymc as pm
import numpy as np

from spearmint.stats import Samples
from spearmint.typing import Tuple


def _get_gamma_prior_params(samples: Samples) -> Tuple[float, float]:
    """Estimate the prior parameters for Beta distribution from samples"""
    alpha = samples.mean**2 / samples.var
    beta = samples.mean / samples.var

    return alpha, beta


def build_poisson_model(
    control_samples: Samples, variation_samples: Samples
) -> pm.Model:
    """
    Compile a Gamma-Poisson Bayesian PyMC model for modeling counted events data.
    The model consists of a Gamma prior over event rate, and a Poisson likelihood.
    For the Gamma prior, we derive parameters separately for the control and
    variation from their observations (i.e. no pooling).

    Parameters
    ----------
    control_observations: Samples, dtype=int
        The control group observations
    variation_observations: Samples, dtype=int
        The variation group observations

    Returns
    -------
    model : pm.Model
        The compiled PyMC model built with the provided data
    hyperparams : Dict
        The prior parameters derived from the Samples

    References
    ----------
    -   https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Gamma.html
    -   https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Poisson.html
    """

    # Informed priors
    alpha_control, beta_control = _get_gamma_prior_params(control_samples)
    alpha_variation, beta_variation = _get_gamma_prior_params(variation_samples)

    with pm.Model() as model:
        # Priors
        lambda_control = pm.Gamma(
            "lambda_control", alpha=alpha_control, beta=beta_control
        )
        lambda_variation = pm.Gamma(
            "lambda_variation", alpha=alpha_variation, beta=beta_variation
        )

        # Likelihoods
        pm.Poisson("control", mu=lambda_control, observed=control_samples.data)
        pm.Poisson("variation", mu=lambda_variation, observed=variation_samples.data)

        # Inference parameters
        pm.Deterministic("delta", lambda_variation - lambda_control)
        effect_size = pm.Deterministic(
            "effect_size", (lambda_variation / lambda_control) - 1.0
        )
        pm.Deterministic("delta_relative", effect_size - 1)

    hyperparams = {
        "alpha_control": alpha_control,
        "alpha_variation": alpha_variation,
        "beta_control": beta_control,
        "beta_variation": beta_variation,
    }

    return model, hyperparams
