import pymc as pm

from scipy import stats

from spearmint.typing import Tuple, Dict
from spearmint.stats import Samples
from .analytic_base import BayesianAnalyticModel


def _get_gamma_prior_params(samples: Samples) -> Tuple[float, float]:
    """Estimate the prior parameters for Beta distribution from samples"""
    alpha = samples.mean**2 / samples.var
    beta = samples.mean / samples.var

    return alpha, beta


class PoissonAnalyticModel(BayesianAnalyticModel):
    """
    Implement analytic posterior updates for Gamma-Poisson model. Namely,
    a Gamma prior and Poisson likelihood result in a Gamma posterior over
    rate parameters, which can be calculated efficiently from the prior and
    descriptive statistics of the observations.

    References
    ----------
    - TBD
    """

    def __init__(
        self, prior_alpha: float = 1, prior_beta: float = 1.0, *args, **kwargs
    ):
        """
        Parameters
        ----------
        prior_alpha : float, optional
            The shape parameter for the Gamma prior distribution
        prior_beta : float, optional
            The shape parametrer for the Gamma prior distribution
        """
        super().__init__(delta_param="lambda", *args, **kwargs)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def calculate_posteriors(
        self, control_samples: Samples, variation_samples: Samples
    ):
        """
        Update the posterior distributions for the control and variation in
        light of Samples

        Parameters
        ----------
        control_samples : Samples
            Observations for the control group
        variation_samples : Samples
            Observations for the variation group
        """

        # Update rules
        def _posterior_alpha(prior_alpha, samples):
            return prior_alpha + samples.sum

        def _posterior_beta(prior_beta, samples):
            return prior_beta + samples.nobs

        control_posterior_alpha = _posterior_alpha(self.prior_alpha, control_samples)
        control_posterior_beta = _posterior_beta(self.prior_beta, control_samples)

        variation_posterior_alpha = _posterior_alpha(
            self.prior_alpha, variation_samples
        )
        variation_posterior_beta = _posterior_beta(self.prior_beta, variation_samples)

        self._control_posterior = stats.gamma(
            a=control_posterior_alpha, scale=1 / control_posterior_beta
        )

        self._variation_posterior = stats.gamma(
            a=variation_posterior_alpha, scale=1 / variation_posterior_beta
        )


def build_poisson_analytic_model(
    control_samples: Samples,
    variation_samples: Samples,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> Tuple[PoissonAnalyticModel, Dict[str, float]]:
    model = PoissonAnalyticModel(prior_alpha=prior_alpha, prior_beta=prior_beta)
    model.calculate_posteriors(control_samples, variation_samples)

    hyperparams = {"prior_alpha": prior_alpha, "prior_beta": prior_beta}
    return model, hyperparams


def build_poisson_pymc_model(
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
