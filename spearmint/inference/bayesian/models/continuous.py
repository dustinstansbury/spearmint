import pymc as pm
from scipy import stats

from spearmint.typing import Dict, Tuple
from spearmint.stats import Samples
from spearmint.inference.bayesian.bayesian_inference import (
    UnsupportedParameterEstimationMethodException,
)
from .analytic_base import BayesianAnalyticModel


class GaussianAnalyticModel(BayesianAnalyticModel):
    """
    Implement analytic posterior updates for hierarchical Gaussian model. Namely,
    a Gaussian prior and likelihood result in a Gaussian posterior over mean
    parameters, which can be calculated from the prior and descriptive statistics
    of the observations.

    References
    ----------
    - https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    """

    def __init__(self, prior_mean: float = 0, prior_var: float = 1.0, *args, **kwargs):
        """
        Parameters
        ----------
        prior_mean : float, optional
            The mean of the prior distribution means, by default 0
        prior_var : float, optional
            The variance of the prior distribution on means, by default 1.0
        """
        super().__init__(delta_param="mu", *args, **kwargs)
        self.prior_mean = prior_mean
        self.prior_var = prior_var

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
        def _posterior_variance(prior_var, samples):
            n = samples.nobs
            sample_var = samples.var

            return 1 / ((n / sample_var) + (1 / prior_var))

        def _posterior_mean(prior_mean, prior_var, posterior_var, samples):
            n = samples.nobs
            sample_mean = samples.mean
            sample_var = samples.var
            return posterior_var * (
                prior_mean / prior_var + n * sample_mean / sample_var
            )

        control_posterior_var = _posterior_variance(self.prior_var, control_samples)
        control_posterior_mean = _posterior_mean(
            self.prior_mean, self.prior_var, control_posterior_var, control_samples
        )

        variation_posterior_var = _posterior_variance(self.prior_var, variation_samples)
        variation_posterior_mean = _posterior_mean(
            self.prior_mean,
            self.prior_var,
            variation_posterior_var,
            variation_samples,
        )

        self._control_posterior = stats.norm(
            control_posterior_mean, control_posterior_var**0.5
        )

        self._variation_posterior = stats.norm(
            variation_posterior_mean, variation_posterior_var**0.5
        )


def build_gaussian_analytic_model(
    control_samples: Samples,
    variation_samples: Samples,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
) -> Tuple[GaussianAnalyticModel, Dict[str, float]]:
    model = GaussianAnalyticModel(prior_mean=prior_mean, prior_var=prior_var)
    model.calculate_posteriors(control_samples, variation_samples)

    hyperparams = {"prior_mean": prior_mean, "prior_var": prior_var}
    return model, hyperparams


def build_gaussian_pymc_model(
    control_samples: Samples,
    variation_samples: Samples,
) -> Tuple[pm.Model, Dict[str, float]]:
    """
    Compiles a Hierarchical Gaussian Bayesian PyMC model for modeling binary data.
    The model consists of a Gaussian data likelihood with a Gaussian prior over
    the means, and Half-Normal prior over variance. For the priors, we derive
    separate location parameters for the control and variation from their
    associated observations (i.e. no pooling).

    Parameters
    ----------
    control_observations: Sample, dtype=float
        The control group observations
    variation_observations: Samples, dtype=float
        The variation group observations

    Returns
    -------
    model : pm.Model
        The compiled PyMC model built with the provided data
    hyperparams : Dict
        The prior parameters derived from the Samples

    References
    ----------
    -   https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Guassian.html
    -   https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.TruncatedNormal.html
    """

    # Empirically-informed priors hyperparams
    control_mean = control_samples.mean
    control_std = control_samples.std

    variation_mean = variation_samples.mean
    variation_std = variation_samples.std

    with pm.Model() as model:
        # Priors
        sigma_control = pm.TruncatedNormal(
            "sigma_control", lower=1e-4, mu=control_std, sigma=1
        )
        sigma_variation = pm.TruncatedNormal(
            "sigma_variation", lower=1e-4, mu=variation_std, sigma=1
        )
        mu_control = pm.Normal("mu_control", mu=control_mean, sigma=control_std)
        mu_variation = pm.Normal("mu_variation", mu=variation_mean, sigma=variation_std)

        # Likelihoods
        pm.Normal(
            "control", mu=mu_control, sigma=sigma_control, observed=control_samples.data
        )
        pm.Normal(
            "variation",
            mu=mu_variation,
            sigma=sigma_variation,
            observed=variation_samples.data,
        )

        # Inference parameters
        delta = pm.Deterministic("delta", mu_variation - mu_control)
        pm.Deterministic("delta_relative", (mu_variation / mu_control) - 1.0)
        pm.Deterministic(
            "effect_size",
            delta / pm.math.sqrt((sigma_control**2.0 + sigma_variation**2.0) / 2.0),
        )

    hyperparams = {
        "sigma_control_mu": control_mean,
        "sigma_control_sigma": 1,
        "sigma_variation_mu": variation_mean,
        "sigma_variation_sigma": 1,
    }
    return model, hyperparams


def build_student_t_analytic_model() -> None:
    raise UnsupportedParameterEstimationMethodException(
        "Analytic parameter estimation not supported for `student_t` model"
    )


def build_student_t_pymc_model(
    control_samples: Samples,
    variation_samples: Samples,
) -> Tuple[pm.Model, Dict[str, float]]:
    """
    Compiles a Hierarchical Student-t Bayesian PyMC model for modeling binary data.
    Using the Student-t model allows for robust inference in the presence of
    outliers. The model consists of a Student-t data likelihood, a Gaussian
    prior over the mean, a Half-Normal prior over variance, and an Exponential
    distribution over the degress of freedom. For the priors, we derive separate
    location parameters for the control and variation from their associated
    observations (i.e. no pooling).

    Parameters
    ----------
    control_observations: Sample, dtype=float
        The control group observations
    variation_observations: Samples, dtype=float
        The variation group observations

    Returns
    -------
    model : pm.Model
        The compiled PyMC model built with the provided data
    hyperparams : Dict
        The prior parameters derived from the Samples

    References
    ----------
    -   https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.StudentT.html
    -   https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Guassian.html
    -   https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.TruncatedNormal.html
    -   https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Exponential.html
    """

    # Empirically-informed priors hyperparams
    control_mean = control_samples.mean
    control_std = control_samples.std

    variation_mean = variation_samples.mean
    variation_std = variation_samples.std

    nu_prior_precision = 0.5
    sigma_prior_var = 1

    with pm.Model() as model:
        # Priors
        sigma_control = pm.TruncatedNormal(
            "sigma_control", lower=1e-4, mu=control_std, sigma=sigma_prior_var
        )
        sigma_variation = pm.TruncatedNormal(
            "sigma_variation", lower=1e-4, mu=variation_std, sigma=sigma_prior_var
        )
        mu_control = pm.Normal("mu_control", mu=control_mean, sigma=control_std)
        mu_variation = pm.Normal("mu_variation", mu=variation_mean, sigma=variation_std)

        nu_control = pm.Exponential("nu_control", lam=1 / nu_prior_precision)
        nu_variation = pm.Exponential("nu_variation", lam=1 / nu_prior_precision)

        # Likelihoods
        pm.StudentT(
            "control",
            nu=nu_control,
            mu=mu_control,
            sigma=sigma_control,
            observed=control_samples.data,
        )
        pm.StudentT(
            "variation",
            nu=nu_variation,
            mu=mu_variation,
            sigma=sigma_variation,
            observed=variation_samples.data,
        )

        # Inference parameters
        delta = pm.Deterministic("delta", mu_variation - mu_control)
        pm.Deterministic("delta_relative", (mu_variation / mu_control) - 1.0)
        pm.Deterministic(
            "effect_size",
            delta / pm.math.sqrt((sigma_control**2.0 + sigma_variation**2.0) / 2.0),
        )

    hyperparams = {
        "sigma_control_mu": control_mean,
        "sigma_control_sigma": sigma_prior_var,
        "sigma_variation_mu": variation_mean,
        "sigma_variation_sigma": sigma_prior_var,
        "nu_precision": nu_prior_precision,
    }
    return model, hyperparams
