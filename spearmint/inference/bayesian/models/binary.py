import pymc as pm

from scipy import stats

from spearmint.typing import Tuple, Dict
from spearmint.stats import Samples
from .analytic_base import BayesianAnalyticModel


def _get_beta_prior_params(
    samples: Samples,
) -> Tuple[float, float, float]:
    """Estimate the prior parameters for Beta distribution from samples"""
    mean = samples.mean
    var = samples.var
    nobs = samples.nobs

    effective_sample_size = nobs * mean * (1 - mean) / var - 1
    alpha = mean * effective_sample_size
    beta = (1 - mean) * effective_sample_size

    return effective_sample_size, alpha, beta


class BinomialAnalyticModel(BayesianAnalyticModel):
    """
    Implement analytic posterior updates for Beta-Binomial model. Namely,
    a Beta prior and Binomial likelihood result in a Beta posterior over
    proportionality parameters, which can be calculated from the prior and
    descriptive statistics of the observations.

    Notes
    -----
    We use the same model for Bernoulli and Binomial Likelihoods, as their
    posterirs updates are calculated in the same way.

    References
    ----------
    - https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall06/lectures/bernoulli-slides.pdf
    """

    def __init__(
        self, prior_alpha: float = 1, prior_beta: float = 1.0, *args, **kwargs
    ):
        """
        Parameters
        ----------
        prior_alpha : float, optional
            The shape parameter for the Beta prior distribution
        prior_beta : float, optional
            The shape parametrer for the Beta prior distribution
        """
        super().__init__(delta_param="p", *args, **kwargs)
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
            n_success = samples.sum
            return prior_alpha + n_success

        def _posterior_beta(prior_beta, samples):
            n = samples.nobs
            n_fail = n - samples.sum

            return prior_beta + n_fail

        control_posterior_alpha = _posterior_alpha(self.prior_alpha, control_samples)
        control_posterior_beta = _posterior_beta(self.prior_beta, control_samples)

        variation_posterior_alpha = _posterior_alpha(
            self.prior_alpha, variation_samples
        )
        variation_posterior_beta = _posterior_beta(self.prior_beta, variation_samples)

        self._control_posterior = stats.beta(
            control_posterior_alpha, control_posterior_beta
        )

        self._variation_posterior = stats.beta(
            variation_posterior_alpha, variation_posterior_beta
        )


def build_binomial_analytic_model(
    control_samples: Samples,
    variation_samples: Samples,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> Tuple[BinomialAnalyticModel, Dict[str, float]]:
    model = BinomialAnalyticModel(prior_alpha=prior_alpha, prior_beta=prior_beta)
    model.calculate_posteriors(control_samples, variation_samples)

    hyperparams = {"prior_alpha": prior_alpha, "prior_beta": prior_beta}
    return model, hyperparams


def build_bernoulli_analytic_model(
    control_samples: Samples,
    variation_samples: Samples,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> Tuple[BinomialAnalyticModel, Dict[str, float]]:
    """Note: we use the same model for Binomial/Bernoulli, as they are equivalent"""
    return build_binomial_analytic_model(
        control_samples=control_samples,
        variation_samples=variation_samples,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )


def build_bernoulli_pymc_model(
    control_samples: Samples, variation_samples: Samples
) -> pm.Model:
    """
    Compiles a Beta-Bernoulli Bayesian PyMC model for modeling binary data. The
    model consists of a Beta prior over proportionality (conversion rate), and
    a Bernoulli likelihood. For the Beta prior, we derive parameters separately
    for the control and variation from their observations (i.e. no pooling).

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

    Notes
    -----
    The Beta-Bernoulli model is equivalent to the Beta-Binomial model, and should
    provide very similar estiamtes of proportionality and dleta, but the
    Beta-Binomial model tends to be more computationally efficient.

    References
    ----------
    -   https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Beta.html
    -   https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Bernoulli.html
    """

    # Empirically-informed priors
    kappa_control, alpha_control, beta_control = _get_beta_prior_params(control_samples)
    kappa_variation, alpha_variation, beta_variation = _get_beta_prior_params(
        variation_samples
    )

    hyperparams = {
        "kappa_control": kappa_control,
        "alpha_control": alpha_control,
        "beta_control": beta_control,
        "kappa_variation": kappa_variation,
        "alpha_variation": alpha_variation,
        "beta_variation": beta_variation,
    }

    with pm.Model() as model:
        # Priors
        p_control = pm.Beta("p_control", alpha=alpha_control, beta=beta_control)
        p_variation = pm.Beta("p_variation", alpha=alpha_variation, beta=beta_variation)

        # Likelihoods
        pm.Bernoulli("control", p=p_control, observed=control_samples.data)
        pm.Bernoulli("variation", p=p_variation, observed=variation_samples.data)

        # Inference parameters
        delta = pm.Deterministic("delta", p_variation - p_control)
        pm.Deterministic("delta_relative", (p_variation / p_control) - 1.0)
        pm.Deterministic(
            "effect_size",
            delta
            / pm.math.sqrt(
                (p_control * (1 - p_control) + p_variation * (1 - p_variation)) / 2
            ),
        )

    return model, hyperparams


def build_binomial_pymc_model(
    control_samples: Samples, variation_samples: Samples
) -> pm.Model:
    """
    Compiles a Beta-Binomial Bayesian PyMC model for modeling binary data. The
    model consists of a Beta prior over proportionality (conversion rate), and
    a Binomial likelihood. For the Beta prior, we derive parameters separately
    for the control and variation from their observations (i.e. no pooling).

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

    Notes
    -----
    The Beta-Binomial model is equivalent to the Beta-Bernoulli model, and should
    provide very similar estiamtes of proportionality and dleta, but the
    Beta-Binomial model tends to be more computationally efficient.

    References
    ----------
    -   https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Beta.html
    -   https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Binomial.html
    """

    # Empirically-informed priors
    n_control = control_samples.nobs
    kappa_control, alpha_control, beta_control = _get_beta_prior_params(
        control_samples,
    )

    n_variation = variation_samples.nobs
    kappa_variation, alpha_variation, beta_variation = _get_beta_prior_params(
        variation_samples,
    )

    hyperparams = {
        "kappa_control": kappa_control,
        "alpha_control": alpha_control,
        "beta_control": beta_control,
        "kappa_variation": kappa_variation,
        "alpha_variation": alpha_variation,
        "beta_variation": beta_variation,
    }

    with pm.Model() as model:
        # Priors
        p_control = pm.Beta("p_control", alpha=alpha_control, beta=beta_control)
        p_variation = pm.Beta("p_variation", alpha=alpha_variation, beta=beta_variation)

        # Likelihoods
        pm.Binomial(
            "control",
            p=p_control,
            n=n_control,
            observed=control_samples.data,
        )
        pm.Binomial(
            "variation",
            p=p_variation,
            n=n_variation,
            observed=variation_samples.data,
        )

        # Inference parameters
        delta = pm.Deterministic("delta", p_variation - p_control)
        pm.Deterministic("delta_relative", (p_variation / p_control) - 1.0)
        pm.Deterministic(
            "effect_size",
            delta
            / pm.math.sqrt(
                (p_control * (1 - p_control) + p_variation * (1 - p_variation)) / 2
            ),
        )

    return model, hyperparams
