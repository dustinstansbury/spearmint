import pymc as pm

from spearmint.stats import Samples


def build_gaussian_model(
    control_samples: Samples,
    variation_samples: Samples,
) -> pm.Model:
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


def build_student_t_model(
    control_samples: Samples,
    variation_samples: Samples,
) -> pm.Model:
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
    sigma_prior_sigma = 1

    with pm.Model() as model:
        # Priors
        sigma_control = pm.TruncatedNormal(
            "sigma_control", lower=1e-4, mu=control_std, sigma=sigma_prior_sigma
        )
        sigma_variation = pm.TruncatedNormal(
            "sigma_variation", lower=1e-4, mu=variation_std, sigma=sigma_prior_sigma
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
        "sigma_control_sigma": sigma_prior_sigma,
        "sigma_variation_mu": variation_mean,
        "sigma_variation_sigma": sigma_prior_sigma,
        "nu_precision": nu_prior_precision,
    }
    return model, hyperparams
