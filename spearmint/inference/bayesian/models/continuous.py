import pymc as pm
import numpy as np


def build_gaussian_model(
    control_observations: np.ndarray,
    variation_observations: np.ndarray,
    prior_std_max: float = 10,
    prior_std_min: float = 1e-4,
) -> pm.Model:
    with pm.Model() as model:
        # Priors
        sigma_control = pm.Uniform(
            "sigma_control", lower=prior_std_min, upper=prior_std_max
        )
        sigma_variation = pm.Uniform(
            "sigma_variation", lower=prior_std_min, upper=prior_std_max
        )
        mu_control = pm.Normal(
            "mu_control",
            mu=np.mean(control_observations),
            sigma=np.std(control_observations),
        )
        mu_variation = pm.Normal(
            "mu_variation",
            mu=np.mean(variation_observations),
            sigma=np.std(variation_observations),
        )

        # Likelihoods
        pm.Normal(
            "control", mu=mu_control, sigma=sigma_control, observed=control_observations
        )
        pm.Normal(
            "variation",
            mu=mu_variation,
            sigma=sigma_variation,
            observed=variation_observations,
        )

        # Inference parameters
        delta = pm.Deterministic("delta", mu_variation - mu_control)
        pm.Deterministic("delta_relative", (mu_variation / mu_control) - 1.0)
        pm.Deterministic(
            "effect_size",
            delta / pm.math.sqrt((sigma_control**2.0 + sigma_variation**2.0) / 2.0),
        )

    return model
