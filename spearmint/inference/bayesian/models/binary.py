import pymc as pm
import numpy as np


def build_bernoulli_model(
    control_observations: np.ndarray,
    variation_observations: np.ndarray,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> pm.Model:
    with pm.Model() as model:
        # Priors
        p_control = pm.Beta("p_control", alpha=prior_alpha, beta=prior_beta)
        p_variation = pm.Beta(
            "p_variation",
            alpha=prior_alpha,
            beta=prior_beta,
        )

        # Likelihoods
        pm.Bernoulli("control", p=p_control, observed=control_observations)
        pm.Bernoulli(
            "variation",
            p=p_variation,
            observed=variation_observations,
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

    return model
