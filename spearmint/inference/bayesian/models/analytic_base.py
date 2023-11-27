from abc import ABC, abstractmethod

import arviz as az
import numpy as np

from spearmint.stats import Samples
from spearmint.typing import Optional


class BayesianAnalyticModel(ABC):
    """Base class for Bayesian models that leverage analytic solutions based
    on conjugate priors to implement posterior updates.
    """

    def __init__(
        self,
        delta_param: str,
        control_name: Optional[str] = None,
        variation_name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.delta_param = delta_param
        self.control_name = control_name
        self.variation_name = variation_name
        self._control_posterior = None
        self._variation_posterior = None
        self._prior = None

    @property
    def control_posterior(self):
        return self._control_posterior

    @property
    def variation_posterior(self):
        return self._variation_posterior

    @property
    def prior(self):
        return self._prior

    def sample(self, n_samples: int = 1000) -> az.InferenceData:
        """
        Draw samples from the posterior, and return the results as InferenceData

        Parameters
        ----------
        n_samples : int
            The number of posterior samples to draw

        Returns
        -------
        inference_data : InferenceData
            An Arviz InferenceData structure with a `.posterior` attribute.
        """
        # Sample the prior (for visualization)
        prior_samples = self.prior.rvs(size=n_samples)

        # Sample delta parameters from the posterior
        control_posterior_samples = self.control_posterior.rvs(size=n_samples)
        variation_posterior_samples = self.variation_posterior.rvs(size=n_samples)

        delta_posterior_samples = (
            variation_posterior_samples - control_posterior_samples
        )
        delta_relative_samples = (
            variation_posterior_samples - control_posterior_samples
        ) / np.abs(np.mean(control_posterior_samples))
        effect_size_posterior_samples = delta_posterior_samples / np.sqrt(
            np.var(control_posterior_samples) + np.var(variation_posterior_samples)
        )

        data_dict = {
            "prior": prior_samples,
            f"{self.delta_param}_control": control_posterior_samples,
            f"{self.delta_param}_variation": variation_posterior_samples,
            "delta": delta_posterior_samples,
            "delta_relative": delta_relative_samples,
            "effect_size": effect_size_posterior_samples,
        }

        return az.convert_to_inference_data(data_dict)

    @abstractmethod
    def calculate_posteriors(
        self, control_samples: Samples, variation_samples: Samples
    ) -> None:
        """
        Update the posterior distributions for the control and variation groups
        in light of observed samples

        Parameters
        ----------
        control_samples : Samples
            Observations for the control group
        variation_samples : Samples
            Observations for the variation group
        """
        pass
