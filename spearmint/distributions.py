from abc import ABC, abstractmethod

import holoviews as hv
import numpy as np
from holoviews.core.overlay import Overlay
from holoviews.element.chart import Bars, Curve
from scipy import stats

from spearmint.config import COLORS, FIGURE_PARAMS
from spearmint.typing import List, Tuple, Union

N_GRID_POINTS = 100
PDF_ZERO = 1e-5
DEFAULT_DISTRIBUTION_COLOR = COLORS.blue


class _RandomVariableClass:
    """Placeholder class for type hinting. Do not extend!"""

    def __init__(self):
        pass

    def ppf(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return values

    def pdf(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return values

    def pmf(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return values

    def cdf(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return values

    def evaluate(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return values

    def rvs(self, size: int) -> np.ndarray:
        return np.ones(size)

    def resample(self, size: int) -> np.ndarray:
        return np.ones(size)


_RandomVariable = _RandomVariableClass()


class ProbabilityDistribution(ABC):
    """Base class for plottable probability distributions"""

    def __init__(self, label: str = "", color: str = DEFAULT_DISTRIBUTION_COLOR):
        self.label = label
        self.color = color
        self.dist = _RandomVariable

    def get_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the default domain and range for plotting the probability
        distribution.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            _description_
        """
        values = self.values_grid.ravel()
        probs = self.density(values)
        return values, probs

    def sample(self, sample_size: int) -> np.ndarray:
        """
        Return a sample of `sample_size` from the probability distribution
        """
        return self.dist.rvs(size=sample_size)

    def cdf(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Return the Cumulative Distribution function for the probability
        distribution, evaluated at `values`
        """
        return self.dist.cdf(values)

    def ppf(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.dist.ppf(values)

    @abstractmethod
    def _get_values_grid(self) -> np.ndarray:
        """Private abstractmethod to get default values grid"""
        pass

    @property
    def values_grid(self) -> np.ndarray:
        """
        Return the default domain values for plotting
        """
        return self._get_values_grid()

    @abstractmethod
    def density(self, values: np.ndarray) -> np.ndarray:
        """
        Public abstract method to evaluate the PDF/PMF at the provided values.
        """
        pass


class Pdf(ProbabilityDistribution):
    """
    Base class for continuous probability density functions.
    """

    def __init__(self, label: str = "PDF", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label

    def density(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:  # type: ignore
        return self.dist.pdf(values)

    def plot(self, **plot_opts) -> Curve:  # pragma: no cover
        plot_data = self.get_series()
        plot_opts.update(FIGURE_PARAMS)
        return hv.Curve(
            data=plot_data, label=self.label, kdims="value", vdims="pdf"
        ).opts(color=self.color, **plot_opts)


class Pmf(ProbabilityDistribution):
    """
    Base class for discrete probability mass functions.
    """

    def __init__(self, label: str = "PMF", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label

    def density(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:  # type: ignore
        return self.dist.pmf(values)

    def plot(self, **plot_opts) -> Bars:  # pragma: no cover
        values, pmf = self.get_series()
        plot_data = zip(values, pmf)
        plot_opts.update(FIGURE_PARAMS)

        return hv.Curve(
            data=plot_data, label=self.label, kdims="value", vdims="pdf"
        ).opts(color=self.color, **plot_opts)


class ProbabilityDistributionGroup:
    """
    Convenience class for plotting a group of `ProbabilityDistribution` instances.
    """

    def __init__(self, pdfs: List[Union[Pdf, Pmf]]):
        self.pdfs = pdfs

    def plot(self, **plot_opts) -> Overlay:  # pragma: no cover
        plot_opts.update(FIGURE_PARAMS)
        overlay = self.pdfs[0].plot(**plot_opts)
        for pdf in self.pdfs[1:]:
            overlay *= pdf.plot(**plot_opts)
        return overlay


class Gaussian(Pdf):
    """
    A Gaussian probability density function
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        label: str = "Gaussian",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label = label
        self.mean = mean
        self.std = std
        self.dist = stats.norm(loc=mean, scale=std)

    def _get_values_grid(self) -> np.ndarray:
        _min = self.ppf(PDF_ZERO)
        _max = self.ppf(1 - PDF_ZERO)
        return np.linspace(_min, _max, N_GRID_POINTS + 1)


class Bernoulli(Pmf):
    """
    A Bernoulli probability mass function
    """

    def __init__(self, p: float = 0.5, label: str = "Bernoulli", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label
        self.p = p
        self.dist = stats.bernoulli(p)

    def _get_values_grid(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, 2)


class Binomial(Pmf):
    """
    A Binomial probability mass function
    """

    def __init__(
        self, n: int = 20, p: float = 0.5, label: str = "Binomial", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.label = label
        self.n = n
        self.p = p
        self.dist = stats.binom(n, p)

    def _get_values_grid(self):
        _min = self.ppf(PDF_ZERO)
        _max = self.ppf(1 - PDF_ZERO)
        resolution = int(_max - _min) + 1

        if resolution > 25:
            resolution = 26

        return np.floor(np.linspace(_min, _max, resolution)).astype(int)


class Poisson(Pmf):
    """
    A Poisson probability mass function
    """

    def __init__(self, mu: float = 1.0, label: str = "Poisson", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label
        self.mu = mu
        self.dist = stats.poisson(mu)

    def _get_values_grid(self):
        _min = self.ppf(PDF_ZERO)
        _max = self.ppf(1 - PDF_ZERO)
        return np.arange(_min, _max + 1)


class Kde(Pdf):
    """
    Estimate the shape of a PDF using a Gaussian kernel density estimator.
    """

    def __init__(self, samples: np.ndarray, label: str = "KDE", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label
        self.dist = stats.gaussian_kde(samples)
        self.low = min(samples)
        self.high = max(samples)

    def _get_values_grid(self):
        return np.linspace(self.low, self.high, N_GRID_POINTS + 1)

    def density(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:  # type: ignore
        """Note, we overload Pdf.density here"""
        return self.dist.evaluate(values)

    def sample(self, sample_size: int) -> np.ndarray:
        """Note: we sample 1D arrays"""
        return self.dist.resample(size=sample_size).flatten()
