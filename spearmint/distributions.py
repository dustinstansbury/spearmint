import numpy as np

from scipy import stats

import holoviews as hv
from holoviews.element.chart import Curve, Bars
from holoviews.core.overlay import Overlay

from spearmint.config import FIGURE_PARAMS, COLORS
from spearmint.typing import Union, List, Tuple

N_GRID_POINTS = 100
DEFAULT_DISTRIBUTION_COLOR = COLORS.blue


class ProbabilityDistribution:
    """Base class for plottable probability distributions"""

    def __init__(self, label: str = "", color: str = DEFAULT_DISTRIBUTION_COLOR):
        self.label = label
        self.color = color

    def get_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the default domain and range for plotting the probability
        distribution.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            _description_
        """
        values = self.values_grid().ravel()
        probs = self.density(values)
        return values, probs

    def sample(self, sample_size: int):
        """
        Return a sample of `sample_size` from the probability distribution
        """
        return self.dist.rvs(size=sample_size)

    def cdf(self, values: np.ndarray) -> np.ndarray:
        """
        Return the Cumulative Distribution function for the probability
        distribution, evaluated at `values`
        """
        return self.dist.cdf(values)

    def ppf(self, values: np.ndarray) -> np.ndarray:
        return self.dist.ppf(values)

    def values_grid(self) -> np.ndarray:
        """
        Return the default domain values for plotting
        """
        raise NotImplementedError("Implement Me")

    def density(self, values: np.ndarray) -> np.ndarray:
        """
        Evaluate the PDF/PMF at the provided values
        """
        raise NotImplementedError("Implement Me")


class Pdf(ProbabilityDistribution):
    """
    Base class for continuous probability density functions.
    """

    def __init__(self, label: str = "PDF", *args, **kwargs):
        super().__init__(label=label, *args, **kwargs)

    def density(self, values: np.ndarray) -> np.ndarray:
        return self.dist.pdf(values)

    def plot(self, **plot_opts) -> Curve:
        curve_data = self.get_series()
        plot_opts.update(FIGURE_PARAMS)
        return hv.Curve(
            data=curve_data, label=self.label, kdims="value", vdims="pdf"
        ).opts(color=self.color, **plot_opts)


class Pmf(ProbabilityDistribution):
    """
    Base class for discrete probability mass functions.
    """

    def __init__(self, label: str = "PMF", *args, **kwargs):
        super().__init__(label=label, *args, **kwargs)

    def density(self, values: np.ndarray) -> np.ndarray:
        return self.dist.pmf(values)

    def plot(self, **plot_opts) -> Bars:
        values, pmf = self.get_series()
        bar_data = zip(values, pmf)
        plot_opts.update(FIGURE_PARAMS)
        return hv.Bars(
            data=bar_data, label=self.label, kdims="value", vdims="pmf"
        ).opts(color=self.color, **plot_opts)


class ProbabilityDistributionGroup:
    """
    Convenience class for plotting a group of `ProbabilityDistribution` instances.
    """

    def __init__(self, pdfs: List[Union[Pdf, Pmf]]):
        self.pdfs = pdfs

    def plot(self, **plot_opts) -> Overlay:
        plot_opts.update(FIGURE_PARAMS)
        overlay = self.pdfs[0].plot(**plot_opts)
        for pdf in self.pdfs[1:]:
            overlay *= pdf.plot(**plot_opts)
        return overlay


class Gaussian(Pdf):
    """
    A Gaussian PDF
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        label: str = "Gaussian",
        *args,
        **kwargs,
    ):
        super().__init__(label=label, *args, **kwargs)
        self.mean = mean
        self.std = std
        self.dist = stats.norm(loc=mean, scale=std)

    def values_grid(self) -> np.ndarray:
        _min = self.ppf(1e-5)
        _max = self.ppf(1 - 1e-5)
        return np.linspace(_min, _max, N_GRID_POINTS + 1)


class Bernoulli(Pmf):
    """
    A Bernoulli probability mass function
    """

    def __init__(self, p: float = 0.5, label: str = "Bernoulli", *args, **kwargs):
        super().__init__(label=label, *args, **kwargs)
        self.p = p
        self.dist = stats.bernoulli(p)

    def values_grid(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, 2)


class Binomial(Pmf):
    """
    A Binomial probability mass function
    """

    def __init__(
        self, n: int = 20, p: float = 0.5, label: str = "Binomial", *args, **kwargs
    ):
        super(Binomial, self).__init__(label=label, *args, **kwargs)
        self.n = n
        self.p = p
        self.dist = stats.binom(n, p)

    def values_grid(self):
        _min = self.ppf(1e-4)
        _max = self.ppf(1 - 1e-4)
        resolution = int(_max - _min) + 1

        if resolution > 25:
            resolution = 26

        return np.floor(np.linspace(_min, _max, resolution)).astype(int)


class Poisson(Pmf):
    """
    A Poisson probability mass function
    """

    def __init__(self, mu: float = 1.0, label: str = "Poisson", *args, **kwargs):
        super().__init__(label=label, *args, **kwargs)
        self.mu = mu
        self.dist = stats.poisson(mu)

    def values_grid(self):
        _min = self.ppf(1e-4)
        _max = self.ppf(1 - 1e-4)
        return np.arange(_min, _max)


class Kde(Pdf):
    """
    Estimate the shape of a PDF using a Gaussian kernel density estimator.
    """

    def __init__(self, samples: np.ndarray, label: str = "KDE", *args, **kwargs):
        super().__init__(label=label, *args, **kwargs)
        self.kde = stats.gaussian_kde(samples)
        low = min(samples)
        high = max(samples)
        self._values_grid = np.linspace(low, high, N_GRID_POINTS + 1)

    def density(self, values: np.ndarray) -> np.ndarray:
        """Note, we overload Pdf.density here"""
        return self.kde.evaluate(values)

    def values_grid(self):
        return self._values_grid
