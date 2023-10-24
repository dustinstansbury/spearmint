import numpy as np

import holoviews as hv
from holoviews.element.chart import Curve

from spearmint.config import (
    FIGURE_PARAMS,
    POINTS_PLOT_PARAMS,
    DEFAULT_COLOR,
)
from spearmint.utils import format_value
from spearmint import distributions


def plot_interval(
    left: float,
    right: float,
    middle: float = None,
    color: str = DEFAULT_COLOR,
    label: str = "Interval",
    show_interval_text: bool = False,
    vertical_offset: float = 0.0,
    fontsize: int = 12,
    **plot_opts,
) -> Curve:
    """Plot an interval spanning (left, right).

    Parameters
    ----------
    left : float
        The lower bound of the interval.
    right : float
        The upper bound of the interval.
    middle : float, optional
        The central anchor point for the interval. If None provided, we assume
        the middle is the average of `left` and `right`.
    color : str, optional
        The color of the interval and any associated text, by default DEFAULT_COLOR
    label : str, optional
        The legend label, by default "Interval"
    show_interval_text : bool, optional
        Whether to display text information describing the interval, by default
        False.
    vertical_offset : float, optional
        The desired vertical location of the interval in the 2D plot, by
        default 0.0
    fontsize : int, optional
        If `show_interval_text=True`, this allows easy customization of the
        interval text font size, by default 12

    Returns
    -------
    interval_curve: Curve
        A holoviews Curve element that can be combined with other elements to
        form an Overlay

    Raises
    ------
    ValueError
        If left and right are invalid values (e.g. -inf/inf), then we can't
        determine reasonable bounds for the visualizatoin
    """
    if middle in (-np.inf, np.inf) and (
        left in (np.inf, -np.inf) or right in (np.inf, -np.inf)
    ):
        raise ValueError("Too many interval values are inf")
    middle = np.mean((left, right)) if middle is None else middle
    INFTY_SCALE = 4

    _left = middle - INFTY_SCALE * np.abs(right) if left in (np.inf, -np.inf) else left
    _right = (
        middle + INFTY_SCALE * np.abs(left) if right in (np.inf, -np.inf) else right
    )

    middle_point = hv.Points(data=(middle, vertical_offset)).opts(
        color=color, **POINTS_PLOT_PARAMS
    )

    interval_data = ((_left, _right), (vertical_offset, vertical_offset))
    plot_opts.update(FIGURE_PARAMS)
    interval = (
        hv.Curve(data=interval_data, label=label).opts(color=color, **plot_opts)
        * middle_point
    )
    if show_interval_text:
        annotation_text = f"{format_value(middle, precision=2)}\n{format_value((left, right), precision=2)}"
        annotation = hv.Text(
            middle,
            vertical_offset,
            annotation_text,
            fontsize=fontsize,
        ).opts(color=color)
        interval *= annotation

    return interval


# Functional interface
def plot_gaussian(
    mean: float = 0,
    std: float = 1,
    color: str = DEFAULT_COLOR,
    label: str = "Gaussian",
    **plot_kwargs,
) -> Curve:
    """
    Plot a Gaussian distribution with parameters `mean` and `std`.

    Parameters
    ----------
    mean : float, optional
        The location of the distribution, by default 0
    std : float, optional
        The spread of the distribution, by default 1
    color : str, optional
        The color of the distrubution line, by default DEFAULT_COLOR
    label : str, optional
        The legend lable for the distribution, by default "Gaussian"

    Returns
    -------
    distribution_curve: Curve
        A holoviews Curve element that can be combined with other elements to
        form an Overlay
    """
    return distributions.Gaussian(mean=mean, std=std, label=label, color=color).plot(
        **plot_kwargs
    )


def plot_bernoulli(
    p: float = 0.5, color: str = DEFAULT_COLOR, label: str = "Bernoulli", **plot_kwargs
) -> Curve:
    """
    Plot a Bernoulli distribution with proportionality parameter `p`.

    Parameters
    ----------
    p : float, optional
        The probability of a positiv outcome, by default 0.5
    color : str, optional
        _description_, by default DEFAULT_COLOR
    color : str, optional
        The color of the distrubution line, by default DEFAULT_COLOR
    label : str, optional
        The legend lable for the distribution, by default "Bernoulli"

    Returns
    -------
    distribution_curve: Curve
        A holoviews Curve element that can be combined with other elements to
        form an Overlay
    """
    return distributions.Bernoulli(p=p, label=label, color=color).plot(**plot_kwargs)


def plot_binomial(
    n: float = 10,
    p: float = 0.5,
    color: str = DEFAULT_COLOR,
    label: str = "Binomial",
    **plot_kwargs,
) -> Curve:
    """
    Plot a Binomial distribution with proportionality parameter `p` and trials
    parameters `n`

    Parameters
    ----------
    n : float, optional
        The number of trials, by default 10
    p : float, optional
        The rate of a positive outcome on a trial, by default 0.5
    color : str, optional
        The color of the distrubution line, by default DEFAULT_COLOR
    label : str, optional
        The legend lable for the distribution, by default "Binomial"

    Returns
    -------
    distribution_curve: Curve
        A holoviews Curve element that can be combined with other elements to
        form an Overlay
    """
    return distributions.Binomial(n=n, p=p, label=label, color=color).plot(
        **plot_kwargs
    )


def plot_poisson(
    mu: float = 10,
    color: str = DEFAULT_COLOR,
    label: str = "Poisson",
    **plot_kwargs,
) -> Curve:
    """Plot a Poisson distribution with rate/mean parameter `mu`.

    Parameters
    ----------
    mu : float, optional
        The location and spread of the distribution, by default 10
    color : str, optional
        The color of the distrubution line, by default DEFAULT_COLOR
    label : str, optional
        The legend lable for the distribution, by default "Poisson"

    Returns
    -------
    distribution_curve: Curve
        A holoviews Curve element that can be combined with other elements to
        form an Overlay
    """
    return distributions.Poisson(mu=mu, label=label, color=color).plot(**plot_kwargs)
