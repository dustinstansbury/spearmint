import os
import numpy as np
import pandas as pd
import string

from spearmint.typing import Any, Iterable, Union, DataFrame


def format_value(val: Union[float, Iterable[float]], precision: int = 4) -> str:
    """Helper function for standardizing the precision of numerical and
    iterable values, returning a string representation.

    Parameters
    ----------
    val : Union[float, Iterable[float]]
        The value to format into a string
    precision : int
        The output numerical precision

    Returns
    -------
    formatted_value : str
        Representation of `val` with proper precision
    """
    if isinstance(val, Iterable):
        return f"{tuple((round(v, precision) for v in val))}"
    return f"{round(val, precision)}"


def ensure_dataframe(data: Any, data_attr: str = "data") -> DataFrame:
    """
    Check if an object is a dataframe, and if not, check if it has an
    attribute that is a dataframe, and return that attribute

    Parameters
    ----------
    data : Union[DataFrame, object]
        The data to check

    Returns
    -------
    data : DataFrame
        A verified dataframe

    Raises
    ------
    - ValueError if data `data.data_attr` isn't a DataFrame
    """
    if not isinstance(data, DataFrame):
        if hasattr(data, data_attr) and isinstance(getattr(data, data_attr), DataFrame):
            data = getattr(data, data_attr)
        else:
            raise ValueError("`data` is incorrect format, must be a DataFrame")

    return data


def set_matplotlib_backend() -> str:
    """
    Set supported matplotlb backend depending on the current platform.

    Returns
    -------
    backend : str
        The matplotlib backend being used
    """
    from sys import platform
    import matplotlib as mpl

    backend = "pdf" if platform == "darwin" else "agg"
    mpl.use(backend)
    return backend


def safe_isnan(val: Any) -> bool:
    """
    Check if a value is a NaN, handling `None` values

    Parameters
    ----------
    val : Any
        A value to check

    Returns
    -------
    is_nan : bool
        Truth value of the the value being a NaN
    """
    if val is not None:
        return np.isnan(val)
    return False


def generate_fake_observations(
    n_observations: int = 10000,
    n_treatments: int = 2,
    n_attributes: int = 2,
    distribution: str = "bernoulli",
    seed: int = 123,
):
    """
    Create a dataframe of artificial observations to be used for testing and demos.
    Treatments have different means, but segments defined by the attributes have
    no effect.

    Parameters
    -----------
    n_observations: int
        number of unique observations (e.g. users)
    n_treatments: int
        the number of simulated treatments. Will create an equivalent
        number of cohorts. Each successive treatment, by definition will
        have a systematically larger mean metric values (see metric
        distribution below). Note: the maximum number of treatments allowed
        is 6.
    n_attributes: int
        the number of attribute columns simulated. The number of distinct
        values taken by each attribute column is sampled uniformly between
        1 and 4. The segments defined by n_attributes have no effect on the
        sample means.
    distribution: str
        the type of metric distributions simulated
            - 'bernoulli': the mean increases from .5 by .1 for each successive treatment
            - 'gaussian': the mean increases from 0 by 1 for each successive treatment
            - 'poisson': the mean increases from 0 by 10 for each successive treatment
    seed: int
        The random number generator seed.

    Returns
    -------
    fake_observations : DataFrame
        Synthethic dataset with the columns:
        -   'metric': metric with support of `distribution`
        -   'treatment': the alphabetic label of treatments (e.g. 'A', 'B', etc.)
        -   'attr_*': randomly-genrated on-hot encoded attributes associated with
            each synthetic observation

    """
    distribution_ = distribution.lower()
    if distribution_ not in ("poisson", "bernoulli", "gaussian"):
        raise ValueError(f"Unsupported distribution {distribution}")
    np.random.seed(seed)

    letters = string.ascii_uppercase
    n_treatments = min(n_treatments, 6)

    data = pd.DataFrame()
    data["id"] = list(range(n_observations))

    # add treatments
    treatments = list(letters[:n_treatments])
    data["treatment"] = np.random.choice(treatments, size=n_observations)

    # add attributes (attributes should have no effect)
    attribute_columns = ["attr_{}".format(i) for i in range(n_attributes)]
    for ai, attr in enumerate(attribute_columns):
        attr_vals = [
            "A{}{}".format(ai, a.lower())
            for a in list(letters[: np.random.randint(1, 4)])
        ]
        data[attr] = np.random.choice(attr_vals, size=n_observations)

    # Set the metric data type
    data["metric"] = 0.0  # Fill data with default value for conversion
    if distribution_ == "poisson":
        dtype = int
    elif distribution_ == "bernoulli":
        dtype = bool
    elif distribution_ == "gaussian":
        dtype = float
    data = data.astype({"metric": dtype})

    # Add measurements, each treatment has successively larger means
    for delta, tr in enumerate(treatments):
        tr_mask = data.treatment == tr
        n_tr = sum(tr_mask)
        if distribution_ == "gaussian":
            data.loc[tr_mask, "metric"] = delta + np.random.randn(n_tr)
        elif distribution_ == "bernoulli":
            data.loc[tr_mask, "metric"] = list(
                map(dtype, np.round(0.1 * delta + np.random.random(n_tr)))
            )
        elif distribution_ == "poisson":
            data.loc[tr_mask, "metric"] = list(
                map(dtype, np.random.poisson(1 + delta, size=n_tr))
            )

    return data
