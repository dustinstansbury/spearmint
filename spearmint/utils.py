import os
import shutil
import numpy as np
import pandas as pd
import string

from spearmint.typing import Any, List, Iterable, Union, DataFrame, Path, Optional

BIG_FLOAT = 2.0**32
SMALL_FLOAT = -(2.0**32)
NAN_TYPE_MAPPING = {
    np.inf: None,
    -np.inf: None,
    np.nan: None,
}


def mkdir(dirname: Union[str, Path]) -> None:
    """Make directory, including full file path

    Parameters
    ----------
    dirname : Union[str, Path]
        The fullfile path to make

    Raises
    ------
    OSError if any trouble creating the directory
    """
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def rmdir(dirname: Union[str, Path]) -> None:
    """Remove the fullpath to the directory

    Parameters
    ----------
    dirname : Union[str, Path]
        The fullfile path to remove

    Raises
    ------
    OSError if any trouble creating the directory
    """
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)


def process_warnings(warnings: Union[str, List[str], None]) -> str:
    """Convert sequence of warnings into a string"""
    if warnings is None:
        return ""

    def _flatten(_warnings):  # noqa
        for w in _warnings:
            if isinstance(w, Iterable) and not isinstance(w, str):
                for subw in _flatten(w):
                    yield subw
            else:
                yield w

    if len(warnings) > 0:
        if isinstance(warnings, list):
            flattened_warnings = _flatten(warnings)
            warnings = "; ".join(flattened_warnings)
        return warnings
    return ""


def isnumeric(val: Any) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False


def coerce_value(val: Any) -> Any:
    """
    Infer and cast value to valid types

    Parameters
    ----------
    val : Any
        The value to coerce

    Returns
    -------
    coerced_value : Any
        The value with inferred type

    Raises
    ------
    ValueError if can't infer the type from the value.
    """

    if val in NAN_TYPE_MAPPING:
        return NAN_TYPE_MAPPING[val]
    if isnumeric(val):
        try:
            return int(val)
        except ValueError:
            return float(val)

    lower_val = str(val.lower())
    if lower_val in ("true", "false"):
        if "f" in lower_val:
            return False
        else:
            return True

    if "," in val:
        return [coerce_value(v.strip()) for v in val.split(",")]
    return val


def format_value(
    val: Optional[Union[float, Iterable[float]]], precision: int = 4
) -> str:
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
    if val is None:
        return ""

    if isinstance(val, Iterable):
        return f"{tuple((round(v, precision) for v in val))}"
    if precision == 0:
        return f"{int(val)}"
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


def infer_variable_type(values: np.ndarray) -> str:
    """Infer the type of variable distribution from `values`

    Parameters
    ----------
    values : np.ndarray
        An array of observations

    Returns
    -------
    str
        The type of variable. Either 'binary', 'counts', or 'continuous'.
    """
    if values.dtype == bool:
        return "binary"
    if len(set(values)) == 2 and values.min() == 0 and values.max() == 1:
        return "binary"
    if sum(values.astype(int) - values) == 0:
        return "counts"
    return "continuous"


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
        return np.isnan(val)  # type: ignore
    return False


def generate_fake_observations(
    n_observations: int = 10000,
    n_treatments: int = 2,
    n_attributes: int = 2,
    distribution: str = "bernoulli",
    random_seed: int = 123,
) -> DataFrame:
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
            -   'bernoulli': binary metric values with mean proporionality increasing
                from .5 by .1 for each successive treatment
            -   'gaussian': continuous metric values with the mean increasing
                from 0 by 1 for each successive treatment
            -   'poisson': counts metric values with the mean increasing from 1
                by 1 for each successive treatment
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
    np.random.seed(random_seed)

    letters = string.ascii_uppercase
    n_treatments = min(n_treatments, 6)

    data = pd.DataFrame()
    data["id"] = list(range(n_observations))

    # Add treatments
    treatments = list(letters[:n_treatments])
    data["treatment"] = np.random.choice(treatments, size=n_observations)

    # Add attributes (attributes should have no effect)
    attribute_columns = ["attr_{}".format(i) for i in range(n_attributes)]
    for ai, attr in enumerate(attribute_columns):
        attr_vals = [
            "A{}{}".format(ai, a.lower())
            for a in list(letters[: np.random.randint(1, 4)])
        ]
        data[attr] = np.random.choice(attr_vals, size=n_observations)

    # Add measurements, each treatment has successively larger means
    for delta, tr in enumerate(treatments):
        tr_mask = data.treatment == tr
        n_tr = sum(tr_mask)
        if distribution_ == "gaussian":
            data.loc[tr_mask, "metric"] = delta + np.random.randn(n_tr)
        elif distribution_ == "bernoulli":
            data.loc[tr_mask, "metric"] = np.round(0.1 * delta + np.random.random(n_tr))
        elif distribution_ == "poisson":
            data.loc[tr_mask, "metric"] = np.random.poisson(1 + delta, size=n_tr)

    # Set the metric data type
    if distribution_ == "poisson":
        dtype = int  # type: ignore
    elif distribution_ == "bernoulli":
        dtype = bool  # type: ignore
    elif distribution_ == "gaussian":
        dtype = float  # type: ignore

    data = data.astype({"metric": dtype})

    # reorder columns
    base_cols = ["id", "treatment", "metric"]
    attr_cols = [c for c in data.columns if c not in base_cols]
    reordered_cols = base_cols + attr_cols

    return data[reordered_cols]
