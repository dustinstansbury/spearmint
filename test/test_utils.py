import os
import shutil
import tempfile

import numpy as np
import pytest

from spearmint import utils


@pytest.fixture()
def testdir():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


def test_rmdir_mkdir(testdir):
    # assert original directory exists
    assert os.path.isdir(testdir)

    # rmdir
    utils.rmdir(testdir)
    assert not os.path.isdir(testdir)

    # mkdir
    utils.mkdir(testdir)
    assert os.path.isdir(testdir)


def test_process_warnings():
    assert utils.process_warnings("a warning") == "a warning"
    assert utils.process_warnings(["a", "warning"]) == "a; warning"
    assert utils.process_warnings([["a"], "warning"]) == "a; warning"
    assert utils.process_warnings([["a"], ["warning"]]) == "a; warning"


def test_coerce_value():
    assert utils.coerce_value("true") is True
    assert utils.coerce_value("false") is False

    assert utils.coerce_value(np.nan) is None
    assert utils.coerce_value(np.inf) is None
    assert utils.coerce_value(-np.inf) is None

    assert isinstance(utils.coerce_value("1.0"), float)
    assert isinstance(utils.coerce_value("1"), int)
    assert isinstance(utils.coerce_value("a,b"), list)


def test_format_value():
    assert utils.format_value(1) == "1"
    assert utils.format_value(1.0) == "1.0"
    assert utils.format_value(1.23456) == "1.2346"
    assert utils.format_value((1, 2)) == "(1, 2)"
    assert utils.format_value((1.0, 2.0)) == "(1.0, 2.0)"
    assert utils.format_value((1.234, 2.345), precision=2) == "(1.23, 2.35)"


def test_ensure_dataframe():
    test_data = utils.generate_fake_observations(distribution="bernoulli")
    with pytest.raises(ValueError):
        utils.ensure_dataframe(None)
    assert utils.ensure_dataframe(test_data).equals(test_data)

    class DataObj:
        data = test_data

    assert utils.ensure_dataframe(DataObj(), "data").equals(test_data)


def test_infer_variable_type():
    assert utils.infer_variable_type(np.array([True, False])) == "binary"
    assert utils.infer_variable_type(np.array([1, 0, 1])) == "binary"
    assert utils.infer_variable_type(np.array([1.0, 0.0, 1.0])) == "binary"
    assert utils.infer_variable_type(np.array([1, 2, 3])) == "counts"
    assert utils.infer_variable_type(np.array([1.0, 2.0, 3.0])) == "counts"
    assert utils.infer_variable_type(np.array([1.0, 2.1, 3.0])) == "continuous"


def test_set_matplotlib_backend():
    assert utils.set_matplotlib_backend() in ("pdf", "agg")


def test_safe_isnan():
    assert utils.safe_isnan(None) is False
    assert utils.safe_isnan(np.inf) == False  # noqa
    assert utils.safe_isnan(np.nan) == True  # noqa


def test_generate_fake_observations():
    fake_bernoulli = utils.generate_fake_observations(distribution="bernoulli")
    assert fake_bernoulli.dtypes["metric"] == bool

    fake_gaussian = utils.generate_fake_observations(distribution="gaussian")
    assert fake_gaussian.dtypes["metric"] == float

    fake_poisson = utils.generate_fake_observations(distribution="poisson")
    assert fake_poisson.dtypes["metric"] == int

    with pytest.raises(ValueError):
        utils.generate_fake_observations(distribution="unsupported distribution")

    n_attrs = 2
    n_treatments = 4
    fake_data = utils.generate_fake_observations(
        n_treatments=n_treatments, n_attributes=n_attrs
    )
    n_columns = len(fake_data.columns)
    n_attr_columns = n_columns - 3  # exclude ID, metric, or treatment columns
    assert n_attr_columns == n_attrs
    assert len(fake_data["treatment"].unique()) == n_treatments
