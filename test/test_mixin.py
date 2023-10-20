import pytest
import json
from spearmint import mixin


def test_init_repr():
    class TestRepr(mixin.InitRepr):
        __ATTRS__ = ["abra"]
        abra = "cadabra"

    ir = TestRepr()
    assert ir.__repr__() == "TestRepr(abra='cadabra')"


def test_jsonable():
    class TestJsonable(mixin.Jsonable):
        def __init__(self):
            self.abra = "cadabra"

    ts = TestJsonable()
    ts_dict = ts.to_dict()
    assert "abra" in ts_dict
    assert ts_dict["abra"] == "cadabra"
    assert json.loads(ts.__repr__()) == ts_dict


def test_dataframeable():
    class ValidDataFrameable(mixin.Dataframeable):
        def to_dict(self):
            return {"abra": ["cadabra", "calamazam"]}

    class InvalidDataframeable(mixin.Dataframeable):
        pass

    with pytest.raises(NotImplementedError):
        InvalidDataframeable().to_dataframe()

    vdf = ValidDataFrameable()
    vdf_dict = vdf.to_dict()
    assert vdf_dict == {"abra": ["cadabra", "calamazam"]}
    df = vdf.to_dataframe()
    assert "abra" in df.columns
    assert df.iloc[0].values == "cadabra"
    assert df.iloc[1].values == "calamazam"
