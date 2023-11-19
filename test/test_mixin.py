import json

import pytest

from spearmint import mixin


def test_jsonable():
    class TestJsonable(mixin.JsonableMixin):
        def __init__(self):
            self.abra = "cadabra"

    ts = TestJsonable()
    ts_dict = ts.to_dict()
    assert "abra" in ts_dict
    assert ts_dict["abra"] == "cadabra"
    assert json.loads(ts.__repr__()) == ts_dict


def test_dataframeable():
    class ValidDataFrameable(mixin.DataframeableMixin):
        def to_dict(self):
            return {"abra": ("cadabra", "calamazam"), "spearmint": "refreshing"}

    class InvalidDataframeable(mixin.DataframeableMixin):
        pass

    with pytest.raises(NotImplementedError):
        InvalidDataframeable().to_dataframe()

    vdf = ValidDataFrameable()
    vdf_dict = vdf.to_dict()

    assert vdf_dict == {"abra": ("cadabra", "calamazam"), "spearmint": "refreshing"}
    df = vdf.to_dataframe()
    assert df.columns.tolist() == ["abra", "spearmint"]
    assert df.iloc[0].tolist() == [("cadabra", "calamazam"), "refreshing"]
