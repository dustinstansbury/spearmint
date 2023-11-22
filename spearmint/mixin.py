import json

from spearmint.typing import Iterable, DataFrame, Any, Dict


class JsonableMixin:
    """
    Objects that can be converted to dict and JSON string
    """

    def to_json(self) -> str:
        """Serialize object to JSON string"""
        return json.dumps(self, default=lambda x: x.__dict__, sort_keys=True, indent=4)

    def to_dict(self) -> Dict[Any, Any]:
        """Deserialize object to dictionary"""
        _dict = json.loads(self.to_json())
        assert isinstance(_dict, dict), "to_json must return a dict-able string"
        return _dict

    def __repr__(self) -> str:
        return self.to_json()


class DataframeableMixin:
    """
    Export an object to tabular format. Must overload/implement a `to_dict`
    method that returns a dictionary representation of the object.
    """

    def to_dict(self) -> dict:
        """
        Returns
        -------
        dictionary : dict
            Representation of the object as a dictionary, e.g.
            {
                "ATTR_1_NAME": [ATTR_1_VALUE],
                "ATTR_2_NAME": [[ATTR_2_VALUE_A, FIELD_2_VALUE_B]],
                "ATTR_3_NAME": [(ATTR_3_VALUE_A, FIELD_3_VALUE_B)], ...
            }

        Raises
        ------
        NotImplementedError unless implemented
        """
        raise NotImplementedError()

    def to_dataframe(self) -> DataFrame:
        """
        Export object to a dataframe. If `safe_cast` is True attempt to
        casting common problematic values (see TYPE_MAPPING); this can be useful
        for ensuring type safety, for example when uploading data to a database.
        """

        def is_valid_value(v):
            if v is None:
                return False
            if isinstance(v, Iterable) and len(v) == 0:
                return False
            return True

        _dict = {k: [v] for k, v in self.to_dict().items() if is_valid_value(v)}
        return DataFrame(_dict)

        # for k, v in self.to_dict().items():
        #     # make values a sequence for dataframe API
        #     if v:
        #         _dict[k] = v

        # return DataFrame(self.to_dict())
