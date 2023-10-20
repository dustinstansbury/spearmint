import json
import numpy as np
from pandas import DataFrame


class InitRepr:
    """
    Objects whose __repr__ returns an intialization command
    """

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        class_len = len(class_name)
        attr_str = [
            "{}={!r}".format(attr, getattr(self, attr)) for attr in self.__ATTRS__
        ]
        attr_str = ",\n{}".format(" " * (1 + class_len)).join(attr_str)
        return f"{class_name}({attr_str})"


class Jsonable:
    """
    Objects that can be converted to dict and JSON string
    """

    def to_json(self) -> str:
        """Serialize object to JSON string"""
        return json.dumps(self, default=lambda x: x.__dict__, sort_keys=True, indent=4)

    def to_dict(self) -> dict:
        """Deserialize object to dictionary"""
        return json.loads(self.to_json())

    def __repr__(self) -> str:
        return self.to_json()


class Dataframeable:
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
        return DataFrame(self.to_dict())

    # def to_dataframe(self, safe_cast: bool = False) -> DataFrame:
    #     """
    #     Export object to a dataframe. If `safe_cast` is True attempt to
    #     casting common problematic values (see TYPE_MAPPING); this can be useful
    #     for ensuring type safety, for example when uploading data to a database.
    #     """
    #     _dict = self.to_dict()
    #     if safe_cast:
    #         from spearmint.utils import safe_cast_json

    #         _dict = safe_cast_json(_dict, TYPE_MAPPING)
    #     return DataFrame(_dict)
