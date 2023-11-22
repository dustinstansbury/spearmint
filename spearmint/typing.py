from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Protocol, Tuple, Union, Optional

from pandas import DataFrame
from numpy import ndarray

Sequence = Union[List, Tuple, ndarray]
DataFrameColumns = Union[str, List[str]]
FilePath = Union[str, Path]


__all__ = [
    "Path",
    "Any",
    "Callable",
    "Dict",
    "Iterable",
    "List",
    "Protocol",
    "Tuple",
    "Union",
    "DataFrame",
    "DataFrameColumns",
    "FilePath",
    "Sequence",
    "Optional",
]
