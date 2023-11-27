from pandas import DataFrame

from spearmint import config
from spearmint.typing import DataFrameColumns, List, Tuple, Optional


def search_config(df: DataFrame, section: str, key: str) -> List[str]:
    """
    Search a dataframe `df` for parameters defined in the global configuration.

    Parameters
    ----------
    df: dataframe
        Data to search
    param_name: str
        type of parameter to search ('entities', 'metrics', or 'attributes')

    Returns
    -------
    parameters : List[str]
        Any found parameters in the dataframe
    """
    available = config.get(section, key)
    available = [available] if not isinstance(available, list) else available
    columns = df.columns
    return [c for c in columns if c in available]


class DatasetException(Exception):
    pass


class Dataset:
    """Interface between global configuration, data observations, and `Experiment`."""

    def __init__(
        self,
        df: DataFrame,
        treatment: Optional[str] = None,
        measures: Optional[DataFrameColumns] = None,
        attributes: Optional[DataFrameColumns] = None,
        metadata: Optional[DataFrameColumns] = None,
    ):
        """
        Parameters
        ----------
        df : DataFrame
            A data set of raw observations data for an experiment.
        treatment : str, optional
            A discrete-valued column in `df` defining the treatment. If None,
            provided, the `experiment.treatment` variable in `spearmint.cfg` file
            will be used.
        measures : DataFrameColumns, optional
            The columns in `df` associated with the measured outcome variables.
            If None provided, the `experiment.measures` variable in `spearmint.cfg`
            file will be used.
        attributes : DataFrameColumns, optional
            The columns in  `df` defining attributes that can be used for
            segmentation. If None provided, the `experiment.attributes` variable
            in `spearmint.cfg` file will be used.
        metadata : DataFrameColumns, optional
            _description_, by default None

        Raises
        ------
        DatasetException
            -   If any of the columns defining the treatment, measures, attributes
                or metadata are not in `df`
        """

        def _check_columns_in_df(columns, column_type):
            cols = [columns] if isinstance(columns, str) else columns
            for c in cols:
                if c not in df.columns:
                    raise DatasetException(f"{column_type} column {c} not in dataframe")

        # treatment
        treatment = (
            treatment
            if treatment
            else search_config(df, "hypothesis_test", "default_treatment_name")[0]
        )
        _check_columns_in_df(treatment, "treatment")
        self.treatment = treatment

        # measures
        measures = [measures] if isinstance(measures, str) else measures
        measures = (
            measures
            if measures
            else search_config(df, "experiment", "default_measure_names")
        )
        _check_columns_in_df(measures, "measure")
        self.measures = measures

        # attributes
        attributes = [attributes] if isinstance(attributes, str) else attributes
        attributes = (
            attributes
            if attributes
            else search_config(df, "hypothesis_test", "default_attribute_names")
        )
        _check_columns_in_df(attributes, "attribute")
        self.attributes = attributes

        metadata = [metadata] if isinstance(metadata, str) else metadata
        metadata = metadata if metadata else []
        _check_columns_in_df(metadata, "metadata")
        self.metadata = metadata

        all_columns = [self.treatment] + self.measures + self.attributes + self.metadata
        self.data = df[all_columns]
        self.columns = set(all_columns)

    def __repr__(self):
        return f"Dataset(treatment='{self.treatment}', measures={self.measures}, attributes={self.attributes})"

    @property
    def cohorts(self) -> List[str]:
        """
        Returns
        -------
        cohorts: List[str]
            A list of cohorts defined by the `treatment` column.

        Example
        -------
        >>> observations = utils.generate_fake_observations()
        >>> observations.head()
            id treatment attr_0 attr_1  metric
        0   0         A    A0b    A1c   False
        1   1         B    A0c    A1a    True
        2   2         A    A0a    A1c    True
        3   3         A    A0a    A1c   False
        4   4         A    A0c    A1c   False
        >>> dataset = Dataset(observations)
        >>>  print(dataset.cohorts)
        ['A', 'B']
        """
        if not hasattr(self, "_cohorts"):
            self._cohorts = sorted(self.data[self.treatment].unique().tolist())
        return self._cohorts

    def segments(self, attribute: str) -> List[Tuple[str, str]]:
        """
        Return a list of tuples containing (treatment, segment) pairs, for a
        given segmentation attribute.

        Returns
        -------
        segments:
            A list of tuples, containing each combination of treatment and the
            values taken by the provided `attribute`.

        Example
        -------
        >>> observations = utils.generate_fake_observations()
        >>> observations.head()
            id treatment attr_0 attr_1  metric
        0   0         A    A0b    A1c   False
        1   1         B    A0c    A1a    True
        2   2         A    A0a    A1c    True
        3   3         A    A0a    A1c   False
        4   4         A    A0c    A1c   False
        >>> dataset = Dataset(observations)
        >>> print(dataset.segments("attr_0"))
        [('A', 'A0a'), ('A', 'A0b'), ('A', 'A0c'), ('B', 'A0a'), ('B', 'A0b'), ('B', 'A0c')]
        """
        return self.data.groupby([self.treatment, attribute]).sum().index.tolist()

    @property
    def cohort_measures(self) -> DataFrame:
        """
        Returns
        -------
        measures: DataFrame
            The metric samples for each cohort. Each row is a cohort, and the
            value is a list of metric values.

        Example
        -------
        >>> observations = utils.generate_fake_observations()
        >>> observations.head()
            id treatment attr_0 attr_1  metric
        0   0         A    A0b    A1c   False
        1   1         B    A0c    A1a    True
        2   2         A    A0a    A1c    True
        3   3         A    A0a    A1c   False
        4   4         A    A0c    A1c   False
        >>> dataset = Dataset(observations)
        >>> print(dataset.cohort_measures)
                                                    metric
        A  [False, True, False, False, False, False, Fals...
        B  [True, False, True, False, False, False, True,...
        """
        measures: dict = {}
        for cohort in self.cohorts:
            measures[cohort] = {}
            for metric in self.measures:
                mask = self.data[self.treatment] == cohort
                measures[cohort][metric] = self.data[mask][metric].values

        return DataFrame(measures).T

    def segment_samples(self, attribute: str) -> DataFrame:
        """
        Returns
        -------
        Returns the metric values for each treatment-attribute pair.

        Example
        -------
        >>> observations = utils.generate_fake_observations()
        >>> observations.head()
            id treatment attr_0 attr_1  metric
        0   0         A    A0b    A1c   False
        1   1         B    A0c    A1a    True
        2   2         A    A0a    A1c    True
        3   3         A    A0a    A1c   False
        4   4         A    A0c    A1c   False
        >>> dataset = Dataset(observations)
        >>> print(dataset.segment_samples('attr_0'))
        print(dataset.segment_samples('attr_0'))
                    metric
        A   A0a     [True, False, True, False, False, False, False...
            A0b     [False, True, False, True, True, False, False,...
            A0c     [False, False, False, False, True, False, True...
        B   A0a     [True, False, False, True, True, False, True, ...
            A0b     [False, False, True, True, False, False, False...
            A0c     [True, False, True, False, True, True, True, F...
        """
        measures: dict = {}
        for segment in self.segments(attribute):
            measures[segment] = {}
            for metric in self.measures:
                mask = (self.data[self.treatment] == segment[0]) & (
                    self.data[attribute] == segment[1]
                )
                measures[segment][metric] = self.data[mask][metric].values
        return DataFrame(measures).T
