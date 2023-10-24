from rich.console import Console
from rich.table import Table as RichTable


class SpearmintTable(RichTable):
    """Consistently-formatted Rich table with history"""

    def __init__(self, title: str = None, **rich_table_kwargs):
        """_summary_

        Parameters
        ----------
        title : str, optional
            An optional title for the the able, by default None

        **rich_table_kwargs
            Any arguments that can be provided to `rich.table.Table`.` See
            https://rich.readthedocs.io/en/stable/reference/table.html# for
            details
        """
        title = title if title is not None else ""
        super().__init__(
            title=f"{title}",
            title_justify="left",
            header_style="italic cyan",
            title_style="bold cyan",
            **rich_table_kwargs,
        )
        self._print_history = []

    def print(self) -> None:
        """Print the table state, appending to print history"""
        console = Console(record=True)
        console.print(self)
        self._print_history.append(console.export_text())