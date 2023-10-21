from spearmint import table


def test_spearmint_table():
    spearmint_table = table.SpearmintTable(title="\ntest title", min_width=10)
    spearmint_table.add_column("test row", justify="right")

    # print once
    spearmint_table.print()

    # print again, with new title
    spearmint_table.title = "\nnew title"
    spearmint_table.print()
    assert len(spearmint_table._print_history) == 2

    # get printed text, excluding table grids
    first_printed_lines = spearmint_table._print_history[0].split("\n")[1::2]

    assert "test title" in first_printed_lines[0]
    assert "test row" in first_printed_lines[1]

    # get printed text, excluding table grids
    second_printed_lines = spearmint_table._print_history[1].split("\n")[1::2]
    assert "new title" in second_printed_lines[0]
    assert "test title" not in second_printed_lines[0]
    assert "test row" in second_printed_lines[1]


def test_spearmint_table_extension():
    class MySpearmintTable(table.SpearmintTable):
        def __init__(self):
            super().__init__(title="\nmy table")

            self.add_column("my first column", justify="right")
            self.add_column("my second column", justify="left")

            self.add_row("first value", "second value")

    my_table = MySpearmintTable()
    my_table.print()

    # get printed text, excluding table grids
    printed_lines = my_table._print_history[0].split("\n")[1::2]
    assert "my table" in printed_lines[0]
    assert "my first column" in printed_lines[1]
    assert "my second column" in printed_lines[1]
    assert "first value" in printed_lines[2]
    assert "second value" in printed_lines[2]
