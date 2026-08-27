"""Tests for the pieces of the TUI that hold rules of their own."""

from __future__ import annotations

from axio_tui.app import _citation_markdown


class TestCitationRendering:
    """What the Citation branch appends to the transcript."""

    def test_a_title_cannot_close_the_label(self) -> None:
        # The `]` ended the label, and the rest of the title became the link target.
        rendered = _citation_markdown("Smith v. Jones (2019) [pdf]", "", "https://ex.com/a")

        assert rendered.count("](") == 1
        assert rendered == " [Smith v. Jones (2019) \\[pdf\\]](https://ex.com/a)"

    def test_a_query_string_cannot_close_the_link(self) -> None:
        # The `)` ended the link, and `b&t=1)` reached the transcript as text.
        rendered = _citation_markdown("Case", "", "https://ex.com/s?q=a)b&t=1")

        assert rendered == " [Case](https://ex.com/s?q=a%29b&t=1)"

    def test_a_citation_with_no_url_is_not_a_link(self) -> None:
        assert _citation_markdown(None, "the cited words", None) == " _[the cited words]_"

    def test_a_citation_with_nothing_at_all_still_says_something(self) -> None:
        assert _citation_markdown(None, "", None) == " _[source]_"

    def test_a_trailing_backslash_cannot_escape_the_closing_bracket(self) -> None:
        # Escaped after the brackets, a title ending in `\` produced `\]`, markdown read it as an
        # escaped bracket, and the label never closed.
        rendered = _citation_markdown("ends with a backslash \\", "", "https://e.com/a")

        assert rendered == " [ends with a backslash \\\\](https://e.com/a)"
