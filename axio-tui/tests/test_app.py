"""Tests for the pieces of the TUI that hold rules of their own."""

from __future__ import annotations

from axio_tui.app import _citation_markdown, _ThinkingWidget


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


class TestTheThinkingWidget:
    """Reasoning is shown while it lasts and folded to one line once the answer starts."""

    def test_it_shows_the_tail_while_the_model_thinks(self) -> None:
        w = _ThinkingWidget()

        w.add("first\nsecond\nthird")

        assert "thinking…" in w.shown and "third" in w.shown

    def test_it_keeps_only_the_last_lines_on_screen(self) -> None:
        # A model that reasons at length otherwise pushes everything else off the screen.
        w = _ThinkingWidget()

        w.add("\n".join(str(n) for n in range(20)))

        assert "19" in w.shown and "0\n" not in w.shown

    def test_the_answer_folds_it_to_one_line(self) -> None:
        w = _ThinkingWidget()
        w.add("a lot of thinking")

        w.collapse(938)

        assert w.shown == "[dim]✻ thinking · 938 tokens[/]"

    def test_the_count_arrives_after_the_fold(self) -> None:
        # reasoning_tokens comes with IterationEnd, which is after the answer has started and the
        # pane has already folded. `collapse` is the one way in: it keeps the figure it has when
        # called without one, and redraws when a later call brings it.
        w = _ThinkingWidget()
        w.add("x")
        w.collapse()

        w.collapse(42)

        assert "42 tokens" in w.shown

    def test_markup_in_the_reasoning_is_not_read_as_markup(self) -> None:
        # A model that writes [dim] or a bracketed citation would otherwise style the pane.
        w = _ThinkingWidget()

        w.add("consider [dim]this[/] and [1]")

        assert "\\[dim]" in w.shown

    def test_it_says_when_the_provider_billed_for_thinking_and_sent_none(self) -> None:
        # OpenAI withholds reasoning summaries from an unverified organisation. Silence there is
        # indistinguishable from a model that did not reason, and the tokens are billed either way.
        w = _ThinkingWidget()

        w.collapse(938)

        assert w.withheld
        assert w.shown == "[dim]✻ thinking · 938 tokens, not sent by the provider[/]"

    def test_it_says_nothing_of_the_sort_when_the_reasoning_arrived(self) -> None:
        w = _ThinkingWidget()
        w.add("some thinking")

        w.collapse(938)

        assert not w.withheld
        assert w.shown == "[dim]✻ thinking · 938 tokens[/]"

    def test_a_turn_that_did_not_reason_claims_nothing(self) -> None:
        w = _ThinkingWidget()

        w.collapse(0)

        assert not w.withheld

    def test_an_iteration_with_no_reasoning_at_all_is_empty(self) -> None:
        # A tool round that neither reasoned nor was billed for it. Shown, a bare marker says
        # nothing and reads as a second turn.
        w = _ThinkingWidget()

        w.collapse(0)

        assert w.empty

    def test_a_marker_with_something_to_report_is_not_empty(self) -> None:
        billed, spoken = _ThinkingWidget(), _ThinkingWidget()
        spoken.add("brief")

        billed.collapse(120)
        spoken.collapse(0)

        assert not billed.empty and not spoken.empty


def test_the_thinking_widget_keeps_only_the_lines_it_shows() -> None:
    # The whole reasoning was kept and split again on every delta, so a model that reasons at
    # length paid for its own output once per chunk.
    from axio_tui.app import _ThinkingWidget

    widget = _ThinkingWidget()
    for at in range(200):
        widget.add(f"line {at}\n")

    assert len(widget._tail) <= _ThinkingWidget.LINES + 1
    widget._draw()
    assert "line 199" in widget.shown
    assert "line 100" not in widget.shown
    assert not widget.empty
