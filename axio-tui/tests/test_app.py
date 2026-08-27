def test_a_citation_cannot_close_its_own_markdown_link() -> None:
    # A `]` in the title and a `)` in the query string ended the link early. The rest of the url
    # then reached the transcript as text.
    from axio_tui.app import _link_safe

    label = _link_safe("Smith v. Jones (2019) [pdf]")
    target = _link_safe("https://ex.com/s?q=a)b&t=1")
    assert "]" not in label and ")" not in target
    assert f"[{label}]({target})".count("](") == 1
