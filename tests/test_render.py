from langsmith_studio_nb._render import link_html, link_text


def test_link_html_contains_url_and_opens_a_new_tab():
    html = link_html("https://smith.langchain.com/studio/?baseUrl=x")

    assert 'href="https://smith.langchain.com/studio/?baseUrl=x"' in html
    assert 'target="_blank"' in html
    assert "<div" not in html


def test_link_html_appends_a_hint():
    html = link_html("https://studio.example", hint="Add *.trycloudflare.com")

    assert "Add *.trycloudflare.com" in html
    assert html.index("Add *.trycloudflare.com") > html.index("Open LangGraph Studio")


def test_link_text():
    assert link_text("https://studio.example") == "https://studio.example"
    assert link_text("https://studio.example", hint="do this") == (
        "https://studio.example\ndo this"
    )
