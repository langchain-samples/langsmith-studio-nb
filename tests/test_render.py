from langsmith_studio_nb._render import LABEL, link_html, link_text


def test_link_html_shows_the_url_and_opens_a_new_tab():
    html = link_html("https://smith.langchain.com/studio/?baseUrl=x")

    assert html == (
        f'{LABEL} <a href="https://smith.langchain.com/studio/?baseUrl=x" '
        'target="_blank" rel="noopener">https://smith.langchain.com/studio/?baseUrl=x</a>'
    )


def test_link_html_escapes_the_url():
    """A Studio URL carries &organizationId; unescaped it is invalid markup."""
    html = link_html("https://studio?baseUrl=x&organizationId=ws-1")

    assert "&amp;organizationId" in html
    assert "=x&organizationId" not in html


def test_link_html_appends_a_hint():
    html = link_html("https://studio.example", hint="Add *.trycloudflare.com")

    assert html.endswith("<br>Add *.trycloudflare.com")


def test_link_text():
    assert link_text("https://studio.example") == f"{LABEL} https://studio.example"
    assert link_text("https://studio.example", hint="do this") == (
        f"{LABEL} https://studio.example\ndo this"
    )
