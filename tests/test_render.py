from langsmith_studio_nb._render import link_html, render


def test_link_html_contains_url_and_opens_a_new_tab():
    html = link_html("https://smith.langchain.com/studio/?baseUrl=x")

    assert 'href="https://smith.langchain.com/studio/?baseUrl=x"' in html
    assert 'target="_blank"' in html


def test_render_shows_html_and_echoes_the_url():
    shown: list[str] = []
    echoed: list[str] = []

    render("https://studio.example", display_html=shown.append, echo=echoed.append)

    assert shown == [link_html("https://studio.example")]
    assert echoed == ["https://studio.example"]
