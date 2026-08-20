import pytest

from langsmith_studio_nb._urls import STUDIO_ORIGIN, is_loopback_url, studio_url


def test_studio_url_without_workspace():
    assert studio_url("http://127.0.0.1:2024") == (
        f"{STUDIO_ORIGIN}/studio/?baseUrl=http://127.0.0.1:2024"
    )


def test_studio_url_with_workspace():
    url = studio_url("https://x.trycloudflare.com", workspace_id="ws-1")

    assert url.endswith("?baseUrl=https://x.trycloudflare.com&organizationId=ws-1")


def test_studio_url_is_not_percent_encoded():
    assert "%3A" not in studio_url("http://127.0.0.1:2024")


def test_studio_url_strips_trailing_slashes():
    url = studio_url("http://127.0.0.1:2024/", origin="https://eu.smith.langchain.com/")

    assert url == "https://eu.smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:2024", True),
        ("http://localhost:2024", True),
        ("http://[::1]:2024", True),
        ("http://0.0.0.0:2024", True),
        ("https://x.trycloudflare.com", False),
    ],
)
def test_is_loopback_url(url, expected):
    assert is_loopback_url(url) is expected
