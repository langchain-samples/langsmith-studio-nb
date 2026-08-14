import langsmith_studio_nb


def test_version_is_reported():
    assert isinstance(langsmith_studio_nb.__version__, str)
    assert langsmith_studio_nb.__version__


def test_public_api_is_importable():
    for name in langsmith_studio_nb.__all__:
        assert hasattr(langsmith_studio_nb, name), name
