import nbstudio


def test_version_is_reported():
    assert isinstance(nbstudio.__version__, str)
    assert nbstudio.__version__


def test_public_api_is_importable():
    for name in nbstudio.__all__:
        assert hasattr(nbstudio, name), name
