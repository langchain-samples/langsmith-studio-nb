from langsmith_studio_nb._ports import resolve_port


def test_resolve_port_keeps_a_free_port():
    assert resolve_port(2024, is_free=lambda _port: True, find_free=lambda: 51234) == 2024


def test_resolve_port_replaces_a_busy_one():
    assert resolve_port(2024, is_free=lambda _port: False, find_free=lambda: 51234) == 51234
