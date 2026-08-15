import pytest

from webapp.parser.utils.shared_logic import safe_parse


class GoodHandler:
    def parse(self, page=None, coordinator=None, **kwargs):
        return (["Col1"], [{"Col1": "1"}], "Contest A", {"meta": True})


def test_good_handler_returns_canonical_tuple():
    h = GoodHandler()
    headers, rows, contest, meta = safe_parse(h)
    assert headers == ["Col1"]
    assert isinstance(rows, list) and rows and rows[0]["Col1"] == "1"
    assert contest == "Contest A"
    assert meta.get("meta") is True


class RaiseHandler:
    def parse(self, *a, **k):
        raise RuntimeError("fail")


def test_handler_exception_returns_error_metadata():
    h = RaiseHandler()
    headers, rows, contest, meta = safe_parse(h)
    assert headers == []
    assert rows == []
    assert isinstance(meta, dict) and meta.get("error") == "exception"


def test_handler_exception_raises_when_requested():
    h = RaiseHandler()
    with pytest.raises(RuntimeError):
        safe_parse(h, raise_on_error=True)


class NoneHandler:
    def parse(self, *a, **k):
        return None


def test_handler_returning_none_normalized_to_error():
    h = NoneHandler()
    headers, rows, contest, meta = safe_parse(h)
    assert headers == []
    assert rows == []
    assert isinstance(meta, dict) and meta.get("error") == "invalid_handler_result"


class MalformedHandler:
    def parse(self, *a, **k):
        return ("only", "two")


def test_malformed_tuple_results_in_error_metadata():
    h = MalformedHandler()
    headers, rows, contest, meta = safe_parse(h)
    assert headers == []
    assert rows == []
    assert isinstance(meta, dict) and meta.get("error") == "invalid_handler_result"
