from energiewende import config
from energiewende.ingest import download


class FakeResponse:
    """Looks like a requests response, but holds fixed data."""

    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def use_fake_internet(monkeypatch, tmp_path):
    """Send all downloads to a fake, and the cache to a temporary folder.
    Returns the list of downloaded URLs."""
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResponse({"call": len(calls)})

    monkeypatch.setattr(download.requests, "get", fake_get)
    return calls


def test_second_call_comes_from_cache(monkeypatch, tmp_path):
    calls = use_fake_internet(monkeypatch, tmp_path)

    first = download.get_json("https://example.com/data")
    second = download.get_json("https://example.com/data")

    assert first == {"call": 1}
    assert second == {"call": 1}  # same data, no new download
    assert len(calls) == 1


def test_old_cache_is_downloaded_again(monkeypatch, tmp_path):
    calls = use_fake_internet(monkeypatch, tmp_path)

    download.get_json("https://example.com/data")
    result = download.get_json("https://example.com/data", max_age_hours=0)

    assert result == {"call": 2}
    assert len(calls) == 2


def test_different_params_are_cached_separately(monkeypatch, tmp_path):
    calls = use_fake_internet(monkeypatch, tmp_path)

    download.get_json("https://example.com/data", params={"day": 1})
    download.get_json("https://example.com/data", params={"day": 2})

    assert len(calls) == 2
    assert len(list(tmp_path.iterdir())) == 2
