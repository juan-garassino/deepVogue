"""Registry parse + hot-reload + append round-trip."""
import time
import pytest

yaml = pytest.importorskip("yaml")
from deepVogue.serve.registry import Registry
from deepVogue.serve.schemas import ModelEntry


def _seed_yaml(path, models):
    path.write_text(yaml.safe_dump(models, sort_keys=False))


def test_parse_list_form(tmp_path):
    y = tmp_path / "models.yaml"
    _seed_yaml(y, [
        {"id": "a", "pkl": "/foo/a.pkl"},
        {"id": "b", "pkl": "/foo/b.pkl", "dataset_kind": "frames", "default_trunc": 0.5},
    ])
    r = Registry(yaml_path=y)
    items = r.list()
    assert {m.id for m in items} == {"a", "b"}
    b = r.get("b")
    assert b.dataset_kind == "frames" and b.default_trunc == 0.5


def test_parse_dict_form(tmp_path):
    y = tmp_path / "models.yaml"
    y.write_text(yaml.safe_dump({"models": [{"id": "z", "pkl": "/x/z.pkl"}]}))
    r = Registry(yaml_path=y)
    assert {m.id for m in r.list()} == {"z"}


def test_hot_reload_on_mtime(tmp_path):
    y = tmp_path / "models.yaml"
    _seed_yaml(y, [{"id": "a", "pkl": "/a.pkl"}])
    r = Registry(yaml_path=y)
    assert {m.id for m in r.list()} == {"a"}
    # bump mtime + content
    time.sleep(0.01)
    _seed_yaml(y, [{"id": "a", "pkl": "/a.pkl"}, {"id": "b", "pkl": "/b.pkl"}])
    # force mtime change in case OS resolution is coarse
    import os
    new_mtime = y.stat().st_mtime + 2
    os.utime(y, (new_mtime, new_mtime))
    assert {m.id for m in r.list()} == {"a", "b"}


def test_append_roundtrip(tmp_path):
    y = tmp_path / "models.yaml"
    _seed_yaml(y, [{"id": "a", "pkl": "/a.pkl"}])
    r = Registry(yaml_path=y)
    r.append_entry(ModelEntry(id="b", pkl="/b.pkl", dataset_kind="frames"))
    on_disk = yaml.safe_load(y.read_text())
    ids = {e["id"] for e in on_disk}
    assert ids == {"a", "b"}
    # registry reflects it on next read
    assert {m.id for m in r.list()} == {"a", "b"}


def test_append_replaces_same_id(tmp_path):
    y = tmp_path / "models.yaml"
    _seed_yaml(y, [{"id": "a", "pkl": "/old.pkl"}])
    r = Registry(yaml_path=y)
    r.append_entry(ModelEntry(id="a", pkl="/new.pkl"))
    a = r.get("a")
    assert a.pkl == "/new.pkl"


def test_missing_yaml_is_empty(tmp_path):
    r = Registry(yaml_path=tmp_path / "nope.yaml")
    assert r.list() == []
    with pytest.raises(KeyError):
        r.get("anything")


def test_malformed_raises(tmp_path):
    y = tmp_path / "models.yaml"
    y.write_text("just a string at top level")
    r = Registry(yaml_path=y)
    with pytest.raises(RuntimeError, match="expected list"):
        r.list()
