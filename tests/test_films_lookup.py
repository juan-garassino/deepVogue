"""Films endpoint path resolution must not be DV_DATASET_NAME-scoped.

Regression: when FastAPI launches with `DV_DATASET_NAME=A` but a request
asks for a model registered under id `B`, the lookup must resolve under
`<unscoped DV_WALKS_DIR>/B/<walk_id>.mp4` (not `.../A/B/...`).
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("yaml")


def test_films_path_uses_unscoped_walks_dir_and_model_id(tmp_path, monkeypatch):
    walks = tmp_path / "walks"
    monkeypatch.setenv("DV_WALKS_DIR", str(walks))
    monkeypatch.setenv("DV_DATASET_NAME", "A")        # different from the model id below
    # api module is import-time only; load fresh.
    import importlib
    from deepVogue.serve import api as api_mod
    importlib.reload(api_mod)
    p = api_mod._resolve_film_path("B", "test")
    # must be under walks/B/, NOT walks/A/B/
    assert "A" not in p.parts, p
    assert p == walks / "B" / "test.mp4", p


def test_films_path_walks_dir_override_from_registry(tmp_path, monkeypatch):
    walks = tmp_path / "walks"; walks.mkdir()
    override = tmp_path / "elsewhere"; override.mkdir()
    monkeypatch.setenv("DV_WALKS_DIR", str(walks))
    monkeypatch.delenv("DV_DATASET_NAME", raising=False)

    # write a registry yaml that overrides walks_dir for model_id "Z"
    import yaml
    reg_yaml = tmp_path / "models.yaml"
    reg_yaml.write_text(yaml.safe_dump([
        {"id": "Z", "pkl": "/dev/null", "walks_dir": str(override)},
    ]))
    monkeypatch.setenv("DV_MODELS_YAML", str(reg_yaml))

    import importlib
    from deepVogue.serve import api as api_mod
    importlib.reload(api_mod)
    p = api_mod._resolve_film_path("Z", "abc")
    assert p == override / "abc.mp4", p
