"""Self-check: HF dataset `config` polje se parsira (default None)."""

from __future__ import annotations

from data.registry import DataSourceSpec


def test_config_parsed() -> None:
    s = DataSourceSpec.from_dict(
        "wiki", {"kind": "hf_text", "dataset_id": "wikitext", "config": "wikitext-103-raw-v1"}
    )
    assert s.config == "wikitext-103-raw-v1"


def test_config_default_none() -> None:
    s = DataSourceSpec.from_dict("ts", {"kind": "hf_text", "dataset_id": "roneneldan/TinyStories"})
    assert s.config is None
