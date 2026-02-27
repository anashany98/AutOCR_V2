from modules.config_normalizer import normalize_config


def test_normalize_lifts_postbatch_sections():
    cfg = {
        "postbatch": {
            "hot_folder": {"enabled": True, "path": "X"},
            "email_importer": {"enabled": True, "host": "imap"},
        }
    }
    out = normalize_config(cfg)
    assert out["hot_folder"]["path"] == "X"
    assert out["email_importer"]["host"] == "imap"


def test_normalize_lifts_llm_vision_to_top_level():
    cfg = {"llm": {"vision": {"enabled": True, "index_path": "data/vision_index.faiss"}}}
    out = normalize_config(cfg)
    assert out["vision"]["enabled"] is True
    assert out["vision"]["index_path"] == "data/vision_index.faiss"


def test_normalize_lifts_llm_output_to_ocr_pipeline_output():
    cfg = {"ocr_pipeline": {}, "llm": {"output": {"formats": ["json"], "save_markdown_in_db": False}}}
    out = normalize_config(cfg)
    assert out["ocr_pipeline"]["output"]["formats"] == ["json"]
    assert out["ocr_pipeline"]["output"]["save_markdown_in_db"] is False


def test_normalize_does_not_override_existing_keys():
    cfg = {
        "hot_folder": {"enabled": False},
        "postbatch": {"hot_folder": {"enabled": True}},
        "vision": {"enabled": False},
        "llm": {"vision": {"enabled": True, "index_path": "X"}},
    }
    out = normalize_config(cfg)
    assert out["hot_folder"]["enabled"] is False
    assert out["vision"]["enabled"] is False

