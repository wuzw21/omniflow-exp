from omniflow.vlm.model_config import resolve_openai_compatible_config


def test_llmthu_profile_uses_openai_base_url_when_not_explicitly_passed():
    api_key, base_url = resolve_openai_compatible_config(
        profile="llmthu",
        environment={
            "LLMTHU_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://llmapi.paratera.com/v1",
        },
    )

    assert api_key == "test-key"
    assert base_url == "https://llmapi.paratera.com/v1"


def test_openai_profile_prefers_dashscope_key_for_dashscope_endpoint():
    api_key, base_url = resolve_openai_compatible_config(
        profile="openai",
        environment={
            "DASHSCOPE_API_KEY": "dashscope-key",
            "OPENAI_API_KEY": "stale-openai-key",
            "OPENAI_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
    )

    assert api_key == "dashscope-key"
    assert base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
