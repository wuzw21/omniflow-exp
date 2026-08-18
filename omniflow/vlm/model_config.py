from __future__ import annotations

from collections.abc import Mapping
import os

_PLACEHOLDER_API_KEYS = {"not-required"}


def resolve_openai_compatible_config(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    profile: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    env = os.environ if environment is None else environment
    resolved_profile = str(
        profile or env.get("OMNIFLOW_MODEL_ENDPOINT_PROFILE") or "auto"
    ).strip().lower()
    if resolved_profile not in {"auto", "openai", "llmthu"}:
        raise ValueError(f"model_endpoint_profile_invalid:{resolved_profile}")

    if resolved_profile == "llmthu":
        key_candidates = (api_key, env.get("LLMTHU_API_KEY"))
        base_url_candidates = (base_url,)
    elif resolved_profile == "openai":
        key_candidates = (api_key, env.get("OPENAI_API_KEY"))
        base_url_candidates = (base_url, env.get("OPENAI_BASE_URL"))
    else:
        key_candidates = (
            api_key,
            env.get("OPENAI_API_KEY"),
            env.get("LLMTHU_API_KEY"),
            env.get("DASHSCOPE_API_KEY"),
        )
        base_url_candidates = (
            base_url,
            env.get("OPENAI_BASE_URL"),
            env.get("OMNIFLOW_OPENAI_BASE_URL"),
        )

    resolved_api_key = next(
        (
            value
            for candidate in key_candidates
            if (value := str(candidate or "").strip())
            and value.lower() not in _PLACEHOLDER_API_KEYS
        ),
        None,
    )
    resolved_base_url = next(
        (
            value
            for candidate in base_url_candidates
            if (value := str(candidate or "").strip())
        ),
        None,
    )
    if resolved_profile != "auto" and (not resolved_api_key or not resolved_base_url):
        raise ValueError(f"model_endpoint_profile_incomplete:{resolved_profile}")
    return resolved_api_key, resolved_base_url


__all__ = ["resolve_openai_compatible_config"]
