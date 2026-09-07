"""AndroidWorld integration; observation/OOB imports do not initialize agents."""


def __getattr__(name):
    if name == "AndroidWorldHost":
        from src.integrations.android_world.host import AndroidWorldHost

        return AndroidWorldHost
    if name == "build_agent":
        from src.integrations.android_world.agent import build_agent

        return build_agent
    if name == "make_agent_result":
        from src.integrations.android_world.host import make_agent_result

        return make_agent_result
    raise AttributeError(name)

__all__ = [
    "AndroidWorldHost",
    "build_agent",
    "make_agent_result",
]
