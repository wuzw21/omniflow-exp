from __future__ import annotations

from dataclasses import replace
import re
from typing import Any
import xml.etree.ElementTree as ET

from omniflow.core.model import Action, Function, FunctionStep, Observation
from omniflow.core.schemas import canonicalize_action, load_canonical_action_schema

FUNCTION_ARTIFACT_VERSION = "omniflow.function.v2"

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "function_id",
    "name",
    "description",
    "input_schema",
    "bindings",
    "render_bindings",
    "steps",
    "agent_visible",
}
_SOURCE_PATH = re.compile(
    r"^\$\.arguments(?P<tail>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+)$"
)
_TARGET_PATH = re.compile(
    r"^\$\.steps\[(?P<action_index>\d+)]\.action\.args"
    r"(?P<tail>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+)$"
)
_PATH_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)]")
_NON_PARAMETERIZABLE_ACTION_ARGUMENTS = {
    "x",
    "x1",
    "x2",
    "y",
    "y1",
    "y2",
}
_RENDER_ATTRIBUTES = {"text", "content-desc"}


def parse_function_artifact(value: dict[str, Any]) -> Function:
    if not isinstance(value, dict):
        raise ValueError("function_artifact_must_be_object")
    unknown = sorted(set(value) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"function_artifact_unknown_fields:{','.join(unknown)}")
    if value.get("schema_version") != FUNCTION_ARTIFACT_VERSION:
        raise ValueError("unsupported_function_artifact_version")
    raw_steps = value.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("function_steps_required")
    canonical_steps: list[dict[str, Any]] = []
    for index, step in enumerate(raw_steps):
        if not isinstance(step, dict) or set(step) - {
            "step_index",
            "source_state_id",
            "action",
        }:
            raise ValueError("function_step_contract_invalid")
        if step.get("step_index") != index:
            raise ValueError("function_step_index_invalid")
        if not str(step.get("source_state_id") or "").strip():
            raise ValueError("function_step_source_state_id_required")
        action = step.get("action")
        if not isinstance(action, dict) or set(action) != {"tool", "args"}:
            raise ValueError("function_action_contract_invalid")
        if not str(action.get("tool") or "").strip():
            raise ValueError("function_action_tool_required")
        if not isinstance(action.get("args"), dict):
            raise ValueError("function_action_args_must_be_object")
        if "target" in action["args"]:
            raise ValueError(f"function_action_target_forbidden:{index}")
        canonical_steps.append(
            {
                "step_index": index,
                "source_state_id": str(step["source_state_id"]),
                "action": canonicalize_action(action, replayable_only=True),
            }
        )
    canonical_value = dict(value)
    canonical_value["steps"] = canonical_steps
    function = Function.from_dict(canonical_value)
    validate_function_artifact(function)
    return function


def validate_function_artifact(function: Function) -> None:
    if function.schema_version != FUNCTION_ARTIFACT_VERSION:
        raise ValueError("unsupported_function_artifact_version")
    if not function.function_id.strip():
        raise ValueError("function_id_required")
    builtin_tool_names = {
        str(tool.get("name") or "").strip()
        for tool in load_canonical_action_schema().get("tools") or ()
        if isinstance(tool, dict)
    }
    if function.id in builtin_tool_names:
        raise ValueError(f"tool_name_reserved:{function.id}")
    if not function.name.strip():
        raise ValueError("function_name_required")
    if not function.description.strip():
        raise ValueError("function_description_required")
    if not function.steps:
        raise ValueError("function_steps_required")
    for step in function.steps:
        if (
            canonicalize_action(step.action.to_dict(), replayable_only=True)
            != step.action.to_dict()
        ):
            raise ValueError("function_action_not_canonical")
    schema = function.input_schema
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("function_parameters_must_be_object_schema")
    if not isinstance(schema.get("properties"), dict):
        raise ValueError("function_parameter_properties_required")
    if schema.get("additionalProperties") is not False:
        raise ValueError("function_parameters_must_forbid_additional_properties")
    required = schema.get("required") or []
    if not isinstance(required, list) or not all(
        isinstance(name, str) for name in required
    ):
        raise ValueError("function_parameter_required_invalid")
    unknown_required = sorted(set(required) - set(schema["properties"]))
    if unknown_required:
        raise ValueError(
            f"function_parameter_required_unknown:{','.join(unknown_required)}"
        )
    _validate_bindings(function)


def bind_function(function: Function, arguments: dict[str, Any]) -> Function:
    validate_function_artifact(function)
    validate_arguments(function.input_schema, arguments)
    steps = [
        FunctionStep(
            step_index=step.step_index,
            source_state_id=step.source_state_id,
            action=Action(step.action.tool, _copy_value(step.action.args)),
        )
        for step in function.steps
    ]
    source_root = {"arguments": arguments}
    for binding in function.bindings:
        source_match = _SOURCE_PATH.fullmatch(binding["source"])
        target_match = _TARGET_PATH.fullmatch(binding["target"])
        if source_match is None or target_match is None:
            raise ValueError("function_binding_path_invalid")
        value = _read_path(
            source_root, _tokens(".arguments" + source_match.group("tail"))
        )
        action_index = int(target_match.group("action_index"))
        params = _copy_value(steps[action_index].action.args)
        _write_path(params, _tokens(target_match.group("tail")), _copy_value(value))
        steps[action_index] = replace(
            steps[action_index],
            action=Action(steps[action_index].action.tool, params),
        )
    render_bindings: list[dict[str, Any]] = []
    for binding in function.render_bindings:
        source_match = _SOURCE_PATH.fullmatch(str(binding.get("source") or ""))
        if source_match is None:
            raise ValueError("function_render_binding_source_invalid")
        value = _read_path(source_root, _tokens(
            ".arguments" + source_match.group("tail")
        ))
        if not isinstance(value, str):
            raise ValueError("function_render_binding_value_invalid")
        bound = dict(binding)
        bound["replacement"] = value
        render_bindings.append(bound)
    return replace(
        function,
        steps=tuple(steps),
        render_bindings=tuple(render_bindings),
    )


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ValueError("function_arguments_must_be_object")
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        raise ValueError("function_parameter_properties_required")
    required = [str(name) for name in schema.get("required") or ()]
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"function_arguments_invalid:missing:{','.join(missing)}")
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ValueError(f"function_arguments_invalid:unknown:{','.join(unknown)}")
    for name, value in arguments.items():
        definition = properties.get(name)
        if not isinstance(definition, dict):
            continue
        expected_type = str(definition.get("type") or "")
        if expected_type and not _matches_json_type(value, expected_type):
            raise ValueError(f"function_arguments_invalid:type:{name}")
        enum_values = definition.get("enum")
        if isinstance(enum_values, list) and value not in enum_values:
            raise ValueError(f"function_arguments_invalid:enum:{name}")


def _validate_bindings(function: Function) -> None:
    properties = function.input_schema["properties"]
    targets: set[str] = set()
    bound_properties: set[str] = set()
    for binding in function.bindings:
        if set(binding) != {"source", "target"}:
            raise ValueError("function_binding_contract_invalid")
        source = str(binding.get("source") or "")
        target = str(binding.get("target") or "")
        source_match = _SOURCE_PATH.fullmatch(source)
        target_match = _TARGET_PATH.fullmatch(target)
        if source_match is None or target_match is None:
            raise ValueError(f"function_binding_path_invalid:{source}->{target}")
        source_tokens = _tokens(".arguments" + source_match.group("tail"))
        target_tokens = _tokens(target_match.group("tail"))
        if (
            target_tokens
            and target_tokens[-1] in _NON_PARAMETERIZABLE_ACTION_ARGUMENTS
        ):
            raise ValueError(
                f"function_binding_target_non_parameterizable:{target}"
            )
        property_name = source_tokens[1] if len(source_tokens) > 1 else ""
        if property_name not in properties:
            raise ValueError(f"function_binding_source_unknown:{source}")
        bound_properties.add(str(property_name))
        action_index = int(target_match.group("action_index"))
        if action_index not in range(len(function.steps)):
            raise ValueError(f"function_binding_target_invalid:{target}")
        if target in targets:
            raise ValueError(f"function_binding_target_duplicate:{target}")
        targets.add(target)
        try:
            _read_path(
                function.steps[action_index].action.args,
                target_tokens,
            )
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(f"function_binding_target_missing:{target}") from error
    _validate_render_bindings(function, bound_properties)
    unbound = sorted(
        set(function.input_schema.get("required") or ()) - bound_properties
    )
    if unbound:
        raise ValueError(f"function_required_parameters_unbound:{','.join(unbound)}")


def _validate_render_bindings(
    function: Function,
    bound_properties: set[str],
) -> None:
    properties = function.input_schema["properties"]
    targets: set[tuple[int, str, str]] = set()
    for binding in function.render_bindings:
        allowed = {
            "source",
            "step_index",
            "node_id",
            "attribute",
            "recorded_value",
            "replacement",
        }
        if not isinstance(binding, dict) or set(binding) - allowed:
            raise ValueError("function_render_binding_contract_invalid")
        source = str(binding.get("source") or "")
        source_match = _SOURCE_PATH.fullmatch(source)
        if source_match is None:
            raise ValueError(f"function_render_binding_path_invalid:{source}")
        source_tokens = _tokens(".arguments" + source_match.group("tail"))
        if len(source_tokens) != 2 or not isinstance(source_tokens[1], str):
            raise ValueError("function_render_binding_source_invalid")
        parameter_name = source_tokens[1]
        if parameter_name not in properties:
            raise ValueError(f"function_render_binding_source_unknown:{source}")
        bound_properties.add(parameter_name)
        step_index = binding.get("step_index")
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            raise ValueError("function_render_binding_step_index_invalid")
        if step_index not in range(len(function.steps)):
            raise ValueError("function_render_binding_step_index_invalid")
        if function.steps[step_index].action.tool not in {"click", "long_press"}:
            raise ValueError("function_render_binding_action_invalid")
        node_id = str(binding.get("node_id") or "").strip()
        if not node_id:
            raise ValueError("function_render_binding_node_id_required")
        attribute = str(binding.get("attribute") or "").strip()
        if attribute not in _RENDER_ATTRIBUTES:
            raise ValueError("function_render_binding_attribute_invalid")
        recorded_value = binding.get("recorded_value")
        if not isinstance(recorded_value, str) or not recorded_value:
            raise ValueError("function_render_binding_recorded_value_invalid")
        target = (step_index, node_id, attribute)
        if target in targets:
            raise ValueError("function_render_binding_target_duplicate")
        targets.add(target)
        if "replacement" in binding and not isinstance(binding["replacement"], str):
            raise ValueError("function_render_binding_replacement_invalid")


def render_bound_source_state(
    source_state: Observation,
    render_bindings: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    step_index: int,
) -> Observation:
    return _render_bound_state(
        source_state,
        render_bindings,
        step_index=step_index,
        endpoint="source",
    )


def render_bound_target_state(
    target_state: Observation,
    render_bindings: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    step_index: int,
) -> Observation:
    return _render_bound_state(
        target_state,
        render_bindings,
        step_index=step_index,
        endpoint="target",
    )


def _render_bound_state(
    state: Observation,
    render_bindings: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    step_index: int,
    endpoint: str,
) -> Observation:
    """Render both endpoint observations into a private semantic mask view."""

    bindings = [
        binding
        for binding in render_bindings
        if isinstance(binding, dict)
        and int(binding.get("step_index", -1)) == int(step_index)
    ]
    if not bindings:
        return state
    xml = str(state.xml or "")
    if not xml:
        raise ValueError(f"function_render_{endpoint}_xml_missing")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise ValueError(f"function_render_{endpoint}_xml_invalid") from error
    for binding in bindings:
        node_id = str(binding.get("node_id") or "").strip()
        attribute = str(binding.get("attribute") or "").strip()
        recorded_value = binding.get("recorded_value")
        replacement = binding.get("replacement")
        if (
            not isinstance(recorded_value, str)
            or not isinstance(replacement, str)
            or not replacement
        ):
            raise ValueError("function_render_binding_unbound")
        parameter_name = _render_binding_parameter_name(binding)
        mask = f"<{parameter_name}>"
        if endpoint == "source":
            node = next(
                (
                    element
                    for element in root.iter("node")
                    if str(element.attrib.get("id") or "").strip() == node_id
                ),
                None,
            )
            if node is None:
                raise ValueError(f"function_render_binding_node_missing:{node_id}")
            current = str(node.attrib.get(attribute) or "")
            if recorded_value not in current:
                raise ValueError(
                    f"function_render_binding_literal_missing:{node_id}:{attribute}"
                )
            node.attrib[attribute] = current.replace(recorded_value, mask)
            continue
        matched = False
        for node in root.iter("node"):
            for target_attribute in (attribute, *_RENDER_ATTRIBUTES):
                current = str(node.attrib.get(target_attribute) or "")
                if replacement not in current:
                    continue
                node.attrib[target_attribute] = current.replace(replacement, mask)
                matched = True
        if not matched:
            raise ValueError(
                f"function_render_binding_target_literal_missing:{parameter_name}"
            )
    rendered_xml = ET.tostring(root, encoding="unicode")
    return Observation(
        xml=rendered_xml,
        package_name=state.package_name,
        activity_name=state.activity_name,
        image_base64=state.image_base64,
        extra=dict(state.extra),
    )


def _render_binding_parameter_name(binding: dict[str, Any]) -> str:
    source_match = _SOURCE_PATH.fullmatch(str(binding.get("source") or ""))
    if source_match is None:
        raise ValueError("function_render_binding_source_invalid")
    tokens = _tokens(".arguments" + source_match.group("tail"))
    parameter_name = str(tokens[-1]) if tokens else ""
    if not parameter_name:
        raise ValueError("function_render_binding_source_invalid")
    return parameter_name


def _tokens(tail: str) -> list[str | int]:
    return [name if name else int(index) for name, index in _PATH_TOKEN.findall(tail)]


def _read_path(value: Any, tokens: list[str | int]) -> Any:
    current = value
    for token in tokens:
        if isinstance(token, str) and isinstance(current, dict):
            if token not in current:
                raise KeyError(token)
            current = current[token]
        elif isinstance(token, int) and isinstance(current, list):
            current = current[token]
        else:
            raise TypeError("path_type_mismatch")
    return current


def _write_path(value: Any, tokens: list[str | int], replacement: Any) -> None:
    if not tokens:
        raise ValueError("function_binding_target_root_forbidden")
    current = value
    for token in tokens[:-1]:
        if isinstance(token, str) and isinstance(current, dict):
            if token not in current:
                raise KeyError(token)
            current = current[token]
        elif isinstance(token, int) and isinstance(current, list):
            current = current[token]
        else:
            raise TypeError("path_type_mismatch")
    final = tokens[-1]
    if isinstance(final, str) and isinstance(current, dict) and final in current:
        current[final] = replacement
        return
    if isinstance(final, int) and isinstance(current, list):
        current[final] = replacement
        return
    raise KeyError(final)


def _copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def _matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return False
