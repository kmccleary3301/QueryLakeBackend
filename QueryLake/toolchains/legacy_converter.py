from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from QueryLake.typing.toolchains import (
    Mapping,
    MappingDestination,
    NodeV2,
    StreamConfig,
    ToolChainV2,
    ValueExpression,
    ValueRef,
)
from QueryLake.typing import toolchains_legacy as legacy


def _route_to_jsonpath(route: Optional[List[Any]]) -> str:
    if not route:
        return "$"
    parts: List[str] = ["$"]
    for elem in route:
        if isinstance(elem, int):
            parts.append(f"[{elem}]")
        else:
            safe = str(elem)
            if safe.isidentifier():
                parts.append(f".{safe}")
            else:
                parts.append(f"['{safe}']")
    return "".join(parts)


def _value_obj_to_literal(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_value_obj_to_literal(item) for item in obj]
    if not isinstance(obj, dict):
        return obj
    obj_type = obj.get("type")
    if obj_type is None:
        return {k: _value_obj_to_literal(v) for k, v in obj.items()}
    if obj_type == "staticValue":
        return _value_obj_to_literal(obj.get("value"))
    if obj_type == "getNodeInput":
        return {"ref": ValueRef(source="inputs", path=_route_to_jsonpath(obj.get("route"))).model_dump()}
    if obj_type == "getNodeOutput":
        return {"ref": ValueRef(source="outputs", path=_route_to_jsonpath(obj.get("route"))).model_dump()}
    if obj_type == "stateValue":
        return {"ref": ValueRef(source="state", path=_route_to_jsonpath(obj.get("route"))).model_dump()}
    if obj_type == "getFile":
        return {"ref": ValueRef(source="files", path=_route_to_jsonpath(obj.get("route"))).model_dump()}
    raise NotImplementedError(f"Unsupported value object type: {obj_type}")


def _value_source_from_feed_map(fm: legacy.feedMappingAtomic) -> ValueExpression:
    if getattr(fm, "value", None) is not None:
        literal = _value_obj_to_literal(fm.value)
        if literal is None:
            return ValueExpression.from_any({"literal": {}})
        return ValueExpression.from_any({"literal": literal})
    for attr, source in (
        ("getFromInputs", "inputs"),
        ("getFromOutputs", "outputs"),
        ("getFromState", "state"),
        ("getFromFiles", "files"),
        ("getFrom", None),
    ):
        value = getattr(fm, attr, None)
        if value is None:
            continue
        if attr == "getFrom" and value.type == "getNodeOutput":
            return ValueExpression(ref=ValueRef(source="outputs", path=_route_to_jsonpath(value.route)))
        route = getattr(value, "route", None)
        src = source or "inputs"
        return ValueExpression(ref=ValueRef(source=src, path=_route_to_jsonpath(route)))
    return ValueExpression.from_any({"literal": {}})


def _set_literal_at_path(container: Any, path: List[Any], expression: ValueExpression) -> None:
    if not path:
        raise ValueError("Path required for insertion.")
    current = container
    for segment in path[:-1]:
        key = segment
        if isinstance(current, list):
            index = int(key)
            while len(current) <= index:
                current.append({})
            if not isinstance(current[index], (dict, list)):
                current[index] = {}
            current = current[index]
        else:
            if key not in current or not isinstance(current[key], (dict, list)):
                current[key] = {}
            current = current[key]
    final_key = path[-1]
    if isinstance(current, list):
        index = int(final_key)
        while len(current) <= index:
            current.append(None)
        current[index] = expression.model_dump()
    else:
        current[final_key] = expression.model_dump()


def _convert_append_action(
    append_action: legacy.appendAction,
    fm: legacy.feedMapping,
) -> ValueExpression:
    if append_action.initialValue is not None:
        base_literal = _value_obj_to_literal(append_action.initialValue)
    else:
        base_literal = {}
    if not isinstance(base_literal, (dict, list)):
        base_literal = {}
    literal_copy = copy.deepcopy(base_literal)
    insertions = append_action.insertions or []
    insertion_values = append_action.insertion_values or []
    src_expression = _value_source_from_feed_map(fm)
    for idx, path in enumerate(insertions):
        insertion_path = path or []
        if idx < len(insertion_values):
            val_spec = insertion_values[idx]
            if val_spec is None:
                expr = src_expression
            else:
                expr = ValueExpression.from_any({"literal": _value_obj_to_literal(val_spec)})
        else:
            expr = src_expression
        _set_literal_at_path(literal_copy, insertion_path, expr)
    return ValueExpression.from_any({"literal": literal_copy})


def _convert_feed_mapping(fm: legacy.feedMapping) -> Mapping:
    if fm.destination == "<<STATE>>":
        destination = MappingDestination(kind="state")
    elif fm.destination == "<<USER>>":
        destination = MappingDestination(kind="user")
    elif fm.destination == "<<FILES>>":
        destination = MappingDestination(kind="files")
    else:
        destination = MappingDestination(kind="node", id=fm.destination)

    stream_cfg = StreamConfig()
    if getattr(fm, "stream", False):
        stream_cfg.enabled = True
        stream_cfg.mode = "append"
        stream_cfg.initial = getattr(fm, "stream_initial_value", None)

    mode = "set"
    path = "$"
    value_expr = _value_source_from_feed_map(fm)

    if fm.sequence:
        for action in fm.sequence:
            if isinstance(action, legacy.appendAction):
                mode = "append"
                path = _route_to_jsonpath(action.route)
                value_expr = _convert_append_action(action, fm)
            elif isinstance(action, legacy.createAction):
                mode = "set"
                path = _route_to_jsonpath(action.route)
            else:
                raise NotImplementedError(f"Unsupported action type: {type(action)}")
    return Mapping(
        destination=destination,
        path=path,
        value=value_expr,
        mode=mode,
        stream=stream_cfg,
    )


def _convert_input_argument(arg: legacy.nodeInputArgument, node_id: str, optional: List[str]) -> Optional[ValueExpression]:
    if arg.value is not None:
        return ValueExpression.from_any({"literal": arg.value})
    if arg.from_state is not None:
        return ValueExpression(ref=ValueRef(source="state", path=_route_to_jsonpath(arg.from_state.route)))
    if getattr(arg, "from_user", False):
        return ValueExpression(ref=ValueRef(source="inputs", path=f"$.{arg.key}"))
    if getattr(arg, "from_server", False):
        return ValueExpression(ref=ValueRef(source="server", path=f"$.{arg.key}"))
    if getattr(arg, "optional", False):
        optional.append(arg.key)
    if arg.default_value is not None:
        return ValueExpression.from_any({"literal": arg.default_value})
    return None


def convert_toolchain(legacy_toolchain: legacy.ToolChain) -> ToolChainV2:
    new_nodes: List[NodeV2] = []
    for legacy_node in legacy_toolchain.nodes:
        node_type = "api" if legacy_node.api_function else "transform"
        optional_inputs: List[str] = []
        inputs: Dict[str, ValueExpression] = {}
        for arg in legacy_node.input_arguments or []:
            expr = _convert_input_argument(arg, legacy_node.id, optional_inputs)
            if expr is not None:
                inputs[arg.key] = expr
        metadata: Dict[str, Any] = {}
        if optional_inputs:
            metadata["optional_inputs"] = optional_inputs
        mappings = [
            _convert_feed_mapping(feed_map)
            for feed_map in legacy_node.feed_mappings or []
        ]
        new_nodes.append(
            NodeV2(
                id=legacy_node.id,
                type=node_type,
                api_function=legacy_node.api_function,
                inputs=inputs,
                mappings=mappings,
                metadata=metadata,
            )
        )
    return ToolChainV2(
        name=legacy_toolchain.name,
        id=legacy_toolchain.id,
        category=legacy_toolchain.category,
        initial_state=legacy_toolchain.initial_state,
        nodes=new_nodes,
        display_configuration=legacy_toolchain.display_configuration,
        suggestions=legacy_toolchain.suggestions,
        first_event_follow_up=legacy_toolchain.first_event_follow_up,
    )


def convert_toolchain_dict(data: Dict[str, Any]) -> ToolChainV2:
    legacy_tc = legacy.ToolChain(**data)
    return convert_toolchain(legacy_tc)
