from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .toolchain_interface import DisplaySection


class ValueRef(BaseModel):
    """Reference to a value within session state, node contexts, or literal data."""

    source: Literal["state", "inputs", "outputs", "files", "server", "const"]
    path: Optional[str] = None
    value: Optional[Any] = None

    @model_validator(mode="after")
    def _validate_reference(cls, values: "ValueRef") -> "ValueRef":
        if values.source == "const":
            if values.value is None:
                raise ValueError("ValueRef with source 'const' requires a 'value'.")
        else:
            if not values.path:
                raise ValueError("ValueRef requires 'path' when source is not 'const'.")
        return values


class ValueExpression(BaseModel):
    """Represents either a reference or an inline literal."""

    ref: Optional[ValueRef] = None
    literal: Optional[Any] = None

    @model_validator(mode="after")
    def _check_exclusive_fields(cls, values: "ValueExpression") -> "ValueExpression":
        has_ref = values.ref is not None
        has_literal = values.literal is not None
        if has_ref == has_literal:
            raise ValueError("ValueExpression requires exactly one of 'ref' or 'literal'.")
        return values

    @classmethod
    def from_any(cls, value: Any) -> "ValueExpression":
        if isinstance(value, ValueExpression):
            return value
        if isinstance(value, dict) and ("ref" in value or "literal" in value):
            return cls.model_validate(value)
        # Treat bare ValueRef dicts as refs for ergonomics
        if isinstance(value, dict) and "source" in value:
            return cls(ref=ValueRef.model_validate(value))
        return cls(literal=value)


class MappingDestination(BaseModel):
    kind: Literal["state", "node", "user", "files"]
    id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_destination(cls, values: "MappingDestination") -> "MappingDestination":
        if values.kind == "node" and not values.id:
            raise ValueError("Mapping destination 'node' requires an 'id'.")
        if values.kind != "node" and values.id is not None:
            raise ValueError("Only 'node' destinations may specify an 'id'.")
        return values


class StreamConfig(BaseModel):
    enabled: bool = False
    mode: Literal["append", "set"] = "append"
    initial: Optional[Any] = None


class Mapping(BaseModel):
    destination: MappingDestination
    path: str
    value: ValueExpression
    mode: Literal["set", "append", "merge"] = "set"
    condition: Optional[Dict[str, Any]] = None
    iterate: Optional[ValueRef] = None
    stream: StreamConfig = Field(default_factory=StreamConfig)


class NodeV2(BaseModel):
    id: str
    type: Literal["api", "transform"] = "api"
    api_function: Optional[str] = None
    inputs: Dict[str, ValueExpression] = Field(default_factory=dict)
    mappings: List[Mapping] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_node(cls, values: "NodeV2") -> "NodeV2":
        if values.type == "api" and not values.api_function:
            raise ValueError("API nodes require 'api_function'.")
        if values.type == "transform" and values.api_function is not None:
            raise ValueError("Transform nodes must not define 'api_function'.")
        return values


class StartScreenSuggestion(BaseModel):
    display_text: str
    event_id: str
    event_parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolChainV2(BaseModel):
    name: str
    id: str
    category: str
    initial_state: Dict[str, Any]
    nodes: List[NodeV2]
    display_configuration: Optional[DisplaySection] = None
    suggestions: List[StartScreenSuggestion] = Field(default_factory=list)
    first_event_follow_up: Optional[str] = None
    ui_spec_v2: Optional[Dict[str, Any]] = None


class ToolChainSessionFile(BaseModel):
    type: Literal["<<||TOOLCHAIN_SESSION_FILE||>>"] = "<<||TOOLCHAIN_SESSION_FILE||>>"
    name: Optional[str] = None
    document_hash_id: str


__all__ = [
    "Mapping",
    "MappingDestination",
    "NodeV2",
    "StreamConfig",
    "ToolChainV2",
    "ToolChainSessionFile",
    "ValueExpression",
    "ValueRef",
]


# Legacy schema exports for compatibility
from QueryLake.typing import toolchains_legacy as _legacy  # noqa: E402
from QueryLake.typing.toolchains_legacy import *  # noqa: E402,F401,F403

ToolChain = _legacy.ToolChain

if hasattr(_legacy, "__all__"):
    __all__ += [name for name in _legacy.__all__ if name not in __all__]
else:
    __all__ += [name for name in dir(_legacy) if not name.startswith("_") and name not in __all__]
