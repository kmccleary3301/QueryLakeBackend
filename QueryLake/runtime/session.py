from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import jsonpatch
from types import SimpleNamespace

try:
    from jsonpath_ng.ext import parse as jsonpath_parse
except ImportError:  # pragma: no cover - fallback used in minimal test environments
    class _MiniJSONPath:
        def __init__(self, path: str) -> None:
            self.tokens = self._tokenize(path)

        def _tokenize(self, path: str):
            if not path or path == "$":
                return []
            if not path.startswith("$"):
                raise ValueError("JSONPath must start with '$'")
            tokens = []
            current = ""
            idx = 1
            while idx < len(path):
                char = path[idx]
                if char == ".":
                    if current:
                        tokens.append(("key", current))
                        current = ""
                    idx += 1
                    continue
                if char == "[":
                    if current:
                        tokens.append(("key", current))
                        current = ""
                    end = path.index("]", idx)
                    content = path[idx + 1 : end]
                    if content == "*":
                        tokens.append(("wildcard", None))
                    elif content.startswith("'") and content.endswith("'"):
                        tokens.append(("key", content[1:-1]))
                    else:
                        tokens.append(("index", int(content)))
                    idx = end + 1
                    continue
                current += char
                idx += 1
            if current:
                tokens.append(("key", current))
            return tokens

        def find(self, data):
            values = [data]
            for kind, value in self.tokens:
                next_values = []
                for item in values:
                    if kind == "key" and isinstance(item, dict) and value in item:
                        next_values.append(item[value])
                    elif kind == "index" and isinstance(item, list):
                        if -len(item) <= value < len(item):
                            next_values.append(item[value])
                    elif kind == "wildcard":
                        if isinstance(item, list):
                            next_values.extend(item)
                        elif isinstance(item, dict):
                            next_values.extend(item.values())
                values = next_values
            return [SimpleNamespace(value=v) for v in values]

    def jsonpath_parse(path: str) -> _MiniJSONPath:
        return _MiniJSONPath(path)

from QueryLake.typing.toolchains import Mapping, NodeV2, ToolChainV2, ValueExpression, ValueRef
from QueryLake.runtime.jobs import JobRegistry, JobStatus


@dataclass
class ExecutionContext:
    state: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    files: Dict[str, Any]
    server: Dict[str, Any]


EmitCallback = Callable[[str, Dict[str, Any]], None]


class ToolchainSessionV2:
    """In-memory executor for a ToolChain v2 session."""

    def __init__(
        self,
        session_id: str,
        toolchain: ToolChainV2,
        *,
        author: str,
        server_context: Dict[str, Any],
        emit_event: Callable[[str, Dict[str, Any], Dict[str, Any]], None],
        job_registry: Optional[JobRegistry] = None,
    ) -> None:
        self.session_id = session_id
        self.toolchain = toolchain
        self.author = author
        self.server_context = server_context
        self._emit_event = emit_event
        self.job_registry = job_registry
        self._logger = logging.getLogger(__name__)

        self.state: Dict[str, Any] = deepcopy(toolchain.initial_state)
        self.files: Dict[str, Any] = {}
        self.local_cache: Dict[str, Any] = {}
        self.node_map: Dict[str, NodeV2] = {node.id: node for node in toolchain.nodes}
        self.node_inbox: Dict[str, Dict[str, Any]] = {}

    async def process_event(
        self,
        node_id: str,
        payload: Dict[str, Any],
        *,
        actor: str,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        queue: Deque[Tuple[str, Dict[str, Any]]] = deque([(node_id, payload)])
        user_payload: Dict[str, Any] = {}

        while queue:
            current_id, current_payload = queue.popleft()
            node = self.node_map.get(current_id)
            if node is None:
                continue

            context = ExecutionContext(
                state=self.state,
                inputs=deepcopy(current_payload),
                outputs={},
                files=self.files,
                server=self.server_context,
            )

            self._emit(
                "NODE_STARTED",
                {
                    "session_id": self.session_id,
                    "node_id": current_id,
                    "inputs": current_payload,
                    "actor": actor,
                    "correlation_id": correlation_id,
                },
                snapshot=False,
            )

            node_result = await (
                self._run_transform_node(node, context, actor, correlation_id)
                if node.type == "transform"
                else self._run_api_node(node, context, actor, correlation_id)
            )

            if node_result.get("user"):
                user_payload.update(node_result["user"])

            for target_node, arguments in node_result.get("next_nodes", []):
                merged = self.node_inbox.get(target_node, {})
                merged.update(arguments)
                self.node_inbox[target_node] = {}
                queue.append((target_node, merged))

            self._emit(
                "NODE_COMPLETED",
                {
                    "session_id": self.session_id,
                    "node_id": current_id,
                    "actor": actor,
                    "correlation_id": correlation_id,
                },
                snapshot=True,
            )

        return user_payload

    async def _run_api_node(
        self,
        node: NodeV2,
        context: ExecutionContext,
        actor: str,
        correlation_id: Optional[str],
    ) -> Dict[str, Any]:
        assert node.api_function, "API node missing function name"
        resolved_inputs = self._resolve_inputs(node, context)

        stream_callbacks, cleanup_streams = self._prepare_stream_callbacks(
            node, context, actor, correlation_id
        )
        if stream_callbacks:
            resolved_inputs.setdefault("stream_callables", stream_callbacks)

        umbrella = self.server_context.get("umbrella")
        call_target = umbrella.api_function_getter(node.api_function)
        track_job = bool(self.job_registry and self._should_track_job(node))
        job_id: Optional[str] = None
        job_signal = None
        signal_bus = self.server_context.get("job_signal_bus")

        if track_job:
            job_id = uuid.uuid4().hex
            if signal_bus is not None:
                job_signal = await signal_bus.create(job_id)
                if job_signal is not None and self.job_registry:
                    job_signal.configure(self.job_registry, self.session_id, node.id)
            signature = inspect.signature(call_target)
            if (
                job_signal is not None
                and "job_signal" in signature.parameters
                and "job_signal" not in resolved_inputs
            ):
                resolved_inputs = dict(resolved_inputs)
                resolved_inputs["job_signal"] = job_signal

        # Inject auth from server context if the target expects it and it's absent
        try:
            signature = inspect.signature(call_target)
            if "auth" in signature.parameters and "auth" not in resolved_inputs:
                auth_from_ctx = self.server_context.get("auth")
                if auth_from_ctx is not None:
                    resolved_inputs = dict(resolved_inputs)
                    resolved_inputs["auth"] = auth_from_ctx
        except Exception:
            pass

        if asyncio.iscoroutinefunction(call_target):
            task = asyncio.create_task(call_target(**resolved_inputs))
        else:
            task = asyncio.create_task(asyncio.to_thread(call_target, **resolved_inputs))

        if track_job and job_id is not None:
            async def cancel_callback() -> None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await self.job_registry.register(
                job_id,
                self.session_id,
                node.id,
                JobStatus.RUNNING,
                cancel_callback=cancel_callback,
            )
            self._logger.debug(
                "job.enqueued",
                extra={
                    "session_id": self.session_id,
                    "job_id": job_id,
                    "node_id": node.id,
                },
            )
            self._emit(
                "JOB_ENQUEUED",
                {
                    "session_id": self.session_id,
                    "node_id": node.id,
                    "job_id": job_id,
                    "actor": actor,
                    "correlation_id": correlation_id,
                },
                snapshot=False,
            )

        try:
            outputs = await task
            context.outputs = outputs if isinstance(outputs, dict) else {"result": outputs}

            result = self._apply_mappings(node, context, actor, correlation_id)

            for finalizer in cleanup_streams:
                await finalizer()

            if job_id and self.job_registry:
                await self.job_registry.update(
                    job_id,
                    self.session_id,
                    node.id,
                    JobStatus.COMPLETED,
                )
                self._logger.info(
                    "job.completed",
                    extra={
                        "session_id": self.session_id,
                        "job_id": job_id,
                        "node_id": node.id,
                    },
                )
                self._emit(
                    "JOB_COMPLETED",
                    {
                        "session_id": self.session_id,
                        "node_id": node.id,
                        "job_id": job_id,
                        "actor": actor,
                        "correlation_id": correlation_id,
                    },
                    snapshot=False,
                )

            return result
        except asyncio.CancelledError:
            for finalizer in cleanup_streams:
                await finalizer()
            if job_id and self.job_registry:
                await self.job_registry.update(
                    job_id,
                    self.session_id,
                    node.id,
                    JobStatus.CANCELLED,
                )
                self._logger.info(
                    "job.cancelled",
                    extra={
                        "session_id": self.session_id,
                        "job_id": job_id,
                        "node_id": node.id,
                    },
                )
                self._emit(
                    "JOB_CANCELLED",
                    {
                        "session_id": self.session_id,
                        "node_id": node.id,
                        "job_id": job_id,
                        "actor": actor,
                        "correlation_id": correlation_id,
                    },
                    snapshot=False,
                )
            raise
        except Exception as exc:
            for finalizer in cleanup_streams:
                await finalizer()
            if job_id and self.job_registry:
                await self.job_registry.update(
                    job_id,
                    self.session_id,
                    node.id,
                    JobStatus.FAILED,
                    result_meta={"error": str(exc)},
                )
                self._logger.error(
                    "job.failed",
                    extra={
                        "session_id": self.session_id,
                        "job_id": job_id,
                        "node_id": node.id,
                        "error": str(exc),
                    },
                )
                self._emit(
                    "JOB_FAILED",
                    {
                        "session_id": self.session_id,
                        "node_id": node.id,
                        "job_id": job_id,
                        "error": str(exc),
                        "actor": actor,
                        "correlation_id": correlation_id,
                    },
                    snapshot=False,
                )
            raise
        finally:
            if track_job and job_id and signal_bus is not None:
                await signal_bus.discard(job_id)

    async def _run_transform_node(
        self,
        node: NodeV2,
        context: ExecutionContext,
        actor: str,
        correlation_id: Optional[str],
    ) -> Dict[str, Any]:
        return self._apply_mappings(node, context, actor, correlation_id)

    def _resolve_inputs(self, node: NodeV2, context: ExecutionContext) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for key, expr in node.inputs.items():
            resolved[key] = self._evaluate_expression(expr, context)
        return resolved

    def _apply_mappings(
        self,
        node: NodeV2,
        context: ExecutionContext,
        actor: str,
        correlation_id: Optional[str],
    ) -> Dict[str, Any]:
        user_payload: Dict[str, Any] = {}
        next_nodes: List[Tuple[str, Dict[str, Any]]] = []

        for mapping in node.mappings:
            if mapping.iterate is not None:
                # Iterate over a collection and apply the mapping for each element.
                seq = self._resolve_ref(mapping.iterate, context)
                if seq is None:
                    continue
                if not isinstance(seq, list):
                    seq = [seq]
                # Allow mapping.value to reference the current loop element via server.loop
                had_prev_loop = "loop" in context.server
                prev_loop = context.server.get("loop")
                try:
                    for elem in seq:
                        context.server["loop"] = elem
                        value_i = self._evaluate_expression(mapping.value, context)
                        if mapping.destination.kind == "state":
                            self._apply_to_state(mapping.path, value_i, mapping.mode, actor, correlation_id)
                        elif mapping.destination.kind == "user":
                            self._apply_to_user(mapping.path, value_i, user_payload, actor, correlation_id)
                        elif mapping.destination.kind == "node":
                            next_nodes.append((mapping.destination.id, self._value_to_nested(mapping.path, value_i, mapping.mode)))
                        elif mapping.destination.kind == "files":
                            self._apply_to_files(mapping.path, value_i, mapping.mode, actor, correlation_id)
                finally:
                    if had_prev_loop:
                        context.server["loop"] = prev_loop
                    else:
                        context.server.pop("loop", None)
                continue

            if mapping.condition and not self._evaluate_condition(mapping.condition, context):
                continue

            value = self._evaluate_expression(mapping.value, context)

            if mapping.destination.kind == "state":
                self._apply_to_state(mapping.path, value, mapping.mode, actor, correlation_id)
            elif mapping.destination.kind == "user":
                self._apply_to_user(mapping.path, value, user_payload, actor, correlation_id)
            elif mapping.destination.kind == "node":
                node_inputs = next_nodes
                node_inputs.append((mapping.destination.id, self._value_to_nested(mapping.path, value, mapping.mode)))
            elif mapping.destination.kind == "files":
                self._apply_to_files(mapping.path, value, mapping.mode, actor, correlation_id)

        return {"user": user_payload, "next_nodes": next_nodes}

    def _apply_to_state(
        self,
        path: str,
        value: Any,
        mode: str,
        actor: str,
        correlation_id: Optional[str],
    ) -> None:
        before = deepcopy(self.state)
        self._assign(self.state, path, value, mode)
        patch = jsonpatch.JsonPatch.from_diff(before, self.state)
        if patch:
            self._emit(
                "STATE_PATCH_APPLIED",
                {
                    "session_id": self.session_id,
                    "path": path,
                    "patch": list(patch),
                    "actor": actor,
                    "correlation_id": correlation_id,
                },
                snapshot=False,
            )

    def _apply_to_files(self, path: str, value: Any, mode: str, actor: str, correlation_id: Optional[str]) -> None:
        self._assign(self.files, path, value, mode)
        self._emit(
            "FILES_UPDATED",
            {
                "session_id": self.session_id,
                "path": path,
                "mode": mode,
                "actor": actor,
                "correlation_id": correlation_id,
            },
            snapshot=False,
        )

    def _apply_to_user(self, path: str, value: Any, user_payload: Dict[str, Any], actor: str, correlation_id: Optional[str]) -> None:
        self._assign(user_payload, path, value, "merge")
        self._emit(
            "USER_DELIVERED",
            {
                "session_id": self.session_id,
                "path": path,
                "value": value,
                "actor": actor,
                "correlation_id": correlation_id,
            },
            snapshot=False,
        )

    def _assign(self, target: Dict[str, Any], path: str, value: Any, mode: str) -> None:
        route = self._jsonpath_to_route(path)
        if not route:
            return
        parent, key = self._navigate(target, route)
        existing = self._get_value(parent, key)

        if mode == "append":
            if existing is None:
                if isinstance(value, list):
                    self._set_value(parent, key, value)
                elif isinstance(value, str):
                    self._set_value(parent, key, value)
                else:
                    self._set_value(parent, key, [value])
                return
            if isinstance(existing, list):
                existing.append(value)
            elif isinstance(existing, str):
                self._set_value(parent, key, existing + str(value))
            else:
                raise TypeError("Append mode requires list or string target")
        elif mode == "merge":
            if isinstance(existing, dict) and isinstance(value, dict):
                existing.update(value)
            else:
                self._set_value(parent, key, value)
        else:
            self._set_value(parent, key, value)

    def _jsonpath_to_route(self, path: str) -> List[Any]:
        if not path or path == "$":
            return []
        expression = path[1:] if path.startswith("$") else path
        tokens: List[Any] = []
        buffer = ""
        idx = 0
        while idx < len(expression):
            char = expression[idx]
            if char == ".":
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                idx += 1
                continue
            if char == "[":
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                end = expression.index("]", idx)
                content = expression[idx + 1 : end]
                if content.startswith("'"):
                    tokens.append(content.strip("'"))
                else:
                    tokens.append(int(content))
                idx = end + 1
                continue
            buffer += char
            idx += 1
        if buffer:
            tokens.append(buffer)
        return [token for token in tokens if token != ""]

    def _evaluate_expression(self, expr: ValueExpression, context: ExecutionContext) -> Any:
        if expr.ref is not None:
            return self._resolve_ref(expr.ref, context)
        return self._resolve_literal(expr.literal, context)

    def _resolve_literal(self, literal: Any, context: ExecutionContext) -> Any:
        if isinstance(literal, dict):
            if set(literal.keys()) == {"ref"}:
                ref = ValueRef.model_validate(literal["ref"])
                return self._resolve_ref(ref, context)
            return {k: self._resolve_literal(v, context) for k, v in literal.items()}
        if isinstance(literal, list):
            return [self._resolve_literal(item, context) for item in literal]
        return literal

    def _resolve_ref(self, ref: ValueRef, context: ExecutionContext) -> Any:
        if ref.source == "const":
            return ref.value
        sources = {
            "state": context.state,
            "inputs": context.inputs,
            "outputs": context.outputs,
            "files": context.files,
            "server": context.server,
        }
        base = sources.get(ref.source)
        if base is None:
            return None
        path = ref.path or "$"
        finder = jsonpath_parse(path)
        matches = [match.value for match in finder.find(base)]
        if not matches:
            return None
        return matches[0] if len(matches) == 1 else matches

    def _prepare_stream_callbacks(
        self,
        node: NodeV2,
        context: ExecutionContext,
        actor: str,
        correlation_id: Optional[str],
    ) -> Tuple[Dict[str, Callable[[str], Any]], List[Callable[[], Any]]]:
        callbacks: Dict[str, Callable[[str], Any]] = {}
        cleanup: List[Callable[[], Any]] = []

        for mapping in node.mappings:
            if not mapping.stream.enabled:
                continue
            if mapping.destination.kind != "state":
                raise NotImplementedError("Streaming currently supported only for state destinations")
            if mapping.value.ref is None or mapping.value.ref.source != "outputs":
                raise ValueError("Streaming mappings must reference node outputs")
            stream_id = uuid.uuid4().hex
            output_key = self._top_level_key(mapping.value.ref.path)
            dest_path = mapping.path

            if mapping.stream.initial is not None:
                self._apply_to_state(dest_path, mapping.stream.initial, "set", actor, correlation_id)

            self._emit(
                "STREAM_OPEN",
                {
                    "session_id": self.session_id,
                    "node_id": node.id,
                    "stream_id": stream_id,
                    "actor": actor,
                    "correlation_id": correlation_id,
                    "path": dest_path,
                },
                snapshot=False,
            )

            async def on_chunk(chunk: str, *, _path=dest_path, _mode=mapping.stream.mode, _stream_id=stream_id):
                before = deepcopy(self.state)
                self._assign(self.state, _path, chunk if _mode == "set" else chunk, _mode)
                patch = jsonpatch.JsonPatch.from_diff(before, self.state)
                self._emit(
                    "STREAM_CHUNK",
                    {
                        "session_id": self.session_id,
                        "stream_id": _stream_id,
                        "chunk": chunk,
                        "path": _path,
                        "actor": actor,
                        "correlation_id": correlation_id,
                        "patch": list(patch),
                    },
                    snapshot=False,
                )

            async def finalize(*, _stream_id=stream_id):
                self._emit(
                    "STREAM_CLOSE",
                    {
                        "session_id": self.session_id,
                        "stream_id": _stream_id,
                        "actor": actor,
                        "correlation_id": correlation_id,
                    },
                    snapshot=False,
                )

            cleanup.append(finalize)

            async def wrapper(chunk: str, callback=on_chunk):
                await callback(chunk)

            callbacks[output_key] = wrapper

        return callbacks, cleanup

    def _top_level_key(self, path: Optional[str]) -> str:
        if not path or path == "$":
            return "result"
        route = self._jsonpath_to_route(path)
        return str(route[0]) if route else "result"

    def _navigate(self, obj: Any, route: List[Any]) -> Tuple[Any, Any]:
        current = obj
        for segment in route[:-1]:
            current = self._get_or_create(current, segment)
        return current, route[-1]

    def _get_or_create(self, obj: Any, key: Any, default: Any = None) -> Any:
        if isinstance(key, int):
            if not isinstance(obj, list):
                raise TypeError("Expected list during traversal")
            while len(obj) <= key:
                obj.append(deepcopy(default) if default is not None else {})
            return obj[key]

        if isinstance(obj, list):
            raise TypeError("Cannot access dict key on list")

        if key not in obj:
            obj[key] = deepcopy(default) if default is not None else {}
        return obj[key]

    def _set_value(self, obj: Any, key: Any, value: Any) -> None:
        if isinstance(key, int):
            if not isinstance(obj, list):
                raise TypeError("Expected list when assigning by index")
            while len(obj) <= key:
                obj.append(None)
            obj[key] = value
        else:
            if isinstance(obj, list):
                raise TypeError("Cannot assign dict key on list")
            obj[key] = value

    def _get_value(self, obj: Any, key: Any) -> Any:
        if isinstance(key, int):
            if not isinstance(obj, list) or key >= len(obj):
                return None
            return obj[key]
        if isinstance(obj, list):
            return None
        return obj.get(key)

    def _value_to_nested(self, path: str, value: Any, mode: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        self._assign(result, path, value, mode)
        return result

    def _emit(self, kind: str, payload: Dict[str, Any], *, snapshot: bool) -> None:
        metadata = {"snapshot": snapshot}
        if snapshot:
            metadata["state"] = self.state
            metadata["files"] = self.files
        self._emit_event(kind, payload, metadata)

    def serialize(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "files": self.files,
            "local_cache": self.local_cache,
        }

    # ------------------------------------------------------------------
    # JSONLogic (subset) evaluation helpers

    def _evaluate_condition(self, condition: Dict[str, Any], context: ExecutionContext) -> bool:
        data = {
            "state": context.state,
            "inputs": context.inputs,
            "outputs": context.outputs,
            "files": context.files,
        }
        return bool(self._jsonlogic(condition, data))

    def _should_track_job(self, node: NodeV2) -> bool:
        if not node.api_function:
            return False
        tracked = {
            "llm",
            "llm_multistep_search",
            "process_pdf_with_surya",
            "process_pdf_with_surya_2",
            "self_guided_search",
            "multi_search",
        }
        return node.api_function in tracked or node.metadata.get("track_job", False)

    def _jsonlogic(self, expr: Any, data: Dict[str, Any]) -> Any:
        if isinstance(expr, dict):
            if len(expr) != 1:
                raise ValueError("Invalid JSONLogic expression")
            op, value = next(iter(expr.items()))
            return self._apply_logic_operator(op, value, data)
        if isinstance(expr, list):
            return [self._jsonlogic(item, data) for item in expr]
        return expr

    def _apply_logic_operator(self, op: str, value: Any, data: Dict[str, Any]) -> Any:
        if op == "var":
            path = value if isinstance(value, str) else value[0]
            return self._lookup_var(path, data)
        if op in {"==", "!="}:  # comparison
            left, right = [self._jsonlogic(arg, data) for arg in value]
            return left == right if op == "==" else left != right
        if op in {">", "<", ">=", "<="}:
            left, right = [self._jsonlogic(arg, data) for arg in value]
            return eval(f"left {op} right")
        if op == "+":
            return sum(self._jsonlogic(arg, data) for arg in value)
        if op == "-":
            left, right = [self._jsonlogic(arg, data) for arg in value]
            return left - right
        if op == "!":
            return not self._jsonlogic(value, data)
        if op == "and":
            for arg in value:
                if not self._jsonlogic(arg, data):
                    return False
            return True
        if op == "or":
            for arg in value:
                if self._jsonlogic(arg, data):
                    return True
            return False
        raise NotImplementedError(f"Unsupported JSONLogic operator: {op}")

    def _lookup_var(self, path: str, data: Dict[str, Any]) -> Any:
        if path.startswith("$"):
            expression = jsonpath_parse(path)
            matches = [m.value for m in expression.find(data)]
            return matches[0] if matches else None
        parts = path.split('.')
        current: Any = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
