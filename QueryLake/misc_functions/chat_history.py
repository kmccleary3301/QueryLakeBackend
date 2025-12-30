from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel

from QueryLake.misc_functions.server_class_functions import (
    construct_functions_available_prompt,
)
from QueryLake.typing.function_calling import FunctionCallDefinition


def format_chat_history(
    chat_history: List[dict],
    sources: List[dict] | None = None,
    functions_available: List[Union[FunctionCallDefinition, dict]] | None = None,
) -> List[dict]:
    sources = sources or []

    for i in range(len(sources)):
        if isinstance(sources[i], BaseModel):
            sources[i] = sources[i].model_dump()

    # Turn chat history entry content fields into lists.
    chat_history = [
        entry
        if not isinstance(entry.get("content"), str)
        else {
            **entry,
            "content": [
                {"type": "text", "text": entry["content"]},
            ],
        }
        for entry in chat_history
    ]

    if len(sources) > 0:
        new_entry = {
            "type": "text",
            "text": (
                "SYSTEM MESSAGE - PROVIDED SOURCES\n"
                "Cite these sources in your response with the following notation "
                "for inline citations: {cite:source_number} (i.e. {cite:3})\n"
                "<SOURCES>\n"
                + "\n\n".join(
                    [
                        f"[{i + 1}] Source {i + 1}\n\n{source['text']}"
                        for i, source in enumerate(sources)
                    ]
                )
                + "\n</SOURCES>\nEND SYSTEM MESSAGE\n"
            ),
        }
        chat_history[-1]["content"] = [new_entry] + chat_history[-1]["content"]

    if functions_available is not None:
        new_entry = {
            "type": "text",
            "text": (
                "SYSTEM MESSAGE - AVAILABLE FUNCTIONS\n"
                f"<FUNCTIONS>{construct_functions_available_prompt(functions_available)}\n"
                "</FUNCTIONS>\nEND SYSTEM MESSAGE\n\n"
            ),
        }
        chat_history[-1]["content"] = [new_entry] + chat_history[-1]["content"]

    stripped_chat_history = [
        {
            **entry,
            "content": "\n".join(
                [
                    part["text"]
                    for part in entry["content"]
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
            ),
        }
        for entry in chat_history
    ]

    # If it's all text, just return the stripped chat history.
    if all(
        part.get("type") == "text"
        for entry in chat_history
        for part in (entry.get("content") or [])
        if isinstance(part, dict)
    ):
        chat_history = stripped_chat_history

    return chat_history

