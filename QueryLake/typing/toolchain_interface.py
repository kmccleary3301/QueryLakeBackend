from typing import List, Union, Optional, Literal
from pydantic import BaseModel, Field

# class BooleanConfigEntryFieldType(BaseModel):
#     name: str
#     type: str = Field("boolean", const=True)
#     default: bool

# class StringConfigEntryFieldType(BaseModel):
#     name: str
#     type: str = Field("string", const=True)
#     default: str

# class NumberConfigEntryFieldType(BaseModel):
#     name: str
#     type: str = Field("number", const=True)
#     default: int

# ConfigEntryFieldType = Union[BooleanConfigEntryFieldType, StringConfigEntryFieldType, NumberConfigEntryFieldType]





# Custom Interface Components
DISPLAY_COMPONENTS = ["chat", "markdown", "text", "graph"]
INPUT_COMPONENTS = ["file_upload", "chat_input"]

DisplayComponents = Literal["chat", "markdown", "text", "graph"]
InputComponenets = Literal["file_upload", "chat_input"]



class DisplayMapping(BaseModel):
    display_route: List[Union[str, int]]
    display_as: DisplayComponents

class InputEvent(BaseModel):
    hook: str
    target_event: str
    store: bool
    target_route: str

class ConfigEntry(BaseModel):
    name: str
    value: Union[str, int, bool]


# TODO: Combine these three into one. Should be easy.
class InputMappingProto(BaseModel):
    hooks: List[InputEvent]
    config: List[ConfigEntry]
    tailwind: Optional[str] = ""

class FileUploadMapping(InputMappingProto):
    display_as: str = "file_upload"

class ChatInputMapping(InputMappingProto):
    display_as: str = "chat_input"

InputMapping = Union[FileUploadMapping, ChatInputMapping]
ContentMapping = Union[DisplayMapping, InputMapping]





# Standard types.

class ContentSection(BaseModel):
    split: str = "none"
    align: str
    tailwind: Optional[str] = ""
    mappings: List[ContentMapping]

class DivisionSection(BaseModel):
    split: str
    sections: List[ContentSection]

class HeaderSection(BaseModel):
    align: str
    tailwind: Optional[str] = ""
    mappings: List[ContentMapping]

DisplaySection = Union[ContentSection, DivisionSection]