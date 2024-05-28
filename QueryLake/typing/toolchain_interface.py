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

# DisplayComponents = Literal["chat", "markdown", "text", "graph"]
DisplayComponents = str
# InputComponenets = Literal["file_upload", "chat_input"]
InputComponenents = str


class DisplayMapping(BaseModel):
    display_route: List[Union[int, str]] # The order here matters, otherwise ints will become string representations.
    display_as: DisplayComponents

class InputEvent(BaseModel):
    hook: str
    target_event: str
    fire_index: int
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

# InputMapping = Union[FileUploadMapping, ChatInputMapping]

class InputMapping(BaseModel):
    display_as: str
    hooks: List[InputEvent]
    config: List[ConfigEntry]
    tailwind: Optional[str] = ""

ContentMapping = Union[DisplayMapping, InputMapping]





# Standard types.



# export type contentDiv = {
#   type: "div",
#   align: alignType,
#   tailwind: string,
#   mappings: (contentMapping | contentDiv)[]
# }

class ContentDiv(BaseModel):
    type: str = "div"
    align: str
    tailwind: Optional[str] = ""
    mappings: List[Union[ContentMapping, "ContentDiv"]]

class HeaderSection(BaseModel):
    align: str
    tailwind: Optional[str] = ""
    mappings: List[Union[ContentMapping, ContentDiv]]

class ContentSection(BaseModel):
    split: str = "none"
    size: float
    align: str
    tailwind: Optional[str] = ""
    mappings: List[Union[ContentMapping, ContentDiv]]
    header: Optional[HeaderSection] = None
    footer: Optional[HeaderSection] = None

class DivisionSection(BaseModel):
    split: str
    size: float
    sections: List[ContentSection]
    header: Optional[HeaderSection] = None
    footer: Optional[HeaderSection] = None

ContentDiv.model_rebuild()

DisplaySection = Union[ContentSection, DivisionSection]