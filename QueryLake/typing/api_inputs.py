from typing import List, Dict, Optional, Union, Tuple, Literal, Annotated, Any
from pydantic import BaseModel

class TextChunks(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class DocumentModifierArgs(BaseModel):
    document_id: str
    text: Optional[Union[str, List[TextChunks]]] = None
    metadata: Optional[Dict[str, Any]] = None
    scan: Optional[bool] = False