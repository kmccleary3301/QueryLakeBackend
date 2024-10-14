from typing import List, Dict, Optional, Union, Tuple, Literal, Annotated, Any
from pydantic import BaseModel


class DocumentModifierArgs(BaseModel):
    document_id: str
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    scan: Optional[bool] = False