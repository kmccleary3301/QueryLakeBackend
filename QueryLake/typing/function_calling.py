from typing import List, Optional
from pydantic import BaseModel

class FunctionCallArgumentDefinition(BaseModel):
    name: str
    type: Optional[str] = None
    default: Optional[str] = None
    description: Optional[str] = None
    
class FunctionCallDefinition(BaseModel):
    name: str
    description: str
    parameters: List[FunctionCallArgumentDefinition]