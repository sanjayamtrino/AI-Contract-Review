from pydantic import BaseModel, Field
from typing import List, Union

class ParseResult(BaseModel):
    """Schema for the result of a parsing operation in the registry service."""

    success: bool = Field(..., description="Indicates if the parsing was successful") 
    chunks: List[dict] = Field(default_factory=list, description="List of parsed data chunks") 
    metadata: dict = Field(default_factory=dict, description="Additional metadata about the parsing operation(if any)") 
    error_message: Union[str, None] = Field(None, description="Error message if the parsing has any failed operations.") 
    processing_time: float = Field(..., description="Time taken to process the parsing operation in seconds.") 