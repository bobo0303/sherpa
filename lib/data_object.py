from pydantic import BaseModel

# Request body model for loading a model
class LoadModelRequest(BaseModel):
    language: str

