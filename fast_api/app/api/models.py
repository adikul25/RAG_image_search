from pydantic import BaseModel
import json

class ImageDescription(BaseModel):
    descriptions: list


class Query(BaseModel):
    query: str

