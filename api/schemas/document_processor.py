from pydantic import BaseModel, HttpUrl

class URLUploadRequest(BaseModel):
    url: HttpUrl

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str