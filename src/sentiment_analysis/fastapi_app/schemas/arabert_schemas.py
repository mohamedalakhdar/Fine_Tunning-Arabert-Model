from pydantic import BaseModel

class PredictRequest(BaseModel):
    comment: str


class PredictResponse(BaseModel):
    comment_status: str 
