from fastapi import APIRouter
from ..schemas.arabert_schemas import PredictRequest, PredictResponse

from ..models.arabert_model import AraBertModel

arabert_model= AraBertModel()

router= APIRouter(
    prefix="/api/v1/sentiment-analysis",
    tags=["Arabic Sentiment Analysis"],
)


@router.post("/predict", response_model=PredictResponse)
def predict_sentiment_analysis(data: PredictRequest, arabert_model=arabert_model):

    text= data.comment
    model_prediction= arabert_model.predict(text=text)

    return PredictResponse(comment_status=model_prediction)
