from fastapi import FastAPI
from .routes import sentiment_analysis

app= FastAPI()
app.include_router(sentiment_analysis.router)


@app.get("/health")
def health_check():
    return{
        "status": "healthy"
    }

