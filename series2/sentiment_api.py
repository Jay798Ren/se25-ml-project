"""
creat api for one api from task series 1
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
nlp = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global nlp
    try:
        logger.info("Loading sentiment analysis model...")
        nlp = pipeline("sentiment-analysis")
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e
    yield
    # Clean up resources on shutdown (if needed)
    # pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Sentiment Analysis API",
    description="Sentiment Analysis Service using Hugging Face - Academic Project",
    version="1.0.0",
    lifespan=lifespan
)

# Define request data model
class TextRequest(BaseModel):
    text: str

# Define response data model
class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    model_used: str = "Hugging Face Sentiment Analysis"

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: TextRequest):
    """
    Analyze text sentiment

    - **text**: Text to analyze for sentiment
    """
    if nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet, please try again later")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty")

    try:
        # Analyze sentiment
        result = nlp(request.text)[0]

        logger.info(f"Analyzed text: '{request.text}' -> {result['label']} (confidence: {result['score']:.4f})")

        return SentimentResponse(
            text=request.text,
            sentiment=result['label'],
            confidence=result['score']
        )
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sentiment Analysis API Service is running",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze - Analyze text sentiment",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if nlp is not None else "loading"
    return {
        "status": status,
        "model_loaded": nlp is not None
    }

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Sentiment Analysis API Service...")
    print("Access API documentation at:")
    print("http://localhost:8000/docs")
    print("http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to stop the service")
    print("=" * 50)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )