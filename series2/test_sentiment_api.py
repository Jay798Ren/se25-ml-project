"""
Tests for Sentiment Analysis API
"""

import pytest
from fastapi.testclient import TestClient
from sentiment_api import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data

def test_health_check():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_sentiment_analysis_positive():
    """Test positive sentiment analysis"""
    test_text = "I love this product! It's amazing!"
    response = client.post("/analyze", json={"text": test_text})

    assert response.status_code == 200
    data = response.json()

    assert data["text"] == test_text
    assert "sentiment" in data
    assert "confidence" in data
    assert data["confidence"] > 0.5  # Confidence should be reasonable

def test_sentiment_analysis_negative():
    """Test negative sentiment analysis"""
    test_text = "This is terrible and I hate it."
    response = client.post("/analyze", json={"text": test_text})

    assert response.status_code == 200
    data = response.json()

    assert data["text"] == test_text
    assert "sentiment" in data
    assert "confidence" in data

def test_empty_text():
    """Test empty text"""
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 400

def test_whitespace_text():
    """Test whitespace-only text"""
    response = client.post("/analyze", json={"text": "   "})
    assert response.status_code == 400

def test_chinese_text():
    """Test Chinese text sentiment analysis"""
    test_text = "This restaurant has delicious food and great service!"
    response = client.post("/analyze", json={"text": test_text})

    assert response.status_code == 200
    data = response.json()

    assert data["text"] == test_text
    assert "sentiment" in data
    assert "confidence" in data