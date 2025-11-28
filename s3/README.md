# LLM Agent Project

## Project Description
This project implements an intelligent Q&A assistant using lightweight Large Language Models (LLMs). The system tests multiple LLMs for performance and selects the optimal model for the final agent implementation.

## Project Structure



## Models Tested
- **gemma3:4b** (3.3GB) - Primary selected model
- **gemma3:1b** (815MB) - Lightweight alternative
- **qwen3-vl:4b** (3.3GB) - Multimodal capability model

## Installation & Setup

### Prerequisites
- Python 3.8+
- Ollama installed locally
- Required models pulled via Ollama

### Installation Steps
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt