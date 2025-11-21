# 2B-8B  bendi yunxing    yu xunlian
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import time

app = FastAPI(title="Gemma 3 4B API", description="使用 Ollama 运行的 Gemma 3 4B 模型")

OLLAMA_URL = "http://localhost:11434"


class ChatRequest(BaseModel):
    message: str
    model: str = "gemma3:4b"  # 改为 gemma3:4b


class ChatResponse(BaseModel):
    response: str
    model: str
    time_taken: float


def check_ollama():
    """检查 Ollama 服务状态"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


@app.get("/")
def home():
    return {"message": "Gemma 3 4B API", "status": "ready"}


@app.get("/models")
def list_models():
    """查看已安装的模型"""
    if not check_ollama():
        raise HTTPException(503, "Ollama 服务未运行")

    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        return response.json()
    except:
        raise HTTPException(500, "无法获取模型列表")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """与 Gemma 3 对话"""
    if not check_ollama():
        raise HTTPException(503, "请先启动 Ollama 服务")

    data = {
        "model": request.model,
        "prompt": request.message,
        "stream": False
    }

    start_time = time.time()

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=data,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            time_taken = time.time() - start_time

            return ChatResponse(
                response=result.get("response", "无响应"),
                model=request.model,
                time_taken=round(time_taken, 2)
            )
        else:
            raise HTTPException(500, f"模型响应错误: {response.text}")

    except requests.exceptions.Timeout:
        raise HTTPException(408, "请求超时，请稍后重试")
    except Exception as e:
        raise HTTPException(500, f"生成响应失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)