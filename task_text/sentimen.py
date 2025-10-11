# task_text/sentiment.py
from transformers import pipeline

def main():
    # 创建一个情感分析 pipeline（会自动下载模型）
    nlp = pipeline("sentiment-analysis")
    samples = [
        "I love this course! It's very helpful.",
        "This is the worst experience I've had."
    ]
    for s in samples:
        print(s)
        print(nlp(s))
        print()

if __name__ == "__main__":
    main()