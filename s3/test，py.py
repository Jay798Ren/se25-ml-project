import ollama
import time


def test_model_intelligence(model_name):
    """Test model intelligence"""
    test_questions = [
        "Explain artificial intelligence in one sentence.",
        "If I have 3 apples, eat 1, and then buy 5 more, how many do I have now?",
        "What is the capital of France?",
        "Create a short poem about the moon"
    ]

    print(f"\n=== Testing Model: {model_name} ===")

    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")

        try:
            start_time = time.time()
            response = ollama.chat(model=model_name, messages=[
                {'role': 'user', 'content': question}
            ])
            end_time = time.time()

            answer = response['message']['content']
            response_time = end_time - start_time

            print(f"Answer: {answer}")
            print(f"Response time: {response_time:.2f} seconds")

        except Exception as e:
            print(f"Error: {e}")


def test_hallucination(model_name):
    """Test hallucination tendency"""
    fake_question = "Please provide detailed information about Zhang San, the 2025 Nobel Prize winner"

    print(f"\n=== Testing Hallucination: {model_name} ===")
    print(f"Question: {fake_question}")

    try:
        response = ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': fake_question}
        ])
        answer = response['message']['content']
        print(f"Answer: {answer}")

        # Simple check for hallucination
        if "don't know" in answer.lower() or "not exist" in answer.lower() or "not announced" in answer.lower() or "cannot confirm" in answer.lower():
            print("✅ Model recognized false information")
        else:
            print("⚠️ Model may be hallucinating")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Use your actual model names
    models = ["gemma3:4b", "gemma3:1b", "qwen3-vl:4b"]

    for model in models:
        test_model_intelligence(model)
        test_hallucination(model)
        print("\n" + "=" * 50)