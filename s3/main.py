from langchain_community.llms import Ollama
import time


class SimpleQAAgent:
    def __init__(self, model_name="gemma3:4b"):
        # 使用新的导入方式避免警告
        try:
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(model=model_name)
        except ImportError:
            # 如果新包不可用，回退到旧方法
            self.llm = Ollama(model=model_name)

    def ask_question(self, question):
        """Process user question"""
        print(f"User: {question}")

        start_time = time.time()
        try:
            response = self.llm.invoke(question)
            end_time = time.time()

            print(f"Assistant: {response}")
            print(f"Response time: {end_time - start_time:.2f} seconds")
            return response

        except Exception as e:
            print(f"Error: {e}")
            return "Sorry, I cannot answer this question right now."


def main():
    agent = SimpleQAAgent("gemma3:4b")

    print("AI Assistant started! Type 'exit' to end the conversation")

    while True:
        user_input = input("\nEnter your question: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if user_input:
            agent.ask_question(user_input)


if __name__ == "__main__":
    main()