from infra.llm.client import BaseLLMClient, ModelTimeoutError
import ollama
import time



class OllamaClient(BaseLLMClient):
    def __init__(self, model_name="llama3.2", system_prompt="You are a helpful assistant."):
        
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]


    def predict(self, prompt: str, retries=3, delay=2) -> str:
        self.messages.append({"role": "user", "content": prompt})
        for attempt in range(1, retries + 1):
            try:
                response = ollama.chat(model=self.model_name, messages=self.messages)
                answer = response['message']['content']
                self.messages.append({"role": "assistant", "content": answer})
                return answer
            except Exception as e:
                print(f"[OllamaClient] Attempt {attempt} failed: {e}")
                if attempt == retries:
                    raise ModelTimeoutError("Ollama request failed after retries")
                time.sleep(delay)
