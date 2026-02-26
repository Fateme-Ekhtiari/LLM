
class Translator:
    """Business logic for translating using LLM."""
    def __init__(self, llm_client):
        self.client = llm_client
