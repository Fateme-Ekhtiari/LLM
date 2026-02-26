
class Poet:
    """Business logic for text summarization using LLM."""
    def __init__(self, llm_client):
        self.client = llm_client
