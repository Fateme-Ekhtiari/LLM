
class Summarizer:
    """Business logic for text summarization using LLM."""
    def __init__(self, llm_client=None):
        self.client = llm_client