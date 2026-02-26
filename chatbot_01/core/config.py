# ai_engine/core/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env

class Settings:
    def __init__(self):
        self.API_KEY = os.getenv("API_KEY")
        self.BASE_URL = os.getenv("BASE_URL")
        self.MODEL_NAME = os.getenv("MODEL_NAME")

        if not self.API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables")

settings = Settings()
