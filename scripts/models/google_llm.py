import os
import asyncio
from typing import Literal, Optional, List

from pydantic import BaseModel, model_validator
from google import genai

from scripts.models.base_llm import BaseGenerationConfig, BaseLLM

class GoogleClientConfig(BaseModel):
    """
    Contains and validates data needed to configure the genai client.
    """
    provider: str = "google"
    api_key: str

    @model_validator(mode="before")
    @classmethod
    def load_from_env(cls, values: dict) -> dict:
        """
        Before validation checks for these variables.
        If not passed directly, attempts to pull from environment.
        """
        if not values.get("api_key"):
            env_key = os.environ.get("GOOGLE_API_KEY")
            if not env_key:
                raise ValueError("Missing env var GOOGLE_API_KEY!")
            values["api_key"] = env_key
        return values

class GoogleGenerationConfig(BaseGenerationConfig):
    """
    Contains and validates data needed to generate text from the genai client.
    """
    provider: str = "google"
    model: Literal["gemini-2.0-flash", "gemini-2.0-flash-lite"]
    tools: List[dict] = []

    def to_genai_config(self) -> genai.types.GenerateContentConfig:
        """
        Returns a genai.types.GenerateContentConfig to be passed to the genai.(aio.)generate_content() method.
        Hardcodes all safety_settings to be "OFF" and response_modality to be "TEXT".
        """
        generate_content_config = genai.types.GenerateContentConfig(
            temperature = self.temperature,
            max_output_tokens = self.max_tokens,
            response_modalities = ["TEXT"],
            safety_settings = [
                genai.types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                genai.types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                genai.types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                genai.types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",threshold="OFF")
            ],
            tools=[genai.types.Tool(function_declarations=[genai.types.FunctionDeclaration(**tool) for tool in self.tools])] if self.tools else []
        )
        return generate_content_config

class GoogleLLM(BaseLLM):
    """
    Inherits BaseLLM and interfaces genai and genai.aio.
    """
    provider: str = "google"

    def __init__(self, config: GoogleClientConfig) -> None:
        """
        Initialization creates a client object per config.
        """
        self.client = self._initialize_client(config)

    def _initialize_client(self, config: GoogleClientConfig) -> Optional[genai.Client]:
        """
        Returns genai.Client object.
        Returns None on failure.
        """
        try:
            client = genai.Client(api_key=config.api_key)
            return client
        except Exception as e:
            print(f"Error initializing genai client: {e}")
            return None
        
    def generate(self, prompt: str, config: GoogleGenerationConfig) -> str:
        """
        Text generation by querying the client using the prompt provided, and generation config.
        """
        model = config.model
        gen_cfg = config.to_genai_config()

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=gen_cfg
        )

        return response.candidates[0].content.parts[0].text
    
    async def generate_async(self, prompt: str, config: GoogleGenerationConfig) -> str:
        """
        Asynchronous text generation by querying the client.aio using the prompt provided, and generation config.
        """
        model = config.model
        gen_cfg = config.to_genai_config()

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=gen_cfg
        )

        return response.candidates[0].content.parts[0].text