import os
import asyncio
from typing import ClassVar, List, Optional

from pydantic import model_validator, field_validator
from openai import AzureOpenAI, AsyncAzureOpenAI

from scripts.models.base_llm import BaseClientConfig, BaseGenerationConfig, BaseLLM

class AzureOAIClientConfig(BaseClientConfig):
    """
    Contains and validates data needed to configure the AzureOpenAI client.
    """
    provider: str = "azureoai"
    api_key: str
    api_version: str
    azure_endpoint: str

    @model_validator(mode="before")
    @classmethod
    def load_from_env(cls, values: dict) -> dict:
        """
        Before validation checks for these variables.
        If not passed directly, attempts to pull from environment.
        """
        required_envs = {
            "api_key": "AZURE_OPENAI_KEY",
            "api_version": "AZURE_OPENAI_API_VERSION",
            "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
        }
        for field_name, env_var in required_envs.items():
            if not values.get(field_name):
                env_value = os.environ.get(env_var)
                if not env_value:
                    raise ValueError(f"Missing env var: {env_var}")
                values[field_name] = env_value
        return values

class AzureOAIGenerationConfig(BaseGenerationConfig):
    """
    Contains and validates data needed to generate text from the AzureOpenAI client.
    """
    _AZURE_MODEL_MAP: ClassVar[dict[str, str]] = {
        "gpt-4o": "card-ai-gpt-4o20241212",
        "gpt-4o-mini": "card-ai-gpt-4o-turbo20250114",
    }
    provider: str = "azureoai"
    model: str
    tools: List[dict] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in cls._AZURE_MODEL_MAP:
            raise ValueError(f"Invalid model '{v}'. Must be one of: {list(cls._AZURE_MODEL_MAP.keys())}")
        return v

    def to_azureoai_dict(self) -> dict:
        """
        Returns a dict to be passed to the (Async)AzureOpenAI.chat.completions.create() method.
        """
        gen_cfg = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tools": self.tools
        }
        return gen_cfg
    
    def get_azure_model(self) -> str:
        """
        Maps plain text model names to our Azure deployed version. 
        """
        return self._AZURE_MODEL_MAP[self.model]
    
class AzureOAILLM(BaseLLM):
    """
    Inherits BaseLLM and interfaces AzureOpenAI and AsyncAzureOpenAI.
    """
    provider: str = "azureoai"

    def __init__(self, config: AzureOAIClientConfig) -> None:
        """
        Initialization creates sync and async client objects.
        """
        self.sync_client = self._initialize_sync_client(config)
        self.async_client = self._initialize_async_client(config)
    
    def _initialize_sync_client(self, config: AzureOAIClientConfig) -> Optional[AzureOpenAI]:
        """
        Returns AzureOpenAI object.
        Returns None on failure.
        """
        try:
            client = AzureOpenAI(api_key=config.api_key, api_version=config.api_version, azure_endpoint=config.azure_endpoint)
            return client
        except Exception as e:
            print(f"Error initializing AzureOpenAI synchronous client: {e}.")
            return None

    def _initialize_async_client(self, config: AzureOAIClientConfig) -> Optional[AsyncAzureOpenAI]:
        """
        Returns AsyncAzureOpenAI.
        Returns None on failure.
        """
        try:
            client = AsyncAzureOpenAI(api_key=config.api_key, api_version=config.api_version, azure_endpoint=config.azure_endpoint)
            return client
        except Exception as e:
            print(f"Error initializing AzureOpenAI asynchronous client: {e}.")
            return None

    def generate(self, prompt: str, config: AzureOAIGenerationConfig) -> str:
        """
        Text generation by querying the sync client using the prompt provided, and generation config.
        """
        model = config.get_azure_model()
        gen_cfg = config.to_azureoai_dict()

        response = self.sync_client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content":prompt}],
            **gen_cfg
        )

        return response.choices[0].message.content
    
    async def generate_async(self, prompt: str, config: AzureOAIGenerationConfig) -> str:
        """
        Asyncrhonous text generation by querying the async client using the prompt provided, and generation config.
        """
        model = config.get_azure_model()
        gen_cfg = config.to_azureoai_dict()

        response = await self.async_client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content":prompt}],
            **gen_cfg
        )

        return response.choices[0].message.content