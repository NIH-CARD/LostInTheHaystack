import os
from typing import ClassVar, List, Optional

from pydantic import Field, model_validator
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

from scripts.models.base_llm import BaseClientConfig, BaseGenerationConfig, BaseLLM

class AzureAIClientConfig(BaseClientConfig):
    """
    Contains and validates data needed to configure the azure.ai.inference ChatCompletionsClient.
    """
    provider: str = "azureai"
    api_key: str
    azure_endpoint: str

    @model_validator(mode="before")
    @classmethod
    def load_from_env(cls, values: dict) -> dict:
        """
        Before validation checks for these variables.
        If not passed directly, attempts to pull from environment.
        """
        required_envs = {
            "api_key": "AZURE_AI_KEY",
            "azure_endpoint": "AZURE_AI_ENDPOINT",
        }
        for field_name, env_var in required_envs.items():
            if not values.get(field_name):
                env_value = os.environ.get(env_var)
                if not env_value:
                    raise ValueError(f"Missing env var: {env_var}")
                values[field_name] = env_value
        return values

class AzureAIGenerationConfig(BaseGenerationConfig):
    """
    Contains and validates data needed to generate text from the AzureOpenAI client.
    """
    provider: str = "azureai"
    model: str

    def to_azureai_config(self) -> dict:
        """
        Returns a dict to be passed to the (Async)AzureOpenAI.chat.completions.create() method.
        """
        gen_cfg = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tools": None,
        }
        return gen_cfg

class AzureAILLM(BaseLLM):
    """
    Inherits BaseLLM and interfaces azure.ai.inference ChatCompletionsClient.
    """
    provider: str = "azureai"

    def __init__(self, config: AzureAIClientConfig):
        """
        Initialization creates sync and async client objects.
        """
        self.client = self._initialize_client(config)
    
    def _initialize_client(self, config: AzureAIClientConfig) -> Optional[ChatCompletionsClient]:
        """
        Returns azure.ai.inference ChatCompletionsClient object.
        Returns None on failure.
        """
        try:
            client = ChatCompletionsClient(endpoint=config.azure_endpoint, credential=AzureKeyCredential(config.api_key))
            return client
        except Exception as e:
            print(f"Error initializing azure.ai.inference ChatCompletionsClient: {e}")
            return None

    def generate(self, prompt: str, config: AzureAIGenerationConfig) -> str:
        """
        Text generation by querying the client using the prompt provided, and generation config.
        """
        model = config.model
        gen_cfg = config.to_azureai_config()

        response = self.client.complete(
            model=config.model,
            messages=[UserMessage(content=prompt)],
            **gen_cfg
        )
        return response.choices[0].message.content

    async def generate_async(self, prompt: str, config: AzureAIGenerationConfig) -> str:
        """
        The inference SDK doesn't have an async client.
        """
        raise NotImplementedError("Azure AI inference client is sync only")