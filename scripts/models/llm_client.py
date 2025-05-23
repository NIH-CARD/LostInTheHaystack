import asyncio
from typing import Literal

from pydantic import BaseModel

from scripts.models.base_llm import BaseLLM, BaseGenerationConfig as GenCfg
from scripts.models.azureai_llm import AzureAILLM, AzureAIClientConfig, AzureAIGenerationConfig
from scripts.models.azureoai_llm import AzureOAILLM, AzureOAIClientConfig, AzureOAIGenerationConfig
from scripts.models.google_llm import GoogleLLM, GoogleClientConfig, GoogleGenerationConfig
from scripts.models.hf_llm import HuggingFaceLLM, HuggingFaceClientConfig, HuggingFaceGenerationConfig

class LLMConfig(BaseModel):
    """
    Unified LLM configuration class.
    """
    provider: Literal["azureai", "azureoai", "google", "huggingface"]
    client_params: dict
    generation_params: dict

    def to_client_config(self):
        """
        Returns the {Provider}ClientConfig based on the provider and client params.
        """
        if self.provider == "azureai":
            return AzureAIClientConfig(**self.client_params)
        elif self.provider == "azureoai":
            return AzureOAIClientConfig(**self.client_params)
        elif self.provider == "google":
            return GoogleClientConfig(**self.client_params)
        elif self.provider == "huggingface":
            return HuggingFaceClientConfig(**self.client_params)

    def to_generation_config(self):
        """
        Returns the {Provider}GenerationConfig based on the provider and generation params.
        """
        if self.provider == "azureai":
            return AzureAIGenerationConfig(**self.generation_params)
        elif self.provider == "azureoai":
            return AzureOAIGenerationConfig(**self.generation_params)
        elif self.provider == "google":
            return GoogleGenerationConfig(**self.generation_params)
        elif self.provider == "huggingface":
            return HuggingFaceGenerationConfig(**self.generation_params)

class LLMFactory:
    """
    Unified LLM factory class.
    """
    @staticmethod
    def create_llm(config: LLMConfig) -> BaseLLM:
        """
        Create the {Provider}LLM given an LLMConfig.
        """
        provider = config.provider
        client_config = config.to_client_config()

        if provider == "azureai":
            return AzureAILLM(client_config)
        elif provider == "azureoai":
            return AzureOAILLM(client_config)
        elif provider == "google":
            return GoogleLLM(client_config)
        elif provider == "huggingface":
            return HuggingFaceLLM(client_cfg)
    
    @staticmethod
    def create_gen_config(config: LLMConfig) -> GenCfg:
        """
        Create the {Provider}GenerationConfig given an LLMConfig.
        """
        return config.to_generation_config()

    @staticmethod
    def create_llm_and_gen_config(config: LLMConfig) -> tuple[BaseLLM, GenCfg]:
        """
        Create both the ({Provider}LLM, {Provider}GenerationConfig) given an LLMConfig.
        """
        provider = config.provider
        client_config = config.to_client_config()
        gen_config = config.to_generation_config()

        if provider == "azureai":
            return AzureAILLM(client_config), gen_config
        elif provider == "azureoai":
            return AzureOAILLM(client_config), gen_config
        elif provider == "google":
            return GoogleLLM(client_config), gen_config
        elif provider == "huggingface":
            return HuggingFaceLLM(client_config), gen_config

def init_llm(client_params: dict = {}, generation_params: dict = {}) -> tuple[BaseLLM, GenCfg]:
    llm_cfg = LLMConfig(
        provider=generation_params.get("provider"),
        client_params=client_params,
        generation_params=generation_params,
    )
    return LLMFactory.create_llm_and_gen_config(llm_cfg)