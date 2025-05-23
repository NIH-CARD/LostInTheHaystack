from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

class BaseClientConfig(BaseModel):
    """
    Basic params needed to configure LLM clients, to be inherited by {Provider}ClientConfig objects.
    """
    provider: str

class BaseGenerationConfig(BaseModel):
    """
    Basic params needed to configure LLM generation, to be inherited by {Provider}GenerationConfig objects.
    """
    provider: str
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=128, ge=1)
    
class BaseLLM(ABC):
    """
    Defines an abstract LLM class, to be inherited by {Provider}LLM objects.
    """
    provider: str

    @abstractmethod
    def __init__(self, config: BaseClientConfig):
        """
        Initialization of a client connection, using a ClientConfig object.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, config: BaseGenerationConfig):
        """
        Generation by querying the client with a prompt, using a GenerationConfig object.
        """
        pass

    @abstractmethod
    async def generate_async(self, prompt: str, config: BaseGenerationConfig):
        """
        Asynchronous generation by querying the client with a prompt, using a GenerationConfig object.
        """
        pass