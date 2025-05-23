import os
import asyncio
from typing import Any, Dict, Optional

os.environ["HF_HOME"] = "./.cache/.huggingface_cache"

import torch
from pydantic import Field, model_validator, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.models.base_llm import BaseClientConfig, BaseGenerationConfig, BaseLLM

class HuggingFaceClientConfig(BaseClientConfig):
    """
    Contains and validates data needed to load a model & tokenizer from HuggingFace.
    """
    provider: str = "huggingface"
    model_name: str
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"
    auth_token: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def load_from_env(cls, values: dict) -> dict:
        """
        Before validation checks for these variables.
        If not passed directly, attempts to pull from environment.
        """
        if values.get("auth_token") is None:
            env_token = os.environ.get("HF_TOKEN")
            if not env_token:
                raise ValueError("Missing env var HF_TOKEN!")
            values["auth_token"] = env_token
        return values


class HuggingFaceGenerationConfig(BaseGenerationConfig):
    """
    Contains and validates data needed to generate text from the HF causal-LM.
    """
    provider: str = "huggingface"
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    pad_token_id: Optional[int] = None

    def to_hf_dict(self) -> Dict[str, Any]:
        """
        Map the object to kwargs accepted by hf `model.generate`.
        """
        gen_cfg = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "pad_token_id": self.pad_token_id,
        }
        return gen_cfg


class HuggingFaceLLM(BaseLLM):
    """
    Inherits BaseLLM and interfaces text generation with any HF causal-LM.
    """
    provider: str = "huggingface"

    def __init__(self, config: HuggingFaceClientConfig) -> None:
        """
        Initialization creates both tokenizer and model objects per config.
        """
        self.model, self.tokenizer = self._initialize_model_and_tokenizer(config)

    def _initialize_model_and_tokenizer(self, cfg: HuggingFaceClientConfig) -> None:
        """
        Returns tokenizer + model, loaded onto whichever devices Accelerate decides.
        """

        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=cfg.torch_dtype,
                device_map=cfg.device_map,
                token=cfg.auth_token,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name,
                token=cfg.auth_token,
                padding_side="left",
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer
        except Exception as exc:
            raise RuntimeError(f"Failed to load {cfg.model_name}: {exc}") from exc


    def _run_generate(
        self, prompt: str, gen_cfg: HuggingFaceGenerationConfig,
    ) -> str:
        """
        Core generation logic used by both sync and async interfaces.
        """

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **gen_cfg.to_hf_dict(),
            )

        text = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        return text[len(prompt):].lstrip()

    def generate(
        self, prompt: str, config: HuggingFaceGenerationConfig,
    ) -> str:
        """
        Synchronous generation.
        """
        return self._run_generate(prompt, config)

    async def generate_async(
        self, prompt: str, config: HuggingFaceGenerationConfig,
    ) -> str:
        """
        Asynchronous generation.
        """
        return await asyncio.to_thread(self._run_generate, prompt, config)

if __name__ == "__main__":
    # To run this exact script I used an interactive node:
    # sinteractive --gres=gpu:p100:2,lscratch:10 --mem=50G
    
    model_id = "meta-llama/Llama-3.1-8B"
    prompt = "What is the meaning of life?"

    client_config = HuggingFaceClientConfig(
        model_name=model_id,
        device_map="auto"
    )

    llm = HuggingFaceLLM(client_config)

    gen_config = HuggingFaceGenerationConfig(
        max_tokens=128,
        temperature=0.0,
        pad_token_id=llm.tokenizer.eos_token_id
    )

    print(f"Generating output for: {prompt}")
    response = llm.generate(prompt, gen_config)
    print(f"Response: {response}")