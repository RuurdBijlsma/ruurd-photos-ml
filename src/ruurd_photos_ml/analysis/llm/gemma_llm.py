# llm/gemma_llm.py

from functools import lru_cache
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Make sure to import your new protocol
from .protocol import LLMProtocol


@lru_cache(maxsize=None)
def get_model_and_tokenizer() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Retrieve and cache the Gemma model and tokenizer.
    This function is cached to ensure the model is loaded into memory only once.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    model_id = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"

    print(f"Loading model and tokenizer for '{model_id}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # Automatically use GPU if available
        torch_dtype=torch.bfloat16,  # Recommended for modern GPUs
        load_in_4bit=True,  # Enable 4-bit quantization
    )
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


class GemmaLLM(LLMProtocol):
    """
    An implementation of the LLMProtocol using the 4-bit quantized Gemma 3 12B model.
    """

    def __init__(self, max_new_tokens: int = 200, temperature: float = 0.7):
        """
        Initializes the GemmaLLM.

        Args:
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: The temperature for sampling-based generation.
        """
        self.model, self.tokenizer = get_model_and_tokenizer()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # Using self.model.device is robust, as device_map="auto" handles placement
        self.device = self.model.device

    def generate(self, prompt: str) -> str:
        """
        Generates a text response based on the given prompt.

        Args:
            prompt: The input text to the model.

        Returns:
            The generated text response as a string.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                top_k=50,
            )

        # Decode the output and skip any special tokens (e.g., padding)
        decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean the output by removing the original prompt
        # Models often include the prompt in their response.
        if decoded_text.startswith(prompt):
            return decoded_text[len(prompt):].strip()

        return decoded_text.strip()