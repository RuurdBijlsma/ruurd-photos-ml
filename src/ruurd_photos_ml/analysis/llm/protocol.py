# llm/protocol.py

from enum import StrEnum, auto
from typing import Protocol


class LLMProtocol(Protocol):
    """A protocol defining the interface for a Large Language Model."""

    def generate(self, prompt: str) -> str:
        """
        Generates a text response based on the given prompt.

        Args:
            prompt: The input text to the model.

        Returns:
            The generated text response as a string.
        """
        ...


class LLMProvider(StrEnum):
    """An enumeration of the supported LLM providers."""

    GEMMA_3 = auto()