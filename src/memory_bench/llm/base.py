from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Schema:
    """Provider-agnostic JSON schema description for structured output."""
    properties: dict[str, Any]   # {name: {"type": "string"|"boolean"|..., "description": ...}}
    required: list[str]


@dataclass
class ToolDef:
    """A callable tool that can be invoked by the LLM during an agentic loop."""
    name: str
    description: str
    parameters: dict[str, Any]   # {param_name: {"type": ..., "description": ...}}
    required: list[str]
    fn: Callable[..., str]       # sync callable; must return a string result


class LLM(ABC):
    """Minimal interface for LLM generation with structured JSON output."""

    @property
    def model_id(self) -> str:
        """Return a 'provider:model' identifier string."""
        return self.__class__.__name__

    @abstractmethod
    def generate(self, prompt: str, schema: Schema) -> dict:
        """
        Send `prompt` to the model and return a parsed dict matching `schema`.
        Implementations must handle rate-limit retries internally.
        """
        ...

    def tool_loop(self, prompt: str, tools: list[ToolDef], max_tool_calls: int = 10) -> str:
        """
        Run an agentic tool-calling loop. The model may call tools zero or more times
        before returning a final text response.

        Returns the model's final text response after all tool calls are resolved.
        Raises NotImplementedError if this LLM does not support tool calling.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support tool_loop")
