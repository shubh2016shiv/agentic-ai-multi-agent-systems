"""
Base Agent Class
==================
Abstract base class providing common infrastructure for all medical agents.
Handles system prompt injection, tool binding, and standardized execution.

Every agent in the system inherits from BaseAgent, which provides:
    1. Consistent LLM configuration via core.config.get_llm()
    2. Automatic Langfuse callback injection
    3. Standardized response formatting via core.models.AgentResponse
    4. Token tracking via resilience.token_manager
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from core.config import get_llm
from core.models import AgentResponse
from observability.callbacks import build_callback_config

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all medical domain agents.

    Subclasses must implement:
        - system_prompt: Property returning the agent's system instructions
        - process(): Method implementing the agent's core logic

    Optional overrides:
        - tools: List of tools to bind to the agent's LLM
        - post_process(): Transform raw LLM output before returning

    Args:
        name: Human-readable agent name (e.g., "TriageAgent").
        description: What this agent does (for orchestrator context).
        llm_provider: Override the default LLM provider.
        temperature: LLM temperature (defaults to 0.0 for medical domain).

    Example:
        class TriageAgent(BaseAgent):
            @property
            def system_prompt(self):
                return "You are a medical triage specialist..."

            def process(self, input_data):
                return self.invoke(input_data["query"])
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        llm_provider: str | None = None,
        temperature: float = 0.0,
    ):
        self.name = name
        self.description = description
        self.llm = get_llm(provider=llm_provider, temperature=temperature)
        self._tools: list = []

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt that defines this agent's behavior."""
        ...

    @abstractmethod
    def process(self, input_data: dict[str, Any]) -> AgentResponse:
        """
        Execute the agent's core logic.

        Args:
            input_data: The input data for this agent to process.

        Returns:
            AgentResponse with the agent's findings and recommendations.
        """
        ...

    @property
    def tools(self) -> list:
        """Override to provide tools for this agent."""
        return self._tools

    def bind_tools(self, tools: list) -> None:
        """Bind tools to this agent's LLM."""
        self._tools = tools
        if tools:
            self.llm = self.llm.bind_tools(tools)

    def invoke(
        self,
        user_message: str,
        trace_name: str | None = None,
    ) -> str:
        """
        Invoke the LLM with the agent's system prompt and a user message.

        This is the primary interface for simple ask/respond interactions.
        For more complex tool-using workflows, subclasses use LangGraph.

        Args:
            user_message: The user/orchestrator query.
            trace_name: Optional Langfuse trace name.

        Returns:
            The LLM's response text.
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message),
        ]

        config = build_callback_config(
            trace_name=trace_name or f"{self.name}_invocation",
            tags=[self.name],
        )

        response = self.llm.invoke(messages, config=config)
        return response.content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
