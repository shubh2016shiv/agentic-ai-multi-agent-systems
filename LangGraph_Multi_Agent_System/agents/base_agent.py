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

Two execution interfaces:
    - process(input_data)          → AgentResponse  (used by handoff/HITL/guardrails scripts)
    - process_with_context(patient_data, context) → str  (used by MAS architecture scripts)

Both are valid; which one a script uses depends on what level of
structure (AgentResponse vs raw string) the pattern needs.
"""

import json
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
        - process_with_context(): Used by MAS architecture pattern scripts

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

    # ── Abstract interface ──────────────────────────────────────────────

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

    # ── Tool management ─────────────────────────────────────────────────

    @property
    def tools(self) -> list:
        """Override to provide tools for this agent."""
        return self._tools

    def bind_tools(self, tools: list) -> None:
        """Bind tools to this agent's LLM."""
        self._tools = tools
        if tools:
            self.llm = self.llm.bind_tools(tools)

    # ── Execution interface 1: invoke() ─────────────────────────────────
    # Used by: handoff scripts, HITL scripts, guardrails scripts
    # Returns: str (raw LLM response text)

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

    # ── Execution interface 2: process_with_context() ───────────────────
    # Used by: MAS architecture pattern scripts (sequential_pipeline,
    #          supervisor_orchestration, parallel_voting, etc.)
    # Returns: str (raw LLM response text)
    #
    # Design rationale:
    #   MAS architecture scripts wire agents into LangGraph graphs where
    #   each agent receives (patient_data, upstream_context). This method
    #   provides that calling convention while reusing the root BaseAgent
    #   infrastructure (LLM config, observability, tools).

    def process_with_context(
        self,
        patient_data: dict,
        context: str = "",
    ) -> str:
        """
        Process a patient case with optional upstream context.

        This is the interface used by MAS architecture pattern scripts
        (e.g., sequential_pipeline, supervisor_orchestration) where agents
        are wired into LangGraph graphs and receive accumulated context
        from upstream agents.

        The default implementation formats the patient data, appends
        any upstream context, and calls the LLM. Subclasses may override
        this for custom pre-processing (e.g., tool calls before LLM).

        Args:
            patient_data: Patient case as a dict (from PatientCase.model_dump()).
            context: Optional accumulated context from upstream agents.

        Returns:
            The agent's response as a raw string.
        """
        patient_summary = self._format_patient_summary(patient_data)
        prompt = f"{patient_summary}"
        if context:
            prompt += f"\n\nUpstream Context:\n{context}"

        return self._invoke_llm_with_context(prompt)

    # ── Shared utilities ────────────────────────────────────────────────

    def _invoke_llm_with_context(
        self,
        prompt: str,
        trace_suffix: str = "",
        max_output_words: int = 150,
    ) -> str:
        """
        Invoke the LLM with the agent's system prompt and a formatted prompt.

        This helper is used internally by process_with_context() and may
        be called directly by subclasses that need custom prompt assembly.

        Args:
            prompt: The formatted prompt (patient data + context).
            trace_suffix: Optional suffix for the Langfuse trace name.
            max_output_words: Suggested word limit included in the prompt.

        Returns:
            The LLM response content as a string.
        """
        llm = get_llm()
        trace_name = f"mas_arch_{self.name}"
        if trace_suffix:
            trace_name = f"{trace_name}_{trace_suffix}"

        config = build_callback_config(
            trace_name=trace_name,
            tags=["mas_architecture", self.name],
        )

        full_prompt = (
            f"{self.system_prompt}\n\n"
            f"{prompt}\n\n"
            f"Keep your response under {max_output_words} words."
        )

        response = llm.invoke(full_prompt, config=config)
        return response.content

    def _format_patient_summary(self, patient_data: dict) -> str:
        """
        Format patient data into a standard clinical summary string.

        This ensures all agents receive patient data in the same format,
        regardless of which architecture pattern is orchestrating them.

        Args:
            patient_data: Patient case dict (from PatientCase.model_dump()).

        Returns:
            Formatted patient summary string.
        """
        age = patient_data.get("age", "?")
        sex = patient_data.get("sex", "?")
        complaint = patient_data.get("chief_complaint", "Not specified")
        symptoms = ", ".join(patient_data.get("symptoms", []))
        history = ", ".join(patient_data.get("medical_history", []))
        medications = ", ".join(patient_data.get("current_medications", []))
        allergies = ", ".join(patient_data.get("allergies", [])) or "NKDA"
        vitals = patient_data.get("vitals", {})
        labs = patient_data.get("lab_results", {})

        return (
            f"Patient: {age}y {sex}\n"
            f"Chief Complaint: {complaint}\n"
            f"Symptoms: {symptoms}\n"
            f"Medical History: {history}\n"
            f"Current Medications: {medications}\n"
            f"Allergies: {allergies}\n"
            f"Vitals: {json.dumps(vitals)}\n"
            f"Lab Results: {json.dumps(labs)}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
