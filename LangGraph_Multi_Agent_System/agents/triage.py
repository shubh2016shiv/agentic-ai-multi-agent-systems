"""
Triage Agent Module
===================
Provides the TriageAgent class which specializes in assessing
patient symptoms, determining case urgency, and calculating health risks.

Used by:
    - Handoff and HITL scripts (via process())
    - MAS architecture scripts (via process_with_context())
"""

from typing import Any

from agents.base_agent import BaseAgent
from core.models import AgentResponse
from tools import analyze_symptoms, assess_patient_risk
from prompts import TRIAGE_SYSTEM_PROMPT


class TriageAgent(BaseAgent):
    """
    Medical Triage Agent.

    Responsible for analyzing initial patient presentation,
    classifying the severity/urgency of symptoms, and determining
    the overall risk level. Follows the Single Responsibility Principle
    by focusing purely on triage assessment before routing to specialists.

    Execution interfaces:
        - process(input_data)  → AgentResponse  (structured output)
        - process_with_context(patient_data, context) → str  (MAS architecture scripts)
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="TriageAgent",
            description="Analyzes symptoms and assesses patient risk and urgency.",
            **kwargs,
        )
        self.bind_tools(self.tools)

    @property
    def system_prompt(self) -> str:
        return TRIAGE_SYSTEM_PROMPT

    @property
    def tools(self) -> list:
        return [analyze_symptoms, assess_patient_risk]

    def process(self, input_data: dict[str, Any]) -> AgentResponse:
        """
        Execute the triage assessment based on the input data.

        Args:
            input_data: Dictionary containing patient 'query' or 'symptoms'.

        Returns:
            AgentResponse containing the LLM's structured analysis.
        """
        user_query = input_data.get("query") or input_data.get("symptoms", "")
        if not user_query:
            raise ValueError("TriageAgent requires a 'query' or 'symptoms' in input_data.")

        response_text = self.invoke(user_query)

        return AgentResponse(
            name=self.name,
            output=response_text,
        )

    def process_with_context(
        self,
        patient_data: dict,
        context: str = "",
    ) -> str:
        """
        Execute triage assessment with optional upstream context.

        Overrides the base implementation to add a domain-specific
        prompt prefix ("Triage Assessment") for traceability.

        Args:
            patient_data: Full patient case dict.
            context: Optional upstream context (usually empty for triage).

        Returns:
            Triage summary with urgency classification.
        """
        patient_summary = self._format_patient_summary(patient_data)

        prompt = f"Triage Assessment:\n\n{patient_summary}"
        if context:
            prompt += f"\n\nAdditional Context:\n{context}"

        return self._invoke_llm_with_context(prompt, trace_suffix="triage_assessment")

