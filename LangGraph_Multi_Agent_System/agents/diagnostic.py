"""
Diagnostic Agent Module
========================
Provides the DiagnosticAgent class which specializes in generating
differential diagnoses, correlating symptoms with lab values, and
assessing risk factors.

Used by:
    - MAS architecture scripts (via process_with_context)
    - Any future script that needs diagnostic capability
"""

from typing import Any

from agents.base_agent import BaseAgent
from core.models import AgentResponse
from prompts import DIAGNOSTIC_SYSTEM_PROMPT


class DiagnosticAgent(BaseAgent):
    """
    Diagnostic Specialist Agent.

    Responsible for generating ranked differential diagnoses based on
    patient symptoms, lab values, vital signs, and medical history.
    Assesses how comorbidities affect the differential and recommends
    further diagnostic workup when needed.

    Execution interfaces:
        - process(input_data)  → AgentResponse  (structured output)
        - process_with_context(patient_data, context) → str  (MAS architecture scripts)
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="DiagnosticAgent",
            description="Generates differential diagnoses and assesses risk factors.",
            **kwargs,
        )

    @property
    def system_prompt(self) -> str:
        return DIAGNOSTIC_SYSTEM_PROMPT

    def process(self, input_data: dict[str, Any]) -> AgentResponse:
        """
        Execute the diagnostic assessment based on the input data.

        Args:
            input_data: Dictionary containing 'query' or 'symptoms'.

        Returns:
            AgentResponse containing the diagnostic analysis.
        """
        user_query = input_data.get("query") or input_data.get("symptoms", "")
        if not user_query:
            raise ValueError("DiagnosticAgent requires a 'query' or 'symptoms' in input_data.")

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
        Execute diagnostic assessment with upstream context.

        Overrides the base implementation to add a domain-specific
        prompt prefix ("Diagnostic Assessment") for traceability.

        Args:
            patient_data: Full patient case dict.
            context: Optional upstream context (e.g., triage + medication results).

        Returns:
            Differential diagnosis with risk assessment.
        """
        patient_summary = self._format_patient_summary(patient_data)

        prompt = f"Diagnostic Assessment:\n\n{patient_summary}"
        if context:
            prompt += f"\n\nUpstream Findings:\n{context}"

        return self._invoke_llm_with_context(prompt, trace_suffix="diagnostic_assessment")
