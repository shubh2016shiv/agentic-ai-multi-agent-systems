"""
Pharmacist Agent Module
========================
Provides the PharmacistAgent class which specializes in medication
review, drug-drug interaction checking, and dose adjustments.

Used by:
    - MAS architecture scripts (via process_with_context)
    - Any future script that needs clinical pharmacy capability

Note:
    This is distinct from PharmacologyAgent (pharmacology.py), which
    focuses on drug lookup, drug info retrieval, and dosage calculation
    tools. PharmacistAgent is a clinical pharmacy role used in MAS
    architecture patterns for medication review within multi-agent
    workflows.
"""

from typing import Any

from agents.base_agent import BaseAgent
from core.models import AgentResponse
from prompts import PHARMACIST_SYSTEM_PROMPT


class PharmacistAgent(BaseAgent):
    """
    Clinical Pharmacist Agent.

    Responsible for reviewing current medication regimens, checking for
    drug-drug interactions, adjusting doses for renal/hepatic function,
    and flagging contraindications.

    Execution interfaces:
        - process(input_data)  → AgentResponse  (structured output)
        - process_with_context(patient_data, context) → str  (MAS architecture scripts)
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="PharmacistAgent",
            description="Reviews medications, checks interactions, and adjusts doses.",
            **kwargs,
        )

    @property
    def system_prompt(self) -> str:
        return PHARMACIST_SYSTEM_PROMPT

    def process(self, input_data: dict[str, Any]) -> AgentResponse:
        """
        Execute the medication review based on the input data.

        Args:
            input_data: Dictionary containing 'query' or 'medications'.

        Returns:
            AgentResponse containing the pharmacist's analysis.
        """
        user_query = input_data.get("query", "")
        if not user_query and "medications" in input_data:
            user_query = f"Review these medications: {input_data['medications']}"

        if not user_query:
            raise ValueError("PharmacistAgent requires a 'query' or 'medications' in input_data.")

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
        Execute medication review with upstream context.

        Overrides the base implementation to add a domain-specific
        prompt prefix ("Medication Review") for traceability.

        Args:
            patient_data: Full patient case dict.
            context: Optional upstream context (e.g., triage results).

        Returns:
            Medication review with interaction flags and dose adjustments.
        """
        patient_summary = self._format_patient_summary(patient_data)

        prompt = f"Medication Review:\n\n{patient_summary}"
        if context:
            prompt += f"\n\nUpstream Findings:\n{context}"

        return self._invoke_llm_with_context(prompt, trace_suffix="medication_review")
