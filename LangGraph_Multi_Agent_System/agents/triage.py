"""
Triage Agent Module
===================
Provides the TriageAgent class which specializes in assessing
patient symptoms, determining case urgency, and calculating health risks.
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

        # Note: In a fully-fledged LangGraph system, this invocation might
        # be handled by the graph itself, binding tools dynamically. Here we
        # provide the direct invoke method for simpler architectures or tests.
        response_text = self.invoke(user_query)
        
        return AgentResponse(
            name=self.name,
            output=response_text
        )
