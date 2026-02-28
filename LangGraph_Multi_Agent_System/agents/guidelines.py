"""
Guidelines Agent Module
=======================
Provides the GuidelinesAgent class which looks up and interprets
official clinical guidelines for evidence-based medicine.
"""

from typing import Any

from agents.base_agent import BaseAgent
from core.models import AgentResponse
from tools import lookup_clinical_guideline
from prompts import GUIDELINES_SYSTEM_PROMPT


class GuidelinesAgent(BaseAgent):
    """
    Clinical Guidelines Agent.

    Responsible for retrieving verified medical guidelines (e.g., AHA, IDSA)
    to support evidence-based decision making for patient care plans.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="GuidelinesAgent",
            description="Retrieves and interprets official clinical practice guidelines.",
            **kwargs,
        )
        self.bind_tools(self.tools)

    @property
    def system_prompt(self) -> str:
        return GUIDELINES_SYSTEM_PROMPT

    @property
    def tools(self) -> list:
        return [lookup_clinical_guideline]

    def process(self, input_data: dict[str, Any]) -> AgentResponse:
        """
        Process the request to find and interpret clinical guidelines.

        Args:
            input_data: Dictionary containing the patient 'query' or 'condition'.

        Returns:
            AgentResponse containing clinical guideline implications.
        """
        user_query = input_data.get("query", "")
        if not user_query and "condition" in input_data:
            user_query = f"Find guidelines for condition: {input_data['condition']}"

        if not user_query:
            raise ValueError("GuidelinesAgent requires a 'query' or 'condition' in input_data.")

        response_text = self.invoke(user_query)

        return AgentResponse(
            name=self.name,
            output=response_text
        )
