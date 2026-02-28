"""
Pharmacology Agent Module
=========================
Provides the PharmacologyAgent class detailing drug interactions,
dosage adjustments, and general pharmacological information.
"""

from typing import Any

from agents.base_agent import BaseAgent
from core.models import AgentResponse
from tools import (
    check_drug_interactions,
    lookup_drug_info,
    calculate_dosage_adjustment,
)
from prompts import PHARMACOLOGY_SYSTEM_PROMPT


class PharmacologyAgent(BaseAgent):
    """
    Medical Pharmacology Agent.

    Responsible for all medication-related inquiries, including checking
    for adverse drug-drug interactions, retrieving drug specifications,
    and calculating necessary renal or hepatic dosage adjustments.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="PharmacologyAgent",
            description="Handles drug information, adverse interactions, and dosages.",
            **kwargs,
        )
        self.bind_tools(self.tools)

    @property
    def system_prompt(self) -> str:
        return PHARMACOLOGY_SYSTEM_PROMPT

    @property
    def tools(self) -> list:
        return [check_drug_interactions, lookup_drug_info, calculate_dosage_adjustment]

    def process(self, input_data: dict[str, Any]) -> AgentResponse:
        """
        Process the pharmacology request based on the input data.

        Args:
            input_data: Dictionary containing 'query', 'medications', etc.

        Returns:
            AgentResponse containing the pharmacological analysis.
        """
        user_query = input_data.get("query", "")
        if not user_query and "medications" in input_data:
            user_query = f"Assess these medications: {input_data['medications']}"
            
        if not user_query:
            raise ValueError("PharmacologyAgent requires a 'query' or 'medications' in input_data.")

        response_text = self.invoke(user_query)
        
        return AgentResponse(
            name=self.name,
            output=response_text
        )
