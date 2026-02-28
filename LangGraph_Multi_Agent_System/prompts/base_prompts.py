"""
Base Prompts Module
===================
Shared system prompt components or base interfaces to ensure
a standardized prompt structure across all agents.
"""

# Common suffix attached to all agents to reinforce enterprise constraints
ENTERPRISE_CONSTRAINTS_SUFFIX = (
    "\n\nSecurity & Compliance Constraints:\n"
    "1. Never output raw PII (Patient Identifiable Information).\n"
    "2. All medical advice must be explicitly flagged as an AI recommendation, not a final clinical decision.\n"
    "3. Fail safely: If you lack information or tools fail, state your limitation clearly rather than guessing."
)
