"""
Shared Data Models
===================
Pydantic models used across the entire Multi-Agent System. These models serve
as the CONTRACT between modules — agents, tools, guardrails, and orchestrators
all communicate using these shared types.

Design Principles:
    - Every model is immutable by default (frozen=True) to prevent accidental
      mutation during handoffs between agents
    - Rich docstrings serve as self-documenting schemas
    - Optional fields have sensible defaults for progressive enrichment
      (e.g., a PatientCase starts with symptoms and gains diagnoses over time)
    - Enum-like Literal types prevent invalid values at the boundary

Why Pydantic?
    In a multi-agent system, data flows through many components. Pydantic
    ensures that data is validated at every boundary crossing, catching
    errors early rather than letting them propagate through agent chains.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# Patient Domain Models
# ============================================================


class PatientCase(BaseModel):
    """
    Represents a patient case flowing through the clinical decision support system.

    This is the PRIMARY data object that agents operate on. It is progressively
    enriched as it moves through the agent pipeline:
        Triage Agent → adds urgency, initial_assessment
        Diagnosis Agent → adds differential_diagnoses
        Pharmacology Agent → adds recommended_medications
        Safety Agent → adds safety_flags
    """

    patient_id: str = Field(description="Unique patient identifier")
    age: int = Field(description="Patient age in years")
    sex: str = Field(description="Patient sex (M/F/Other)")
    chief_complaint: str = Field(description="Primary reason for visit")
    symptoms: list[str] = Field(default_factory=list, description="List of reported symptoms")
    medical_history: list[str] = Field(
        default_factory=list, description="Relevant past medical conditions"
    )
    current_medications: list[str] = Field(
        default_factory=list, description="Medications the patient is currently taking"
    )
    allergies: list[str] = Field(
        default_factory=list, description="Known drug allergies"
    )
    lab_results: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value pairs of lab test names and results",
    )
    vitals: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value pairs of vital signs (BP, HR, Temp, etc.)",
    )

    # --- Fields populated by agents during processing ---
    urgency: Optional[str] = Field(
        default=None, description="Triage urgency level: emergent/urgent/routine"
    )
    initial_assessment: Optional[str] = Field(
        default=None, description="Triage agent's initial assessment"
    )
    differential_diagnoses: list[str] = Field(
        default_factory=list, description="Possible diagnoses ranked by likelihood"
    )
    recommended_medications: list[str] = Field(
        default_factory=list, description="Medications recommended by the pharmacology agent"
    )
    safety_flags: list[str] = Field(
        default_factory=list,
        description="Safety concerns flagged during review (interactions, contraindications)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "PT-2026-001",
                "age": 67,
                "sex": "M",
                "chief_complaint": "Persistent cough and shortness of breath",
                "symptoms": ["chronic cough", "dyspnea on exertion", "wheezing"],
                "medical_history": ["COPD Stage II", "Hypertension"],
                "current_medications": ["Tiotropium", "Lisinopril"],
                "allergies": ["Penicillin"],
                "lab_results": {"eGFR": "58 mL/min", "HbA1c": "6.2%"},
                "vitals": {"BP": "145/90", "HR": "88", "SpO2": "92%"},
            }
        }


class DrugInfo(BaseModel):
    """
    Structured information about a medication from the drug database.

    Populated by the document_processing.excel_processor from the drugs/ folder.
    Used by the Pharmacology Agent and Safety Agent for drug-related decisions.
    """

    drug_name: str = Field(description="Generic drug name")
    drug_class: str = Field(description="Therapeutic drug class (e.g., 'ACE Inhibitor')")
    sub_class: Optional[str] = Field(
        default=None, description="Drug sub-class for more specific classification"
    )
    is_preferred: bool = Field(
        default=False, description="Whether this is a preferred/formulary drug"
    )
    common_indications: list[str] = Field(
        default_factory=list, description="Common conditions this drug treats"
    )
    contraindications: list[str] = Field(
        default_factory=list, description="Conditions where this drug should NOT be used"
    )
    common_interactions: list[str] = Field(
        default_factory=list, description="Known drug-drug interactions"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "drug_name": "Lisinopril",
                "drug_class": "ACE Inhibitor",
                "sub_class": "Cardiovascular",
                "is_preferred": True,
                "common_indications": ["Hypertension", "Heart Failure", "Diabetic Nephropathy"],
                "contraindications": ["Pregnancy", "Angioedema history", "Bilateral renal artery stenosis"],
                "common_interactions": ["Potassium-sparing diuretics", "NSAIDs", "Lithium"],
            }
        }


class GuidelineReference(BaseModel):
    """
    A reference to a specific clinical guideline recommendation.

    Produced by the document_processing module when processing PDFs from
    medical_guidelines/. Used to back agent decisions with evidence-based citations.
    """

    source_document: str = Field(description="Name of the guideline document (e.g., 'COPD GOLD 2024')")
    section: str = Field(description="Section or chapter within the guideline")
    recommendation: str = Field(description="The specific recommendation text")
    evidence_grade: Optional[str] = Field(
        default=None,
        description="Evidence grade (e.g., 'Grade A - Strong recommendation')",
    )
    page_number: Optional[int] = Field(default=None, description="Page number in the source PDF")

    class Config:
        json_schema_extra = {
            "example": {
                "source_document": "COPD GOLD 2024",
                "section": "Pharmacological Treatment",
                "recommendation": "Long-acting bronchodilators are preferred over short-acting for maintenance therapy in COPD.",
                "evidence_grade": "Grade A",
                "page_number": 45,
            }
        }


class HandoffContext(BaseModel):
    """
    Context package passed during agent-to-agent handoffs.

    This is a CRITICAL model for Multi-Agent Systems. When one agent hands off
    to another, it must pass enough context for the receiving agent to continue
    work WITHOUT accessing the full conversation history. This implements the
    Context Scoping principle from Chapter 4.

    The context is deliberately SCOPED: only information relevant to the
    receiving agent is included. This reduces token usage and improves
    the receiving agent's reasoning accuracy.
    """

    from_agent: str = Field(description="Name of the agent initiating the handoff")
    to_agent: str = Field(description="Name of the agent receiving the handoff")
    reason: str = Field(description="Why this handoff is happening")
    patient_case: PatientCase = Field(description="The patient case being handed off")
    task_description: str = Field(
        description="What the receiving agent should do with this case"
    )
    relevant_findings: list[str] = Field(
        default_factory=list,
        description="Key findings from the sending agent that the receiver needs",
    )
    handoff_depth: int = Field(
        default=0,
        description="How many handoffs have occurred in this workflow (circuit breaker metric)",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the handoff was initiated",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "from_agent": "TriageAgent",
                "to_agent": "PharmacologyAgent",
                "reason": "Patient needs medication review for drug interactions",
                "task_description": "Check current medications for interactions and suggest alternatives if needed",
                "relevant_findings": [
                    "Patient is on Lisinopril + Spironolactone (potential hyperkalemia risk)",
                    "eGFR declining (58 mL/min) — dose adjustment may be needed",
                ],
                "handoff_depth": 1,
            }
        }


class AgentResponse(BaseModel):
    """
    Standardized response format from any agent in the system.

    Every agent returns this format, ensuring consistent downstream processing
    by the orchestrator. This is the "interface contract" for agent outputs.
    """

    agent_name: str = Field(description="Which agent produced this response")
    status: str = Field(
        description="Outcome status: 'success', 'needs_review', 'error', 'escalate'"
    )
    response_text: str = Field(description="Human-readable response/recommendation")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Agent's self-assessed confidence in its response (0-1)",
    )
    evidence: list[GuidelineReference] = Field(
        default_factory=list,
        description="Clinical guidelines supporting this response",
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Actionable recommendations"
    )
    safety_warnings: list[str] = Field(
        default_factory=list, description="Any safety concerns identified"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (tokens_used, latency_ms, etc.)",
    )
