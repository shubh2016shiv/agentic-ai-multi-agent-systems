"""
============================================================
Shared Models for Orchestration Patterns
============================================================
Pydantic and TypedDict models used across all orchestration
sub-modules. These provide a consistent data contract so that
any orchestration pattern can process the same patient workload
and produce comparable results.

════════════════════════════════════════
  WHERE THIS FITS IN THE MAS ARCHITECTURE
════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────┐
    │                  MAS Architecture                       │
    │                                                         │
    │  orchestration/models.py  <── YOU ARE HERE              │
    │  ─────────────────────────────────────────────────────  │
    │  Data contracts shared by ALL orchestration patterns.   │
    │  Every agent, every runner, every synthesis node uses   │
    │  these models to pass data through the graph.           │
    │                                                         │
    │  PatientWorkload ──► BaseOrchestrator.invoke_specialist  │
    │       ▼                        ▼                        │
    │  (routing metadata)     OrchestrationResult             │
    │                              ▼                          │
    │              BaseOrchestrator.invoke_synthesizer         │
    │                              ▼                          │
    │                     final_report (str)                  │
    └─────────────────────────────────────────────────────────┘

Imported by:
    - orchestration/orchestrator.py  (BaseOrchestrator methods return OrchestrationResult)
    - scripts/orchestration/supervisor_orchestration/agents.py (STAGE 1.2)
    - scripts/orchestration/peer_to_peer_orchestration/agents.py (STAGE 2.2)
    - scripts/orchestration/dynamic_router_orchestration/agents.py (STAGE 3.2)
    - scripts/orchestration/graph_of_subgraphs_orchestration/agents.py (STAGE 4.2)
    - scripts/orchestration/hybrid_orchestration/agents.py (STAGE 5.2)
    - All runner.py files (SHARED_PATIENT seed data)

Design:
    - OrchestrationResult: standard envelope for agent outputs
    - PatientWorkload: wraps PatientCase with routing metadata
    - SHARED_PATIENT: the common patient case used by all patterns
============================================================
"""

import json
from typing import Optional
from pydantic import BaseModel, Field

import sys


from core.models import PatientCase


# ============================================================
# Shared Data Models
# ============================================================

class OrchestrationResult(BaseModel):
    """
    Standard result envelope returned by any agent in any
    orchestration pattern.

    This ensures uniform formatting across Supervisor, P2P,
    Dynamic Router, Subgraph, and Hybrid patterns.

    CONNECTION: BaseOrchestrator.invoke_specialist() returns this.
    CONNECTION: BaseOrchestrator.invoke_synthesizer() receives a list of these.
    """
    agent_name: str = Field(description="Name of the agent that produced this result")
    specialty: str = Field(description="Clinical specialty (e.g., pulmonology, cardiology)")
    output: str = Field(description="The agent's clinical assessment or recommendation")
    duration_seconds: float = Field(default=0.0, description="Time taken for processing")
    was_successful: bool = Field(default=True, description="Whether the agent completed successfully")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class PatientWorkload(BaseModel):
    """
    Wraps a PatientCase with orchestration metadata.

    This couples the clinical data with routing hints so that
    orchestrators know what type of processing is needed.

    CONNECTION: Used by runner.py files to seed the LangGraph state.
    """
    patient_case: PatientCase
    required_specialties: list[str] = Field(
        default_factory=lambda: ["pulmonology", "cardiology", "nephrology"],
        description="Which specialties should assess this patient",
    )
    urgency_level: str = Field(
        default="standard",
        description="Routing hint: 'emergency', 'urgent', or 'standard'",
    )
    orchestration_notes: str = Field(
        default="",
        description="Free-text notes for the orchestrator",
    )


# ============================================================
# Shared Patient Case
# ============================================================
# All 5 orchestration patterns process the SAME patient so
# learners can compare how different orchestrations handle
# identical clinical data.

SHARED_PATIENT = PatientCase(
    patient_id="PT-ORCH-001",
    age=68, sex="M",
    chief_complaint="Worsening cough, ankle swelling, and fatigue",
    symptoms=["chronic cough", "bilateral ankle edema", "fatigue", "dyspnea on exertion"],
    medical_history=["COPD Stage II", "Heart Failure NYHA II", "CKD Stage 3a"],
    current_medications=["Tiotropium 18mcg daily", "Furosemide 40mg daily",
                         "Lisinopril 10mg daily", "Carvedilol 12.5mg BID"],
    allergies=["Penicillin"],
    lab_results={
        "eGFR": "45 mL/min",
        "BNP": "580 pg/mL",
        "K+": "4.8 mEq/L",
        "Cr": "1.5 mg/dL",
        "SpO2": "92%",
    },
    vitals={"BP": "138/85", "HR": "78", "SpO2": "92%", "Temp": "37.0C"},
)

SHARED_WORKLOAD = PatientWorkload(
    patient_case=SHARED_PATIENT,
    required_specialties=["pulmonology", "cardiology", "nephrology"],
    urgency_level="urgent",
    orchestration_notes="Multi-system overlap: COPD exacerbation vs HF decompensation vs CKD progression",
)


def format_patient_for_prompt(patient: PatientCase) -> str:
    """
    Format a PatientCase into a standardized prompt string.

    Used by all orchestration patterns to ensure agents receive
    patient data in exactly the same format.

    CONNECTION: Called by BaseOrchestrator.invoke_specialist() and by
    subgraph node factories in graph_of_subgraphs_orchestration/agents.py.
    """
    return (
        f"Patient: {patient.age}y {patient.sex}\n"
        f"Chief Complaint: {patient.chief_complaint}\n"
        f"Symptoms: {', '.join(patient.symptoms)}\n"
        f"Medical History: {', '.join(patient.medical_history)}\n"
        f"Current Medications: {', '.join(patient.current_medications)}\n"
        f"Allergies: {', '.join(patient.allergies) or 'NKDA'}\n"
        f"Vitals: {json.dumps(patient.vitals)}\n"
        f"Lab Results: {json.dumps(patient.lab_results)}"
    )
