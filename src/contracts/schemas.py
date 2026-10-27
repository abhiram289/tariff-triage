from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator
from datetime import datetime


class DisputeType(str, Enum):
    TARIFF_CATEGORY = "tariff_category"
    METER_ACCURACY = "meter_accuracy"
    LOAD_VIOLATION = "load_violation"
    ARREARS = "arrears"
    DISCONNECTION = "disconnection"


class PremisesType(str, Enum):
    DOMESTIC = "domestic"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    AGRICULTURAL = "agricultural"
    EDUCATIONAL_OR_CHARITABLE = "educational_or_charitable"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class RoutingDecision(str, Enum):
    BILL_STANDS = "bill-stands-with-reason"
    REVISION_RECOMMENDED = "revision-recommended"
    SITE_INSPECTION_REQUIRED = "site-inspection-required"
    ESCALATE_TO_HUMAN = "escalate-to-human"


class RunStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class DisputeSubmissionRequest(BaseModel):
    """
    Published Data Contract for Dispute Ingestion.
    Supports standard and convenient alias field names (F1, F2, D1).
    """
    consumer_id: str = Field(
        ...,
        min_length=3,
        max_length=32,
        pattern=r"^[A-Za-z0-9\-_]+$",
        description="Unique identifier for the consumer / meter connection account"
    )
    disputed_tariff_category: Optional[str] = Field(
        default=None,
        description="The tariff category printed on the disputed bill (e.g., LT-2(a), LT-3, LT-4)"
    )
    billed_category: Optional[str] = Field(
        default=None,
        description="Alias for disputed_tariff_category"
    )
    billing_dispute_type: DisputeType = Field(
        default=DisputeType.TARIFF_CATEGORY,
        description="Categorical dispute type (tariff_category, meter_accuracy, load_violation, etc.)"
    )
    sanctioned_load_kw: float = Field(
        ...,
        ge=0.05,
        le=5000.0,
        description="Sanctioned connected load in kiloWatts (range 0.05 kW to 5000 kW)"
    )
    supply_voltage_kv: float = Field(
        default=0.23,
        ge=0.20,
        le=220.0,
        description="Nominal supply voltage in kV"
    )
    premises_type: Optional[PremisesType] = Field(
        default=None,
        description="Primary nature of the premises"
    )
    connection_type: Optional[str] = Field(
        default=None,
        description="Alias for premises_type"
    )
    premises_activity_description: Optional[str] = Field(
        default=None,
        description="Detailed description of the actual activity conducted at the premises"
    )
    description: Optional[str] = Field(
        default=None,
        description="Alias for premises_activity_description"
    )
    has_secondary_activity: bool = Field(
        default=False,
        description="Whether a secondary or mixed activity is taking place on the same premises"
    )
    secondary_activity_description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Description of any secondary activity"
    )
    disputed_amount: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Amount in dispute, if applicable"
    )
    billing_month: Optional[str] = Field(
        default=None,
        description="Optional billing month (e.g. 2026-09)"
    )

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "DisputeSubmissionRequest":
        # Resolve aliases
        if not self.disputed_tariff_category and self.billed_category:
            self.disputed_tariff_category = self.billed_category
        elif not self.disputed_tariff_category:
            self.disputed_tariff_category = "LT-2(a)"

        if not self.premises_activity_description and self.description:
            self.premises_activity_description = self.description
        elif not self.premises_activity_description:
            raise ValueError("Premises activity description or 'description' field is required.")

        if self.connection_type and not self.premises_type:
            try:
                self.premises_type = PremisesType(self.connection_type.lower())
            except ValueError:
                self.premises_type = PremisesType.DOMESTIC
        elif not self.premises_type:
            self.premises_type = PremisesType.DOMESTIC

        # Auto-detect secondary activity from description if not explicitly set
        desc_lower = self.premises_activity_description.lower()
        if not self.has_secondary_activity and any(t in desc_lower for t in ["tailor", "shop", "clinic", "office", "commercial", "cottage"]):
            self.has_secondary_activity = True
            if not self.secondary_activity_description:
                self.secondary_activity_description = "Extracted from description"

        # Cross-field rule 1: If has_secondary_activity is True, description must be provided
        if self.has_secondary_activity:
            if not self.secondary_activity_description or len(self.secondary_activity_description.strip()) < 3:
                raise ValueError("secondary_activity_description is required when has_secondary_activity is true")

        # Cross-field rule 2: Domestic connections with huge sanctioned load (> 50 kW on LT) require explicit industrial/commercial declaration
        if self.premises_type == PremisesType.DOMESTIC and self.sanctioned_load_kw > 50.0:
            raise ValueError(
                f"Domestic premises cannot have a sanctioned load exceeding 50.0 kW (given {self.sanctioned_load_kw} kW)."
            )

        # Cross-field rule 3: LT supply voltage (> 0.5 kV cannot be standard LT single/three-phase without HT classification)
        if self.sanctioned_load_kw <= 50.0 and self.supply_voltage_kv > 1.0:
            raise ValueError(
                f"Low tension connection with load {self.sanctioned_load_kw} kW cannot have supply voltage of {self.supply_voltage_kv} kV."
            )

        return self


class Citation(BaseModel):
    document_name: str = Field(..., description="Corpus source (e.g. Tariff Order 2024, Supply Code 2004)")
    section_or_clause: str = Field(..., description="Specific clause or section reference (e.g. Clause 4.2.1, Section 31.06)")
    page_number: Optional[int] = Field(default=None, description="Page number in the original PDF")
    quoted_text: str = Field(..., description="Exact or verbatim passage retrieved from the corpus supporting the position")


class TriageStepTrace(BaseModel):
    step_number: int
    agent_name: str
    action: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output_summary: Optional[str] = None
    step_latency_sec: float
    step_cost_usd: float = 0.0


class TriageResult(BaseModel):
    run_id: str
    consumer_id: str
    status: RunStatus
    predicted_tariff_category: Optional[str] = None
    applicable_tariff_schedule: Optional[str] = None
    routing_decision: RoutingDecision
    routing_justification: str
    citations: List[Citation] = Field(default_factory=list)
    human_escalation_reason: Optional[str] = None
    prompt_version: str
    trace_steps: List[TriageStepTrace] = Field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    wall_clock_latency_sec: float = 0.0
    verification_passed: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class SubmissionAckResponse(BaseModel):
    run_id: str
    consumer_id: str
    status: RunStatus
    submission_timestamp: datetime = Field(default_factory=datetime.utcnow)
    estimated_poll_interval_sec: float = 2.0
    status_url: str
    correlation_id: str


class QuarantineRecord(BaseModel):
    quarantine_id: str
    raw_payload: Dict[str, Any]
    validation_errors: List[Dict[str, Any]]
    client_ip: Optional[str] = None
    correlation_id: str
    quarantined_at: datetime = Field(default_factory=datetime.utcnow)


class FeedbackRequest(BaseModel):
    run_id: str
    is_satisfactory: bool
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    officer_remarks: Optional[str] = None
    officer_id: Optional[str] = None
