import json
from src.contracts.schemas import DisputeSubmissionRequest, DisputeType, PremisesType
from src.triage_engine import process_dispute_triage

def test_difficult_case():
    print("\n--- Running Test: Section 4.1 Difficult Case (Tailoring in Domestic Connection) ---")
    request = DisputeSubmissionRequest(
        consumer_id="CONS-KA-BLR-0941",
        disputed_tariff_category="LT-3",
        billing_dispute_type=DisputeType.TARIFF_CATEGORY,
        sanctioned_load_kw=2.5,
        supply_voltage_kv=0.23,
        premises_type=PremisesType.DOMESTIC,
        premises_activity_description="I have a domestic connection. I run a small tailoring shop from one room of the house. Which tariff category should I be billed under?",
        has_secondary_activity=True,
        secondary_activity_description="One room utilized for tailoring machine operated by occupant"
    )

    result = process_dispute_triage(
        dispute=request,
        run_id="run-test-401",
        correlation_id="corr-test-01"
    )

    print(f"Status: {result.status.value}")
    print(f"Predicted Category: {result.predicted_tariff_category}")
    print(f"Routing Decision: {result.routing_decision.value}")
    print(f"Routing Justification:\n  {result.routing_justification}")
    print(f"Citations Verified ({len(result.citations)}):")
    for c in result.citations:
        print(f"  - [{c.document_name} | {c.section_or_clause} | P.{c.page_number}]: \"{c.quoted_text}\"")

    assert result.predicted_tariff_category == "LT-1", f"Expected LT-1, got {result.predicted_tariff_category}"
    assert result.routing_decision.value == "revision-recommended", f"Unexpected routing: {result.routing_decision.value}"
    assert len(result.citations) >= 1, "Expected at least 1 verified citation"
    print("\n[SUCCESS] Difficult Case test passed successfully!")

if __name__ == "__main__":
    test_difficult_case()
