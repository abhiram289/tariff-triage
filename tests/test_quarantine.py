import json
from fastapi.testclient import TestClient
from src.api import app
from src.ingestion.quarantine import quarantine_manager

client = TestClient(app)

def test_valid_submission():
    payload = {
        "consumer_id": "CONS-10029",
        "disputed_tariff_category": "LT-3",
        "billing_dispute_type": "tariff_category",
        "sanctioned_load_kw": 2.0,
        "supply_voltage_kv": 0.23,
        "premises_type": "domestic",
        "premises_activity_description": "Residential home with tailoring in single room",
        "has_secondary_activity": True,
        "secondary_activity_description": "Tailoring machine used by occupant"
    }
    response = client.post("/api/v1/dispute/submit", json=payload)
    assert response.status_code == 202, f"Expected 202, got {response.status_code}"
    data = response.json()
    assert "run_id" in data
    assert data["status"] == "queued"
    print("\n[SUCCESS] Valid submission accepted and queued.")

def test_malformed_submission_quarantined():
    initial_count = quarantine_manager.count()

    # Malformed payload: has_secondary_activity is True but description is missing (cross-field validation failure)
    malformed_payload = {
        "consumer_id": "CONS-BAD-999",
        "disputed_tariff_category": "LT-2",
        "billing_dispute_type": "tariff_category",
        "sanctioned_load_kw": -5.0,  # Range error (must be >= 0.05)
        "supply_voltage_kv": 0.23,
        "premises_type": "domestic",
        "premises_activity_description": "Short",
        "has_secondary_activity": True,
        "secondary_activity_description": ""  # Cross-field violation
    }

    response = client.post("/api/v1/dispute/submit", json=malformed_payload)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    res_data = response.json()
    assert res_data["status"] == "quarantined"
    assert "quarantine_id" in res_data
    print(f"DEBUG res_data errors: {res_data.get('errors')}")
    assert len(res_data["errors"]) >= 1

    # Verify written to quarantine audit store (F2, D1)
    new_count = quarantine_manager.count()
    assert new_count == initial_count + 1
    print(f"\n[SUCCESS] Malformed request rejected at boundary and quarantined with ID: {res_data['quarantine_id']}")

if __name__ == "__main__":
    test_valid_submission()
    test_malformed_submission_quarantined()
