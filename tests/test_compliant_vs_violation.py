from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_compliant_vs_violation():
    print("\n" + "=" * 65)
    print("TEST: Compliant Case vs Commercial Violation Case")
    print("=" * 65)

    # 1. Compliant Case
    p_comp = {
        "consumer_id": "BESCOM-DOM-01",
        "connection_type": "domestic",
        "sanctioned_load_kw": 2.5,
        "billed_category": "LT-2(a)",
        "description": "I have a domestic connection and reside in the house. I operate a single tailoring machine from one room for local neighborhood orders."
    }
    r_comp = client.post("/api/v1/dispute/submit", json=p_comp)
    assert r_comp.status_code == 202
    st_comp = client.get(f"/api/v1/dispute/status/{r_comp.json()['run_id']}").json()

    print("\n[+] SCENARIO 1: COMPLIANT TAILORING (Schedule LT-1 Exemption)")
    print(f"    - Route: {st_comp['routing_decision']}")
    print(f"    - Predicted Category: {st_comp['predicted_tariff_category']}")
    print(f"    - Justification: {st_comp['routing_justification']}")
    print("    - Citations:")
    for c in st_comp["citations"]:
        print(f"      * [{c['document_name']} | {c['section_or_clause']} | P.{c['page_number']}]: \"{c['quoted_text']}\"")

    assert st_comp["routing_decision"] == "bill-stands-with-reason"
    assert st_comp["predicted_tariff_category"] == "LT-1"
    assert len(st_comp["citations"]) == 2

    # 2. Violation Case
    p_viol = {
        "consumer_id": "BESCOM-VIOL-02",
        "connection_type": "domestic",
        "sanctioned_load_kw": 12.0,
        "billed_category": "LT-2(a)",
        "description": "Domestic residential connection. Operating a commercial tailoring boutique with 6 industrial sewing machines and 4 non-resident workers."
    }
    r_viol = client.post("/api/v1/dispute/submit", json=p_viol)
    assert r_viol.status_code == 202
    st_viol = client.get(f"/api/v1/dispute/status/{r_viol.json()['run_id']}").json()

    print("\n[+] SCENARIO 2: TAILORING VIOLATION (Commercial/MSME Load Exceedance)")
    print(f"    - Route: {st_viol['routing_decision']}")
    print(f"    - Predicted Category: {st_viol['predicted_tariff_category']}")
    print(f"    - Justification: {st_viol['routing_justification']}")
    print("    - Citations:")
    for c in st_viol["citations"]:
        print(f"      * [{c['document_name']} | {c['section_or_clause']} | P.{c['page_number']}]: \"{c['quoted_text']}\"")

    assert st_viol["routing_decision"] == "site-inspection-required"
    assert st_viol["predicted_tariff_category"] == "LT-5"
    assert len(st_viol["citations"]) == 2

    print("\n[SUCCESS] Both Compliant and Violation scenarios verified!")

if __name__ == "__main__":
    test_compliant_vs_violation()
