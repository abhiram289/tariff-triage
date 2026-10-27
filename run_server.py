import uvicorn
import os

if __name__ == "__main__":
    port = 8085
    print("\n" + "=" * 65)
    print("  Starting Electricity Tariff & Billing Dispute Triage Web Service")
    print(f"  Open in browser: http://localhost:{port}")
    print("=" * 65 + "\n")
    uvicorn.run("src.api:app", host="0.0.0.0", port=port, reload=False)
