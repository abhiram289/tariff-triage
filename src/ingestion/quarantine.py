import json
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import uuid

from src.contracts.schemas import QuarantineRecord
from src.config import settings


class QuarantineManager:
    def __init__(self, quarantine_dir: str = settings.QUARANTINE_DIR):
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.quarantine_dir / "quarantine_records.jsonl"

    def quarantine(
        self,
        raw_payload: Dict[str, Any],
        validation_errors: List[Dict[str, Any]],
        correlation_id: str,
        client_ip: str = "unknown"
    ) -> QuarantineRecord:
        record = QuarantineRecord(
            quarantine_id=f"quar-{uuid.uuid4().hex[:12]}",
            raw_payload=raw_payload,
            validation_errors=validation_errors,
            client_ip=client_ip,
            correlation_id=correlation_id,
            quarantined_at=datetime.utcnow()
        )

        # Append to JSONL for auditability
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

        return record

    def list_records(self, limit: int = 50) -> List[QuarantineRecord]:
        if not self.jsonl_path.exists():
            return []
        records = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines[-limit:]):
                if line.strip():
                    records.append(QuarantineRecord.model_validate_json(line))
        return records

    def count(self) -> int:
        if not self.jsonl_path.exists():
            return 0
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


quarantine_manager = QuarantineManager()
