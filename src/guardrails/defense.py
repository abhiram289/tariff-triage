import re
from typing import Tuple, Optional
from src.config import settings


class GuardrailsManager:
    # Common prompt injection signatures
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"disregard\s+(the\s+)?(prompt|system|instructions?)",
        r"system\s*override",
        r"you\s+are\s+now\s+(an\s+unrestricted|a\s+different|in\s+dan\s+mode)",
        r"bypass\s+(all\s+)?rules",
        r"jailbreak",
        r"pretend\s+you\s+are\s+not",
        r"reveal\s+your\s+(initial|system)\s+prompt",
        r"what\s+are\s+your\s+hidden\s+instructions",
        r"repeat\s+the\s+words\s+above",
        r"simulate\s+an\s+unfiltered\s+ai"
    ]

    # Unauthorized determinations (system must NEVER revise a bill directly)
    UNAUTHORIZED_OUTPUT_PATTERNS = [
        r"i\s+have\s+revised\s+your\s+bill",
        r"your\s+bill\s+is\s+hereby\s+(cancelled|zeroed|recalculated)",
        r"you\s+do\s+not\s+have\s+to\s+pay",
        r"we\s+have\s+adjusted\s+the\s+billed\s+amount\s+to",
        r"refund\s+has\s+been\s+credited"
    ]

    TOXICITY_PATTERNS = [
        r"\b(threat|kill|destroy|stupid|idiot|fraudulent\s+scam|cheat)\b"
    ]

    def check_input_injection(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Scans free-text fields for adversarial prompt injections.
        Returns (is_safe, violation_reason).
        """
        if not settings.ENABLE_PROMPT_INJECTION_FILTER:
            return True, None

        normalized = text.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, normalized):
                return False, f"Prompt injection signature detected: matching pattern '{pattern}'"
        return True, None

    def check_input_toxicity(self, text: str) -> Tuple[bool, Optional[str]]:
        """Scans input text for abusive or toxic language."""
        if not settings.ENABLE_TOXICITY_FILTER:
            return True, None

        normalized = text.lower()
        for pattern in self.TOXICITY_PATTERNS:
            if re.search(pattern, normalized):
                return False, f"Toxic or abusive language detected: matching '{pattern}'"
        return True, None

    def filter_output(self, response_text: str) -> Tuple[str, bool]:
        """
        Ensures model output never issues an unauthorized bill adjustment.
        Returns (sanitized_text, was_modified).
        """
        if not settings.ENABLE_OUTPUT_FILTER:
            return response_text, False

        modified = False
        sanitized = response_text
        for pattern in self.UNAUTHORIZED_OUTPUT_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                sanitized = re.sub(
                    pattern,
                    "[REDACTED: System cannot revise billing amounts directly. Tariff position recommendation only]",
                    sanitized,
                    flags=re.IGNORECASE
                )
                modified = True

        return sanitized, modified


guardrails = GuardrailsManager()
