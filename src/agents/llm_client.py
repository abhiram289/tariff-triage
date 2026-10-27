import os
import time
import json
from typing import Dict, Any, Optional, Tuple
import google.generativeai as genai

from src.config import settings
from src.agents.circuit_breaker import circuit_breaker, CircuitBreakerOpenException

# Estimated pricing for Gemini 2.0 Flash / 1.5 Flash (USD per 1M tokens)
COST_PER_1M_PROMPT_TOKENS = 0.075  # $0.075 per 1M tokens
COST_PER_1M_COMPLETION_TOKENS = 0.30  # $0.30 per 1M tokens


class LLMClient:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.is_stub = settings.USE_LLM_STUB
        self._configured = False

    def _ensure_configured(self):
        if not self._configured:
            if not self.api_key:
                # Check environment directly as fallback
                self.api_key = os.getenv("GEMINI_API_KEY")
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self._configured = True

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = (prompt_tokens / 1_000_000 * COST_PER_1M_PROMPT_TOKENS) + \
               (completion_tokens / 1_000_000 * COST_PER_1M_COMPLETION_TOKENS)
        return round(cost, 6)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Executes an LLM call through the circuit breaker, with latency, token & cost tracking.
        """
        start_time = time.time()

        # Handle Stub mode for D6 load testing
        if self.is_stub or settings.USE_LLM_STUB:
            time.sleep(settings.LLM_STUB_DELAY_SEC)
            elapsed = time.time() - start_time
            return {
                "text": "STUB_RESPONSE: [Grounding: Verified against KERC Clause 4.2.1] Billing category LT-2(a) applies.",
                "prompt_tokens": 150,
                "completion_tokens": 30,
                "cost_usd": self.calculate_cost(150, 30),
                "latency_sec": round(elapsed, 3),
                "is_stub": True
            }

        self._ensure_configured()
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured in .env. Please set GEMINI_API_KEY to execute agent reasoning."
            )

        def _call_model():
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction
            )
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response

        # Run within Circuit Breaker
        response = circuit_breaker.execute(_call_model)
        elapsed = time.time() - start_time

        # Token counting estimation if not directly in metadata
        text = response.text if response and response.text else ""
        prompt_tokens = len(prompt.split()) * 2
        completion_tokens = len(text.split()) * 2

        cost = self.calculate_cost(prompt_tokens, completion_tokens)

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "latency_sec": round(elapsed, 3),
            "is_stub": False
        }


llm_client = LLMClient()
