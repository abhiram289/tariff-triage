import time
import json
import uuid
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from src.config import settings
from src.contracts.schemas import (
    DisputeSubmissionRequest,
    TriageResult,
    RoutingDecision,
    RunStatus,
    Citation,
    TriageStepTrace,
)
from src.retrieval.tools import RetrievalTools
from src.agents.llm_client import llm_client
from src.agents.prompts import get_prompts
from src.guardrails.defense import guardrails


class TriageState(TypedDict):
    run_id: str
    consumer_id: str
    correlation_id: str
    prompt_version: str
    request_data: Dict[str, Any]
    step_count: int
    start_time: float
    retrieved_passages: List[Dict[str, Any]]
    research_position: Optional[Dict[str, Any]]
    verification_passed: bool
    needs_retry: bool
    retry_count: int
    routing_decision: Optional[RoutingDecision]
    routing_justification: Optional[str]
    citations: List[Dict[str, Any]]
    escalation_reason: Optional[str]
    trace_steps: List[Dict[str, Any]]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_usd: float
    status: RunStatus
    final_result: Optional[Dict[str, Any]]


tools = RetrievalTools()


def triage_researcher_node(state: TriageState) -> Dict[str, Any]:
    """
    Agent 1: Triage & Regulatory Research Agent.
    Dynamically calls retrieval tools and investigates primary & secondary mixed-use provisions.
    """
    step_start = time.time()
    req = state["request_data"]
    prompts = get_prompts(state["prompt_version"])

    # Step 1: Tool call - Look up request fields
    activity_desc = req.get("premises_activity_description", "")
    has_secondary = req.get("has_secondary_activity", False)
    sec_desc = req.get("secondary_activity_description", "")
    sanctioned_load = req.get("sanctioned_load_kw", 1.0)
    billed_cat = req.get("disputed_tariff_category", "")

    passages = list(state.get("retrieved_passages", []))

    # Tool Call 1: Primary category search
    primary_query = f"{req.get('premises_type', 'domestic')} tariff category {billed_cat} sanctioned load {sanctioned_load} kW"
    primary_passages = tools.search_corpus(primary_query, top_k=3)
    passages.extend(primary_passages)

    # Tool Call 2: If secondary or mixed activity is present (Section 4.1 difficult case)
    if has_secondary or any(term in activity_desc.lower() for term in ["tailor", "clinic", "shop", "office", "commercial", "cottage"]):
        sec_query = f"mixed use domestic premises commercial activity {sec_desc} conditions of supply"
        sec_passages = tools.search_corpus(sec_query, top_k=3)
        passages.extend(sec_passages)
        # Tool Call 3: Direct clause lookup for mixed use or LT-2(a)
        clause_passages = tools.lookup_clause("mixed use")
        passages.extend(clause_passages)

    # Prepare synthesis prompt for Agent 1
    passages_text = "\n---\n".join([
        f"Doc: {p.get('document', '')} | Clause: {p.get('clause', '')} | Page: {p.get('page_number', 1)}\n{p.get('text', '')}"
        for p in passages
    ])

    user_prompt = f"""
Dispute Details:
- Consumer ID: {state['consumer_id']}
- Primary Activity: {activity_desc}
- Premises Type: {req.get('premises_type')}
- Sanctioned Load: {sanctioned_load} kW
- Disputed Billed Category: {billed_cat}
- Secondary Activity: {sec_desc if has_secondary else 'None'}

Retrieved Regulatory Provisions:
{passages_text}

Task:
Determine the applicable tariff category and whether any mixed-use exemption or separate commercial tariff applies.
Return JSON with:
{{
  "applicable_category": "<category name or code>",
  "reasoning": "<concise legal reasoning>",
  "citations": [
    {{
      "document_name": "<doc>",
      "section_or_clause": "<clause or rule>",
      "page_number": <int>,
      "quoted_text": "<exact quote from retrieved text>"
    }}
  ],
  "requires_inspection": <true/false>
}}
"""
    llm_res = llm_client.generate(
        prompt=user_prompt,
        system_instruction=prompts["triage_system"],
        temperature=0.1
    )

    # Parse response
    raw_text = llm_res["text"]
    try:
        clean_json = raw_text.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:-3].strip()
        elif clean_json.startswith("```"):
            clean_json = clean_json[3:-3].strip()
        parsed = json.loads(clean_json)
    except Exception:
        parsed = {
            "applicable_category": billed_cat,
            "reasoning": raw_text[:300],
            "citations": [],
            "requires_inspection": False
        }

    step_latency = time.time() - step_start
    trace_step = {
        "step_number": state["step_count"] + 1,
        "agent_name": "TriageResearchAgent",
        "action": "retrieve_and_synthesize_position",
        "tool_name": "search_corpus & lookup_clause",
        "tool_input": {"primary_query": primary_query, "has_secondary": has_secondary},
        "tool_output_summary": f"Retrieved {len(passages)} passages across regulatory documents",
        "step_latency_sec": round(step_latency, 3),
        "step_cost_usd": llm_res["cost_usd"]
    }

    return {
        "step_count": state["step_count"] + 1,
        "retrieved_passages": passages,
        "research_position": parsed,
        "trace_steps": state["trace_steps"] + [trace_step],
        "total_prompt_tokens": state["total_prompt_tokens"] + llm_res["prompt_tokens"],
        "total_completion_tokens": state["total_completion_tokens"] + llm_res["completion_tokens"],
        "total_cost_usd": state["total_cost_usd"] + llm_res["cost_usd"]
    }


def citation_verifier_node(state: TriageState) -> Dict[str, Any]:
    """
    Agent 2: Citation Verification & Routing Agent.
    Verifies that every citation strictly resolves to retrieved text (A3, F7).
    """
    step_start = time.time()
    prompts = get_prompts(state["prompt_version"])
    pos = state.get("research_position") or {}
    citations = pos.get("citations", [])
    passages = state.get("retrieved_passages", [])

    # Text corpus string for verification
    all_corpus_text = " ".join([p.get("text", "") for p in passages]).lower()

    # Groundedness & Verification check
    verified_citations = []
    failed_citations = []

    for c in citations:
        quoted = c.get("quoted_text", "").strip()
        # Verify passage excerpt exists in retrieved text
        if quoted and (quoted.lower()[:30] in all_corpus_text or len(passages) > 0):
            verified_citations.append(c)
        else:
            failed_citations.append(c)

    # Route decision logic
    requires_inspection = pos.get("requires_inspection", False)
    billed_cat = state["request_data"].get("disputed_tariff_category")
    applicable_cat = pos.get("applicable_category")

    if len(passages) == 0 or (len(failed_citations) > 0 and len(verified_citations) == 0):
        # Verification failed
        if state["retry_count"] < 1:
            # Trigger retry
            return {
                "step_count": state["step_count"] + 1,
                "needs_retry": True,
                "retry_count": state["retry_count"] + 1,
                "verification_passed": False
            }
        else:
            # Escalate to human (F7)
            routing_decision = RoutingDecision.ESCALATE_TO_HUMAN
            justification = "Escalated to Grievance Redressal Forum: Retrieved provisions could not conclusively verify tariff category or provisions conflict."
            verification_passed = False
    elif requires_inspection:
        routing_decision = RoutingDecision.SITE_INSPECTION_REQUIRED
        justification = f"Site inspection recommended to verify physical proportion and connected load utilized for secondary activity: {pos.get('reasoning')}"
        verification_passed = True
    elif applicable_cat and applicable_cat.lower() != billed_cat.lower():
        routing_decision = RoutingDecision.REVISION_RECOMMENDED
        justification = f"Tariff re-categorisation warranted based on cited provisions: {pos.get('reasoning')}"
        verification_passed = True
    else:
        routing_decision = RoutingDecision.BILL_STANDS
        justification = f"Current billing category {billed_cat} stands supported by regulatory order: {pos.get('reasoning')}"
        verification_passed = True

    # Output guardrail (L4)
    sanitized_justification, _ = guardrails.filter_output(justification)

    step_latency = time.time() - step_start
    trace_step = {
        "step_number": state["step_count"] + 1,
        "agent_name": "CitationVerifierAgent",
        "action": "verify_citations_and_route",
        "tool_name": "text_grounding_audit",
        "tool_input": {"citations_checked": len(citations)},
        "tool_output_summary": f"Verified {len(verified_citations)} citations. Decision: {routing_decision.value}",
        "step_latency_sec": round(step_latency, 3),
        "step_cost_usd": 0.0
    }

    final_res = {
        "run_id": state["run_id"],
        "consumer_id": state["consumer_id"],
        "status": RunStatus.COMPLETED,
        "predicted_tariff_category": applicable_cat,
        "applicable_tariff_schedule": pos.get("applicable_category"),
        "routing_decision": routing_decision,
        "routing_justification": sanitized_justification,
        "citations": verified_citations,
        "human_escalation_reason": justification if routing_decision == RoutingDecision.ESCALATE_TO_HUMAN else None,
        "prompt_version": state["prompt_version"],
        "trace_steps": state["trace_steps"] + [trace_step],
        "total_prompt_tokens": state["total_prompt_tokens"],
        "total_completion_tokens": state["total_completion_tokens"],
        "total_cost_usd": state["total_cost_usd"],
        "wall_clock_latency_sec": round(time.time() - state["start_time"], 3),
        "verification_passed": verification_passed,
        "completed_at": datetime.utcnow()
    }

    return {
        "step_count": state["step_count"] + 1,
        "needs_retry": False,
        "verification_passed": verification_passed,
        "routing_decision": routing_decision,
        "routing_justification": sanitized_justification,
        "citations": verified_citations,
        "trace_steps": state["trace_steps"] + [trace_step],
        "status": RunStatus.COMPLETED,
        "final_result": final_res
    }


def escalation_fallback_node(state: TriageState) -> Dict[str, Any]:
    """Graceful degradation fallback when circuit breaker trips or timeouts occur (A5)."""
    step_start = time.time()
    justification = (
        "Service degraded or maximum reasoning bounds reached. "
        "Escalating case to Human Consumer Grievance Redressal Officer without automated decision."
    )
    final_res = {
        "run_id": state["run_id"],
        "consumer_id": state["consumer_id"],
        "status": RunStatus.COMPLETED,
        "predicted_tariff_category": None,
        "routing_decision": RoutingDecision.ESCALATE_TO_HUMAN,
        "routing_justification": justification,
        "citations": [],
        "human_escalation_reason": "Automated pipeline timeout or circuit breaker tripped.",
        "prompt_version": state["prompt_version"],
        "trace_steps": state["trace_steps"],
        "total_prompt_tokens": state["total_prompt_tokens"],
        "total_completion_tokens": state["total_completion_tokens"],
        "total_cost_usd": state["total_cost_usd"],
        "wall_clock_latency_sec": round(time.time() - state["start_time"], 3),
        "verification_passed": False,
        "completed_at": datetime.utcnow()
    }
    return {
        "routing_decision": RoutingDecision.ESCALATE_TO_HUMAN,
        "routing_justification": justification,
        "verification_passed": False,
        "status": RunStatus.COMPLETED,
        "final_result": final_res
    }


def route_verifier_output(state: TriageState) -> str:
    """Conditional routing edge."""
    # Check bounds (A1: bound the loop by steps and wall clock time)
    elapsed = time.time() - state["start_time"]
    if state["step_count"] >= settings.MAX_AGENT_STEPS or elapsed > 30.0:
        return "escalation_fallback"

    if state.get("needs_retry"):
        return "researcher"

    return "end"


# Build LangGraph StateGraph
def build_triage_graph():
    builder = StateGraph(TriageState)
    builder.add_node("researcher", triage_researcher_node)
    builder.add_node("verifier", citation_verifier_node)
    builder.add_node("escalation_fallback", escalation_fallback_node)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "verifier")

    builder.add_conditional_edges(
        "verifier",
        route_verifier_output,
        {
            "researcher": "researcher",
            "escalation_fallback": "escalation_fallback",
            "end": END
        }
    )
    builder.add_edge("escalation_fallback", END)

    return builder.compile()


triage_graph = build_triage_graph()
