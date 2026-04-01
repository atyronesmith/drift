"""Hybrid executor: rules for deterministic fields, LLM for reasoning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from drift.hybrid.rules import execute_rules, set_nested


@dataclass
class HybridResult:
    extracted: dict
    llm_output: dict
    merged: dict
    rule_results: list
    raw_llm_response: str
    rules_applied: int
    rules_matched: int
    response: object = None  # provider response for timing


def deep_merge(base: dict, overlay: dict) -> dict:
    """Merge overlay into base, preferring overlay values."""
    result = dict(base)
    for key, val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def build_hybrid_prompt(
    original_input: str,
    extracted: dict,
    llm_fields: list[str],
    hybrid_prompt_template: str | None = None,
) -> str:
    """Build a focused LLM prompt that includes extracted fields."""
    if hybrid_prompt_template:
        return hybrid_prompt_template.replace(
            "{extracted_fields}", json.dumps(extracted, indent=2)
        ).replace("{input}", original_input).replace(
            "{llm_fields}", ", ".join(llm_fields)
        )

    # Default template
    return (
        f"The following fields have already been extracted from the input:\n"
        f"```json\n{json.dumps(extracted, indent=2)}\n```\n\n"
        f"Based on the input below, provide ONLY the following analysis fields as valid JSON:\n"
        f"{', '.join(llm_fields)}\n\n"
        f"Do not repeat the already-extracted fields. Output valid JSON only.\n\n"
        f"{original_input}"
    )


async def run_hybrid(
    provider,
    system_prompt: str,
    input_text: str,
    rules: list[dict],
    llm_fields: list[str],
    hybrid_prompt_template: str | None = None,
    temperature: float = 0,
    merge_mode: str = "llm_over_rules",
    output_mode: str = "merge",
) -> HybridResult:
    """Execute the hybrid pipeline: rules first, then focused LLM call.

    Args:
        provider: LLM provider instance
        system_prompt: System prompt for LLM
        input_text: Original input text
        rules: List of rule definitions
        llm_fields: Fields the LLM should produce
        hybrid_prompt_template: Optional custom prompt template
        temperature: LLM temperature
        merge_mode: "llm_over_rules" (LLM wins conflicts) or "rules_over_llm"
        output_mode: "merge" (combine rules+LLM as JSON) or "llm_passthrough" (LLM output is final)
    """
    # Phase 1: Execute rules
    extracted, rule_results = execute_rules(rules, input_text)
    rules_matched = sum(1 for r in rule_results if r.matched)

    # Phase 2: Resolve template vars in system prompt, build user prompt, call LLM
    resolved_system = system_prompt.replace(
        "{extracted_fields}", json.dumps(extracted, indent=2)
    ).replace("{input}", input_text).replace(
        "{llm_fields}", ", ".join(llm_fields)
    )
    prompt = build_hybrid_prompt(
        input_text, extracted, llm_fields, hybrid_prompt_template
    )
    response = await provider.complete(resolved_system, prompt, temperature=temperature)

    # Phase 3: Build output based on mode
    if output_mode == "llm_passthrough":
        # LLM output IS the final output (e.g., YAML manifests)
        return HybridResult(
            extracted=extracted,
            llm_output={},
            merged={},
            rule_results=rule_results,
            raw_llm_response=response.content,
            rules_applied=len(rules),
            rules_matched=rules_matched,
            response=response,
        )

    # Default merge mode: parse and combine
    llm_output = _parse_json_response(response.content)
    if merge_mode == "rules_over_llm":
        merged = deep_merge(llm_output, extracted)
    else:
        merged = deep_merge(extracted, llm_output)

    return HybridResult(
        extracted=extracted,
        llm_output=llm_output,
        merged=merged,
        rule_results=rule_results,
        raw_llm_response=response.content,
        rules_applied=len(rules),
        rules_matched=rules_matched,
        response=response,
    )


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": text}
