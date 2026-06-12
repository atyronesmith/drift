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
    hidden_ms: float = 0.0  # time spent fetching hidden states (0 if none)


def _fetch_hidden_states_remote(system, prompt, base_url=None, layers=None,
                                positions="last_input", prefill_only=True):
    """Indirection over probe_hidden_remote so tests can monkeypatch it."""
    from drift.probing.mlx_hidden import probe_hidden_remote
    return probe_hidden_remote(
        system=system, prompt=prompt, base_url=base_url,
        layers=layers, positions=positions, prefill_only=prefill_only,
    )


def deep_merge(base, overlay):
    """Merge overlay into base, preferring overlay values.

    Handles dicts (recursive merge) and lists (element-wise merge).
    """
    if isinstance(base, dict) and isinstance(overlay, dict):
        result = dict(base)
        for key, val in overlay.items():
            if key in result:
                result[key] = deep_merge(result[key], val)
            else:
                result[key] = val
        return result
    if isinstance(base, list) and isinstance(overlay, list):
        result = list(overlay)
        for i, item in enumerate(base):
            if i < len(result):
                result[i] = deep_merge(item, result[i])
            else:
                result.append(item)
        return result
    return overlay


def build_hybrid_prompt(
    original_input: str,
    extracted: dict,
    llm_fields: list[str],
    hybrid_prompt_template: str | None = None,
    escalation_hints: list[dict] | None = None,
) -> str:
    """Build a focused LLM prompt that includes extracted fields.

    Args:
        original_input: Raw input text
        extracted: Already-extracted fields from rules
        llm_fields: Fields the LLM should produce
        hybrid_prompt_template: Optional custom template
        escalation_hints: Tentative classifier predictions that were below
            confidence threshold, each with path/hint/confidence keys
    """
    hint_block = _format_escalation_hints(escalation_hints)

    if hybrid_prompt_template:
        return hybrid_prompt_template.replace(
            "{extracted_fields}", json.dumps(extracted, indent=2)
        ).replace("{input}", original_input).replace(
            "{llm_fields}", ", ".join(llm_fields)
        ).replace("{escalation_hints}", hint_block)

    # Default template
    parts = [
        f"The following fields have already been extracted from the input:\n"
        f"```json\n{json.dumps(extracted, indent=2)}\n```\n",
    ]
    if hint_block:
        parts.append(hint_block + "\n")
    parts.append(
        f"Based on the input below, provide ONLY the following analysis fields as valid JSON:\n"
        f"{', '.join(llm_fields)}\n\n"
        f"Do not repeat the already-extracted fields. Output valid JSON only.\n\n"
        f"{original_input}"
    )
    return "\n".join(parts)


def _format_escalation_hints(hints: list[dict] | None) -> str:
    """Format escalation hints for inclusion in LLM prompt."""
    if not hints:
        return ""
    lines = ["A classifier made tentative predictions for the following fields "
             "but was not confident enough. Verify or correct these:"]
    for h in hints:
        pct = round(h["confidence"] * 100, 1)
        lines.append(f"  - {h['path']}: tentatively \"{h['hint']}\" ({pct}% confidence)")
    return "\n".join(lines)


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
    hidden_provider: dict | None = None,
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
        hidden_provider: Config for hidden_probe rules, e.g. {"base_url": ...}.
            One prefill pass over (system_prompt, input_text) is shared by all
            probe rules. Probe training must use the same prompt convention.
    """
    import time as _time

    # Phase 0: One shared hidden-state fetch if any hidden_probe rules exist.
    # On failure, probes escalate to LLM — no new failure mode.
    hidden_states = None
    hidden_ms = 0.0
    probe_layers = sorted({r["layer"] for r in rules if r.get("type") == "hidden_probe"})
    if probe_layers:
        t0 = _time.perf_counter()
        try:
            hs = _fetch_hidden_states_remote(
                system=system_prompt, prompt=input_text,
                base_url=(hidden_provider or {}).get("base_url"),
                layers=probe_layers, positions="last_input",
            )
            pos = hs.positions[0]
            hidden_states = {layer: hs.vector(layer, pos) for layer in probe_layers}
        except Exception:
            hidden_states = None
        hidden_ms = (_time.perf_counter() - t0) * 1000

    # Phase 1: Execute rules
    extracted, rule_results = execute_rules(rules, input_text, hidden_states=hidden_states)
    rules_matched = sum(1 for r in rule_results if r.matched)

    # Collect escalation hints from classifier rules that were below threshold
    escalation_hints = [
        {"path": r.path, "hint": r.escalation_hint,
         "confidence": r.escalation_confidence}
        for r in rule_results
        if r.escalation_hint is not None
    ]

    # Phase 2: Resolve template vars in system prompt, build user prompt, call LLM
    resolved_system = system_prompt.replace(
        "{extracted_fields}", json.dumps(extracted, indent=2)
    ).replace("{input}", input_text).replace(
        "{llm_fields}", ", ".join(llm_fields)
    )
    prompt = build_hybrid_prompt(
        input_text, extracted, llm_fields, hybrid_prompt_template,
        escalation_hints=escalation_hints,
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
            hidden_ms=hidden_ms,
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
        hidden_ms=hidden_ms,
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
