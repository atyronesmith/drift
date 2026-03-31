"""
Drift analysis — prescriptive recommendations from metrics.

Takes metrics output from `drift measure` and produces actionable
recommendations: should you distill to code, keep the LLM, or fix
the pipeline before deploying?
"""

from __future__ import annotations


# Thresholds from THEORY.md distillation viability table
BANDS = [
    (0.99, "distill", "Distill to code. LLM adds cost without value for this task."),
    (0.95, "distill_with_fallback", "Distill the common path to deterministic code. Use LLM as fallback for edge cases."),
    (0.80, "keep_with_gates", "LLM is doing real reasoning. Keep it, but add review gates."),
    (0.00, "unreliable", "LLM is unreliable for this task. Improve the prompt, model, or task decomposition before deploying."),
]


def _classify_equivalence(rate: float) -> dict:
    """Classify a functional equivalence rate into a distillation band."""
    for threshold, band, recommendation in BANDS:
        if rate >= threshold:
            return {
                "band": band,
                "threshold": threshold,
                "recommendation": recommendation,
            }
    return {"band": "unknown", "threshold": 0, "recommendation": "No data."}


def _analyze_text_drift(metrics: dict) -> list[dict]:
    """Analyze text-level drift signals."""
    findings = []
    exact = metrics.get("exact_match_rate")
    num_unique = metrics.get("num_unique")
    num_outputs = metrics.get("num_outputs")
    mean_ed = metrics.get("mean_edit_distance")

    if exact is not None:
        if exact == 1.0:
            findings.append({
                "signal": "deterministic",
                "detail": f"100% exact match across {num_outputs} runs. Output is fully deterministic.",
            })
        elif exact >= 0.9:
            findings.append({
                "signal": "low_drift",
                "detail": f"{exact:.0%} exact match ({num_unique} unique variants). Minor textual variation.",
            })
        else:
            findings.append({
                "signal": "significant_drift",
                "detail": f"{exact:.0%} exact match ({num_unique} unique). Mean edit distance: {mean_ed:.4f}.",
            })

    length_cv = metrics.get("length_cv")
    if length_cv is not None and length_cv > 0.1:
        findings.append({
            "signal": "length_variance",
            "detail": f"Length CV {length_cv:.3f} — outputs vary significantly in size.",
        })

    return findings


def _analyze_functional(func_metrics: dict) -> dict:
    """Analyze functional equivalence and produce distillation recommendation."""
    rate = func_metrics.get("functional_equivalence_rate")
    if rate is None:
        return {"available": False}

    classification = _classify_equivalence(rate)

    # Find weak tests (pass rate < 1.0)
    weak_tests = []
    for name, info in func_metrics.get("per_test", {}).items():
        if info.get("pass_rate") is not None and info["pass_rate"] < 1.0:
            weak_tests.append({
                "test": name,
                "pass_rate": info["pass_rate"],
                "failures": info["fail"] + info["error"],
            })

    return {
        "available": True,
        "equivalence_rate": rate,
        "variants_all_pass": func_metrics["variants_all_pass"],
        "variants_with_bugs": func_metrics["variants_with_bugs"],
        "num_variants": func_metrics["num_variants"],
        "classification": classification,
        "weak_tests": weak_tests,
    }


def _analyze_pipeline_steps(metrics: dict) -> list[dict]:
    """Analyze per-step drift in pipeline results."""
    steps = []
    for step in metrics.get("per_step", []):
        entry = {"name": step.get("name", "unknown")}

        if step.get("type") == "loop":
            entry["type"] = "loop"
            entry["iteration_count_cv"] = step.get("iteration_count_cv")
            entry["termination_agreement_rate"] = step.get("termination_agreement_rate")
        else:
            entry["type"] = "step"
            entry["exact_match_rate"] = step.get("exact_match_rate")
            entry["num_unique"] = step.get("num_unique")
            entry["mean_edit_distance"] = step.get("mean_edit_distance")

        steps.append(entry)

    # Check for drift amplification across steps
    exact_rates = [s["exact_match_rate"] for s in steps
                   if s["type"] == "step" and s.get("exact_match_rate") is not None]
    if len(exact_rates) >= 2 and exact_rates[0] > exact_rates[-1]:
        amplification = exact_rates[0] - exact_rates[-1]
        steps.append({
            "name": "_drift_amplification",
            "type": "observation",
            "detail": f"Exact match drops {exact_rates[0]:.0%} → {exact_rates[-1]:.0%} across pipeline. "
                      f"Drift amplifies by {amplification:.0%} through the chain.",
        })

    return steps


def _analyze_probe(probe_data: dict) -> dict | None:
    """Analyze probe data to add entropy-based recommendations."""
    stats = probe_data.get("stats", {})
    if not stats:
        return None

    pct_det = stats.get("pct_deterministic", 0)
    mean_entropy = stats.get("mean_entropy", 0)
    num_tokens = stats.get("num_tokens", 0)

    if pct_det >= 0.95:
        assessment = "rule_extractable"
        detail = (f"{pct_det:.0%} of tokens are deterministic (mean entropy {mean_entropy:.3f} bits). "
                  f"This task can be replaced with a parser/template.")
    elif pct_det >= 0.80:
        assessment = "partially_extractable"
        detail = (f"{pct_det:.0%} deterministic tokens. Core extraction is rule-based; "
                  f"stochastic tokens are formatting choices a schema would resolve.")
    elif pct_det >= 0.50:
        assessment = "mixed"
        detail = (f"{pct_det:.0%} deterministic tokens. Output mixes extraction with "
                  f"reasoning/generation. Schema constraints would help but can't fully replace the LLM.")
    else:
        assessment = "llm_required"
        detail = (f"Only {pct_det:.0%} deterministic tokens (mean entropy {mean_entropy:.3f} bits). "
                  f"Output requires LLM reasoning/generation.")

    return {
        "assessment": assessment,
        "pct_deterministic": pct_det,
        "mean_entropy": mean_entropy,
        "num_tokens": num_tokens,
        "high_entropy_count": len(probe_data.get("high_entropy_tokens", [])),
        "detail": detail,
    }


def analyze_metrics(metrics: dict, probe_data: dict | None = None) -> dict:
    """Produce a full analysis from metrics output.

    Returns a structured report with findings and recommendations.
    """
    report = {"type": metrics.get("type", "unknown")}

    # Text drift analysis
    if metrics.get("type") == "pipeline":
        report["text_drift"] = _analyze_text_drift(metrics.get("final", {}))
        report["pipeline_steps"] = _analyze_pipeline_steps(metrics)
    else:
        report["text_drift"] = _analyze_text_drift(metrics)

    # Functional equivalence
    func = metrics.get("functional")
    if func:
        report["functional"] = _analyze_functional(func)
    else:
        report["functional"] = {"available": False}

    # Probe / entropy analysis
    if probe_data:
        probe_analysis = _analyze_probe(probe_data)
        if probe_analysis:
            report["probe"] = probe_analysis

    # Top-level verdict
    report["verdict"] = _produce_verdict(report)

    return report


def _produce_verdict(report: dict) -> dict:
    """Synthesize findings into a single verdict."""
    func = report.get("functional", {})
    text_findings = report.get("text_drift", [])

    if func.get("available"):
        classification = func["classification"]
        verdict = {
            "band": classification["band"],
            "recommendation": classification["recommendation"],
            "confidence": "high" if func["num_variants"] >= 20 else "medium" if func["num_variants"] >= 5 else "low",
            "sample_size": func["num_variants"],
        }

        if func["weak_tests"]:
            worst = min(func["weak_tests"], key=lambda t: t["pass_rate"])
            verdict["warning"] = (
                f"Test '{worst['test']}' has {worst['pass_rate']:.0%} pass rate "
                f"({worst['failures']} failures). Investigate before distilling."
            )

        # Enhance with probe data
        probe = report.get("probe")
        if probe and classification["band"] == "distill":
            if probe["assessment"] == "rule_extractable":
                verdict["probe_detail"] = (
                    f"Probe confirms: {probe['pct_deterministic']:.0%} of tokens are deterministic. "
                    f"Only {probe['high_entropy_count']} tokens need schema constraints."
                )

        return verdict

    # No functional data — fall back to text-level heuristics
    deterministic = any(f["signal"] == "deterministic" for f in text_findings)
    if deterministic:
        return {
            "band": "likely_distillable",
            "recommendation": "Output is textually deterministic. Run functional tests to confirm distillation viability.",
            "confidence": "low",
            "note": "No functional tests available — recommendation is based on text metrics only.",
        }

    significant = any(f["signal"] == "significant_drift" for f in text_findings)
    if significant:
        return {
            "band": "needs_functional_tests",
            "recommendation": "Significant textual drift detected but functional impact is unknown. Add functional tests to determine if drift is cosmetic or meaningful.",
            "confidence": "low",
            "note": "No functional tests available — cannot distinguish cosmetic from functional drift.",
        }

    return {
        "band": "insufficient_data",
        "recommendation": "Add functional tests to assess distillation viability.",
        "confidence": "low",
    }


def format_report(report: dict) -> str:
    """Format analysis report as readable text."""
    lines = []
    lines.append("DRIFT ANALYSIS REPORT")
    lines.append("=" * 50)

    # Verdict
    verdict = report.get("verdict", {})
    band = verdict.get("band", "unknown")
    lines.append("")
    lines.append(f"  Verdict:        {band}")
    lines.append(f"  Confidence:     {verdict.get('confidence', 'unknown')}")
    lines.append(f"  Sample size:    {verdict.get('sample_size', 'n/a')}")
    lines.append("")
    lines.append(f"  {verdict.get('recommendation', '')}")
    if verdict.get("warning"):
        lines.append(f"  WARNING: {verdict['warning']}")
    if verdict.get("note"):
        lines.append(f"  Note: {verdict['note']}")

    # Text drift
    lines.append("")
    lines.append("TEXT DRIFT")
    lines.append("-" * 50)
    for f in report.get("text_drift", []):
        lines.append(f"  [{f['signal']}] {f['detail']}")
    if not report.get("text_drift"):
        lines.append("  No text drift data.")

    # Functional equivalence
    func = report.get("functional", {})
    if func.get("available"):
        lines.append("")
        lines.append("FUNCTIONAL EQUIVALENCE")
        lines.append("-" * 50)
        lines.append(f"  Rate:           {func['equivalence_rate']:.1%}")
        lines.append(f"  Pass/Total:     {func['variants_all_pass']}/{func['num_variants']}")
        lines.append(f"  With bugs:      {func['variants_with_bugs']}")
        if func.get("weak_tests"):
            lines.append("  Weak tests:")
            for t in func["weak_tests"]:
                lines.append(f"    - {t['test']}: {t['pass_rate']:.0%} ({t['failures']} failures)")

    # Probe / entropy analysis
    probe = report.get("probe")
    if probe:
        lines.append("")
        lines.append("TOKEN ENTROPY (probe)")
        lines.append("-" * 50)
        lines.append(f"  Assessment:     {probe['assessment']}")
        lines.append(f"  Deterministic:  {probe['pct_deterministic']:.0%} of {probe['num_tokens']} tokens")
        lines.append(f"  Mean entropy:   {probe['mean_entropy']:.3f} bits")
        lines.append(f"  High entropy:   {probe['high_entropy_count']} tokens")
        lines.append(f"  {probe['detail']}")

    # Pipeline steps
    steps = report.get("pipeline_steps", [])
    if steps:
        lines.append("")
        lines.append("PIPELINE STEPS")
        lines.append("-" * 50)
        for s in steps:
            if s["type"] == "observation":
                lines.append(f"  {s['detail']}")
            elif s["type"] == "loop":
                lines.append(f"  {s['name']}: loop (iter CV: {s.get('iteration_count_cv', 'n/a')}, "
                           f"termination agreement: {s.get('termination_agreement_rate', 'n/a')})")
            else:
                lines.append(f"  {s['name']}: {s.get('exact_match_rate', 0):.0%} exact match, "
                           f"{s.get('num_unique', '?')} unique")

    lines.append("")
    return "\n".join(lines)
