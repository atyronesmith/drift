"""Per-field edit distance analysis for structured pipeline outputs.

Answers the critic's question: does JSON schema actually reduce variance
in the reasoning fields, or just dilute it with boilerplate?

Compares:
- Aggregate ED (whole response)
- Structural ED (JSON keys, brackets, fixed tokens)
- Per-field ED (each field individually, especially variable ones like description/fix/summary)
"""

import json
import sys
from itertools import combinations
from pathlib import Path


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein edit distance normalized to [0, 1]."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0
    # Standard DP
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb] / max(la, lb)


def load_runs(results_dir: Path) -> list[dict]:
    """Load all run JSON files from a results directory."""
    runs = []
    for f in sorted(results_dir.glob("run_*.json")):
        runs.append(json.loads(f.read_text()))
    return runs


def get_step_content(run: dict, step_name: str) -> str:
    """Get the raw content from a specific step."""
    for step in run["steps"]:
        if step["name"] == step_name:
            return step["response"]["content"]
    return ""


def parse_json_content(content: str) -> dict | None:
    """Parse JSON from LLM response, handling code fences."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def mean_pairwise_ed(strings: list[str]) -> float:
    """Compute mean pairwise normalized edit distance."""
    if len(strings) < 2:
        return 0.0
    dists = [normalized_edit_distance(a, b) for a, b in combinations(strings, 2)]
    return sum(dists) / len(dists)


def analyze_reviewer_fields(runs: list[dict]) -> dict:
    """Analyze per-field ED for the reviewer step (JSON issues array)."""
    contents = [get_step_content(r, "reviewer") for r in runs]
    parsed = [parse_json_content(c) for c in contents]
    parsed = [p for p in parsed if p is not None]

    if not parsed:
        return {"error": "no valid JSON outputs"}

    # Aggregate ED on full content
    aggregate_ed = mean_pairwise_ed(contents)

    # Per-issue analysis: align by index
    max_issues = max(len(p.get("issues", [])) for p in parsed)

    field_eds = {}
    for field in ["severity", "line", "category", "description", "fix"]:
        for idx in range(max_issues):
            key = f"issues[{idx}].{field}"
            values = []
            for p in parsed:
                issues = p.get("issues", [])
                if idx < len(issues) and field in issues[idx]:
                    values.append(str(issues[idx][field]))
                else:
                    values.append("")
            field_eds[key] = {
                "mean_ed": round(mean_pairwise_ed(values), 4),
                "unique": len(set(values)),
                "total": len(values),
                "sample": values[0][:80] if values else "",
            }

    # Classify fields
    structural_fields = []  # severity, line, category
    reasoning_fields = []   # description, fix
    for key, data in field_eds.items():
        if any(f in key for f in [".description", ".fix"]):
            reasoning_fields.append((key, data))
        else:
            structural_fields.append((key, data))

    structural_ed = sum(d["mean_ed"] for _, d in structural_fields) / max(len(structural_fields), 1)
    reasoning_ed = sum(d["mean_ed"] for _, d in reasoning_fields) / max(len(reasoning_fields), 1)

    # Number of issues found per run
    issue_counts = [len(p.get("issues", [])) for p in parsed]

    return {
        "aggregate_ed": round(aggregate_ed, 4),
        "structural_fields_ed": round(structural_ed, 4),
        "reasoning_fields_ed": round(reasoning_ed, 4),
        "issue_counts": issue_counts,
        "issue_count_variance": round(max(issue_counts) - min(issue_counts), 0),
        "fields": field_eds,
    }


def analyze_verifier_fields(runs: list[dict]) -> dict:
    """Analyze per-field ED for the verifier step (JSON status/summary)."""
    contents = [get_step_content(r, "verifier") for r in runs]
    parsed = [parse_json_content(c) for c in contents]
    parsed = [p for p in parsed if p is not None]

    if not parsed:
        return {"error": "no valid JSON outputs"}

    aggregate_ed = mean_pairwise_ed(contents)

    field_eds = {}
    for field in ["status", "summary"]:
        values = [str(p.get(field, "")) for p in parsed]
        field_eds[field] = {
            "mean_ed": round(mean_pairwise_ed(values), 4),
            "unique": len(set(values)),
            "total": len(values),
            "sample": values[0][:100] if values else "",
        }

    # remaining_issues as serialized string
    ri_values = [json.dumps(p.get("remaining_issues", [])) for p in parsed]
    field_eds["remaining_issues"] = {
        "mean_ed": round(mean_pairwise_ed(ri_values), 4),
        "unique": len(set(ri_values)),
        "total": len(ri_values),
    }

    structural_ed = (field_eds["status"]["mean_ed"] + field_eds["remaining_issues"]["mean_ed"]) / 2
    reasoning_ed = field_eds["summary"]["mean_ed"]

    return {
        "aggregate_ed": round(aggregate_ed, 4),
        "structural_fields_ed": round(structural_ed, 4),
        "reasoning_fields_ed": round(reasoning_ed, 4),
        "fields": field_eds,
    }


def analyze_free_form(runs: list[dict], step_name: str) -> dict:
    """Analyze ED for a free-form (non-JSON) step."""
    contents = [get_step_content(r, step_name) for r in runs]
    return {
        "aggregate_ed": round(mean_pairwise_ed(contents), 4),
        "unique": len(set(contents)),
        "total": len(contents),
    }


def main():
    base = Path(__file__).parent.parent / "results"

    # Structured pipeline (JSON sandwich)
    structured_dir = base / "pipeline-structured" / "20260407T175205Z"
    # Homogeneous pipeline (free-form)
    homogeneous_dir = base / "pipeline-homogeneous" / "20260306T180616Z"

    print("=" * 70)
    print("PER-FIELD EDIT DISTANCE ANALYSIS")
    print("Does JSON schema reduce reasoning field variance,")
    print("or just dilute it with boilerplate?")
    print("=" * 70)

    # Load data
    structured_runs = load_runs(structured_dir)
    homogeneous_runs = load_runs(homogeneous_dir)

    print(f"\nStructured pipeline: {len(structured_runs)} runs from {structured_dir.name}")
    print(f"Homogeneous pipeline: {len(homogeneous_runs)} runs from {homogeneous_dir.name}")

    # --- REVIEWER STEP ---
    print("\n" + "=" * 70)
    print("STEP 1: REVIEWER")
    print("=" * 70)

    print("\n--- Structured (JSON) ---")
    sr = analyze_reviewer_fields(structured_runs)
    print(f"Aggregate ED:          {sr['aggregate_ed']}")
    print(f"Structural fields ED:  {sr['structural_fields_ed']}  (severity, line, category)")
    print(f"Reasoning fields ED:   {sr['reasoning_fields_ed']}  (description, fix)")
    print(f"Issue counts:          {sr['issue_counts']}")
    print(f"\nPer-field breakdown:")
    for key, data in sr["fields"].items():
        field_type = "REASONING" if any(f in key for f in [".description", ".fix"]) else "structural"
        print(f"  {key:30s}  ED={data['mean_ed']:.4f}  unique={data['unique']}/{data['total']}  [{field_type}]")

    print("\n--- Homogeneous (free-form) ---")
    hr = analyze_free_form(homogeneous_runs, "reviewer")
    print(f"Aggregate ED:          {hr['aggregate_ed']}")

    print(f"\n>>> Structured reasoning ED ({sr['reasoning_fields_ed']:.4f}) vs "
          f"free-form aggregate ED ({hr['aggregate_ed']:.4f})")
    if sr['reasoning_fields_ed'] < hr['aggregate_ed']:
        print(f"    Reasoning fields ARE less variable in structured output.")
        print(f"    Reduction: {(1 - sr['reasoning_fields_ed'] / hr['aggregate_ed']) * 100:.1f}%")
        print(f"    This is NOT just boilerplate dilution.")
    else:
        print(f"    Reasoning fields are NOT less variable — the critic is right.")
        print(f"    The structured ED reduction IS boilerplate dilution.")

    # --- VERIFIER STEP ---
    print("\n" + "=" * 70)
    print("STEP 3: VERIFIER")
    print("=" * 70)

    print("\n--- Structured (JSON) ---")
    sv = analyze_verifier_fields(structured_runs)
    print(f"Aggregate ED:          {sv['aggregate_ed']}")
    print(f"Structural fields ED:  {sv['structural_fields_ed']}  (status, remaining_issues)")
    print(f"Reasoning fields ED:   {sv['reasoning_fields_ed']}  (summary)")
    print(f"\nPer-field breakdown:")
    for key, data in sv["fields"].items():
        field_type = "REASONING" if key == "summary" else "structural"
        print(f"  {key:30s}  ED={data['mean_ed']:.4f}  unique={data['unique']}/{data['total']}  [{field_type}]")

    print("\n--- Homogeneous (free-form) ---")
    hv = analyze_free_form(homogeneous_runs, "verifier")
    print(f"Aggregate ED:          {hv['aggregate_ed']}")

    print(f"\n>>> Structured reasoning ED ({sv['reasoning_fields_ed']:.4f}) vs "
          f"free-form aggregate ED ({hv['aggregate_ed']:.4f})")
    if sv['reasoning_fields_ed'] < hv['aggregate_ed']:
        print(f"    Reasoning fields ARE less variable in structured output.")
        print(f"    Reduction: {(1 - sv['reasoning_fields_ed'] / hv['aggregate_ed']) * 100:.1f}%")
    else:
        print(f"    Reasoning fields are NOT less variable — the critic is right.")

    # --- SUMMARY TABLE ---
    print("\n" + "=" * 70)
    print("SUMMARY: BOILERPLATE DILUTION TEST")
    print("=" * 70)
    print(f"\n{'Metric':<40s} {'Reviewer':>10s} {'Verifier':>10s}")
    print("-" * 62)
    print(f"{'Free-form aggregate ED':<40s} {hr['aggregate_ed']:>10.4f} {hv['aggregate_ed']:>10.4f}")
    print(f"{'Structured aggregate ED':<40s} {sr['aggregate_ed']:>10.4f} {sv['aggregate_ed']:>10.4f}")
    print(f"{'Structured structural fields ED':<40s} {sr['structural_fields_ed']:>10.4f} {sv['structural_fields_ed']:>10.4f}")
    print(f"{'Structured reasoning fields ED':<40s} {sr['reasoning_fields_ed']:>10.4f} {sv['reasoning_fields_ed']:>10.4f}")
    print()
    print("If structured reasoning ED < free-form aggregate ED,")
    print("then the schema constrains reasoning variance, not just dilutes it.")
    print()
    print("If structured reasoning ED >= free-form aggregate ED,")
    print("then the critic is right: the ED drop is boilerplate dilution.")

    # Save results
    results = {
        "reviewer": {
            "structured": sr,
            "free_form": hr,
        },
        "verifier": {
            "structured": sv,
            "free_form": hv,
        },
    }
    out_path = base / "per_field_ed_analysis.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
