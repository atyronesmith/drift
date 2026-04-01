# Design: Automatic Logic Engine Generation

## Vision

Given an LLM prompt and sample inputs, automatically generate an optimized execution engine that produces equivalent output using the minimum necessary LLM calls. The engine combines regex extraction, lookup tables, conditional logic, and targeted LLM calls into a single executable artifact.

```
  TODAY:

  Input ──────────────────────────▶ LLM ──────────────────▶ Output
                                    21s
                                    100% LLM cost
                                    all fields exposed to injection


  GOAL:

                    ┌──▶ Regex rules ──────▶ 6 fields ──┐
                    │    (< 1ms)                         │
  Input ──────┬─────┤                                    │
              │     ├──▶ Lookup tables ────▶ 1 field  ──┼──▶ merge ──▶ Output
              │     │    (< 1ms)                         │
              │     └──▶ Conditionals ─────▶ N fields ──┤     7s
              │          (< 1ms)                         │     30% LLM cost
              │                                          │     only reasoning
              └──────▶ LLM (focused) ──────▶ M fields ──┘     fields exposed
```

The engine is generated automatically. No human writes rules.

## The Compiler Pipeline

```
  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
  │ Step 1      │     │ Step 2       │     │ Step 3       │     │ Step 4    │
  │ PROBE       │────▶│ MULTI-PROBE  │────▶│ CLASSIFY     │────▶│ COMPILE   │
  │             │     │              │     │              │     │           │
  │ 1 input     │     │ N inputs     │     │ Tier fields  │     │ Generate  │
  │ Get entropy │     │ Find patterns│     │ into 3 tiers │     │ engine    │
  └─────────────┘     └──────────────┘     └──────────────┘     └───────────┘

  drift probe          drift probe          drift compile        drift compile
  config.yaml          config.yaml          (internal)           -o engine.py
  -o probe.json        --input ticket_N                          or engine.yaml
                       (run N times)
```

### Step 1: Single-Ticket Probe (existing)

Run the LLM once with logprobs. Measure per-token entropy. Identify which output fields are deterministic (copying from input) vs stochastic (reasoning).

### Step 2: Multi-Ticket Probe (new)

Run the LLM on N diverse inputs (10-50 tickets). For each field:
- Collect the output value and its entropy
- Record the input features (priority, category, etc.) that were present
- Build a matrix: `(input_features) → output_value`

### Step 3: Field Classification (new)

Classify each output field into one of three tiers:

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │ TIER 1: Deterministic Extraction                                    │
  │                                                                     │
  │ Signal: Low entropy on every ticket. Value found in input text.     │
  │ Rule:   regex                                                       │
  │ Example: ticket.id, ticket.subject, requester_email                 │
  │ Cost:   0 (no LLM)                                                  │
  └──────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │ TIER 2: Conditionally Deterministic                                 │
  │                                                                     │
  │ Signal: High entropy per ticket, but consistent output when         │
  │         grouped by input features. e.g., Medium→P3 always.          │
  │ Rule:   mapping or conditional (case statement)                     │
  │ Example: severity, assignment_group                                 │
  │ Cost:   0 (no LLM)                                                  │
  └──────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │ TIER 3: True Reasoning                                              │
  │                                                                     │
  │ Signal: High entropy per ticket AND varies across tickets with      │
  │         the same input features. Output depends on free-text        │
  │         content, not structured fields.                             │
  │ Rule:   LLM call (no substitute)                                    │
  │ Example: summary, root_cause_analysis, next_steps                   │
  │ Cost:   Full LLM inference (but only for these fields)              │
  └──────────────────────────────────────────────────────────────────────┘
```

### Step 4: Compile (new)

Generate an executable artifact from the classified fields.

## Output Format: Code vs Rules Engine vs YAML

### Option A: YAML Spec + Runtime (current)

```yaml
rules:
  - path: ticket.id
    type: regex
    pattern: "Ticket\\ ID:\\s*(\\S+)\\s*$"
  - path: assessment.severity
    type: mapping
    source: ticket.stated_priority
    map: { High: P2, Medium: P3 }

llm_fields:
  - assessment.summary
  - assessment.next_steps
```

Executed by `drift run` with the hybrid executor.

| Pros | Cons |
|------|------|
| Readable, inspectable | Limited expressiveness |
| Easy to hand-edit | Requires custom runtime to execute |
| Portable across languages | Can't express complex logic (loops, string transforms) |
| Good for prototyping | YAML escaping is fragile for regex |

### Option B: Rules Engine (Drools, Rete, etc.)

```drools
rule "Extract Ticket ID"
  when
    $input : Input( text matches ".*Ticket ID:\\s*(\\S+).*" )
  then
    insert(new Field("ticket.id", extract($input, "Ticket ID:\\s*(\\S+)")));
end

rule "Classify Severity"
  when
    $priority : Field( path == "ticket.stated_priority" )
  then
    if ($priority.value.equals("High"))
      insert(new Field("assessment.severity", "P2"));
end
```

| Pros | Cons |
|------|------|
| Industry standard for rule systems | JVM dependency (Drools) |
| Forward chaining, conflict resolution | Python alternatives are immature |
| Good for very complex rule interactions | Overkill for extraction + classification |
| Tooling ecosystem (rule editors, debuggers) | Learning curve, heavy runtime |

Python rules engines exist (`durable_rules`, `business-rules`, `experta`) but none are widely adopted or well-maintained.

### Option C: Generated Python Code (recommended)

```python
"""Auto-generated engine for helpdesk triage.

Generated by: drift compile
Source: 39 probed tickets, 6 regex rules, 1 conditional, 4 LLM fields
"""

import re
import json


def process(input_text: str, llm=None) -> dict:
    """Process a helpdesk ticket. Returns structured triage output."""
    result = {}

    # --- Tier 1: Regex extraction (< 1ms) ---
    result["ticket"] = {
        "id": _extract(r"Ticket ID:\s*(\S+)", input_text),
        "subject": _extract(r"Subject:\s*(.+?)$", input_text),
        "requester_email": _extract(r"From:\s*([\w.+-]+@[\w.-]+)", input_text),
        "created_at": _extract(r"Created:\s*(\S+)", input_text),
        "stated_priority": _extract(r"Priority:\s*(\S+)", input_text),
        "stated_category": _extract(r"Category:\s*(\S+)", input_text),
    }

    # --- Tier 2: Conditional logic (< 1ms) ---
    priority = result["ticket"]["stated_priority"]
    category = result["ticket"]["stated_category"]

    if priority == "High" and category == "Account":
        severity = "P3"
    elif priority == "High":
        severity = "P2"
    elif priority == "Medium":
        severity = "P3"
    elif priority == "Low":
        severity = "P4"
    elif priority == "Urgent":
        severity = "P1"
    else:
        severity = "P3"

    # --- Tier 3: LLM for reasoning fields only ---
    if llm:
        llm_result = llm.complete(
            system="You are an IT helpdesk triage system.",
            prompt=_build_focused_prompt(input_text, result, severity),
        )
        assessment = json.loads(llm_result)
    else:
        assessment = {}

    result["assessment"] = {
        "severity": severity,
        **assessment,
    }

    return result


def _extract(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _build_focused_prompt(input_text, extracted, severity):
    return (
        f"Ticket metadata (already extracted):\n"
        f"{json.dumps(extracted, indent=2)}\n\n"
        f"Severity has been classified as {severity}.\n\n"
        f"Based on the ticket below, provide ONLY these fields as JSON:\n"
        f"- root_cause_category\n"
        f"- assignment_group\n"
        f"- summary (1-2 sentences)\n"
        f"- next_steps (list of 1-3 actions)\n\n"
        f"{input_text}"
    )
```

| Pros | Cons |
|------|------|
| No runtime dependency — it's just Python | Code generation is harder to get right |
| Full expressiveness (any logic) | Generated code needs review before production |
| Testable with pytest, type-checkable | Regex escaping in string literals |
| Debuggable with standard tools | |
| Version-controllable (git diff shows changes) | |
| Can be optimized further by hand | |
| Single file, self-contained | |

**This is the recommended approach.** The generated Python module has a `process()` function that takes input text and an optional LLM provider. It runs all tiers in order and returns the complete output. No custom runtime, no YAML parsing, no rules engine dependency.

## Why Generated Code Wins

The YAML spec was right for prototyping and proving the concept. But as rules get more complex (conditionals, multi-field dependencies, string transforms), YAML becomes a worse fit:

```
  Complexity spectrum:

  regex ──────── mapping ──────── conditional ──────── decision tree ──────── code
  ◄── YAML works fine here ──►   ◄── YAML gets awkward ──►   ◄── need code ──►
```

A Drools-style rules engine solves the complexity problem but adds a heavy dependency for what is fundamentally a compilation step. We don't need forward chaining or conflict resolution — we need sequential extraction, then lookup, then LLM. A Python function is the natural expression of that.

The generated code is also **the artifact that gets deployed**. It's not an intermediate representation that needs a runtime — it IS the runtime. This makes deployment simpler: copy one .py file, provide an LLM client, call `process()`.

## Implementation Plan

### Phase 1: Multi-Ticket Synthesizer

Extend `drift synthesize` to accept multiple probe results and discover conditional patterns.

```bash
# Probe N tickets
for i in $(seq 0 19); do
  drift probe config.yaml --input tickets/ticket_$i.txt -o probes/probe_$i.json
done

# Synthesize from all probes
drift synthesize probes/ --input-dir tickets/ -o hybrid.yaml
```

Changes:
- `synthesize_rules()` accepts a list of `(probe_data, input_text)` pairs
- Groups fields by output value across tickets
- Detects conditional determinism (same value given same input features)
- Emits mapping/conditional rules alongside regex rules

### Phase 2: Code Generator

New command: `drift compile`

```bash
drift compile hybrid.yaml --prompt triage_system.md -o engine.py
```

Takes the hybrid spec (with all three tiers of rules) and generates a Python module with:
- `process(input_text, llm=None) -> dict` — main entry point
- `_extract()` helpers for regex
- Conditional logic as plain if/elif
- A focused LLM prompt for remaining fields
- Docstring with provenance (which tickets were probed, confidence stats)

### Phase 3: Validation Harness

```bash
# Generate engine from 20 training tickets
drift compile hybrid.yaml --prompt triage_system.md -o engine.py

# Validate against 100 test tickets
drift validate engine.py --input-dir test_tickets/ --expected-fields ticket,assessment
```

Runs the generated engine against held-out tickets and reports:
- Field accuracy (do regex rules still match?)
- Tier 2 accuracy (do conditionals agree with what the LLM would have said?)
- Overall equivalence (is engine output ≈ pure-LLM output?)

### Phase 4: Iterative Refinement

The compiler should support re-running on more data to improve rules:

```bash
# Initial compile from 20 tickets
drift compile --from-probes probes_v1/ -o engine_v1.py

# Validate, find failures
drift validate engine_v1.py --input-dir new_tickets/

# Re-probe failures + new tickets, recompile
drift compile --from-probes probes_v2/ -o engine_v2.py
```

Each iteration potentially moves more fields from Tier 3 (LLM) to Tier 2 (conditional) or discovers new regex patterns, reducing LLM dependency further.

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                        drift compile                             │
  │                                                                  │
  │  Inputs:                          Output:                        │
  │  ┌──────────────┐                 ┌──────────────────────────┐  │
  │  │ probe_1.json │                 │ engine.py                │  │
  │  │ probe_2.json │                 │                          │  │
  │  │ ...          │──── compile ───▶│ def process(input, llm): │  │
  │  │ probe_N.json │                 │     # Tier 1: regex      │  │
  │  │              │                 │     # Tier 2: conditions │  │
  │  │ prompt.md    │                 │     # Tier 3: llm call   │  │
  │  │ hybrid.yaml  │                 │     return merged        │  │
  │  └──────────────┘                 └──────────────────────────┘  │
  │                                                                  │
  │  The generated engine.py is self-contained.                      │
  │  It has no dependency on drift — just re, json, and an           │
  │  LLM client for Tier 3 fields.                                   │
  │                                                                  │
  │  Deploying the engine = copying one file.                        │
  └─────────────────────────────────────────────────────────────────┘
```

## Cost Model

For a helpdesk triage task (12 output fields):

| Approach | LLM calls | Fields via LLM | Latency | Cost |
|----------|-----------|----------------|---------|------|
| Pure baseline | 1 (full) | 12/12 | 21s | 100% |
| Hybrid (regex only) | 1 (focused) | 6/12 | 13s | ~60% |
| Hybrid (regex + conditional) | 1 (focused) | 4/12 | ~10s | ~40% |
| Fully compiled engine | 1 (minimal) | 4/12 | ~8s | ~30% |

The compiled engine's LLM call is smaller because the prompt only asks for 4 fields instead of 12, and it includes the already-extracted context so the LLM doesn't need to re-read the input for metadata.

## Open Questions

1. **How many probe tickets are enough?** The conditional severity mapping was clean with 39 tickets (3 per group). For more complex mappings, we may need more. Need to define a convergence criterion.

2. **What about fields that are ~90% deterministic?** Severity was 92% conditionally deterministic (1 outlier in 39). Should the engine include a confidence threshold — use the rule if confident, fall back to LLM if not?

3. **Versioning.** When the input format changes (new fields, different labels), the generated engine breaks. Need a way to detect this and trigger recompilation.

4. **Testing the generated code.** The engine needs tests. Should `drift compile` also generate a test file? Or should validation be a separate step?

5. **Multi-LLM-call engines.** Some fields might need different prompts or different models. The current design assumes one LLM call for all Tier 3 fields. Could split into multiple focused calls.
