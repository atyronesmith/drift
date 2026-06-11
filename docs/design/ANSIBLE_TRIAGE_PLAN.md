# Plan: Drift + Denoise for Ansible Log Analysis Quickstart

**Target**: [rh-ai-quickstart/ansible-log-analysis](https://github.com/rh-ai-quickstart/ansible-log-analysis)
**Corpus**: 103 real AAP failed job logs (`data/logs/failed/job_*.txt`), secrets already redacted
**Goal**: Reduce LLM calls from 4 per log cluster to 1, with measurable latency/token savings

## Background

The quickstart runs a 4-stage LangGraph pipeline per log cluster:

| Stage | Node | LLM? | Output |
|-------|------|-------|--------|
| 1. Embed & Cluster | `cluster_logs_node` | No (sentence-transformers + FAISS) | Grouped log clusters |
| 2. Summarize | `summarize_log_node` | Yes | `[Error Type]: [Brief description]` |
| 3. Classify | `classify_log_node` | Yes | 1 of 8 categories (Literal enum) |
| 4. Route | `router_step_by_step_solution_node` | Yes | "No More Context Needed" or "Need More Context" |
| 5. Solve | `suggest_step_by_step_solution_node` | Yes | Step-by-step remediation |

Stages 2-4 use structured output (Pydantic schemas: `SummarySchema`, `ClassifySchema`, `RouterStepByStepSolutionSchema`). Stage 5 is free-form text generation with optional context enrichment via MCP tools (Loki, cheat sheets).

The upstream truncates logs to the last 50 characters for embedding (`texts = [summary[-50:] for summary in logs]`). Full logs are passed to the summarizer.

### Why This is a Good Drift Target

- **Summarization** output follows a rigid `[Error Type]: [Brief description]` pattern. The prompt includes 8 concrete examples. Error types are a finite set (Variable Error, SSH Error, AWS API Error, etc.) that regex can match directly from log content.
- **Classification** maps to exactly 8 categories with keyword hints already documented in the prompt. The upstream prompt itself includes "keyword matching guidelines" per category.
- **Routing** is a binary decision (2 possible outputs) with explicit decision criteria that map to log completeness signals.
- **Solve** is genuine reasoning -- must stay as LLM.

Existing drift tasks `ansible_classify` and `ansible_route` already have system/user prompts adapted from this upstream. This plan extends them into a full pipeline with functional tests, hybrid compilation, and denoise integration.

---

## Phase 1: Data Acquisition & Task Setup

### Work

1. Copy the 103 log files into `tasks/ansible_triage/inputs/` as a multi-input corpus
2. Create three drift tasks mirroring the upstream stages:
   - `ansible_summarize` -- error log in, structured summary out
   - `ansible_classify` -- (exists) update with functional tests
   - `ansible_route` -- (exists) update with functional tests
3. Write `functional_tests.py` for each task

### Functional Tests

**ansible_summarize** (~10 tests):
- EXTRACTABLE: output matches `[Error Type]: [description]` format
- EXTRACTABLE: error type is from known set (Variable Error, SSH Error, AWS API Error, Repository Error, etc.)
- EXTRACTABLE: hostnames/IPs from input appear in summary when present
- EXTRACTABLE: file paths from input appear in summary when present
- EXTRACTABLE: empty/malformed logs produce "Empty log" summary
- REASONING: root cause description is accurate (spot-check against known logs)

**ansible_classify** (~8 tests):
- EXTRACTABLE: output is valid JSON with `category` key
- EXTRACTABLE: category is one of the 8 valid literals
- EXTRACTABLE: AWS/EC2 errors map to "Cloud Infrastructure / AWS Engineers"
- EXTRACTABLE: SSH/DNS errors map to "Networking / Security Engineers"
- EXTRACTABLE: Ansible variable/role errors map to "DevOps / CI/CD Engineers"
- EXTRACTABLE: subscription/repo errors map to "System Administrators / OS Engineers"
- EXTRACTABLE: k8s/operator errors map to "Kubernetes / OpenShift Cluster Admins"
- REASONING: ambiguous logs get reasonable classification

**ansible_route** (~6 tests):
- EXTRACTABLE: output is valid JSON with `suggestion` key
- EXTRACTABLE: output is one of 2 valid values
- EXTRACTABLE: logs with explicit root cause + component route to "No More Context Needed"
- EXTRACTABLE: empty/no_log/censored outputs route to "Need More Context"
- REASONING: edge cases with partial information get reasonable routing

### Experiment Configs

```yaml
# experiments/baseline-ansible-summarize.yaml
experiment:
  name: baseline-ansible-summarize
  model: qwen2.5:32b
  temperature: 0
  task: ansible_summarize
  system_prompt_file: system.md
  user_prompt_file: user.md
```

Same pattern for classify and route. Pipeline config chains all three.

### Deliverables
- `tasks/ansible_summarize/` with input.md, system.md, user.md, functional_tests.py
- Updated `tasks/ansible_classify/functional_tests.py`
- Updated `tasks/ansible_route/functional_tests.py`
- 3 baseline experiment YAMLs + 1 pipeline YAML

### Success Criteria
- All functional tests pass on baseline LLM runs
- EXTRACTABLE/REASONING split is documented per task

---

## Phase 2: Baseline Measurement

### Work

```bash
# Run baselines (10 runs each)
drift run experiments/baseline-ansible-summarize.yaml --runs 10
drift run experiments/baseline-ansible-classify.yaml --runs 10
drift run experiments/baseline-ansible-route.yaml --runs 10

# Measure
drift measure results/baseline-ansible-summarize/ -o metrics-summarize.json
drift measure results/baseline-ansible-classify/ -o metrics-classify.json
drift measure results/baseline-ansible-route/ -o metrics-route.json

# Probe for extractability
drift probe experiments/baseline-ansible-summarize.yaml -o probe-summarize.json
drift probe experiments/baseline-ansible-classify.yaml -o probe-classify.json
drift probe experiments/baseline-ansible-route.yaml -o probe-route.json
```

### Expected Findings

| Stage | Expected Extractability | Rationale |
|-------|------------------------|-----------|
| Summarize | 40-60% | Error type is extractable (finite set); description requires reasoning |
| Classify | 90%+ | 8 categories with keyword rules; prompt itself documents keyword matching |
| Route | 70-80% | Binary output; decision criteria map to signal presence/absence |
| Solve | <10% | Free-form remediation requires genuine reasoning |

### Deliverables
- Baseline latency per stage (expect ~2-5s each on qwen2.5:32b)
- Token consumption per stage
- Probe JSON files with per-token entropy analysis
- Functional test pass rates with bootstrap CIs

### Success Criteria
- Classification probe shows >85% deterministic tokens (entropy < 0.1)
- Route probe shows >60% deterministic tokens
- Baselines establish reproducible latency/accuracy floor

---

## Phase 3: Denoise Pre-digestion

### Work

1. Add Ansible log domain to denoise (may already exist from pod_log_triage work)
2. Run denoise on the 103 raw logs, measure token savings
3. Compare LLM summarization quality: raw input vs pre-digested input

```bash
# Check if domain exists
drift-denoise detect-domain data/logs/failed/job_1460308.txt

# Pre-digest all 103 logs
for f in tasks/ansible_triage/inputs/*.txt; do
  drift-denoise predigest "$f" -o "${f%.txt}.digest.txt"
done

# Measure savings
drift-denoise stats tasks/ansible_triage/inputs/
```

### Ansible Log Noise Profile

Ansible logs are extremely low-entropy. Common noise patterns:
- `ok:` / `changed:` / `skipping:` task status lines (bulk of output)
- `PLAY RECAP` blocks with all-zero counters
- `TASK [Gathering Facts]` boilerplate
- Dependency install stdout (pip, dnf, subscription-manager)
- Playbook path/role resolution messages

The actual failure signal is typically 5-20 lines in a log that may be thousands of lines long. The upstream already truncates to 5000 chars as a crude pre-digestion.

### Expected Results

| Metric | Expected Value | Basis |
|--------|---------------|-------|
| Token savings | 75-85% | Comparable to denoise's 83% on 500 Ansible logs |
| Quality delta | +5-10% functional accuracy | Pre-digestion removes noise that confuses summarization |
| Truncation elimination | 100% | Pre-digested logs fit without truncation |

### Deliverables
- Token savings report (per-log and aggregate)
- Side-by-side quality comparison on 20 representative logs
- Denoise domain config for Ansible AAP logs (if not already present)

### Success Criteria
- Token savings > 70%
- Functional test pass rate equal or better with pre-digested input
- No information loss for any of the 103 failure signals

---

## Phase 4: Hybrid Optimization

### Work

```bash
# Synthesize rules from probe data
drift synthesize probe-summarize.json --input tasks/ansible_summarize/input.md -o hybrid-summarize.yaml
drift synthesize probe-classify.json --input tasks/ansible_classify/input.md -o hybrid-classify.yaml
drift synthesize probe-route.json --input tasks/ansible_route/input.md -o hybrid-route.yaml

# Compile hybrid engines
drift compile hybrid-classify.yaml --prompt tasks/ansible_classify/user.md -o engines/ansible_classify.py
drift compile hybrid-route.yaml --prompt tasks/ansible_route/user.md -o engines/ansible_route.py

# Validate against corpus
drift validate engines/ansible_classify.py --input-dir tasks/ansible_triage/inputs/
drift validate engines/ansible_route.py --input-dir tasks/ansible_triage/inputs/
```

### Hybrid Strategy Per Stage

| Stage | Strategy | LLM Calls |
|-------|----------|-----------|
| Embed & Cluster | Already deterministic (no change) | 0 |
| Summarize | Hybrid: extract error type via regex, LLM for description | 1 (reduced scope) |
| Classify | Full replacement: keyword rules to 8 categories | 0 |
| Route | Full replacement: signal-presence heuristic | 0 |
| Solve | LLM only (no change) | 1 |

**Classification rules** (high confidence -- the upstream prompt already documents these):
- `ec2|cloudformation|aws_access_key|vpc|iam.*role|bastion` -> Cloud Infrastructure / AWS Engineers
- `cluster|operator|olm|pod|rbac|serviceaccount|openshift` -> Kubernetes / OpenShift Cluster Admins
- `ansible|playbook|runner|role|variable.*undefined|task.*include` -> DevOps / CI/CD Engineers
- `ssh|dns|certificate|tls|cert-manager|firewall|hostname` -> Networking / Security Engineers
- `subscription|repository|yum|dnf|package|registration` -> System Administrators / OS Engineers
- `gitea|argocd|helm|tssc|deployment.*failed` -> Application Developers / GitOps / Platform Engineers
- `azure.*ad|sso|authentication|token.*expired|login|federation` -> Identity & Access Management Engineers
- fallback -> Other / Miscellaneous

**Route rules** (binary decision based on signal completeness):
- Has explicit error message + specific component + identifiable cause -> "No More Context Needed"
- Empty log / no_log / censored / generic timeout -> "Need More Context"

### Expected Results

| Metric | Baseline (4 LLM calls) | Hybrid (1 LLM call) | Reduction |
|--------|------------------------|----------------------|-----------|
| LLM calls per cluster | 4 | 1 (solve only) | 75% |
| Latency per cluster | ~12-20s | ~3-5s | ~75% |
| Token consumption | ~4000-6000 | ~1000-1500 | ~75% |
| Classification accuracy | 100% (LLM) | 100% (rules) | No loss |
| Route accuracy | 100% (LLM) | 95%+ (rules) | Acceptable |

### Deliverables
- Hybrid YAML configs for classify and route
- Compiled engines (`engines/ansible_classify.py`, `engines/ansible_route.py`)
- Validation report: extraction rate on all 103 logs
- Accuracy comparison: hybrid vs LLM baseline

### Success Criteria
- Classification: 100% accuracy on 103 logs (keyword rules cover all categories)
- Route: >= 95% agreement with LLM baseline
- Summarize hybrid: error type extraction >= 90% accuracy
- Combined latency reduction >= 70%

---

## Phase 5: Safety Guard Analysis

### Work

Model the impact of enterprise safety guards on the original vs hybrid pipeline. This uses drift's existing security model projections from `security_models/*.yaml`.

```bash
drift safety-overhead --pipeline experiments/pipeline-ansible-triage.yaml
```

### Guard Overhead Model

The original pipeline makes 4 LLM calls per cluster. Each call in an enterprise deployment requires input + output guards:

| Security Model | Guards per LLM Call | Original (4 calls) | Hybrid (1 call) |
|----------------|--------------------|--------------------|-----------------|
| Baseline (Llama Guard only) | 2 (in + out) | 8 guard checks | 2 guard checks |
| LlamaFirewall (3-layer) | 6 (3 in + 3 out) | 24 guard checks | 6 guard checks |
| Microsoft Foundry | 4 (2 in + 2 out) | 16 guard checks | 4 guard checks |
| FINOS Regulated | 10 (5 in + 5 out) | 40 guard checks | 10 guard checks |

At 200-400ms per Llama Guard check:
- Original + baseline guards: 4 calls + 8 checks = **1.6-3.2s guard overhead**
- Hybrid + baseline guards: 1 call + 2 checks = **0.4-0.8s guard overhead**
- Guard overhead reduction: **75%** (mirrors call reduction)

At LlamaFirewall latency:
- Original: 24 checks x ~200ms = **4.8s guard overhead** (may exceed productive inference time)
- Hybrid: 6 checks x ~200ms = **1.2s guard overhead**

### The Compound Effect

The key finding: drift's hybrid optimization doesn't just save LLM compute -- it eliminates guard checks proportionally. Every LLM call eliminated also eliminates its entire guard stack. For heavily guarded pipelines, the guard savings may exceed the LLM savings.

| Component | Original | Hybrid | Savings |
|-----------|----------|--------|---------|
| Productive LLM calls | 4 | 1 | 75% |
| Guard checks (LlamaFirewall) | 24 | 6 | 75% |
| Guard latency | 4.8s | 1.2s | 3.6s |
| Total pipeline latency | ~17-25s | ~4-6s | ~75% |

### Deliverables
- Safety overhead projection for all 4 security models
- Compound savings visualization (LLM + guard stack)
- Comparison table for the safety-tax document

### Success Criteria
- Quantified guard overhead per security model
- Demonstrated that hybrid savings multiply through the safety stack

---

## Phase 6: Integration & Upstream

### Work

1. Package `ansible_triage` as a complete drift example with:
   - Task definitions, functional tests, experiment configs
   - Hybrid engines, validation results
   - Before/after metrics

2. Propose upstream PR to `rh-ai-quickstart/ansible-log-analysis`:
   - Add `--hybrid` flag to the pipeline
   - Rules engine replaces classify and route nodes
   - Denoise pre-digestion as optional preprocessing
   - Fall back to LLM for logs where rules don't match (confidence threshold)

3. Update `safety-tax` visualizer:
   - Add ansible-log-analysis as a pipeline case study
   - Show 4-call vs 1-call with guard stack overlay

### Upstream PR Structure

```
ansible-log-analysis/
  src/alm/hybrid/
    classify_rules.py    # compiled from drift
    route_rules.py       # compiled from drift
    summarize_rules.py   # error type extraction only
  src/alm/denoise/
    ansible_predigest.py # denoise integration
  src/alm/agents/
    node.py              # modified: hybrid path for classify/route
    graph.py             # modified: conditional hybrid nodes
```

### Deliverables
- Complete drift example in `tasks/ansible_triage/`
- PR to upstream repo with hybrid option
- Updated safety-tax visualization
- Entry in `docs/RESULTS.md` as a new experiment

### Success Criteria
- Upstream PR is reviewable and self-contained
- Hybrid mode is opt-in (existing behavior unchanged by default)
- All 103 logs produce equivalent results in hybrid mode

---

## Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|-------------|
| Phase 1: Task Setup | 1 day | None |
| Phase 2: Baseline | 1 day | Phase 1 |
| Phase 3: Denoise | 1 day | Phase 1 (parallel with Phase 2) |
| Phase 4: Hybrid | 2 days | Phase 2 |
| Phase 5: Safety Guards | 0.5 day | Phase 4 |
| Phase 6: Integration | 1.5 days | Phase 4, Phase 5 |
| **Total** | **~7 days** | |

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Classification rules miss edge cases | Low | The upstream prompt already documents keyword matching; 8 categories are well-separated |
| Route heuristic disagrees with LLM on ambiguous cases | Medium | Accept 95% threshold; ambiguous cases fall through to LLM |
| Denoise strips failure signal from logs | Low | Validate all 103 failure signals preserved; denoise is conservative by design |
| Upstream repo changes pipeline structure | Low | Pin to specific commit; plan works with current LangGraph architecture |
