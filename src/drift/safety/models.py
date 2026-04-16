"""Security model evaluation for multi-agent safety overhead projection.

Loads security model definitions (YAML) and computes projected guard
overhead for a given pipeline shape. Supports comparison across models
(e.g., Microsoft Azure Foundry vs FINOS Regulated vs LlamaFirewall).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GuardCheck:
    name: str
    latency_ms: float
    type: str  # llm, classifier, deterministic, crypto, static_analysis
    description: str = ""


@dataclass
class SecurityModel:
    name: str
    version: str
    description: str
    source: str
    checkpoints: dict[str, list[GuardCheck]]

    @classmethod
    def from_yaml(cls, path: Path) -> SecurityModel:
        data = yaml.safe_load(path.read_text())
        checkpoints = {}
        for cp_name, checks in data.get("checkpoints", {}).items():
            checkpoints[cp_name] = [
                GuardCheck(
                    name=c["name"],
                    latency_ms=c["latency_ms"],
                    type=c["type"],
                    description=c.get("description", ""),
                )
                for c in checks
            ]
        return cls(
            name=data["name"],
            version=data.get("version", ""),
            description=data.get("description", ""),
            source=data.get("source", ""),
            checkpoints=checkpoints,
        )


@dataclass
class PipelineShape:
    """Describes a pipeline's shape for overhead projection."""
    steps: int = 3
    code_steps: int = 1  # steps that produce code (trigger code_output checks)
    tool_calls_per_step: float = 0  # avg MCP/tool calls per step
    iterations: int = 1  # loop iterations (multiplies everything)
    human_approval_steps: int = 0  # steps requiring human sign-off


@dataclass
class OverheadProjection:
    """Result of projecting guard overhead for a model + pipeline."""
    model_name: str
    pipeline: PipelineShape
    checks_per_iteration: int
    total_checks: int
    latency_per_iteration_ms: float
    total_latency_ms: float
    breakdown: list[dict]  # per-checkpoint-type breakdown
    llm_checks: int  # checks requiring LLM inference
    deterministic_checks: int  # fast deterministic checks
    cost_per_month_1m: float  # rough cost at 1M pipeline runs


def project_overhead(model: SecurityModel, pipeline: PipelineShape) -> OverheadProjection:
    """Project total guard overhead for a security model applied to a pipeline."""
    breakdown = []
    total_checks = 0
    total_latency = 0.0
    llm_checks = 0
    deterministic_checks = 0

    # Per-step checks (fire once per step per iteration)
    for cp_name in ("step_input", "step_output"):
        checks = model.checkpoints.get(cp_name, [])
        count = len(checks) * pipeline.steps
        latency = sum(c.latency_ms for c in checks) * pipeline.steps
        for c in checks:
            if c.type in ("llm",):
                llm_checks += pipeline.steps
            else:
                deterministic_checks += pipeline.steps
        if checks:
            breakdown.append({
                "checkpoint": cp_name,
                "checks_per_step": len(checks),
                "total_checks": count,
                "latency_ms": latency,
                "guards": [c.name for c in checks],
            })
        total_checks += count
        total_latency += latency

    # Code output checks (fire only on code-producing steps)
    checks = model.checkpoints.get("code_output", [])
    if checks and pipeline.code_steps:
        count = len(checks) * pipeline.code_steps
        latency = sum(c.latency_ms for c in checks) * pipeline.code_steps
        deterministic_checks += count  # static analysis is not LLM
        breakdown.append({
            "checkpoint": "code_output",
            "checks_per_step": len(checks),
            "total_checks": count,
            "latency_ms": latency,
            "guards": [c.name for c in checks],
        })
        total_checks += count
        total_latency += latency

    # Tool call/response checks (fire per tool invocation per step)
    for cp_name in ("tool_call", "tool_response"):
        checks = model.checkpoints.get(cp_name, [])
        if checks and pipeline.tool_calls_per_step > 0:
            invocations = pipeline.steps * pipeline.tool_calls_per_step
            count = int(len(checks) * invocations)
            latency = sum(c.latency_ms for c in checks) * invocations
            for c in checks:
                n = int(invocations)
                if c.type == "llm":
                    llm_checks += n
                else:
                    deterministic_checks += n
            breakdown.append({
                "checkpoint": cp_name,
                "checks_per_invocation": len(checks),
                "invocations": invocations,
                "total_checks": count,
                "latency_ms": latency,
                "guards": [c.name for c in checks],
            })
            total_checks += count
            total_latency += latency

    # Inter-agent checks (fire between each pair of steps)
    checks = model.checkpoints.get("inter_agent", [])
    if checks and pipeline.steps > 1:
        boundaries = pipeline.steps - 1
        count = len(checks) * boundaries
        latency = sum(c.latency_ms for c in checks) * boundaries
        deterministic_checks += count
        breakdown.append({
            "checkpoint": "inter_agent",
            "boundaries": boundaries,
            "total_checks": count,
            "latency_ms": latency,
            "guards": [c.name for c in checks],
        })
        total_checks += count
        total_latency += latency

    # Memory and evaluation checks (fire once per step)
    for cp_name in ("memory", "evaluation"):
        checks = model.checkpoints.get(cp_name, [])
        if checks:
            count = len(checks) * pipeline.steps
            latency = sum(c.latency_ms for c in checks) * pipeline.steps
            for c in checks:
                if c.type == "llm":
                    llm_checks += pipeline.steps
                else:
                    deterministic_checks += pipeline.steps
            breakdown.append({
                "checkpoint": cp_name,
                "checks_per_step": len(checks),
                "total_checks": count,
                "latency_ms": latency,
                "guards": [c.name for c in checks],
            })
            total_checks += count
            total_latency += latency

    # Audit/observability checks (fire once per step)
    checks = model.checkpoints.get("audit", [])
    if checks:
        count = len(checks) * pipeline.steps
        latency = sum(c.latency_ms for c in checks) * pipeline.steps
        deterministic_checks += count
        breakdown.append({
            "checkpoint": "audit",
            "checks_per_step": len(checks),
            "total_checks": count,
            "latency_ms": latency,
            "guards": [c.name for c in checks],
        })
        total_checks += count
        total_latency += latency

    checks_per_iter = total_checks
    latency_per_iter = total_latency

    total_checks *= pipeline.iterations
    total_latency *= pipeline.iterations
    llm_checks *= pipeline.iterations
    deterministic_checks *= pipeline.iterations

    # Rough cost: LLM checks ~$0.001 each, deterministic ~$0.00001
    cost_per_run = llm_checks * 0.001 + deterministic_checks * 0.00001
    cost_per_month = cost_per_run * 1_000_000

    return OverheadProjection(
        model_name=model.name,
        pipeline=pipeline,
        checks_per_iteration=checks_per_iter,
        total_checks=total_checks,
        latency_per_iteration_ms=latency_per_iter,
        total_latency_ms=total_latency,
        breakdown=breakdown,
        llm_checks=llm_checks,
        deterministic_checks=deterministic_checks,
        cost_per_month_1m=cost_per_month,
    )


def load_all_models(models_dir: Path) -> list[SecurityModel]:
    """Load all security model YAML files from a directory."""
    models = []
    for f in sorted(models_dir.glob("*.yaml")):
        models.append(SecurityModel.from_yaml(f))
    return models
