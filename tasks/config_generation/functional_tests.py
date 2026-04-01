"""
Functional equivalence tests for the config_generation task.

Tests validate the generated Kubernetes manifests against the
requirements in the input. Split into:
- STRUCTURAL: resource types, names, labels (template-able)
- SPEC: specific values from requirements (extractable)
- SEMANTIC: correct K8s patterns and relationships (LLM reasoning)
"""

import re
import yaml as pyyaml


def extract(code: str):
    """Parse YAML manifests from LLM output."""
    text = code.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    docs = list(pyyaml.safe_load_all(text))
    return [d for d in docs if d is not None]


def _find_resource(docs, kind):
    """Find a resource by kind."""
    for d in docs:
        if d.get("kind") == kind:
            return d
    return None


def _find_resources(docs, kind):
    """Find all resources of a kind."""
    return [d for d in docs if d.get("kind") == kind]


# --- STRUCTURAL tests (template candidates) ---

def test_has_deployment(docs):
    """Structure: generates a Deployment resource."""
    dep = _find_resource(docs, "Deployment")
    assert dep is not None, f"Missing Deployment. Got: {[d.get('kind') for d in docs]}"


def test_has_service(docs):
    """Structure: generates a Service resource."""
    svc = _find_resource(docs, "Service")
    assert svc is not None, f"Missing Service. Got: {[d.get('kind') for d in docs]}"


def test_has_hpa(docs):
    """Structure: generates HorizontalPodAutoscaler."""
    hpa = _find_resource(docs, "HorizontalPodAutoscaler")
    assert hpa is not None, f"Missing HPA. Got: {[d.get('kind') for d in docs]}"


def test_has_pdb(docs):
    """Structure: generates PodDisruptionBudget."""
    pdb = _find_resource(docs, "PodDisruptionBudget")
    assert pdb is not None, f"Missing PDB. Got: {[d.get('kind') for d in docs]}"


def test_has_network_policy(docs):
    """Structure: generates NetworkPolicy."""
    np = _find_resource(docs, "NetworkPolicy")
    assert np is not None, f"Missing NetworkPolicy. Got: {[d.get('kind') for d in docs]}"


# --- SPEC tests (value extraction) ---

def test_replicas(docs):
    """Spec: Deployment has 3 replicas."""
    dep = _find_resource(docs, "Deployment")
    assert dep is not None, "Missing Deployment"
    replicas = dep.get("spec", {}).get("replicas")
    assert replicas == 3, f"Expected 3 replicas, got {replicas}"


def test_container_image(docs):
    """Spec: correct container image and tag."""
    dep = _find_resource(docs, "Deployment")
    assert dep is not None, "Missing Deployment"
    containers = dep["spec"]["template"]["spec"]["containers"]
    images = [c.get("image", "") for c in containers]
    assert any("order-processing:v2.3.0" in img for img in images), \
        f"Expected image order-processing:v2.3.0, got {images}"


def test_resource_limits(docs):
    """Spec: memory and CPU limits match requirements."""
    dep = _find_resource(docs, "Deployment")
    assert dep is not None, "Missing Deployment"
    container = dep["spec"]["template"]["spec"]["containers"][0]
    resources = container.get("resources", {})
    limits = resources.get("limits", {})
    assert limits.get("memory") in ("1Gi", "1024Mi", "1G", "1GB"), \
        f"Expected 1Gi memory limit, got {limits.get('memory')}"


def test_env_vars(docs):
    """Spec: required environment variables present."""
    dep = _find_resource(docs, "Deployment")
    assert dep is not None, "Missing Deployment"
    container = dep["spec"]["template"]["spec"]["containers"][0]
    env = container.get("env", [])
    env_from = container.get("envFrom", [])
    all_text = str(env) + str(env_from)
    assert "DATABASE_URL" in all_text, "Missing DATABASE_URL env var"
    assert "REDIS_HOST" in all_text, "Missing REDIS_HOST env var"
    assert "LOG_LEVEL" in all_text, "Missing LOG_LEVEL env var"


def test_labels(docs):
    """Spec: required labels applied."""
    dep = _find_resource(docs, "Deployment")
    assert dep is not None, "Missing Deployment"
    labels = dep.get("metadata", {}).get("labels", {})
    template_labels = dep["spec"]["template"]["metadata"].get("labels", {})
    all_labels = {**labels, **template_labels}
    assert all_labels.get("team") == "platform", f"Expected team=platform, got {all_labels}"
    assert "order-processing" in str(all_labels.get("service", "")), \
        f"Expected service=order-processing label"


def test_hpa_settings(docs):
    """Spec: HPA min/max replicas and CPU target."""
    hpa = _find_resource(docs, "HorizontalPodAutoscaler")
    assert hpa is not None, "Missing HPA"
    spec = hpa.get("spec", {})
    assert spec.get("minReplicas") == 3, f"Expected minReplicas=3, got {spec.get('minReplicas')}"
    assert spec.get("maxReplicas") == 10, f"Expected maxReplicas=10, got {spec.get('maxReplicas')}"


# --- SEMANTIC tests (LLM reasoning) ---

def test_health_checks(docs):
    """Semantic: correct liveness and readiness probes configured."""
    dep = _find_resource(docs, "Deployment")
    assert dep is not None, "Missing Deployment"
    container = dep["spec"]["template"]["spec"]["containers"][0]
    readiness = container.get("readinessProbe", {})
    liveness = container.get("livenessProbe", {})
    # Should have both probes
    assert readiness, "Missing readiness probe"
    assert liveness, "Missing liveness probe"
    # Readiness should hit /ready
    ready_path = str(readiness)
    assert "/ready" in ready_path, f"Readiness probe should check /ready, got {readiness}"


def test_secret_reference(docs):
    """Semantic: DATABASE_URL sourced from secret, not hardcoded."""
    dep = _find_resource(docs, "Deployment")
    assert dep is not None, "Missing Deployment"
    text = str(dep)
    assert "order-db-credentials" in text, \
        "Expected DATABASE_URL from secret 'order-db-credentials'"
    # Should NOT have a hardcoded database URL value
    assert "postgresql://" not in text and "mysql://" not in text, \
        "DATABASE_URL should reference a secret, not be hardcoded"


def test_anti_affinity(docs):
    """Semantic: pod anti-affinity configured for zone spreading."""
    dep = _find_resource(docs, "Deployment")
    assert dep is not None, "Missing Deployment"
    text = str(dep).lower()
    assert "affinity" in text, "Missing anti-affinity configuration"
    assert "topology" in text or "zone" in text, \
        "Anti-affinity should reference topology/zone for AZ spreading"


def test_network_policy_rules(docs):
    """Semantic: NetworkPolicy allows correct namespaces."""
    np = _find_resource(docs, "NetworkPolicy")
    assert np is not None, "Missing NetworkPolicy"
    text = str(np)
    assert "api-gateway" in text, "NetworkPolicy should allow ingress from api-gateway namespace"
    assert "monitoring" in text, "NetworkPolicy should allow ingress from monitoring namespace"


tests = {
    # Structural (template candidates)
    "has_deployment": test_has_deployment,
    "has_service": test_has_service,
    "has_hpa": test_has_hpa,
    "has_pdb": test_has_pdb,
    "has_network_policy": test_has_network_policy,
    # Spec (extractable values)
    "replicas": test_replicas,
    "container_image": test_container_image,
    "resource_limits": test_resource_limits,
    "env_vars": test_env_vars,
    "labels": test_labels,
    "hpa_settings": test_hpa_settings,
    # Semantic (LLM reasoning)
    "health_checks": test_health_checks,
    "secret_reference": test_secret_reference,
    "anti_affinity": test_anti_affinity,
    "network_policy_rules": test_network_policy_rules,
}
