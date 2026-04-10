Namespace: demo-ns
Scope: namespace-specific

Firing Alerts:
- VLLMHighAverageInferenceTime, severity: warning, namespace: demo-ns
  Labels: {"alertname": "VLLMHighAverageInferenceTime", "container": "kserve-container", "endpoint": "vllm-serving-runtime-metrics", "engine": "0", "expr": "rate(vllm:request_inference_time_seconds_sum[5m]) / rate(vllm:request_inference_time_seconds_count[5m]) > 2", "for": "5m", "instance": "10.0.0.42:8080", "job": "llama-3-1-8b-instruct-metrics", "model_name": "meta-llama/Llama-3.1-8B-Instruct", "namespace": "demo-ns", "pod": "llama-3-1-8b-instruct-predictor-a1b2c3d4-x1y2z", "prometheus": "openshift-user-workload-monitoring/user-workload", "service": "llama-3-1-8b-instruct-metrics", "severity": "warning"}

- KubePodCrashLooping, severity: critical, namespace: demo-ns
  Labels: {"alertname": "KubePodCrashLooping", "container": "kserve-container", "namespace": "demo-ns", "pod": "llama-3-1-8b-instruct-predictor-a1b2c3d4-p9q8r", "severity": "critical", "reason": "CrashLoopBackOff", "restart_count": "8"}

- KubeMemoryOvercommit, severity: warning, namespace: demo-ns
  Labels: {"alertname": "KubeMemoryOvercommit", "namespace": "demo-ns", "severity": "warning", "node": "node-3", "requested_bytes": "68719476736", "capacity_bytes": "67645734912"}
