Namespace: m3
Scope: namespace-specific

Firing Alerts:
- VLLMHighAverageInferenceTime, severity: warning, namespace: m3
  Labels: {"alertname": "VLLMHighAverageInferenceTime", "container": "kserve-container", "endpoint": "vllm-serving-runtime-metrics", "engine": "0", "expr": "rate(vllm:request_inference_time_seconds_sum[5m]) / rate(vllm:request_inference_time_seconds_count[5m]) > 2", "for": "5m", "instance": "10.129.2.73:8080", "job": "llama-3-1-8b-instruct-metrics", "model_name": "meta-llama/Llama-3.1-8B-Instruct", "namespace": "m3", "pod": "llama-3-1-8b-instruct-predictor-9fd74489-c6dwt", "prometheus": "openshift-user-workload-monitoring/user-workload", "service": "llama-3-1-8b-instruct-metrics", "severity": "warning"}

- KubePodCrashLooping, severity: critical, namespace: m3
  Labels: {"alertname": "KubePodCrashLooping", "container": "kserve-container", "namespace": "m3", "pod": "llama-3-1-8b-instruct-predictor-9fd74489-q7kfn", "severity": "critical", "reason": "CrashLoopBackOff", "restart_count": "8"}

- KubeMemoryOvercommit, severity: warning, namespace: m3
  Labels: {"alertname": "KubeMemoryOvercommit", "namespace": "m3", "severity": "warning", "node": "node-3", "requested_bytes": "68719476736", "capacity_bytes": "67645734912"}
