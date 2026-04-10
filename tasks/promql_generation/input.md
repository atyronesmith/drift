User Question: What is the current CPU usage across all nodes in the cluster, and are there any nodes running hot over the past 2 hours?

Time Range: past 2 hours
Rate Syntax: 2h
Namespace: openshift-monitoring
Scope: fleet-wide

Current Metrics Status:
- cluster:cpu_usage_cores:sum: Avg=47.32, Latest=49.15, Trend=stable, normal
- node_cpu_seconds_total (idle, node-1): Avg=0.82, Latest=0.79, Trend=slightly_decreasing, normal
- node_cpu_seconds_total (idle, node-2): Avg=0.91, Latest=0.90, Trend=stable, normal
- node_cpu_seconds_total (idle, node-3): Avg=0.65, Latest=0.58, Trend=decreasing, WARNING: below 0.70 threshold
- node_cpu_seconds_total (idle, node-4): Avg=0.88, Latest=0.87, Trend=stable, normal
- kube_node_status_condition (Ready, node-1): Latest=1, normal
- kube_node_status_condition (Ready, node-2): Latest=1, normal
- kube_node_status_condition (Ready, node-3): Latest=1, normal
- kube_node_status_condition (Ready, node-4): Latest=1, normal
- container_cpu_usage_seconds_total (top consumers): vllm-predictor=12.4 cores, elasticsearch=8.7 cores, prometheus=3.2 cores

Common OpenShift/Kubernetes Metrics:
- CPU: 100 - (avg(rate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)
- Memory: 100 - (sum(node_memory_MemAvailable_bytes) / sum(node_memory_MemTotal_bytes) * 100)
- Pods Running: sum(kube_pod_status_phase{phase="Running"})
- Nodes: sum(kube_node_info)

Correlated Logs/Traces:
2026-04-10 14:22:17 WARNING node-3 kubelet: eviction manager: attempting to reclaim ephemeral-storage
2026-04-10 14:23:01 WARNING node-3 kubelet: Pod vllm-predictor-7b4f9-xk2m evicted due to resource pressure
2026-04-10 14:25:33 INFO node-3 kernel: CPU throttling detected: cgroup /kubepods/burstable exceeded quota
