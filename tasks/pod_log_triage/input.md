"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
"GET /readyz" 200 OK 1ms
"GET /healthz" 200 OK 3ms
"GET /healthz" 200 OK 1ms
"GET /readyz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
"GET /livez" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
"GET /readyz" 200 OK 1ms
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
I0413 09:00:01.234567 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:00:03.456789 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:00:05.678901 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:00:07.890123 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:00:09.012345 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:00:11.234567 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:00:13.456789 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:00:15.678901 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:01:00.111111 1 reflector.go:535] caches populated
I0413 09:01:00.222222 1 shared_informer.go:240] Starting shared informer factory
I0413 09:01:00.333333 1 reflector.go:188] Listing and watching v1.Pods
I0413 09:01:00.444444 1 reflector.go:188] Listing and watching v1.Services
I0413 09:01:00.555555 1 reflector.go:188] Listing and watching v1.ConfigMaps
I0413 09:01:01.111111 1 reflector.go:535] caches populated
I0413 09:01:01.222222 1 reflector.go:535] caches populated
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /readyz" 200 OK 1ms
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
I0413 09:02:00.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:02:02.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:02:04.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
Throttling request took 1.1s, tokens=2, queueLength=0
Throttling request took 1.3s, tokens=2, queueLength=0
"GET /metrics" 200 OK 5ms
"GET /metrics" 200 OK 4ms
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
"GET /readyz" 200 OK 1ms
"GET /healthz" 200 OK 1ms
I0413 09:03:00.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:03:02.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
controller-runtime Starting EventSource component=controller source=kind/v1/Pod
controller-runtime Starting Controller component=controller name=pod-controller
controller-runtime Starting workers component=controller name=pod-controller workers=4
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
"GET /readyz" 200 OK 1ms
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
  resourceVersion: "482910"
  uid: a3f8b29c-4e71-4a8b-9f2c-1d3e5f7a8b9c
  creationTimestamp: "2026-04-13T09:00:00Z"
  selfLink: /api/v1/namespaces/demo-ns/pods/llama-3-1-8b-instruct-predictor-a1b2c3d4-x1y2z
  managedFields:
  - manager: kube-controller-manager
    operation: Update
    fieldsType: FieldsV1
    fieldsV1:
      f:metadata:
        f:labels:
          f:app: {}
          f:pod-template-hash: {}
      f:spec:
        f:containers:
          k:{"name":"kserve-container"}:
            f:image: {}
            f:resources: {}
  - manager: kubelet
    operation: Update
    fieldsType: FieldsV1
    fieldsV1:
      f:status:
        f:conditions:
          k:{"type":"ContainersReady"}: {}
          k:{"type":"Initialized"}: {}
          k:{"type":"Ready"}: {}
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
I0413 09:04:00.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:04:02.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:04:04.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /readyz" 200 OK 1ms
E0413 09:05:14.789012 1 memcg_linux.go:61] memory cgroup out of memory: container kserve-container in pod llama-3-1-8b-instruct-predictor-a1b2c3d4-x1y2z exceeded limit 16Gi
level=error msg="OOMKilled: container kserve-container terminated with exit code 137" pod="demo-ns/llama-3-1-8b-instruct-predictor-a1b2c3d4-x1y2z" container="kserve-container" memory_limit="16Gi" memory_usage="16.8Gi"
level=error msg="pod failed readiness probe: connection refused on port 8080" pod="demo-ns/llama-3-1-8b-instruct-predictor-a1b2c3d4-x1y2z"
W0413 09:05:15.123456 1 event.go:285] Event(v1.ObjectReference{Kind:"Pod", Namespace:"demo-ns", Name:"llama-3-1-8b-instruct-predictor-a1b2c3d4-x1y2z"}): type: 'Warning' reason: 'OOMKilling' memory cgroup out of memory
level=error msg="container restart initiated" pod="demo-ns/llama-3-1-8b-instruct-predictor-a1b2c3d4-x1y2z" restart_count=5 back_off="5m0s"
CrashLoopBackOff: back-off 5m0s restarting failed container=kserve-container pod=llama-3-1-8b-instruct-predictor-a1b2c3d4-x1y2z
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
"GET /readyz" 200 OK 1ms
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
I0413 09:06:00.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:06:02.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:06:04.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /readyz" 200 OK 1ms
level=error msg="GPU memory fragmentation detected: 94% utilized, largest contiguous block 1.2Gi" pod="demo-ns/llama-3-1-8b-instruct-predictor-a1b2c3d4-p9q8r" gpu=0
level=error msg="CUDA out of memory: tried to allocate 2.4Gi but only 1.2Gi free" pod="demo-ns/llama-3-1-8b-instruct-predictor-a1b2c3d4-p9q8r" gpu=0
level=error msg="inference request failed: model loading incomplete, GPU memory insufficient" pod="demo-ns/llama-3-1-8b-instruct-predictor-a1b2c3d4-p9q8r" model="meta-llama/Llama-3.1-8B-Instruct"
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
"GET /readyz" 200 OK 1ms
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
"GET /readyz" 200 OK 1ms
I0413 09:07:00.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:07:02.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
Throttling request took 2.4s, tokens=2, queueLength=3
Throttling request took 1.8s, tokens=2, queueLength=2
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
"GET /metrics" 200 OK 4ms
"GET /readyz" 200 OK 1ms
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
I0413 09:08:22.456789 1 node_lifecycle_controller.go:175] Node node-3 condition Ready changed to Unknown
level=error msg="node node-3 is NotReady: kubelet stopped posting status" duration="5m0s"
level=error msg="pod eviction triggered for node node-3" pods_affected=7 namespace="demo-ns"
W0413 09:08:23.567890 1 event.go:285] Event(v1.ObjectReference{Kind:"Node", Name:"node-3"}): type: 'Warning' reason: 'NodeNotReady' Node node-3 status is now: NodeNotReady
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
"GET /readyz" 200 OK 1ms
"GET /healthz" 200 OK 1ms
I0413 09:09:00.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:09:02.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
I0413 09:09:04.000001 1 controller.go:297] successfully renewed lease openshift-kube-scheduler/kube-scheduler
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
"GET /readyz" 200 OK 1ms
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
caches populated
Starting shared informer factory
Listing and watching v1.Endpoints
caches populated
caches populated
serving cert rotation: certificate signed by unknown authority (during bootstrap)
TLS handshake error from 10.0.0.42: remote error: tls: bad certificate
"GET /healthz" 200 OK 1ms
"GET /healthz" 200 OK 2ms
"GET /healthz" 200 OK 1ms
