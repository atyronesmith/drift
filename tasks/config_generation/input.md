```
REQUEST: Deploy new microservice for order processing

We need to deploy the order-processing-service (v2.3.0) to our
Kubernetes production cluster. Here are the requirements:

- The service needs 3 replicas for high availability
- Each replica needs 512MB memory and 250m CPU minimum,
  with limits of 1GB memory and 500m CPU
- It listens on port 8080 for HTTP and 8443 for gRPC
- Health check endpoint is /healthz on port 8080
- Readiness check is /ready on port 8080, should wait 10 seconds
  before first check
- It needs these environment variables:
  * DATABASE_URL from the secret "order-db-credentials" key "url"
  * REDIS_HOST set to "redis-cluster.cache.svc.cluster.local"
  * REDIS_PORT set to "6379"
  * LOG_LEVEL set to "info"
  * ENABLE_METRICS set to "true"
- Mount the config file from ConfigMap "order-processing-config"
  at /etc/config/settings.yaml
- The service should have a PodDisruptionBudget allowing at most
  1 unavailable pod
- Add these labels: team=platform, service=order-processing,
  environment=production
- The service needs a horizontal pod autoscaler that scales
  between 3 and 10 replicas based on CPU utilization at 70%
- It should have anti-affinity to spread across availability zones
- Set up a Service of type ClusterIP exposing both ports
- Add a NetworkPolicy allowing ingress only from the namespace
  "api-gateway" on port 8080 and from "monitoring" namespace
  on port 9090 (metrics)

Container image: registry.example.com/platform/order-processing:v2.3.0

Please generate all the Kubernetes YAML manifests needed.
```
