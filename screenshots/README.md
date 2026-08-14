# Validation screenshots

Terminal output from running the pipeline end to end on a single-node Kubernetes
cluster (k3s v1.30). Container images were built with `nerdctl`, which is the
Docker-compatible CLI for containerd - the Dockerfiles are standard and build the
same way with `docker build`.

The host name, user name, paths and every IP address in these captures are
placeholders. The node shows as 127.0.0.1, pod and Service addresses use the
example ranges from the Kubernetes docs (10.244.0.0/16 and 10.96.0.0/12).

| Screenshot | What it shows |
| --- | --- |
| `00_cluster_and_tests.png` | Cluster version, node status, `pytest tests/ -v` (5 passed) |
| `01_build_training_image.png` | Multi-stage build of `mlops-train:v1` |
| `02_build_serving_image.png` | Multi-stage build of `mlops-serve:v1` |
| `03_images.png` | Both images tagged, plus layer history of the serving image |
| `04_kubectl_apply_core.png` | Namespace, PVCs and ConfigMap applied and bound |
| `05_training_job_running.png` | Training Job submitted, pod Running, first log lines |
| `06_training_logs.png` | Per-epoch train/val loss and accuracy from the Job |
| `07_training_job_complete.png` | Job `Complete` 1/1 with duration |
| `08_serving_rollout.png` | Deployment rollout, 2 pods Ready, ClusterIP Service |
| `09_health_and_predict.png` | `GET /health` and `POST /predict` through the Service |
| `10_deployment_details.png` | Replicas, rolling update strategy, probes, CPU/memory limits |
