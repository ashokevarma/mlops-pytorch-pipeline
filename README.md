# mlops-pytorch-pipeline

A CIFAR-10 image classifier taken through the full deployment lifecycle: local
training → Docker containerization → training + serving on Kubernetes.

## Architecture

```
 docker/Dockerfile.train ──► mlops-train:v1 ──► Job: cifar10-training ─┐
                                                                       │ writes classifier_v1.pt
                                                              checkpoint PVC
                                                                       │ reads (RO)
 docker/Dockerfile.serve ──► mlops-serve:v1 ──► Deployment: model-serving (2 replicas)
                                                        │
                                              Service (ClusterIP :80 → :8080)
```

The training Job writes the checkpoint to a PVC; the serving Deployment mounts
the same PVC read-only and is exposed by a Service.

## Layout

```
src/         model.py, dataset.py, train.py, serve.py
configs/     training_config.yaml
docker/      Dockerfile.train, Dockerfile.serve
k8s/         namespace, configmap, pvc, training-job,
             training-gpu-job (bonus), serving-deployment, serving-service
requirements/train.txt, serve.txt
tests/       test_model.py
```

## Local run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/train.txt pytest
python src/train.py          # reads configs/training_config.yaml
pytest
```

## Docker

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/checkpoints:/app/checkpoints" mlops-train:v1

docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 -v "$(pwd)/checkpoints:/app/checkpoints" mlops-serve:v1

curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
kubectl -n ml-training wait --for=condition=complete job/cifar10-training --timeout=14400s

kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml

kubectl get pods -n ml-training
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

`k8s/training-gpu-job.yaml` is an optional GPU variant of the training Job; apply
it only on a cluster with GPU nodes labelled `accelerator=nvidia-gpu`.

## Validation

The whole flow was run on a single-node Kubernetes cluster (k3s v1.30): both images
built, the training Job completed all 10 epochs on CPU (87.6% validation accuracy)
and wrote the checkpoint to the PVC, then the serving Deployment came up with 2
ready replicas and answered `/health` and `/predict` through the Service. The
terminal output for each step is in the submission write-up.

## API

- `GET /health` → `200` with `{status, checkpoint, classes}` when the model is loaded, else `503`.
- `POST /predict` → multipart form field `image`; returns `{top, predictions:[{class, probability}]}`.

All hyperparameters live in `configs/training_config.yaml` (mounted as the
`training-config` ConfigMap in Kubernetes).
