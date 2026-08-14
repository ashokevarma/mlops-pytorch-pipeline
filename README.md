# mlops-pytorch-pipeline

A CIFAR-10 image classifier taken from local training to Docker images to
training and serving on Kubernetes.

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

## Layout

```
src/         model.py, dataset.py, train.py, serve.py
configs/     training_config.yaml
docker/      Dockerfile.train, Dockerfile.serve
k8s/         namespace, configmap, pvc, training-job,
             training-gpu-job (bonus), serving-deployment, serving-service
requirements/train.txt, serve.txt
tests/       test_model.py
screenshots/ terminal output from the validation run
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

Run end to end on single-node k3s v1.30: both images built, the training Job
completed 10 epochs on CPU at 87.6% validation accuracy and wrote the checkpoint
to the PVC, then 2 serving replicas came up and answered `/health` and
`/predict` through the Service. Screenshots of every step are in
`screenshots/` and in the description of the final PR.

## API

- `GET /health` → `200` with `{status, checkpoint, classes}` when the model is loaded, else `503`.
- `POST /predict` → multipart form field `image`; returns `{top, predictions:[{class, probability}]}`.

Hyperparameters live in `configs/training_config.yaml`, mounted as the
`training-config` ConfigMap in Kubernetes.
