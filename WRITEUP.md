# Reflection: Deploying a PyTorch Classifier with Docker & Kubernetes

This project took a CIFAR-10 image classifier from a single training script all
the way to a scalable, orchestrated service, and the most valuable lesson was
how much of "MLOps" is really about drawing clean seams between *code*, *data*,
*configuration*, and *environment*. The Python was the easy part; the
interesting engineering was everything around it.

**The most challenging part** was the storage and configuration boundary
between the training Job and the serving Deployment. Training produces a
checkpoint, and serving must consume it — but the two run as completely separate
Kubernetes workloads with independent lifecycles. I solved this with a shared
`PersistentVolumeClaim`: the Job mounts it read-write at `/app/checkpoints`, and
the Deployment mounts the *same* claim read-only. This immediately surfaced a
production caveat: a `ReadWriteOnce` volume only works when both workloads land
on the same node, so a real multi-node cluster needs a `ReadWriteMany` backend
(NFS/EFS) or, better, an object store plus a model registry. Designing for the
single-node minikube case while documenting the multi-node path was a useful
exercise in being honest about a solution's limits.

A close second was **keeping the two Docker images honest about their jobs**.
The training image is multi-stage so the pinned dependency layer is cached and
only the source layer rebuilds on code changes. The serving image is a
different discipline entirely: inference-only dependencies (no `tensorboard`),
a non-root user, an explicit `HEALTHCHECK`, and a fixed port. Splitting
`requirements/train.txt` from `requirements/serve.txt` made the serving image
noticeably leaner and reinforced why you rarely want one "does-everything"
container in production.

Getting the **health probes** right also took iteration. The serving app
returns `503` from `/health` until the checkpoint is loaded, which is exactly
what a Kubernetes *readiness* probe needs so the Service withholds traffic until
a replica can actually answer. Pairing that with a *liveness* probe (restart a
wedged container) and a rolling-update strategy of `maxUnavailable: 0` gave
zero-downtime deploys — something that only becomes obvious once you watch pods
cycle during `kubectl rollout`.

Finally, the **Git discipline** — a `develop` branch, one hypothesis per
feature branch, and Conventional Commits — felt like overhead at first but paid
off: each change was small, reviewable, and independently verifiable. If I
extended this project I would add DVC for dataset versioning, push images to a
registry with immutable tags, and add request-latency-based autoscaling, since
CPU is a poor proxy for inference load.
