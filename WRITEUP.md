# Reflection

Writing the model was the easy part. Almost all of my time went on the plumbing
around it: where the data sits, where the checkpoint sits, and how one workload
hands a file to another.

That handover was the most challenging part. Training writes a checkpoint and
serving has to read it, but a Job and a Deployment are separate objects with
separate lifecycles. I used one PersistentVolumeClaim: the Job mounts it
read-write, the Deployment mounts the same claim read-only. It works, but only
because everything runs on one node. The claim is ReadWriteOnce, so on a real
multi-node cluster the serving pods could be scheduled elsewhere and would never
see the file. The proper fix is a ReadWriteMany volume, or pushing the
checkpoint to object storage or a model registry. I kept the simple version and
wrote down where it breaks.

Splitting the Docker images was the other thing I changed my mind about. I
started with one image for both jobs, then saw that serving does not need the
training dependencies. Now there are two requirements files and two
Dockerfiles. The training image is multi-stage, so torch stays in a cached layer
and only the source layer rebuilds after a code change. That took a rebuild from
minutes down to seconds.

The probes needed a couple of tries. `/health` returns 503 until the checkpoint
is loaded, which is what makes the readiness probe useful: the Service will not
send traffic to a pod that cannot answer yet. Liveness uses the same endpoint
for a different reason, to restart a container that has stopped responding. With
maxUnavailable set to 0, a new pod is ready before an old one goes away.

The Git workflow felt like overhead at first: four feature branches, four pull
requests into develop, then a release pull request into main. By the end I
preferred it, because each change was small enough to review on its own.

If I took this further I would push the images to a registry with fixed tags
instead of building them on the node, version the dataset, and scale on request
latency rather than CPU.
