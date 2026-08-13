# Reflection

This assignment took a CIFAR-10 classifier from a single training script to a
containerised training Job and a serving Deployment running on Kubernetes. The
PyTorch part was the part I already knew how to do. Almost all the time went
into everything around it: where the data lives, where the model file lives,
and how one workload hands something to another.

The hardest part was the boundary between training and serving. Training writes
a checkpoint, serving has to read it, but the two are separate Kubernetes
objects with separate lifecycles. I used one PersistentVolumeClaim for it: the
Job mounts it read-write at /app/checkpoints and the Deployment mounts the same
claim read-only. That works, but it only works because everything runs on one
node. The claim is ReadWriteOnce, so on a real multi-node cluster the serving
pods could be scheduled somewhere else and would not see the file. A proper
setup would use a ReadWriteMany volume or push the checkpoint to object storage
or a model registry. I decided it was better to keep the simple version and be
clear about where it breaks.

Splitting the two Docker images also took some thought. At first I had one
image for both, then it was obvious that serving does not need the training
dependencies. So there are two requirements files and two Dockerfiles. The
training image is multi-stage, so the layer with torch in it is cached and only
the source layer rebuilds when I change code. The serving image adds a non-root
user, a fixed port and a HEALTHCHECK. Rebuilding after a code change went from
minutes to seconds once the dependency layer stopped being invalidated.

The probes took a couple of tries to get right. /health returns 503 until the
checkpoint is actually loaded into memory, which is what makes the readiness
probe useful: the Service does not send traffic to a pod that cannot answer
yet. The liveness probe uses the same endpoint but for a different reason, to
restart a container that has stopped responding. With maxUnavailable set to 0
in the rolling update, a new version comes up before an old pod goes away.

The Git workflow felt like extra work at the start. Four feature branches, four
pull requests into develop, then one release pull request into main. By the end
I was glad I did it that way, because each change was small enough to check on
its own. If I continued this project I would version the dataset properly, push
the images to a registry with fixed tags instead of building on the node, and
scale on request latency rather than CPU.
