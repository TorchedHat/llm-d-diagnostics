# Hardware inventory

Collect pod-visible equipment and configuration before interpreting inference
measurements. This reads Kubernetes pod metadata and runs a small standard-library
Python probe inside each explicitly named pod; it does not install software,
change deployments or run transfers.

```sh
python3 toolkit/hardware_profile.py --cli oc --namespace "$NS" \
  --pods "$PREFILL_POD,$DECODE_POD" --container vllm \
  --output-dir /tmp/hardware-inventory
```

The local machine needs Python and `oc` (or `kubectl`) with permission to get and
exec into the selected pods. The selected container needs Python. The output
directory must not exist. `profile.json` records pod UIDs, nodes, image IDs,
restart counts, declared resources, visible NVIDIA GPU details and topology,
RDMA ports/devices, CPU affinity, cgroup limits, software versions and an explicit
allowlist of UCX settings. Missing commands and data remain missing. If a pod
changes during its probe, collection fails and writes `error.json`.

These are pod-visible capabilities and declarations: they do not establish cluster
capacity, GPU isolation, bandwidth, the network route used by a request, RDMA or
GPUDirect. The output contains cluster, node, pod and IP metadata; review it before
sharing. UCX environment collection is limited to named settings and does not read
credential variables.

## Example: two H100 pods

A short inventory smoke on two separately scheduled OpenShift pods, each with
one GPU claim and one shared RDMA resource, found an H100 80GB and active
Ethernet RDMA ports visible in each container. Both pods had a one-CPU cgroup
limit and a 2 GiB memory limit. Pod identities and restart counts were stable
through collection. The probe did not run inference or transfer data; visible
ports do not establish the transport used by a workload.
