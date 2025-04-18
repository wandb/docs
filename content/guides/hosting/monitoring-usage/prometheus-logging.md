---
menu:
  default:
    identifier: prometheus-logging
    parent: monitoring-and-usage
title: Use Prometheus monitoring
weight: 2
---

Use [Prometheus](https://prometheus.io/docs/introduction/overview/) with W&B Server. Prometheus installs are exposed as a [kubernetes ClusterIP service](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225).

{{% alert color="secondary" %}}
Prometheus monitoring is only available with [Self-managed instances]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}).
{{% /alert %}}


Follow the procedure below to access your Prometheus metrics endpoint (`/metrics`):

1. Connect to the cluster with Kubernetes CLI toolkit, [kubectl](https://kubernetes.io/docs/reference/kubectl/). See kubernetes' [Accessing Clusters](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/) documentation for more information.
2. Find the internal address of the cluster with:

    ```bash
    kubectl describe svc prometheus
    ```

3. Start a shell session inside your container running in your Kubernetes cluster with [`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands). Hit the endpoint at `<internal address>/metrics`.

   Copy the command below and execute it in your terminal and replace `<internal address>` with your internal address:

   ```bash
   kubectl exec <internal address>/metrics
   ```

A test pod starts, which you can exec into just to access anything in the network:

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```

From there you can choose to keep access internal to the network or expose it yourself with a kubernetes nodeport service.