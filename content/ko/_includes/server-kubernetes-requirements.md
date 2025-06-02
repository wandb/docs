The W&B Server application is deployed as a [Kubernetes Operator]({{< relref "/guides/hosting/hosting-options/self-managed/kubernetes-operator/" >}}) that deploys multiple pods. The Kubernetes Operator can be managed _only_ using Terraform or Helm.

The Kubernetes cluster requires:
- A fully configured and functioning Ingress controller.
- The capability to provision Persistent Volumes.


