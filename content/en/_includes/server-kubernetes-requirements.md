The W&B Server application is deployed as a [Kubernetes Operator]({{< relref "/guides/hosting/hosting-options/self-managed/kubernetes-operator/" >}}) that deploys multiple pods. The W&B Operator can be managed _only_ using Terraform or Helm.

These instructions require:
- A Kubernetes cluster running Kubernetes v1.28 or newer with:
  - A fully configured and functioning Ingress controller
  - The ability to provision Persistent Volumes.
- A supported version of `kubectl`. Refer to [Version skew policy](https://kubernetes.io/releases/version-skew-policy/#kubectl) in the Kubernetes documentation.
- Helm v3.x or newer
- Terraform v1.9 or newer
- A valid [W&B Server license]({{< relref "../#obtain-your-wb-server-license" >}})

For details, refer to [Reference Architecture]({{< relref "/guides/hosting/hosting-options/self-managed/ref-arch.md" >}}).

