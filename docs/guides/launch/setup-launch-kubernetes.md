---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up Kubernetes

You can use W&B Launch to execute W&B Launch Jobs as a [Kubernetes Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) or [Custom workload](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) resource in a Kubernetes cluster. This is particularly useful if you want to use Kubernetes to manage your compute cluster and want a simple interface to execute training, transformation, or ML workloads in your cluster. 

W&B maintains an [official launch agent image](https://hub.docker.com/r/wandb/launch-agent) that can be deployed to your cluster with a [helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) that is managed by W&B. 

:::info
It is possible to start a launch agent outside of a Kubernetes cluster. However, we recommend that you deploy the launch agent directly into the Kubernetes cluster that is responsible for running the Kubernetes Job or Custom workload. 
:::

The launch agent submits workloads to the cluster specified by the current Kubernetes cluster context.

W&B uses the [Kaniko](https://github.com/GoogleContainerTools/kaniko) builder to enable the launch agent to build Docker images in a Kubernetes cluster. To learn more on how to set up Kaniko for the launch agent, see [Advanced agent set up](./setup-agent-advanced.md).

<!-- Future: insert diagram here -->

## Configure a queue for Kubernetes
The launch queue configuration for a Kubernetes target resource will resemble either a Kubernetes Job spec or a Kubernetes Custom Resource spec. You can control any aspect of the Kubernetes workload resource spec when you create a launch queue. 

<Tabs
  defaultValue="job"
  values={[
    {label: 'Kubernetes Job Spec', value: 'job'},
    {label: 'Custom Resource Spec', value: 'custom'},
  ]}>
  <TabItem value="job">

```yaml
spec:
  template:
    spec:
      containers:
        - env:
            - name: MY_ENV_VAR
              value: some-value
          resources:
            requests:
              cpu: 1000m
              memory: 1Gi
metadata:
  labels:
    queue: k8s-test
namespace: wandb
```

  </TabItem>
  <TabItem value="custom">

In some use cases, you might want to use `CustomResource` definitions. `CustomResource` definitions are useful if, for example, you want to perform multi-node distributed training. See the tutorial for using Launch with multi-node jobs using Volcano for an example application. Another use case might be that you want to use W&B Launch with Kubeflow.

 The following YAML snippet shows a sample launch queue config that uses Kubeflow:

```yaml
kubernetes:
  kind: PyTorchJob
  spec:
    pytorchReplicaSpecs:
      Master:
        replicas: 1
        template:
          spec:
            containers:
              - name: pytorch
                image: '${image_uri}'
                imagePullPolicy: Always
        restartPolicy: Never
      Worker:
        replicas: 2
        template:
          spec:
            containers:
              - name: pytorch
                image: '${image_uri}'
                imagePullPolicy: Always
        restartPolicy: Never
    ttlSecondsAfterFinished: 600
  metadata:
    name: '${run_id}-pytorch-job'
  apiVersion: kubeflow.org/v1
```

  </TabItem>
</Tabs>


For security reasons, W&B will inject the following resources into your launch queue if they are not specified:

* `securityContext`
* `backOffLimit`
* `ttlSecondsAfterFinished`

The following YAML snippet demonstrates how these values will appear in your launch queue:

```yaml title="example-spec.yaml"
spec: 
  template:
    `backOffLimit`: 0
    ttlSecondsAfterFinished: 60
    securityContext:
      allowPrivilegeEscalation: False,
      capabilities:
        drop:
          - ALL,
      seccompProfile: 
        type: "RuntimeDefault"
```



## Create a queue 

Create a queue in the W&B App that uses Kubernetes as its compute resource:

1. Navigate to the [Launch page](https://wandb.ai/launch).
2. Click on the **Create Queue** button.
3. Select the **Entity** you would like to create the queue in.
4. Provide a name for your queue in the **Name** field.
5. Select **Kubernetes** as the **Resource**.
6. Within the **Configuration** field, provide the Kubernetes Job workflow spec or Custom Resource spec you [configured in the previous section](#configure-a-queue-for-kubernetes).


## Configure a launch agent with helm
Use the helm chart provided by W&B to deploy the launch agent into your Kubernetes cluster. Control the behavior of the launch agent with the `values.yaml` [file](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml).

Specify the contents that would normally by defined in your launch agent config file (`~/.config/wandb/launch-config.yaml`) within the `launchConfig` key in the`values.yaml` file.

For example, suppose you have launch agent config that enables you to run a launch agent in EKS that uses the Kaniko Docker image builder:

```yaml title="launch-config.yaml"
queues:
	- <queue name>
max_jobs: <n concurrent jobs>
environment:
	type: aws
	region: us-east-1
registry:
	type: ecr
	uri: <my-registry-uri>
builder:
	type: kaniko
	build-context-store: <s3-bucket-uri>
```

Within your `values.yaml` file, this might look like:

```yaml title="values.yaml"
agent:
  labels: {}
  # W&B API key.
  apiKey: ""
  # Container image to use for the agent.
  image: wandb/launch-agent-dev:latest
  # Image pull policy for agent image.
  imagePullPolicy: Always
  # Resources block for the agent spec.
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

# Namespace to deploy launch agent into
namespace: wandb

# W&B api url (Set yours here)
baseUrl: https://api.wandb.ai

# Additional target namespaces that the launch agent can deploy into
additionalTargetNamespaces:
  - default
  - wandb

# This should be set to the literal contents of your launch agent config.
launchConfig: |
  queues:
    - <queue name>
  max_jobs: <n concurrent jobs>
  environment:
    type: aws
    region: <aws-region>
  registry:
    type: ecr
    uri: <my-registry-uri>
  builder:
    type: kaniko
    build-context-store: <s3-bucket-uri>


# Set to false to disable volcano install.
volcano: true

# The contents of a git credentials file. This will be stored in a k8s secret
# and mounted into the agent container. Set this if you want to clone private
# repos.
gitCreds: |

# Annotations for the wandb service account. Useful when setting up workload identity on gcp.
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account:
    azure.workload.identity/client-id:

# Set to access key for azure storage if using kaniko with azure.
azureStorageAccessKey: ""
```

:::note
You can control whether the Volcano scheduler is installed into your cluster.
:::


For more information on registries, environments and required agent permissions see [Advanced agent set up](./setup-agent-advanced.md).

Follow the instructions in the [helm chart repository](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) to deploy your agent.
