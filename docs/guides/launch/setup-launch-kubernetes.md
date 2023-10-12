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

W&B uses the [Kaniko](https://github.com/GoogleContainerTools/kaniko) builder to enable the launch agent to build Docker images in a Kubernetes cluster. To learn more on how to set up Kaniko for the launch agent, see Advanced agent set up[LINK].

<!-- Future: insert diagram here -->

## Configure a queue for Kubernetes
The launch queue configuration for a Kubernetes target resource will resemble either a Kubernetes Job spec or a Kubernetes Custom Resource spec.  You can control any aspect of the Kubernetes workload resource spec when you create a launch queue. 

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

In some use cases, you might want to use `CustomResource` definitions. `CustomResource` definitions are useful if, for example, you want to perform multi-node distributed training. See the tutorial for using Launch with multinode jobs using Volcano for an example application.  Another use case might be that you want to use W&B Launch with Kubeflow.

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
6. Within the **Configuration** field, provide the Kubernetes Job workflow spec or custom resource spec you defined in the previous section[LINK].


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


For more information on registries, environments and required agent permissions see Advanced agent set up[LINK].

Follow the instructions in the [helm chart repo](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) to deploy your agent.

<!-- 
# Set up for Kubernetes

The following sections outline how to configure a launch queue and agent to execute jobs on a Kubernetes cluster such as Amazon Elastic Kubernetes Service (Amazon EKS) or Google Kubernetes Engine (GKE). 

:
## Prerequisites
Before you get started, ensure you have the:
1. **Kubernetes cluster** [LINK]
2. **W&B API Key** [LINK]

## Configure a queue for Kubernetes
Create a queue in the W&B App that uses Kubernetes as its compute resource:

1. Navigate to the [Launch page](https://wandb.ai/launch).
2. Click on the **Create Queue** button.
3. Select the **Entity** you would like to create the queue in.
4. Provide a name for your queue in the **Name** field.
5. Select **Kubernetes** as the **Resource**.
6. Within the **Configuration** field, provide information about your Kubernetes job. The launch queue configuration follows the same schema as a [Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/#writing-a-job-spec), except that it also supports additional universal queue configuration fields, such as `builder`. 
W&B will populate a YAML and JSON a [Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/#writing-a-job-spec) request body:


<Tabs
  defaultValue="YAML"
  values={[
    {label: 'JSON', value: 'JSON'},
    {label: 'YAML', value: 'YAML'},
  ]}>
  <TabItem value="JSON">

```json title='Queue configuration'
{
  "spec": {
    "backoffLimit": 0,
    "ttlSecondsAfterFinished": 60,
    "template": {
      "spec": {
        "restartPolicy": "Never"
      }
    }
  }
}
```

  </TabItem>
  <TabItem value="YAML">

```yaml title='Queue configuration'
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 60
  template:
    spec:
      restartPolicy: Never
```

  </TabItem>
</Tabs>



## Configure a launch agent for Kubernetes
Create and configure a YAML configuration file for the launch agent. The config should contain, at a minimum your W&B entity and a list of all queues to poll. 

Copy and paste the code block below. Replace the values based on your use case:

```yaml title="~/.config/wandb/launch-config.yaml"
# W&B entity (i.e. user or team) name
entity: entity-name

# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

# List of queues to poll.
queues:
  - default
```

<!-- <Tabs
  defaultValue="manually"
  values={[
    {label: 'Manually', value: 'manually'},
    {label: 'Helm charts', value: 'helm'},
  ]}>
  <TabItem value="manually">



  </TabItem>
  <TabItem value="helm">This is an orange 🍊</TabItem>
</Tabs> -->

<!-- 
### Agent environments

You can also specify a specific environment to run the agent on, specify a container registry, and specify a specific Docker builder. 



For more information on optional launch agent configuration options, see the Configure a launch agent page. [LINK]

## Deploy your agent

Unlike managed compute resources (such as SageMaker), with Kubernetes you will need to deploy your agent to your Kubernetes/compute resource.


There are two ways to deploy your launch agent:

1. Helm charts
2. Deploy with a manual cluster configuration

:::tip
We **strongly recommended** that you install the launch agent through the [official helm repository](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent). Consult the [`README.md` in the chart directory](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent/README.md) for detailed instructions on how to configure and deploy your agent.
:::


:::note
The launch agent uses [Kaniko](https://github.com/GoogleContainerTools/kaniko) to build container images inside of Kubernetes. Kaniko is a tool that builds container images from a Dockerfile, inside a container or Kubernetes cluster. For more information about Kaniko, see the [Kaniko](https://github.com/GoogleContainerTools/kaniko) documentation.

If you want to use the Launch agent without the ability to build new images, you can use the `noop` builder type when you configure your launch agent. More info [here](../launch/run-agent.md#builders).
:::

### Deploy your launch agent with helm charts
Deploy your agent with the launch agent chart from [W&B's official helm-charts repository](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent).

1. Install the wandb/helm-charts repo:
```bash
helm repo add wandb https://wandb.github.io/helm-charts
```
2. Add your W&B API key and the literal contents of your launch config (`launch-config.yaml`) to the Helm chart `values.yml` in  [`wandb/helm-charts/charts/launch-agent/`](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml). For more information, see the [README.md](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/README.md). 

    For example, suppose we use launch agent config file defined earlier in this page:

    ```yaml title="~/.config/wandb/launch-config.yaml"
    # W&B entity (i.e. user or team) name
    entity: awesome-person-entity

    # Max number of concurrent runs to perform. -1 = no limit
    max_jobs: -1

    # List of queues to poll.
    queues:
    - queue-name
    ```

    The helm `values.yaml` file we created looks like: 

    ```yaml title='values.yaml'
    agent:
    labels: {}
    # W&B API key.
    # highlight-next-line
    apiKey: "<Your-W&B-API-key>"
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

    # Additional target namespaces that the launch 
    # agent can deploy into
    additionalTargetNamespaces:
    - default
    - wandb

    # Literal contents of your launch agent config.
    # highlight-start
    launchConfig: |
    metadata:
        name: queue-name
    entity: awesome-person-entity
    max_jobs: -1
    queues:
    - queue-name
    # highlight-end  

    # Set to false to disable volcano install.
    volcano: true

    # The contents of a git credentials file. This is stored
    # in a Kubernetes secret and mounted in the agent container. 
    # Set this if you want to clone private repos.
    gitCreds: |

    # Annotations for the wandb service account. 
    # Useful when setting up workload identity on gcp.
    serviceAccount:
    annotations:
        iam.gke.io/gcp-service-account:
        azure.workload.identity/client-id:

    # Set to access key for azure storage if 
    # you use kaniko with azure.
    azureStorageAccessKey: ""
    ```


3. Navigate to the terminal where you will deploy the launch agent to. Use the helm upgrade (with `--install` flag specified) command to create install and create a release namespace for the helm chart:

    ```bash
    helm upgrade --namespace=wandb \ 
    --create-namespace --install wandb-launch wandb/launch-agent \
    -f ./values.yaml --namespace=wandb-launch
    ```


### Deploy your launch agent with manual cluster configuration
In order to run a launch agent in your cluster without the use of Helm, you will need to create a few other resources in your cluster:

* Namespace
* Service account and roles
* W&B API Key
* Agent configuration


:::tip
In this guide we separated the different resources. However, you can aggregate them into a single file and apply them all at once.
:::

#### Namespace

The following Kubernetes manifest will create a namespace called `wandb` with the `pod-security.kubernetes.io/enforce` and `pod-security.kubernetes.io/warn` labels set to `baseline` and `latest`. This will ensure that all pods created in this namespace will be subject to the baseline pod security policy.

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: wandb
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/enforce-version: latest
    pod-security.kubernetes.io/warn: baseline
    pod-security.kubernetes.io/warn-version: latest
```

#### Service account and roles

The following Kubernetes manifest will create a role named `wandb-launch-agent` in the `wandb` namespace. This role will allow the agent to create pods, configmaps, secrets, and pods/log in the `wandb` namespace. The `wandb-cluster-role` will allow the agent to create pods, pods/log, secrets, jobs, and jobs/status in any namespace of your choice. Make you sure fill in the TODO in the `ClusterRoleBinding` to specify the namespace you want to launch your runs into.

This role will be bound to the `wandb-launch-agent` service account.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: wandb
  name: wandb-launch-agent
rules:
  - apiGroups: [""]
    resources: ["pods", "configmaps", "secrets", "pods/log"]
    verbs: ["create", "get", "watch", "list", "update", "delete", "patch"]
  - apiGroups: ["batch"]
    resources: ["jobs", "jobs/status"]
    verbs: ["create", "get", "watch", "list", "update", "delete", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: job-creator
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "secrets"]
    verbs: ["create", "get", "watch", "list", "update", "delete", "patch"]
  - apiGroups: ["batch"]
    resources: ["jobs", "jobs/status"]
    verbs: ["create", "get", "watch", "list", "update", "delete", "patch"]
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: wandb-launch-serviceaccount
  namespace: wandb
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: wandb-launch-role-binding
  namespace: wandb
subjects:
  - kind: ServiceAccount
    name: wandb-launch-serviceaccount
    namespace: wandb
roleRef:
  kind: Role
  name: wandb-launch-agent
  apiGroup: rbac.authorization.k8s.io

apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: wandb-launch-cluster-role-binding
  namespace: default #TODO: SET YOUR TRAINING NAMESPACE
subjects:
  - kind: ServiceAccount
    name: wandb-launch-serviceaccount
    namespace: wandb
roleRef:
  kind: ClusterRole
  name: job-creator
  apiGroup: rbac.authorization.k8s.io
---
```

#### W&B API key

You will need to create a secret in the `wandb` namespace that contains your W&B API key. This secret will be used by the agent to authenticate with the W&B API so it can pop jobs from your queue and report metrics from launched runs.

```sh
kubectl -n wandb create secret  \
    generic wandb-api-key       \
    --from-literal=password=<your-wandb-api-key>
```

#### Agent configuration

  To run a launch agent in your cluster without the use of Helm, you will need to create a configmap in the `wandb` namespace that contains the configuration for your agent. This configmap will be used by the agent to configure the agent itself. This configuration will depend heavily on your cloud provider and the resources you have available to you. 



  ```yaml title="~/.config/wandb/launch-config.yaml"   
  apiVersion: v1
  kind: ConfigMap
  metadata:
    name: wandb-launch-configmap
    namespace: wandb
  data:
    wandb-base-url: https://api.wandb.ai # TODO: set your base_url here
    launch-config.yaml: |
      max_jobs: -1 # TODO: set max concurrent jobs here
      queues:
      - default # TODO: set queue name here
      environment:
        type: gcp
        region: us-central1 # TODO: set gcp region here
      registry:
        type: gcr
        repository: # TODO: set name of artifact repository name here
        image-name: launch-images # TODO: set name of image here
      builder:
        type: kaniko
        build-context-store: gs://my-bucket/... # TODO: set your build context store here   
  ```

  You can find more information in our [agent documentation](../launch/run-agent.md#agent-configuration).


Now that you have created all the resources needed to run the agent, you can deploy the agent to your cluster. 

The following manifest defines a Kubernetes cluster deployment that will run the agent in your cluster in one container. The agent will run in the `wandb` namespace, use the `wandb-launch-agent` service account. Our API key will be mounted as the `WANDB_API_KEY` environment variable in the container. Our configmap will be mounted as a volume in the container at `/home/launch-agent/launch-config.yaml`.

```yaml title="launch-agent.yaml"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: launch-agent
  namespace: wandb
spec:
  replicas: 1
  selector:
    matchLabels:
      app: launch-agent
  template:
    metadata:
      labels:
        app: launch-agent
    spec:
      serviceAccountName: wandb-launch-serviceaccount
      containers:
        - name: launch-agent
          image: <latest-agent-release>
          resources:
            limits:
              memory: "2Gi"
              cpu: "1000m"
          securityContext:
            allowPrivilegeEscalation: false
            runAsNonRoot: true
            runAsUser: 1000
            capabilities:
              drop: ["ALL"]
            seccompProfile:
              type: RuntimeDefault
          env:
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wandb-api-key
                  key: password
            - name: WANDB_BASE_URL
              valueFrom:
                configMapKeyRef:
                  name: wandb-launch-configmap
                  key: wandb-base-url
          volumeMounts:
            - name: wandb-launch-config
              mountPath: /home/launch_agent/.config/wandb
              readOnly: true
      volumes:
        - name: wandb-launch-config
          configMap:
            name: wandb-launch-configmap
```
We recommend you pull the latest agent image from our public docker registry. You can find the latest image tag [here](https://hub.docker.com/r/wandb/launch-agent-dev/tags?page=1&ordering=last_updated).


Check the status of your deployment with the following command:

```sh
kubectl -n wandb describe deployment launch-agent
``` -->





<!-- 
::tip
The launch agent creates a [Kubernetes Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) for each W&B run that is removed from a launch queue that targets Kubernetes cluster.
:::


The launch agent will automatically set the following values in the top level of a Kubernetes Job spec:

```yaml title="job.yaml"
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 60
  template:
    spec:
      restartPolicy: Never
      containers:  # These security defaults are applied to all containers in the pod spec.
      - securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
          seccompProfile:
            type: RuntimeDefault
``` -->
