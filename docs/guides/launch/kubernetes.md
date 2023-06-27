---
displayed_sidebar: default
---

# Launch on Kubernetes

This guide demonstrates how to use W&B Launch to run ML workloads on a kubernetes cluster.

:::tip
* You can use Launch with any Kubernetes system if you use an image based job and you do not require a build for your launch job.
:::

## Prerequisites
Before you get started, ensure you have:
1. **Kubernetes cluster**
2. **W&B API Key**: If you created a queue in your personal entity, head to [wandb.ai/authorize](wandb.ai/authorize) to get a personal API key. If you created a queue for a W&B, you will need to create a service account in that team or use an API key from a prior service account. For more information on generating service accounts, see these docs.


## 1. Create a queue

Before you can launch a job on Kubernetes cluster, you need to create a Kubernetes cluster queue in the W&B App. To create a Kubernetes cluster queue:

1. Navigate to the [Launch application](https://wandb.ai/launch).
2. Click on the **Queues** tab.
3. Click on the **Create Queue** button.
4. Select the **entity** you would like to create the queue in.
5. Enter a name for your queue.
6. Enter a configuration for your queue. Scroll down to the [Queue configuration for Kubernetes](#queue-configuration-for-kubernetes) for more information on how to define your config.
7. Click on the **Create Queue** button.

### Queue configuration for Kubernetes

The launch agent creates a [Kubernetes Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) for each run that is popped from the launch queue that is configured to use Kubernetes.  The configuration you define for a Kubernetes queue is used to modify the [Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/#writing-a-job-spec) that the launch agent submits to your Kubernetes cluster.

:::note
The launch queue configuration follows the same schema as a Kubernetes Job spec, except that it also supports additional, universal queue configuration fields, such as `builder`.
:::

The launch agent will automatically apply set the following values in the top level of the Kubernetes job spec:

```yaml
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
```
### Example configuration

The following code snippet demonstrates an example of how to set a custom environment variable, resource requests, and labels for all runs launched from a queue:

```json
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "env": [
              {
                "name": "MY_ENV_VAR",
                "value": "some-value"
              }
            ],
            "resources": {
              "requests": {
                "cpu": "1000m",
                "memory": "1Gi"
              }
            }
          }
        ]
      }
    }
  },
  "metadata": {
    "labels": {
      "queue": "k8s-test"
    }
  },
  "namespace": "wandb"
}
```



## 2. Configure the agent
Create a yaml configuration file for the agent you will deploy. Refer to the [Start an agent](./run-agent.md#agent-configuration) page for complete documentation of the agent configuration file. 

The config should contain, at a minimum:

```yaml title="~/.config/wandb/launch-config.yaml"
entity: <your-entity>
queues: [ <your-queue> ]
```


## 3. Adds jobs to your queue
[INSERT]


## 4. Deploy your agent
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

### Deploy with helm
The easiest way to deploy your agent is with the launch agent chart from [W&B's official helm-charts repository](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent).

1. Install the wandb/helm-charts repo:
```bash
helm repo add wandb https://wandb.github.io/helm-charts
```
2. Add your W&B API key and the literal contents of your launch config to `values.yml` in  [`wandb/helm-charts/charts/launch-agent/`](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml). For more information, see the [README.md](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/README.md). 

3. [INSERT]



### Deploy with a manual cluster configuration
In order to run a launch agent in your cluster without the use of Helm, you will need to create a few other resources in your cluster:

* Namespace
* Service account and roles
* W&B API Key
* Agent configuration


:::tip
In this guide we separated the different resources. However, you can aggregate them into a single file and apply them all at once.
:::

#### Namespace

The following kubernetes manifest will create a namespace called `wandb` with the `pod-security.kubernetes.io/enforce` and `pod-security.kubernetes.io/warn` labels set to `baseline` and `latest`. This will ensure that all pods created in this namespace will be subject to the baseline pod security policy.

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

The following kubernetes manifest will create a role named `wandb-launch-agent` in the `wandb` namespace. This role will allow the agent to create pods, configmaps, secrets, and pods/log in the `wandb` namespace. The `wandb-cluster-role` will allow the agent to create pods, pods/log, secrets, jobs, and jobs/status in any namespace of your choice. Make you sure fill in the TODO in the `ClusterRoleBinding` to specify the namespace you want to launch your runs into.

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

```yaml title="launch-config.yaml"
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
```

