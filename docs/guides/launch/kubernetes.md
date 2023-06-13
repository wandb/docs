---
displayed_sidebar: default
---

# Launch on Kubernetes

This guide demonstrates how to use W&B Launch to run ML workloads on a kubernetes (k8s) cluster.

## Building images in Kubernetes

The launch agent uses [Kaniko](https://github.com/GoogleContainerTools/kaniko) to build container images inside of k8s. Kaniko is a tool to build container images from a Dockerfile, inside a container or k8s cluster. Kaniko doesn’t depend on a Docker daemon and executes each command within a Dockerfile completely in userspace. This enables building container images in environments that can’t easily or securely run a Docker daemon, such as a standard k8s cluster.

:::tip
* If you want to use the Launch agent without the ability to build new images, you can use the `noop` builder type when you configure your launch agent. More info [here](../launch/run-agent.md#builders).

* You can use Launch with any Kubernetes system if you use an image based job and you do not require a build for your launch job.
:::


## Create a queue

Before you can launch a job on k8s, you need to create a k8s queue in the W&B App. To create a k8s queue:

1. Navigate to the [Launch application](https://wandb.ai/launch).
2. Click on the **Queues** tab.
3. Click on the **Create Queue** button.
4. Select the **entity** you would like to create the queue in.
5. Enter a name for your queue.
6. Enter a configuration for your queue.
7. Click on the **Create Queue** button.

Congratulations! You have created a k8s queue.

### Queue configuration

The launch agent will create a [Kubernetes Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) for each run that is popped from a Kubernetes queue. The JSON configuration for a Kubernetes queue is used to modify the Job spec that the agent submits to your cluster. The configuration follows the same schema as a [Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/#writing-a-job-spec), except that it is formatted as JSON rather than YAML and supports additional, universal queue configuration fields, e.g. `builder`.

Control over the job spec allows you to specify resource requests, volume mounts, retry strategies, and more for your runs at the queue level. For example, to set a custom environment variable, resource requests, and labels for all runs launched from a queue, you can use a variation of the following configuration:

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

The agent will automatically apply set the following values in the top level of the job spec:

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

## Deploying an agent

Before you can launch a run on k8s, you need to deploy an agent to your cluster.

:::tip
It is **strongly recommended** that you install the launch agent through the [official helm repository](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent). Consult the [`README.md` in the chart directory](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent/README.md) for detailed instructions on how to configure and deploy your agent.
:::

### Manual cluster configuration

In order to run a launch agent in your cluster without the use of Helm, you will need to create a few other resources in your cluster. Here, these are all laid out separately but the purpose of demonstration, but you can aggregate them into a single file and apply them all at once.

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

Lastly, you will need to create a configmap in the `wandb` namespace that contains the configuration for your agent. This configmap will be used by the agent to configure the agent itself. This configuration will depend heavily on your cloud provider and the resources you have available to you. You can find more information in our [agent documentation](../launch/run-agent.md#agent-configuration).

```yaml
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

### Deploying the agent

Now that you have created all the resources needed to run the agent, you can deploy the agent to your cluster. The following manifest defines a k8s deployment that will run the agent in your cluster in one container. The agent will run in the `wandb` namespace, use the `wandb-launch-agent` service account. Our API key will be mounted as the `WANDB_API_KEY` environment variable in the container. Our configmap will be mounted as a volume in the container at `/home/launch-agent/launch-config.yaml`.

We recommend you pull the latest agent image from our public docker registry. You can find the latest image tag [here](https://hub.docker.com/r/wandb/launch-agent-dev/tags?page=1&ordering=last_updated).

```yaml
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

After you have created the deployment, you can check the status of the agent by running the following command:

```sh
kubectl -n wandb describe deployment launch-agent
```
