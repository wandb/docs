---
description: Deploy W&B Platform with Kubernetes Operator (Airgapped)
menu:
  default:
    identifier: operator-airgapped
    parent: kubernetes-operator
title: Kubernetes operator for air-gapped instances
---

## Introduction

This guide provides step-by-step instructions to deploy the W&B Platform in air-gapped customer-managed environments. 

Use an internal repository or registry to host the Helm charts and container images. Run all commands in a shell console with proper access to the Kubernetes cluster.

You could utilize similar commands in any continuous delivery tooling that you use to deploy Kubernetes applications.

## Step 1: Prerequisites

Before starting, make sure your environment meets the following requirements:

- Kubernetes version >= 1.28
- Helm version >= 3
- Access to an internal container registry with the required W&B images
- Access to an internal Helm repository for W&B Helm charts

## Step 2: Prepare internal container registry

Before proceeding with the deployment, you must ensure that the following container images are available in your internal container registry:
* [`docker.io/wandb/controller`](https://hub.docker.com/r/wandb/controller)
* [`docker.io/wandb/local`](https://hub.docker.com/r/wandb/local)
* [`docker.io/wandb/console`](https://hub.docker.com/r/wandb/console)
* [`redis`](https://hub.docker.com/_/redis)
* [`docker.io/otel/opentelemetry-collector-contrib`](https://hub.docker.com/r/otel/opentelemetry-collector-contrib)
* [`quay.io/prometheus/prometheus`](https://quay.io/repository/prometheus/prometheus)
* [`quay.io/prometheus-operator/prometheus-config-reloader`](https://quay.io/repository/prometheus-operator/prometheus-config-reloader)

These images are critical for the successful deployment of W&B components. W&B recommends that you use WSM to prepare the container registry. 

If your organization already uses an internal container registry, you can add the images to it. Otherwise, follow the proceeding section to use a called WSM to prepare the container repository.

You are responsible for tracking the Operator's requirements and for checking for and downloading image upgrades, either by [using WSM]({{< relref "#list-images-and-their-versions" >}}) or by using your organization's own processes.

### Install WSM

Install WSM using one of these methods.

{{% alert %}}
WSM requires a functioning Docker installation.
{{% /alert %}}

#### Bash
Run the Bash script directly from GitHub:

```bash
curl -sSL https://raw.githubusercontent.com/wandb/wsm/main/install.sh | bash
```
The script downloads the binary to the folder in which you executed the script. To move it to another folder, execute:

```bash
sudo mv wsm /usr/local/bin
```

#### GitHub
Download or clone WSM from the W&B managed `wandb/wsm` GitHub repository at `https://github.com/wandb/wsm`. See the `wandb/wsm` [release notes](https://github.com/wandb/wsm/releases) for the latest release.

### List images and their versions

Get an up to date list of image versions using `wsm list`.

```bash
wsm list
```

The output looks similar to the following:

```text
:package: Starting the process to list all images required for deployment...
Operator Images:
  wandb/controller:1.16.1
W&B Images:
  wandb/local:0.62.2
  redis:7.2
  quay.io/prometheus-operator/prometheus-config-reloader:v0.67
  quay.io/prometheus/prometheus:v2.47
  otel/opentelemetry-collector-contrib:0.97
  wandb/console:2.13.1
Here are the images required to deploy W&B. Ensure these images are available in your internal container registry and update the values.yaml accordingly.
```

### Download images

Download all images in the latest versions using `wsm download`.

```bash
wsm download
```

The output looks similar to the following:

```text
Downloading operator helm chart
Downloading wandb helm chart
✓ wandb/controller:1.16.1
✓ redis:7.2
✓ otel/opentelemetry-collector-contrib:0.97
✓ quay.io/prometheus-operator/prometheus-config-reloader:v0.67
✓ wandb/console:2.13.1
✓ quay.io/prometheus/prometheus:v2.47

  Done! Installed 7 packages.
```

WSM downloads a `.tgz` archive for each image to the `bundle` directory.

## Step 3: Prepare internal Helm chart repository

Along with the container images, you also must ensure that the following Helm charts are available in your internal Helm Chart repository. The WSM tool introduced in the last step can also download the Helm charts. Alternatively, download them here:


- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B Platform](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)


The `operator` chart is used to deploy the W&B Operator, which is also referred to as the Controller Manager. The `platform` chart is used to deploy the W&B Platform using the values configured in the custom resource definition (CRD).

## Step 4: Set up Helm repository

Now, configure the Helm repository to pull the W&B Helm charts from your internal repository. Run the following commands to add and update the Helm repository:

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## Step 5: Install the Kubernetes operator

The W&B Kubernetes operator, also known as the controller manager, is responsible for managing the W&B platform components. To install it in an air-gapped environment, 
you must configure it to use your internal container registry.

To do so, you must override the default image settings to use your internal container registry and set the key `airgapped: true` to indicate the expected deployment type. Update the `values.yaml` file as shown below:

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

Replace the tag with the version that is available in your internal registry.

Install the operator and the CRD:
```bash
helm upgrade --install operator wandb/operator -n wandb --create-namespace -f values.yaml
```

For full details about the supported values, refer to the [Kubernetes operator GitHub repository](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml).

## Step 6: Configure W&B Custom Resource 

After installing the W&B Kubernetes operator, you must configure the Custom Resource (CR) to point to your internal Helm repository and container registry.

This configuration ensures that the Kubernetes operators uses your internal registry and repository are when it deploys the required components of the W&B platform. 

Copy this example CR to a new file named `wandb.yaml`.

```yaml
apiVersion: apps.wandb.com/v1
kind: WeightsAndBiases
metadata:
  labels:
    app.kubernetes.io/instance: wandb
    app.kubernetes.io/name: weightsandbiases
  name: wandb
  namespace: default

spec:
  chart:
    url: http://charts.yourdomain.com
    name: operator-wandb
    version: 0.18.0

  values:
    global:
      host: https://wandb.yourdomain.com
      license: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      bucket:
        accessKey: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        secretKey: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        name: s3.yourdomain.com:port #Ex.: s3.yourdomain.com:9000
        path: bucket_name
        provider: s3
        region: us-east-1
      mysql:
        database: wandb
        host: mysql.home.lab
        password: password
        port: 3306
        user: wandb
      extraEnv:
        ENABLE_REGISTRY_UI: 'true'
    
    # If install: true, Helm installs a MySQL database for the deployment to use. Set to `false` to use your own external MySQL deployment.
    mysql:
      install: false

    app:
      image:
        repository: registry.yourdomain.com/local
        tag: 0.59.2

    console:
      image:
        repository: registry.yourdomain.com/console
        tag: 2.12.2

    ingress:
      annotations:
        nginx.ingress.kubernetes.io/proxy-body-size: 64m
      class: nginx

    
```

To deploy the W&B platform, the Kubernetes Operator uses the values from your CR to configure the `operator-wandb` Helm chart from your internal repository.

Replace all tags/versions with the versions that are available in your internal registry.

For more information, see the [W&B Operator Configuration reference]({{< relref "/guides/hosting/hosting-options/self-managed/kubernetes-operator/#configuration-reference-for-wb-server" >}}) and the [example `wandb.yaml` file](https://github.com/wandb/helm-charts/blob/main/charts/operator/crds/wandb.yaml) in the Helm chart repository.

## Step 7: Deploy the W&B platform

Now that the Kubernetes operator and the CR are configured, apply the `wandb.yaml` configuration to deploy the W&B platform:

```bash
kubectl apply -f wandb.yaml
```

## FAQ

Refer to the below frequently asked questions (FAQs) and troubleshooting tips during the deployment process:

### There is another ingress class. Can that class be used?
Yes, you can configure your ingress class by modifying the ingress settings in `values.yaml`.

### The certificate bundle has more than one certificate. Would that work?
You must split the certificates into multiple entries in the `customCACerts` section of `values.yaml`.

### How do you prevent the Kubernetes operator from applying unattended updates. Is that possible?
You can turn off auto-updates from the W&B console. Reach out to your W&B team for any questions on the supported versions. W&B supports a major W&B Server release for 12 months from its initial release date. Customers with **Self-managed** instances are responsible for upgrading in time to maintain support. Avoid staying on an unsupported version. Refer to [Release policies and processes]({{< relref "/ref/release-notes/release-policies.md" >}}).

{{% alert %}}
W&B strongly recommends customers with **Self-managed** instances to update their deployments with the latest release at minimum once per quarter to maintain support and receive the latest features, performance improvements, and fixes.
{{% /alert %}}

### Does the deployment work if the environment has no connection to public repositories?
If your configuration sets `airgapped` to `true`, the Kubernetes operator uses only your internal resources and does not attempt to connect to public repositories. 
