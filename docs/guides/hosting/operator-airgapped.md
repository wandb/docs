---
description: Hosting W&B Server with Kubernetes Operator (Airgapped)
displayed_sidebar: default
---

# Kubernetes operator for air-gapped instances

## Introduction

This guide provides step-by-step instructions to deploy the W&B Platform in air-gapped customer-managed environments. 
As the air-gapped environments are assumed to be disconnected from the public network, special configuration is required to ensure the successful deployment and functioning of the W&B platform.
Use an internal repository or registry to host the helm charts and container images.
All commands must be executed in a shell console with proper access to the Kubernetes cluster.
You could utilize similar commands in any continuous delivery tooling that you use to deploy Kubernetes applications.

## Step 1: Prerequisites

Before starting, make sure your environment meets the following requirements:

- Kubernetes version >= 1.28
- Helm version >= 3
- Access to an internal container registry with the required W&B images
- Access to an internal Helm repository for W&B Helm charts

## Step 2: Prepare Internal Container Registry

Before proceeding with the deployment, you must ensure that the following container images are available in your internal container registry. 
These images are critical for the successful deployment of W&B components.

```bash
wandb/local                                             0.59.2
wandb/console                                           2.12.2
wandb/controller                                        1.13.0
otel/opentelemetry-collector-contrib                    0.97.0
bitnami/redis                                           7.2.4-debian-12-r9
quay.io/prometheus/prometheus                           v2.47.0
quay.io/prometheus-operator/prometheus-config-reloader  v0.67.0
```

## Step 2: Prepare internal helm chart repository

Along with the container images, you also must ensure that the following helm charts are available in your internal Helm Chart repository. 


- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B Platform](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)


The `operator` chart is used to deploy the W&B Operator (Controller Manager) while the `operator-wandb` chart will be used with the values configured in the CRD to deploy W&B Platform.

## Step 3: Set Up Helm Repository

Now, configure the Helm repository to pull the W&B Helm charts from your internal repository. Run the following commands to add and update the Helm repository:

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## Step 4: Install the kubernetes operator aka controller manager

The W&B kubernetes operator i.e. the controller manager is responsible for managing the W&B platform components. To install it in an air-gapped environment, 
you need to configure it to use your internal container registry.

To do so, you must override the default image settings to use your internal container registry and set the key `airgapped: true` to indicate the expected deployment type. Update the `values.yaml` file as shown below:

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

You can find all supported values in the [official kubernetes operator repository](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml).

## Step 5: Configure Custom Resource Definitions (CRDs)

After installing the W&B kubernetes operator, you must configure the Custom Resource Definitions (CRDs) to point to your internal Helm repository and container registry. 
This configuration ensures that your internal registry & repository are utilized to deploy the required components of the W&B platform. Below is an example of how to configure the CRD.

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
        name: s3.yourdomain.com
        provider: s3
      mysql:
        database: wandb
        host: mysql.home.lab
        password: password
        port: 3306
        user: wandb
    
    # Ensre it's set to use your own MySQL
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

To deploy the W&B platform, the kubernetes Operator will use the `operator-wandb` chart from your internal repository and use the values from your CRD to configure the helm chart.

You can find all supported values in the [official kubernetes operator repository](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml).

## Step 6: Deploy the W&B platform

Finally, after setting up the kubernetes operator and the CRD, deploy the W&B platform using the following command:

```bash
kubectl apply -f wandb.yaml
```

## Troubleshooting and FAQ

Refer to the below frequently asked questions (FAQs) and troubleshooting tips during the deployment process:

**We have another ingress class. Can we use that?**  
Yes, you can configure your ingress class by modifying the ingress settings in `values.yaml`.

**Our certificate bundle has more than one certificate. Would that work?**  
You must split the certificates into multiple entries in the `customCACerts` section of `values.yaml`.

**I don't want the W&B Operator to apply unattended updates.**  
You can disable auto-updates by ensuring you have the latest versions and adjusting the W&B Console settings.

**What if my environment has no connection to external repositories?**  
As long as the `airgapped: true` configuration is enabled, the W&B Operator will not attempt to reach public repositories and will use your internal resources.
