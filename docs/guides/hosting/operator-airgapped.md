---
description: Deploy W&B Platform with Kubernetes Operator (Airgapped)
displayed_sidebar: default
title: "Tutorial: Deploy W&B in airgapped environment with Kubernetes"
---

## Introduction

This guide provides a comprehensive and user-friendly step-by-step process for deploying the Weights & Biases (W&B) Platform in airgapped environments. 
Since airgapped environments are isolated from the internet, special configurations are required to ensure the proper deployment of W&B components using 
internal repositories for both Helm charts and container images.
The document assume all commands will be executed in a shell console with proper access to the Kubernetes.
Although using command line for the matter of documentation, the process also applies to any continuous deliver tooling used to deploy Kubernetes application.

## Step 1: Prerequisites

Before starting, make sure your environment meets the following requirements:

- Kubernetes version >= 1.29
- Helm version >= 3
- Kubernetes Metrics installed (required for future support of Horizontal Pod Autoscaler support)
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

## Step 2: Prepare Internal Helm Chart repository

Along with the container images, you also must ensure that the following helm charts are available in your internal Helm Chart repository. 


- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [Operator W&B](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)


The `operator` chart is used to deploy the W&B Operator (Controller Manager) while the `operator-wandb` chart will be used with the values configured in the CRD to deploy W&B Platform.

## Step 3: Set Up Helm Repository

Now, configure the Helm repository to pull the W&B Helm charts from your internal repository. Run the following commands to add and update the Helm repository:

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## Step 4: Install W&B Operator (Controller Manager)

The W&B Operator (Controller Manager) is responsible for managing the W&B platform components. To install it in an airgapped environment, 
you need to configure it to use your internal container registry.

To do so, uou must override the default image settings to use your internal container registry and set the key `airgapped: true` to indicate this is the deployment type you expect. Update the `values.yaml` file as shown below:

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

All supported values can be found in the official W&B Operator [repository](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml).

## Step 5: Configure Custom Resource Definitions (CRDs)

Once the W&B Operator is installed, configure the Custom Resource Definitions (CRDs) to point to your internal Helm repository and container registry. 
This configuration ensures that all required components are deployed from your internal resources. Below is an example of how to configure the CRD.

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

To deploy the W&B Platform, the W&B Operator will use the `operator-wandb` chart from your internal repository and use the values from your CRD to configure the helm chart.

All supported values can be found in the Operator W&B [repository](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml).

## Step 6: Deploy the W&B Platform

Finally, after setting up the CRD and W&B Operator, deploy the W&B platform using the following command:

```bash
kubectl apply -f wandb.yaml
```

## Troubleshooting and FAQ

Below are some frequently asked questions (FAQs) and troubleshooting tips to help you during the deployment process:

**I have another ingress class, can I use it?**  
Yes, you can configure your ingress class by modifying the ingress settings in `values.yaml`.

**My certificate bundle has more than one certificate.**  
Split the certificates into multiple entries in the `customCACerts` section of `values.yaml`.

**I don't want the W&B Operator to apply unattended updates.**  
You can disable auto-updates by ensuring you have the latest versions and adjusting the W&B Console settings.

**What if my environment has no connection to external repositories?**  
As long as the `airgapped: true` configuration is enabled, the W&B Operator will not attempt to reach public repositories and will use your internal resources.
