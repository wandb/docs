---
description: Deploy W&B Platform with Kubernetes Operator (Airgapped)
displayed_sidebar: default
---

# Kubernetes operator for air-gapped instances

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

## Step 2: Prepare internal Helm chart repository

Along with the container images, you also must ensure that the following Helm charts are available in your internal Helm Chart repository. 


- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B Platform](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)


The `operator` chart is used to deploy the W&B Operator, or the Controller Manager. While the `platform` chart is used to deploy the W&B Platform using the values configured in the custom resource definition (CRD).

## Step 3: Set up Helm repository

Now, configure the Helm repository to pull the W&B Helm charts from your internal repository. Run the following commands to add and update the Helm repository:

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## Step 4: Install the Kubernetes operator

The W&B Kubernetes operator, also known as the controller manager, is responsible for managing the W&B platform components. To install it in an air-gapped environment, 
you must configure it to use your internal container registry.

To do so, you must override the default image settings to use your internal container registry and set the key `airgapped: true` to indicate the expected deployment type. Update the `values.yaml` file as shown below:

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

You can find all supported values in the [official Kubernetes operator repository](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml).

## Step 5: Configure CustomResourceDefinitions 

After installing the W&B Kubernetes operator, you must configure the Custom Resource Definitions (CRDs) to point to your internal Helm repository and container registry. 

This configuration ensures that the Kubernetes operators uses your internal registry and repository are when it deploys the required components of the W&B platform. 

Below is an example of how to configure the CRD.

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

To deploy the W&B platform, the Kubernetes Operator uses the `operator-wandb` chart from your internal repository and use the values from your CRD to configure the Helm chart.

You can find all supported values in the [official Kubernetes operator repository](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml).

## Step 6: Deploy the W&B platform

Finally, after setting up the Kubernetes operator and the CRD, deploy the W&B platform using the following command:

```bash
kubectl apply -f wandb.yaml
```

## Troubleshooting and FAQ

Refer to the below frequently asked questions (FAQs) and troubleshooting tips during the deployment process:

**There is another ingress class. Can that class be used?**  
Yes, you can configure your ingress class by modifying the ingress settings in `values.yaml`.

**The certificate bundle has more than one certificate. Would that work?**  
You must split the certificates into multiple entries in the `customCACerts` section of `values.yaml`.

**How do you prevent the Kubernetes operator from applying unattended updates. Is that possible?**  
You can turn off auto-updates from the W&B console. Reach out to your W&B team for any questions on the supported versions. Also, note that W&B supports platform versions released in last 6 months. W&B recommends performing periodic upgrades. 

**Does the deployment work if the environment has no connection to public repositories?**  
As long as you have enabled the `airgapped: true` configuration, the Kubernetes operator does not attempt to reach public repositories. The Kubernetes operator attempts to use your internal resources.
