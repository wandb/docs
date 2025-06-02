---
description: Guide for updating W&B (Weights & Biases) version and license across
  different installation methods.
menu:
  default:
    identifier: server-upgrade-process
    parent: self-managed
title: Update W&B license and version
url: guides/hosting/server-upgrade-process
weight: 6
---

This page explains how to update your W&B Server version and license using the same method you used to install W&B Server.

## Supported deployment mechanisms
The following table lists how to update your license and version based on your deploymenth method.

<a id="supported-deployment-types"></a>
| Release Type    | Description         |
| ---------------- | ------------------ |
| [Terraform]({{< relref "#update-with-terraform" >}}) | W&B supports three public Terraform modules for cloud deployment: [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest), [GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest), and [Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest). |
| [Helm]({{< relref "#update-with-helm" >}})              | You can use the [Helm Chart](https://github.com/wandb/helm-charts) to install W&B into an existing Kubernetes cluster.  |

{{% readfile "/_includes/server-kubernetes-requirements.md" %}}

For details, refer to [Reference Architecture]({{< relref "ref-arch.md" >}}).

## Update license and version
Select your deployment method to continue.

{{< tabpane text=true >}}
  {{% tab header="Update with Terraform" value="update-with-terraform" %}}

1. Select your cloud provider to obtain the appropriate Terraform module.

   |Cloud provider| Terraform module|
   |-----|-----|
   |AWS|[AWS Terraform module](https://registry.terraform.io/modules/wandb/wandb/aws/latest)|
   |GCP|[GCP Terraform module](https://registry.terraform.io/modules/wandb/wandb/google/latest)|
   |Azure|[Azure Terraform module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)|
1. Within your Terraform configuration, update `wandb_version` and `license` in your Terraform `wandb_app` module configuration:
   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "new_version"
       license       = "new_license_key" # Your new license key
       wandb_version = "new_wandb_version" # Desired W&B version
       ...
   }
   ```
1. Apply the Terraform configuration with `terraform plan` and `terraform apply`:
   ```bash
   terraform init
   terraform apply
   ```
1. (Optional) If you use a `terraform.tfvars` or other `.tfvars` file:
   1. Update or create a `terraform.tfvars` file with the new W&B version and license key.
      ```bash
      terraform plan -var-file="terraform.tfvars"
      ```
   1. Apply the configuration. In your Terraform workspace directory execute:  
      ```bash
      terraform apply -var-file="terraform.tfvars"
      ```

  {{% /tab %}}
  {{% tab header="Update with Helm" value="update-with-helm" %}}

This section shows how to update W&B Server with Helm using one of these methods:
- [Update the W&B spec]({{< relref "#update-wb-with-spec" >}})
- [Update using environment variables]({{< relref "#update-license-and-version-directly" >}})

For more details, see the [upgrade guide](https://github.com/wandb/helm-charts/blob/main/upgrade.md) in the public repository.

### Update the W&B spec {#update-wb-with-spec}
1. Specify a new version by modifying the `image.tag` and/or `license` values in your Helm chart `*.yaml` configuration file:
    ```yaml
    license: 'new_license'
    image:
      repository: wandb/local
      tag: 'new_version'
    ```
1. Execute the Helm upgrade with the following commands:
    ```bash
    helm repo update
    
    helm upgrade --namespace=wandb --create-namespace \
      --install wandb wandb/wandb --version $\{chart_version\} \
      -f $\{wandb_install_spec.yaml\}
    ```

### Update using environment variables {#update-license-and-version-directly}
1. Set the new license key and image tag as environment variables:

   ```bash
   export LICENSE='new_license'
   export TAG='new_version'
   ```
1. Upgrade your Helm release with the command below, merging the new values with the existing configuration:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```
  {{% /tab %}}
{{< /tabpane >}}

## Advanced: Update with admin UI {#update-with-admin-ui}

{{% alert %}}
This section is provided for historical purposes. Self-hosted Docker installations are no longer supported. Refer to [Supported deployment mechanisms]({{< relref "#supported-deployment-mechanisms" >}}).
{{% /alert %}}

This method works only for updating licenses that are not set with an environment variable in the W&B server container, typically in self-hosted Docker installations.

1. Obtain a new license from the [W&B Deployment Page](https://deploy.wandb.ai/), ensuring it matches the correct organization and deployment ID for the deployment you are looking to upgrade.
1. Access the W&B Admin UI at `<host-url>/system-settings`.
1. Navigate to the license management section.
1. Enter the new license key and save your changes.