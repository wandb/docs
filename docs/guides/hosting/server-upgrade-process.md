---
description: Guide for updating W&B (Weights & Biases) version and license across different installation methods.
displayed_sidebar: default
---

# Server upgrade process

When updating the W&B Server Version and License information, tailor the process to the initial installation method. Here are the primary methods for installing W&B and the corresponding update processes:

| Release Type                                               | Description                                                                                                                                                                                                                                                                                   |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [terraform](./how-to-guides#wb-production-and-development) | W&B supports three public Terraform modules for cloud deployment: [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest), [GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest), and [Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest). |
| [helm](./how-to-guides/bare-metal#helm-chart)              | You can use the [Helm Chart](https://github.com/wandb/helm-charts) to install W&B into an existing Kubernetes cluster.                                                                                                                                                                        |
| [docker](./how-to-guides/bare-metal#docker-deployment)     | Docker latest docker image can found in the [W&B Docker Registry](https://hub.docker.com/r/wandb/local/tags).                                                                                                                                                                                 |

## Updating via Terraform

1. To upgrade, adjust the `wandb_version` and `license` for your specific cloud provider, update the following in your Terraform `wandb_app` module configuration:

   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "~> new_version"

       license       = "new_license_key" # Your new license key
       wandb_version = "new_wandb_version" # Desired W&B version
       ...
   }
   ```

2. Apply the Terraform configuration with `terraform plan` and `terraform apply`.

If you are opting to use a `terraform.tfvars` or other `.tfvars` file:

1. **Modify `*.tfvars` File:** Update or create a `terraform.tfvars` file with the new W&B version and license key.

2. **Apply Configuration:** Execute:  
   `terraform plan -var-file="terraform.tfvars"`  
   `terraform apply -var-file="terraform.tfvars"`  
    in your Terraform workspace directory.

## Updating via Helm

### Update W&B via spec

1. Specify a new version by modifying the `image.tag` and/or `license` values in your Helm chart `*.yaml` configuration file:

   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```

2. Execute the Helm upgrade with the following command:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     -f ${wandb_install_spec.yaml}
   ```

### Update license and version directly

1. Set the new license key and image tag as environment variables:

   ```bash
   export LICENSE='new_license'
   export TAG='new_version'
   ```

2. Upgrade your Helm release with the command below, merging the new values with the existing configuration:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```

For more details, see the [upgrade guide](https://github.com/wandb/helm-charts/blob/main/UPGRADE.md) in the public repository.

## Updating via Docker container

1. Choose a new version from the [W&B Docker Registry](https://hub.docker.com/r/wandb/local/tags).
2. Pull the new Docker image version with:

   ```bash
   docker pull wandb/local:<new_version>
   ```

3. Update your Docker container to run the new image version, ensuring you follow best practices for container deployment and management.

For docker `run` examples and further details, refer to the [Docker deployment](./how-to-guides/bare-metal##docker-deployment).

## Updating via admin UI

This method is only works for updating licenses that are not set via an environment variable in the W&B server container, typically in self-hosted Docker installations.

1. Obtain a new license from the [W&B Deployment Page](https://deploy.wandb.ai/), ensuring it matches the correct organization and deployment ID for the deployment you are looking to upgrade.
2. Access the W&B Admin UI at `<host-url>/system-settings`.
3. Navigate to the license management section.
4. Enter the new license key and save your changes.

## W&B Dedicated Cloud Updates

:::note
For dedicated installations, W&B upgrades your server version on a monthly basis. More information can be found in the [release process](./server-release-process) docs.
:::
