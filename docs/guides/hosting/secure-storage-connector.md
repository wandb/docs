---
displayed_sidebar: default
---

# Secure storage connector

The team-level secure storage connector allows teams within W&B to utilize a separate cloud file storage bucket from the rest of the W&B instance. This provides greater data access control and data isolation for teams with highly sensitive data or strict compliance requirements.

:::info
This feature is currently only available for Google Cloud Platform and Amazon Web Services. To request a license to enable this feature, email contact@wandb.com.
:::

A cloud storage bucket can be configured only once for a team at the time of team creation. To provision a bucket, W&B recommends that you use a Terraform configuration managed by W&B for [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector) or [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector).

Select **External Storage** when you create a team tp configure a cloud storage bucket. Select your provider and fill out your bucket name and storage encryption key ID, if applicable, and select **Create Team**.

An error or warning will appear at the bottom of the page if there are issues accessing the bucket or the bucket has invalid settings.

![](/images/hosting/prod_setup_secure_storage.png)

Only system administrators have the permissions to configure the secure storage connector. The same cloud storage bucket can be used amongst multiple teams by selecting an existing cloud storage bucket from the dropdown.

:::note
W&B uses a garbage collection process to delete W&B artifacts. For more information on how W&B deletes artifacts, and for information on how to enable garbage collection for W&B Server, see the [Delete artifacts](../artifacts/delete-artifacts.md) page.
:::