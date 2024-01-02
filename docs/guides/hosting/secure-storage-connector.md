---
displayed_sidebar: default
---

# Secure storage connector

## Introduction
The secure storage connector (referred to as SSC henceforth) allows you to store the artifacts and other files pertaining to your W&B runs within a storage bucket that's managed by you. It provides you with more control over where the files for your AI workflows are stored, and may help conform to your enterprise governance requirements. There are two levels of secure storages:

* Instance level: The instance level secure storage allows you to use your own managed bucket to store the files that are accessed as part of any runs in your W&B Server instance. This capability is only available for **Dedicated Cloud** and **Self-managed** instances. In the case of **SaaS Cloud**, the instance level bucket is fully managed by W&B, which is also the default for **Dedicated Cloud**. 
* Team level: The team level secure storage allows teams within your organization to utilize a separate storage bucket from the one used at the instance level. This provides greater data access control and data isolation for teams with highly sensitive data or strict compliance requirements. This capability is available for all kinds of W&B organizations, including on **SaaS Cloud**.

:::info
In case of **Dedicated Cloud** or **Self-managed** instances, you could configure SSC at both the instance level and separately for any or all teams within your organization. Taking an example, assume there are two teams called _Omega_ & _Kappa_ in a **Dedicated Cloud** instance, and the SSC is configured at the instance level. If the SSC is also configured separately for team _Omega_, then the files pertaining to that team’s runs will be stored on the team level secure storage. On the other hand if SSC is not configured for the team _Kappa_, then the files pertaining to that team's runs will be stored on the instance level secure storage.
:::

:::note
When it comes to **Self-managed** instances, the instance level SSC is the default since the deployment is fully managed by the customer. In such a case the capability doesn't have a special significance, as compared to when it is used with **Dedicated Cloud**. Though the team level SSC provides the same intended benefits even for **Self-managed** instances, especially when different business units and departments share the same instance to efficiently utilize the infrastructure and administrative resources. That would also apply to firms that have separate project teams managing ML workflows for separate customer engagements.
:::

## Availability matrix
The following table shows the availability of different kinds of SSC across different W&B Server deployment types (X means it is available):

| W&B Server deployment type | Instance level SSC | Team level SSC | Additional information |
|----------------------------|--------------------|----------------|------------------------|
| Dedicated Cloud | X | X | Both the instance and team level SSC are available only for AWS and GCP. For Azure, the default bucket fully managed by W&B is used for all file storage. |
| SaaS Cloud | | X | The team level SSC is available only for AWS and GCP. For Azure, the default bucket fully managed by W&B is used for all file storage. |
| Self-managed | X | X | Refer to the `note` above on significance of SSC for **Self-managed** instances. Also, you could use the S3-compatible secure storage like [MinIO](https://github.com/minio/minio) with such instances. |

:::note
When using **Dedicated Cloud**, you can configure the instance or team level SSC with the bucket from the same cloud in which your instance is hosted. It is not possible to configure a bucket from a different cloud.
:::

:::note
W&B uses a garbage collection process to delete W&B artifacts. For more information on how W&B deletes artifacts, and for information on how to enable garbage collection based on how you W&B, see the [How to enable garbage collection based on how you host W&B](../artifacts/delete-artifacts.md#how-to-enable-garbage-collection-based-on-how-wb-is-hosted) page.
:::

## Configure team level SSC
A cloud storage bucket can be configured only once for a team at the time of team creation. To provision a bucket, W&B recommends that you use a Terraform module managed by W&B for [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector) or [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector).

Select **External Storage** when you create a team to configure a cloud storage bucket. Select your provider and fill out your bucket name and storage encryption key ID, if applicable, and select **Create Team**.

An error or warning will appear at the bottom of the page if there are issues accessing the bucket or the bucket has invalid settings.

![](/images/hosting/prod_setup_secure_storage.png)

Only system administrators have the permissions to configure the secure storage connector. The same cloud storage bucket can be used amongst multiple teams by selecting an existing cloud storage bucket from the dropdown.

:::note
Reach out to your W&B team to configure the instance level SSC for your **Dedicated Cloud** or **Self-managed** instance.
:::