---
displayed_sidebar: default
---

# Secure storage connector

## Introduction
The secure storage connector allows you to store the artifacts and other files pertaining to your W&B runs within a storage bucket that's managed by you. It provides you with more control over where you store the files for your AI workflows, and may help conform to your enterprise governance requirements. There are two levels of secure storage:

* Instance level: The instance level secure storage allows you to use your own managed bucket to store any files that your users may access as part of any runs in your W&B Server instance. This capability is only available for **Dedicated Cloud** and **Self-managed** instances. In the case of **SaaS Cloud**, the instance level bucket is fully managed by W&B, which is also the default for **Dedicated Cloud**. 
* Team level: The team level secure storage allows teams within your organization to utilize a separate storage bucket from the one used at the instance level. This provides greater data access control and data isolation for teams with highly sensitive data or strict compliance requirements. This capability is available for all kinds of W&B organizations, including on **SaaS Cloud**.

:::info
In case of **Dedicated Cloud** or **Self-managed** instances, you could configure secure storage connector at both the instance level and separately for any or all teams within your organization. Taking an example, assume there are two teams called _Omega_ & _Kappa_ in a **Dedicated Cloud** instance, and you have configured secure storage connector at the instance level. If you also configure the secure storage connector separately for team _Omega_, team level secure storage is used to store the files pertaining to that team’s runs. On the other hand if you do not configure the secure storage connector for team _Kappa_, instance level secure storage is used to store the files pertaining to that team’s runs.
:::

:::note
When it comes to **Self-managed** instances, the instance level secure storage connector is the default since the deployment is fully managed by the customer. In such a case the capability doesn't have a special significance, as compared to when you use it with **Dedicated Cloud**. Though the team level secure storage connector provides the same intended benefits even for **Self-managed** instances, especially when different business units and departments share an instance to efficiently utilize the infrastructure and administrative resources. That would also apply to firms that have separate project teams managing ML workflows for separate customer engagements.
:::

## Availability matrix
The following table shows the availability of different kinds of secure storage connector across different W&B Server deployment types. X means the feature is available on the specific deployment type.

| W&B Server deployment type | Instance level | Team level | Additional information |
|----------------------------|--------------------|----------------|------------------------|
| Dedicated Cloud | X | X | Both the instance and team level secure storage connector are available only for Amazon Web Services and Google Cloud Platform. W&B fully manages the default and only bucket for Azure cloud. |
| SaaS Cloud | | X | The team level secure storage connector is available only for Amazon Web Services and Google Cloud Platform. W&B fully manages the default and only bucket for Azure cloud. |
| Self-managed | X | X | Refer to the preceding `note` on significance of secure storage connector for **Self-managed** instances. Also, you could use the S3-compatible secure storage like [MinIO](https://github.com/minio/minio) with such instances. |

:::note
You should configure your instance or team level secure storage connector with the bucket from the same cloud as your instance in **Dedicated Cloud**. It is not possible to configure a bucket from a different cloud.
:::

:::note
W&B uses a garbage collection process to delete W&B artifacts. For more information on how W&B deletes artifacts, and for information on how to enable garbage collection based on how you W&B, see the [How to enable garbage collection based on how you host W&B](../artifacts/delete-artifacts.md#how-to-enable-garbage-collection-based-on-how-wb-is-hosted) page.
:::

## Configure team level secure storage connector
A cloud storage bucket can be configured only once for a team at the time of team creation. To provision a bucket, W&B recommends that you use a Terraform module managed by W&B for [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector) or [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector).

Select **External Storage** when you create a team to configure a cloud storage bucket. Select your provider and fill out your bucket name and storage encryption key ID, if applicable, and select **Create Team**.

An error or warning will appear at the bottom of the page if there are issues accessing the bucket or the bucket has invalid settings.

![](/images/hosting/prod_setup_secure_storage.png)

Only system administrators have the permissions to configure the secure storage connector. The same cloud storage bucket can be used amongst multiple teams by selecting an existing cloud storage bucket from the dropdown.

:::note
Reach out to your W&B team to configure the instance level secure storage connector for your **Dedicated Cloud** or **Self-managed** instance.
:::