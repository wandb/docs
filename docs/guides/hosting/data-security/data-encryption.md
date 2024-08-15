---
displayed_sidebar: default
---

# Encryption on Dedicated Cloud

W&B uses a W&B-managed cloud-native key to encrypt the W&B-managed database and object storage in every [Dedicated Cloud](../hosting-options/dedicated_cloud.md), by using the customer-managed encryption key (CMEK) capability in each cloud. In this case, W&B acts as a `customer` of the cloud provider, while providing the W&B platform as a service to you. Using a W&B-managed key means that W&B has control over the keys that are used to encrypt the data in each cloud, thus doubling down on its promise to provide a highly safe and secure platform to all of its customers.

A `unique key` is used for encrypting the data in each customer instance, providing another layer of isolation between Dedicated Cloud tenants. The capability is available on AWS, Azure and GCP.

:::info
Dedicated Cloud instances on GCP and Azure that were provisioned before August 2024 use the default cloud provider managed key for encrypting the W&B-managed database and object storage. Only new instances that are created from August 2024 onwards use the W&B-managed cloud-native key for the relevant encryption.

Dedicated Cloud instances on AWS have been using the W&B-managed cloud-native key for encryption from before August 2024.
:::

:::info
W&B doesn't generally allow customers to bring their own cloud-native key to encrypt the W&B-managed database and object storage in their Dedicated Cloud instance. Reason being, multiple teams and personas in an organization could have access to its cloud infrastructure for various reasons. Some of those teams or personas may not have context on W&B as a critical component in the organization's technology stack, and thus may remove the cloud-native key completely or revoke W&B's IAM access to it. Such an action could corrupt all data in the organization's W&B instance and thus leave it in a irrecoverable state.

If your organization needs to use their own cloud-native key to encrypt the W&B-managed database and object storage in order to approve the use of Dedicated Cloud for your AI workflows, W&B can review it on a exception basis. If approved, use of your cloud-native key for encryption will be bound to the `shared responsibility model` of W&B Dedicated Cloud. If your key is removed or if W&B's IAM access to that key is revoked at any point when your Dedicated Cloud instance is live, W&B will not be liable for any resulting data loss or corruption and will not be responsible for recovery of such data.
:::