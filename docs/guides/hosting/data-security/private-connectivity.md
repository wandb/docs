---
title: Configure private connectivity to Dedicated Cloud
---

You can connect to your [Dedicated Cloud](../hosting-options/dedicated_cloud.md) instance over the cloud provider's secure private network. This applies to the access from your AI workloads to the W&B APIs and optionally from your user browsers to the W&B app UI as well. When using private connectivity, the relevant requests and responses do not transit through the public network or internet.

:::info
Secure private connectivity is available in preview as an advanced security option with Dedicated Cloud.
:::

Secure private connectivity is available on Dedicated Cloud instances on AWS, GCP and Azure:

* Using [AWS Privatelink](https://aws.amazon.com/privatelink/) on AWS
* Using [GCP Private Service Connect](https://cloud.google.com/vpc/docs/private-service-connect) on GCP
* Using [Azure Private Link](https://azure.microsoft.com/en-us/products/private-link) on Azure

Once enabled, W&B creates a private endpoint service for your instance and provides you the relevant DNS URI to connect to. With that, you can create private endpoints in your cloud accounts that can route the relevant traffic to the private endpoint service. Private endpoints are easier to setup for your AI training workloads running within your cloud VPC or VNet. To use the same mechanism for traffic from your user browsers to the W&B app UI, you must configure appropriate DNS based routing from your corporate network to the private endpoints in your cloud accounts.

:::info
If you would like to use this feature, contact your W&B team.
:::

You can use secure private connectivity with [IP allowlisting](./ip-allowlisting.md). If you use secure private connectivity for IP allowlisting, W&B recommends that you secure private connectivity for all traffic from your AI workloads and majority of the traffic from your user browsers if possible, while using IP allowlisting for instance administration from privileged locations.