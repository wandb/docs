---
displayed_sidebar: default
---

# IP allowlisting for Dedicated Cloud

You can restrict access to your [Dedicated Cloud](../hosting-options/dedicated_cloud.md) instance from only an authorized list of IP addresses. This applies to the access from your AI workloads to the W&B APIs and from your user browsers to the W&B app UI as well. Once you configure IP allowlisting for your Dedicated Cloud instance, any requests from other unauthorized locations are denied.

IP allowlisting is available on Dedicated Cloud instances on AWS, GCP and Azure.

You can use IP allowlisting with [secure private connectivity](./private-connectivity.md). If you use IP allowlisting with secure private connectivity, W&B recommends using secure private connectivity for all traffic from your AI workloads and majority of the traffic from your user browsers if possible, while using IP allowlisting for instance administration from privileged locations.

:::important
W&B strongly recommends to use [CIDR blocks](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing) assigned to your corporate or business egress gateways rather than individual `/32` IP addresses. Using individual IP addresses is not scalable and has strict limits per cloud.
:::