---
menu:
  default:
    identifier: ip-allowlisting
    parent: data-security
title: Configure IP allowlisting for Dedicated Cloud
weight: 3
---

You can restrict access to your [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) instance from only an authorized list of IP addresses. This applies to the access from your AI workloads to the W&B APIs and from your user browsers to the W&B app UI as well. Once IP allowlisting has been set up for your Dedicated Cloud instance, W&B denies any requests from other unauthorized locations. Reach out to your W&B team to configure IP allowlisting for your Dedicated Cloud instance.

IP allowlisting is available on Dedicated Cloud instances on AWS, GCP and Azure.

You can use IP allowlisting with [secure private connectivity]({{< relref "./private-connectivity.md" >}}). If you use IP allowlisting with secure private connectivity, W&B recommends using secure private connectivity for all traffic from your AI workloads and majority of the traffic from your user browsers if possible, while using IP allowlisting for instance administration from privileged locations.

{{% alert color="secondary" %}}
W&B strongly recommends to use [CIDR blocks](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing) assigned to your corporate or business egress gateways rather than individual `/32` IP addresses. Using individual IP addresses is not scalable and has strict limits per cloud.
{{% /alert %}}
