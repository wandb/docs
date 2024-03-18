---
slug: /guides/hosting
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# W&B Server for enterprise users

W&B Server is designed for [enterprise-tier users](https://wandb.ai/site/for-enterprise) who want fine-tuned control over: authenticating and authorizing users, managing users, data security compliance, and monitoring tooling. With W&B Server, you can: [bring your own bucket to store sensitive data](./secure-storage-connector.md), [authenticate users with LDAP](./ldap.md), [manage users with SCIM or W&B Python SDK API](./scim.md), and more. 

W&B Server is a resource isolated environment managed by W&B or by yourself. There are three ways you can host W&B Server:
* [W&B Dedicated Cloud](#wb-dedicated-cloud)
* [W&B SaaS Cloud](#wb-saas-cloud)
* [W&B Customer-managed](#wb-customer-managed) (on-prem private cloud or on-prem bare metal)

The following responsibility matrix outlines some of the key differences between the different deployment options:
![](/images/hosting/shared_responsibility_matrix.png)



## Hosting options
The following sections provide an overview of each deployment type. 

### W&B Dedicated Cloud
W&B Dedicated Cloud is a single-tenant, fully managed solution available to enterprise customers who have sensitive use cases and stringent enterprise security controls. INSERT.


For more information about W&B Dedicated Cloud, see [INSERT].

### W&B SaaS Cloud
W&B SaaS Cloud is a multi-tenant, Software as a Service (SaaS) solution where you can access to a fast, secure version of W&B with all of the latest features. INSERT.

For more information about W&B SaaS, see [INSERT].

### W&B Customer-Managed
Deploy and manage W&B to your exact specifications, either on your own customer-managed cloud or on-prem servers. INSERT.

<Tabs
  defaultValue="privatecloud"
  values={[
    {label: 'Private cloud', value: 'privatecloud'},
    {label: 'Bare metal', value: 'bare_metal'},
  ]}>
  <TabItem value="privatecloud">Deploy W&B Server in your private cloud infrastructure. INSERT.</TabItem>
  <TabItem value="bare_metal">Deploy W&B Server on your on-prem, bare metal infrastructure. INSERT.</TabItem>
</Tabs>

For more information about W&B Customer-manged deployment options, see [INSERT].


<!-- OLD -->
<!-- W&B Server is shipped as a packaged Docker image that can be deployed easily into any underlying infrastructure. -->

<!-- :::info
Production-grade features for W&B Server are available for enterprise-tier only.

See the [Basic Setup guide](./how-to-guides/basic-setup.md) to set up a developer or trial environment.
::: -->

<!-- With W&B Server you can configure and leverage features such as:

- [Secure Storage Connector](./secure-storage-connector.md)
- [Single Sign-On](./sso.md)
- [Role-based Access Control via LDAP](./ldap.md)
- [Audit Logs](./audit-logging.md)
- [User Management](./manage-users.md)
- [Prometheus Monitoring](./prometheus-logging.md)
- [Slack Alerts](./slack-alerts.md)
- [Garbage Collect Deleted Artifacts](../artifacts/delete-artifacts.md#how-to-enable-garbage-collection-based-on-how-you-host-wb) -->

<!-- The following sections of the documentation describes different options on how to install W&B Server, the shared responsibility model, step-by-step installation and configuration guides. -->

<!-- ## Recommendations

W&B recommends the following when configuring W&B Server:

1. Run the W&B Server Docker container with an external storage and an external MySQL database in order to preserve the state outside of a container. This protects the data from being accidentally deleted if the container dies or crashes.
2. Leverage Kubernetes to run the W&B Server Docker image and expose the `wandb` service.
3. Set up and manage a scale-able file system if you plan on using W&B Server for production-related work. -->

<!-- ## System Requirements

W&B Server requires a machine with at least

- 4 cores of CPU &
- 8GB of memory (RAM)

Your W&B data will be saved on a persistent volume or external database, ensuring that it is preserved across different versions of the container.

:::tip
For enterprise customers, W&B offers extensive technical support and frequent installation updates for privately hosted instances.
::: -->

<!-- ## Releases

Subscribe to receive notifications from the [W&B Server GitHub repository](https://github.com/wandb/server/releases) when a new W&B Server release comes out.

To subscribe, select the **Watch** button at the top of the GitHub page and select **All Activity**. -->

## Considerations

### Dedicated, SaaS or Customer managed?
W&B recommends that you consider a W&B managed hosting option (W&B SaaS Cloud or W&B Dedicated Cloud) before you privately host W&B Server on your infrastructure. INSERT.


## Next steps

1. Obtain your W&B license. You need a W&B license to complete your configuration of a W&B Server. [INSERT link to page]
2. See the Basic Setup guide[LINK] to set up a developer or trial environment.