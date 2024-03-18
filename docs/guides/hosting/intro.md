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

## How do I choose between a Dedicated, SaaS or Customer managed hosting type?
W&B recommends that you consider a W&B managed hosting option (W&B SaaS Cloud or W&B Dedicated Cloud) before you privately host W&B Server on your infrastructure. INSERT.


## Next steps

1. Obtain your W&B license. You need a W&B license to complete your configuration of a W&B Server. [INSERT link to page]
2. See the Basic Setup guide[LINK] to set up a developer or trial environment.