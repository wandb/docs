---
description: Release process for W&B Server
displayed_sidebar: default
---

# Server release process

## Frequency and deployment types
W&B Server releases apply to the **Dedicated Cloud** and **Self-managed** deployments. There are following kinds of server releases:

| Release type | Description |
|--------------|-------------|
| Monthly | Monthly releases include new features, medium and low severity bug fixes, and enhancements. |
| Patch | Patch releases include critical and high severity bug fixes. Patches are only rarely released, as needed. |
| Feature | The feature release targets a specific release date for a new product feature, which occasionally happens before the standard monthly release. |

All releases are immediately deployed to all **Dedicated Cloud** instances once the acceptance testing phase is complete. It keeps those managed instances fully updated, making the latest features and fixes available to relevant customers. Customers with **Self-managed** instances are responsible for the update process on their own schedule, where they can use[the latest Docker image](https://hub.docker.com/r/wandb/local). Refer to [release support and end of life](#release-support-and-end-of-life).

:::info
Some advanced features are available only with the enterprise license. So even if you get the latest docker image but don't have an enterprise license, you would not be able to take advantage of the relevant advanced capabilities.
:::

:::note
Some new features start in private preview, which means they are only available to design partners or early adopters. You may not have such a feature until the W&B team enables it for your instance.
:::

## Release notes
The release notes for all releases are available at [W&B Server Releases on GitHub](https://github.com/wandb/server/releases). Customers who use Slack can receive automatic release announcements in their W&B Slack channel. Ask your W&B team to enable these updates.

## Release update and downtime
A server release doesn't generally require instance downtime for **Dedicated Cloud** instances and for customers with **Self-managed** deployments who have implemented a proper rolling update process. Downtime may be needed in the following scenarios:

* When a new feature or enhancement requires changes to the underlying infrastructure like compute, storage or network, account teams work with **Dedicated Cloud** customers in advance of such updates.
* When a infrastructure change is required as part of a security patch or to avoid hitting `support end-of-life` for a particular version. For urgent changes, **Dedicated Cloud** customers may not receive notification in advance. The priority here is to keep the fleet secure and fully supported.

For both cases, such updates are applied to the whole **Dedicated Cloud** fleet in batches, and we can not exclude a particular instance from the process. Customers with **Self-managed** instances are responsible to manage such updates on their own schedule. Refer to [release support and end of life](#release-support-and-end-of-life).

## Release support and end of life
W&B supports server releases for six months. **Dedicated Cloud** instances are automatically updated. Customers with **Self-managed** instances need to run a process to update deployments with the latest released. Staying on a version older than six months will significantly limit support.