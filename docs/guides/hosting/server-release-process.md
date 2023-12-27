---
description: Release process for Server
displayed_sidebar: default
---

# Server Release Process

## Frequency and deployment types
W&B Server releases apply to the **Dedicated Cloud** and **Self-managed** deployments. There are following kinds of server releases:

| Release type | Description |
|--------------|-------------|
| Monthly | The monthly general release includes new features, enhancements, and medium & low severity bug fixes. |
| Patch | The patch release includes critical & high severity bug fixes, and is cut only when needed. These could be anywhere from 0 to a few in a month, though we try and eliminate the need altogether by finding & fixing the relevant bugs as part of multi-level testing process (unit, integration and acceptance testing). |
| Feature | The feature release includes specific features that may have been promised to be available by a certain date and thus it needs to be separated from the monthly general release. It's similar to a patch release in that way. |

All releases are immediately deployed to all **Dedicated Cloud** instances once the acceptance testing phase is complete. It keeps those managed instances fully updated, making the latest features and fixes available to relevant customers. Customers with **Self-managed** instances are responsible for the update process on their own schedule, where they can use the latest docker image from [wandb/local](https://hub.docker.com/r/wandb/local). Refer to [release support and end of life](#release-support-and-end-of-life).

:::info
Some advanced features are available only with the enterprise license. So even if you get the latest docker image but don't have an enterprise license, you would not be able to take advantage of the relevant advanced capabilities.
:::

:::note
Some of the new features may be released in private preview and thus are only available to design partners or early adopters. You may not be able to use such a feature until your W&B team enables it for your instance.
:::

## Release notes
The release notes for all releases are available at [Wandb Server Releases on Github](https://github.com/wandb/server/releases). The notes are automatically published to customers with **Dedicated Cloud** or **Self-managed** deployments who collaborate with their W&B teams using Slack. For rest, we recommend to keep track of the latest on the above linked Github page.

Let your W&B team know if you have a collaboration slack channel with them but you are not getting the automatic release note updates.

## Release update and downtime
A server release doesn't generally require instance downtime, at least in case of **Dedicated Cloud** instances and for customers with **Self-managed** deployments who have implemented a proper rolling update process. But downtime may be needed in the following scenarios:

* When a new feature or enhancement requires changes to the underlying infrastructure like compute, storage or network. We try our best to notify the **Dedicated Cloud** customers in advance of such updates, so they can be well prepared.
* When a infrastructure change is required as part of a security patch or to ensure that we're not hitting `support end-of-life` for a particular version. If such a change needs urgent attention, we may not be able to notify the **Dedicated Cloud** customers in advance. The goal here is to keep the fleet secure & fully supportable at all times.

For both cases above, such updates are applied to the whole **Dedicated Cloud** fleet in batches, and we can not exclude a particular instance from the process. Customers with **Self-managed** instances are responsible to manage such updates on their own schedule. Refer to [release support and end of life](#release-support-and-end-of-life).

## Release support and end of life
We officially support server releases from last six months. This should not affect the **Dedicated Cloud** instances as we update those as part of the release process. But customers with **Self-managed** instances should ensure that they're updating their deployment(s) with the latest release when possible. Staying on a version older than six months would mean that they have limited to no means of support.