---
description: Release process for W&B Server
menu:
  default:
    identifier: server-release-process
    parent: w-b-platform
title: Release policies and processes
weight: 20
date: 2025-05-01
---
This page gives details about W&B Server releases and W&B's release policies. This page relates to [W&B Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}}) and [Self-Managed]({{< relref "/guides/hosting/hosting-options/self-managed/" >}}) deployments. To learn more about an individual W&B Server release, refer to [W&B release notes]({{< relref "/ref/release-notes/" >}}).

W&B fully manages [W&B Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) and the details in this page do not apply.

## Release support and end of life policy
W&B supports a major W&B Server release for 12 months from its initial release date.
- **Dedicated Cloud** instances are automatically updated to maintain support.
- Customers with **Self-managed** instances are responsible for upgrading in time to maintain support. Avoid staying on an unsupported version.

  {{% alert %}}
  W&B strongly recommends customers with **Self-managed** instances to update their deployments with the latest release at minimum once per quarter to maintain support and receive the latest features, performance improvements, and fixes.
  {{% /alert %}}

## Release types and frequencies
- **Major releases** are produced monthly, and may include new features, enhancements, performance improvements, medium and low severity bug fixes, and deprecations. An example of a major release is `0.68.0`.
- **Patch releases** within a major version are produced as needed, and include critical and high severity bug fixes. An example of a patch release is `0.67.1`.

## Release rollout
1. After testing and validation are complete, a release is first rolled out to all **Dedicated Cloud** instances to keep them fully updated.
1. After additional observation, the release is published, and **Self-managed** deployments can upgrade to it on their own schedule, and are responsible for upgrading in time to comply with the [Release support and End of Life (EOL) policy]({{< relref "#release-support-and-end-of-life-policy" >}}). Learn more about [upgrading W&B Server]({{< relref "/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" >}}).

## Downtime during upgrades
- When a **Dedicated Cloud** instance is upgraded, downtime is generally not expected, but may occur in certain situations:
  - If a new feature or enhancement requires changes to the underlying infrastructure, such as compute, storage or network.
  - To roll out a critical infrastructure change such as a security fix.
  - If the instance's current version has reached its [End of Life (EOL)]({{< relref "/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" >}}) and is upgraded by W&B to maintain support.
- For **Self-managed** deployments, the customer is responsible for implementing a rolling update process that meets their service level objectives (SLOs), such as by [running W&B Server on Kubernetes]({{< relref "/guides/hosting/hosting-options/self-managed/kubernetes-operator/" >}}).

## Feature availability
After installing or upgrading, certain features may not be immediately available.

### Enterprise features
An Enterprise license includes support for important security capabilities and other enterprise-friendly functionality. Some advanced features require an Enterprise license.

- **Dedicated Cloud** includes an Enterprise license and no action is required.
- On **Self-managed** deployments, features that require an Enterprise license are not available until it is set. To learn more or obtain an Enterprise license, refer to [Obtain your W&B Server license]({{< relref "/guides/hosting/hosting-options/self-managed.md#obtain-your-wb-server-license" >}}).

### Private preview and opt-in features
Most features are available immediately after installing or upgrading W&B Server. The W&B team must enable certain features before you can use them in your instance.

{{% alert color="warning" %}}
Any feature in a preview phase is subject to change. A preview feature is not guaranteed to become generally available.
{{% /alert %}}

- **Private preview**: W&B invites design partners and early adopters to test these features and provide feedback. Private preview features are not recommended for production environments.

    The W&B team must enable a private preview feature for your instance before you can use it. Public documentation is not available; instructions are provided directly. Interfaces and APIs may change, and the feature may not be fully implemented.
- **Public preview**: Contact W&B to opt in to a public preview to try it out before it is generally available.

    The W&B team must enable a public preview feature before you can use it in your instance. Documentation may not be complete, interfaces and APIs may change, and the feature may not be fully implemented.

To learn more about an individual W&B Server release, including any limitations, refer to [W&B Release notes]({{< relref "/ref/release-notes/" >}}).


