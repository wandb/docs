---
title: Disable automatic updates for W&B Server
description: Learn how to disable automatic updates for W&B Server.
menu:
  default:
    identifier: disable-automatic-app-version-updates
    parent: self-managed
weight: 99
---

This page shows how to disable automatic version upgrades for W&B Server and pin its version. These instructions work for deployments managed by the [W&B Kubernetes Operator]({{< relref "/guides/hosting/hosting-options/self-managed/kubernetes-operator/" >}}) only.

{{% alert %}}
W&B supports a major W&B Server release for 12 months from its initial release date. Customers with **Self-managed** instances are responsible for upgrading in time to maintain support. Avoid staying on an unsupported version. W&B strongly recommends customers with **Self-managed** instances to update their deployments with the latest release at minimum once per quarter to maintain support and receive the latest features, performance improvements, and fixes.
{{% /alert %}}

## Requirements

- W&B Kubernetes Operator `v1.13.0` or newer
- System Console `v2.12.2` or newer

To verify that you meet these requirements, refer to the W&B Custom Resource or Helm chart for your instance. Check the `version` values for the `operator-wandb` and `system-console` components.

## Disable automatic updates
1. Log in to the W&B App as a user with the `admin` role.
2. Click the user icon at the top, then click **System Console**.
3. Go to **Settings** > **Advanced**, then select the **Other** tab.
4. In the **Disable Auto Upgrades** section, turn on **Pin specific version**.
5. Click the **Select a version** drop-down, select a W&B Server version.
6. Click **Save**.

    {{< img src="/images/hosting/disable_automatic_updates_saved_and_enabled.png" alt="Disable Automatic Updates Saved" >}}

    Automatic upgrades are turned off and W&B Server is pinned at the version you selected.
1. Verify that automatic upgrades are turned off. Go to the **Operator** tab and search the reconciliation logs for the string `Version pinning is enabled`.

```
│info 2025-04-17T17:24:16Z wandb default No changes found
│info 2025-04-17T17:24:16Z wandb default Active spec found
│info 2025-04-17T17:24:16Z wandb default Desired spec
│info 2025-04-17T17:24:16Z wandb default License
│info 2025-04-17T17:24:16Z wandb default Version Pinning is enabled
│info 2025-04-17T17:24:16Z wandb default Found Weights & Biases instance, processing the spec...
│info 2025-04-17T17:24:16Z wandb default === Reconciling Weights & Biases instance...
```