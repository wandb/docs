---
title: Disable Automatic Application Version Updates
description: Disable W&B Automatice Application Version Updates
menu:
  default:
    identifier: disable-automatic-app-version-updates
    parent: self-managed
weight: 1
---

This page describes the proceess for how to disable the W&B server application version automatic updates and how to pin a specific version of the application.

## Requirements

This feature requires the [kubernetes operator deployment model](https://docs.wandb.ai/guides/hosting/operator/) running at least `v1.13.0` of the operator and `v2.12.2` of the System Console.

## Disabling Automatic Updates

Begin by navigating to the System Console by logging into the W&B application and clicking on the user Icon in the upper right screen then selecting `System Console`

    {{< img src="/images/hosting/access_system_console_directly.png" alt="" >}}

From the `System Console` navigate to `Settings`, `Advanced`, and then the `Other` tab. Scrolling down will reveal the section titled `Disable Auto Upgrades`. This will be in an untoggled state if the automatic application version upgrades are enabled and the version is not pinned.

    {{< img src="/images/hosting/disable_automatic_updates_untoggled.png" alt="" >}}

To disable automatic application version updates, click the toggle and select the desired version to pin the application to.

    {{< img src="/images/hosting/disable_automatic_updates_enabling.png" alt="" >}}

Once the desired application version has been selected, click the `Save` button to complete the process.

    {{< img src="/images/hosting/disable_automatic_updates_unsaved.png" alt="" >}}

With the desired application version pinned and the `Save` button clicked, automatic application version upgrades will successfully be disabled on the W&B server.

    {{< img src="/images/hosting/disable_automatic_updates_saved_and_enabled.png" alt="" >}}

This can be verified by going to the `Operator` tab and reviewing the reconciliation logs from the kubernetes operator that manages the W&B server installation.

    {{< img src="/images/hosting/disable_automatic_updates_operator_logs.png" alt="" >}}