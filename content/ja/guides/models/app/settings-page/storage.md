---
description: Ways to manage W&B data storage.
menu:
  default:
    identifier: ja-guides-models-app-settings-page-storage
    parent: settings
title: Manage storage
weight: 60
---

If you are approaching or exceeding your storage limit, there are multiple paths forward to manage your data. The path that's best for you will depend on your account type and your current project setup.

## Manage storage consumption
W&B offers different methods of optimizing your storage consumption:

-  Use [reference artifacts]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ja" >}}) to track files saved outside the W&B system, instead of uploading them to W&B storage.
- Use an [external cloud storage bucket]({{< relref path="teams.md" lang="ja" >}}) for storage. *(Enterprise only)*

## Delete data
You can also choose to delete data to remain under your storage limit. There are several ways to do this:

- Delete data interactively with the app UI.
- [Set a TTL policy]({{< relref path="/guides/core/artifacts/manage-data/ttl.md" lang="ja" >}}) on Artifacts so they are automatically deleted.