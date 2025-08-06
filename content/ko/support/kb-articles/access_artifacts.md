---
menu:
  support:
    identifier: ko-support-kb-articles-access_artifacts
support:
- artifacts
title: Who has access to my artifacts?
toc_hide: true
type: docs
url: /support/:filename
---

Artifacts inherit access permissions from their parent project:

* In a private project, only team members can access artifacts.
* In a public project, all users can read artifacts, while only team members can create or modify them.
* In an open project, all users can read and write artifacts.

## Artifacts Workflows

This section outlines workflows for managing and editing Artifacts. Many workflows utilize [the W&B API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ko" >}}), a component of [the client library]({{< relref path="/ref/python/" lang="ko" >}}) that provides access to W&B-stored data.