---
title: "Who has access to my artifacts?"
tags:
   - artifacts
---

Artifacts inherit the access to their parent project:

* If the project is private, then only members of the project's team have access to its artifacts.
* For public projects, all users have read access to artifacts but only members of the project's team can create or modify them.
* For open projects, all users have read and write access to artifacts.

## Questions about Artifacts workflows

This section describes workflows for managing and editing Artifacts. Many of these workflows use [the W&B API](../track/public-api-guide.md), the component of [our client library](../../ref/python/README.md) which provides access to data stored with W&B.