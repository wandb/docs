---
title: "How can admins restrict what ML engineers have access to modify? For example, changing an image tag may be fine but other job settings may not be."
tags: []
---

### How can admins restrict what ML engineers have access to modify? For example, changing an image tag may be fine but other job settings may not be.
This can be controlled by [queue config templates](./setup-queue-advanced.md), which expose certain queuefields for non-team-admin users to edit within limits defined by admin users. Only team admins can create or edit queues, including defining which fields are exposed and the limits for them.