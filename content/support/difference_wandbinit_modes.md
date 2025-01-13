---
title: "What is the difference between wandb.init modes?"
toc_hide: true
type: docs
tags:
   - experiments
---
These modes are available:

* `online` (default): The client sends data to the wandb server.
* `offline`: The client stores data locally on the machine instead of sending it to the wandb server. Use the [`wandb sync`](../ref/cli/wandb-sync.md) command to synchronize the data later.
* `disabled`: The client simulates operation by returning mocked objects and prevents any network communication. All logging is turned off, but all API method stubs remain callable. This mode is typically used for testing.