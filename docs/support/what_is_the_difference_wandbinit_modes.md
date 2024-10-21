---
title: "What is the difference between wandb.init modes?"
tags: []
---

### What is the difference between wandb.init modes?
Modes can be "online", "offline" or "disabled", and default to online.

`online`(default): In this mode, the client sends data to the wandb server.

`offline`: In this mode, instead of sending data to the wandb server, the client will store data on your local machine which can be later synced with the [`wandb sync`](../../ref/cli/wandb-sync.md) command.

`disabled`: In this mode, the client returns mocked objects and prevents all network communication. The client will essentially act like a no-op. In other words, all logging is entirely disabled. However, stubs out of all the API methods are still callable. This is usually used in tests.