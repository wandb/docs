---
title: "Can I turn off wandb when testing my code?"
displayed_sidebar: support
tags:
   - artifacts
---
Use `wandb.init(mode="disabled")` or set `WANDB_MODE=disabled` to configure W&B as a no-operation (NOOP) for testing purposes.

:::note
Using `wandb.init(mode="disabled")` does not prevent W&B from saving artifacts to `WANDB_CACHE_DIR`.
:::