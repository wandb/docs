---
title: "Can I disable wandb when testing my code?"
tags:
   - None
---
Use `wandb.init(mode="disabled")` or set `WANDB_MODE=disabled` to configure W&B as a no-operation (NOOP) for testing purposes.

**Note**: Using `wandb.init(mode="disabled")` does not prevent W&B from saving artifacts to `WANDB_CACHE_DIR`.