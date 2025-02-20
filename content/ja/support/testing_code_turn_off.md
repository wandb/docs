---
menu:
  support:
    identifier: ja-support-testing_code_turn_off
tags:
- artifacts
title: Can I turn off wandb when testing my code?
toc_hide: true
type: docs
---

Use `wandb.init(mode="disabled")` or set `WANDB_MODE=disabled` to configure W&B as a no-operation (NOOP) for testing purposes.

{{% alert %}}
Using `wandb.init(mode="disabled")` does not prevent W&B from saving artifacts to `WANDB_CACHE_DIR`.
{{% /alert %}}