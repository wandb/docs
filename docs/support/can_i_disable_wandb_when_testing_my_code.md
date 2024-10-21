---
title: "Can I disable wandb when testing my code?"
tags: []
---

### Can I disable wandb when testing my code?
By using `wandb.init(mode="disabled")` or by setting `WANDB_MODE=disabled` you will make wandb act like a NOOP which is perfect for testing your code.

**Note**: Setting `wandb.init(mode=“disabled”)` does not prevent `wandb` from saving artifacts to `WANDB_CACHE_DIR`