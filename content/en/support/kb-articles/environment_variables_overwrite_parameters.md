---
url: /support/:filename
title: Do environment variables overwrite the parameters passed to wandb.init()?
toc_hide: true
type: docs
support:
- environment variables
---
Arguments passed to `wandb.init` override environment variables. To set a default directory other than the system default when the environment variable isn't set, use `wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))`.