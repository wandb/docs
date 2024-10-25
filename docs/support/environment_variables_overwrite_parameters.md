---
title: Do environment variables overwrite the parameters passed to wandb.init()?
tags:
- environment variables
---
Arguments passed to `wandb.init` take precedence over the environment. You could call `wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))` if you want to have a default other than the system default when the environment variable isn't set.