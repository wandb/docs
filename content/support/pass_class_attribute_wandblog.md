---
title: "What happens if I pass a class attribute into wandb.log()?"
toc_hide: true
type: docs
tags:
   - experiments
---
Avoid passing class attributes into `wandb.log()`. Attributes may change before the network call executes. When storing metrics as class attributes, use a deep copy to ensure the logged metric matches the attribute's value at the time of the `wandb.log()` call.