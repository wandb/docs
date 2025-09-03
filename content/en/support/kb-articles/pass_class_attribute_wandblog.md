---
url: /support/:filename
title: "What happens if I pass a class attribute into wandb.Run.log()?"
toc_hide: true
type: docs
support:
   - experiments
---
Avoid passing class attributes into `wandb.Run.log()`. Attributes may change before the network call executes. When storing metrics as class attributes, use a deep copy to ensure the logged metric matches the attribute's value at the time of the `wandb.Run.log()` call.