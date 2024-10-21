---
title: "What happens if I pass a class attribute into wandb.log()?"
tags:
   - experiments
---

It is generally not recommended to pass class attributes into `wandb.log()` as the attribute may change before the network call is made. If you are storing metrics as the attribute of a class, it is recommended to deep copy the attribute to ensure the metric logged matches the value of the attribute at the time that `wandb.log()` was called.