---
title: "How do I get the random run name in my script?"
displayed_sidebar: support
tags:
   - experiments
---
Call `wandb.run.save()` to save the current run. Retrieve the name using `wandb.run.name`.