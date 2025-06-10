---
url: /support/:filename
title: "How do I get the random run name in my script?"
toc_hide: true
type: docs
support:
   - experiments
---
Call `wandb.run.save()` to save the current run. Retrieve the name using `wandb.run.name`.