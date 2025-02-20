---
menu:
  support:
    identifier: ko-support-random_run_name_script
tags:
- experiments
title: How do I get the random run name in my script?
toc_hide: true
type: docs
---

Call `wandb.run.save()` to save the current run. Retrieve the name using `wandb.run.name`.