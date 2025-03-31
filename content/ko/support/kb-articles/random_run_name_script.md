---
menu:
  support:
    identifier: ko-support-kb-articles-random_run_name_script
support:
- experiments
title: How do I get the random run name in my script?
toc_hide: true
type: docs
url: /support/:filename
---

Call `wandb.run.save()` to save the current run. Retrieve the name using `wandb.run.name`.