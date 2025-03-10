---
menu:
  support:
    identifier: ko-support-runs_state_crashed_ui_running_machine_get_data
tags:
- experiments
title: My run's state is `crashed` on the UI but is still running on my machine. What
  do I do to get my data back?
toc_hide: true
type: docs
---

You likely lost connection to your machine during training. Recover data by running [`wandb sync [PATH_TO_RUN]`]({{< relref path="/ref/cli/wandb-sync.md" lang="ko" >}}). The path to your run is a folder in your `wandb` directory that matches the Run ID of the ongoing run.