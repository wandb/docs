---
menu:
  support:
    identifier: ko-support-kb-articles-run_sweeps_slurm
support:
- sweeps
title: How should I run sweeps on SLURM?
toc_hide: true
type: docs
url: /support/:filename
---

When using sweeps with the [SLURM scheduling system](https://slurm.schedmd.com/documentation.html), run `wandb agent --count 1 SWEEP_ID` in each scheduled job. This command executes a single training job and then exits, facilitating runtime predictions for resource requests while leveraging the parallelism of hyperparameter searches.