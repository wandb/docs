---
menu:
  support:
    identifier: ko-support-kb-articles-rerun_grid_search
support:
- sweeps
- hyperparameter
- runs
title: Can I rerun a grid search?
toc_hide: true
type: docs
url: /support/:filename
---

If a grid search completes but some W&B Runs need re-execution due to crashes, delete the specific W&B Runs to re-run. Then, select the **Resume** button on the [sweep control page]({{< relref path="/guides/models/sweeps/sweeps-ui.md" lang="ko" >}}). Start new W&B Sweep agents using the new Sweep ID.

W&B Run parameter combinations that completed are not re-executed.