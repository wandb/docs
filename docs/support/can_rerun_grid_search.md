---
title: "Can I rerun a grid search?"
displayed_sidebar: support
tags:
   - sweeps
---
If a grid search completes but some W&B Runs need re-execution due to crashes, delete the specific W&B Runs to re-run. Then, select the **Resume** button on the [sweep control page](../guides/sweeps/sweeps-ui.md). Start new W&B Sweep agents using the new Sweep ID.

W&B Run parameter combinations that completed are not re-executed.