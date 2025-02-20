---
menu:
  support:
    identifier: ko-support-run_finished_alerts
tags:
- alerts
- notebooks
title: Do "Run Finished" alerts work in notebooks?
toc_hide: true
type: docs
---

No. **Run Finished** alerts (activated with the **Run Finished** setting in User Settings) operate only with Python scripts and remain turned off in Jupyter Notebook environments to avoid notifications for each cell execution. 

Use `wandb.alert()` in notebook environments instead.