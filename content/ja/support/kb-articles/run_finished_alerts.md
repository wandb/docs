---
menu:
  support:
    identifier: ja-support-kb-articles-run_finished_alerts
support:
- alerts
- notebooks
title: Do "Run Finished" alerts work in notebooks?
toc_hide: true
type: docs
url: /support/:filename
---

No. **Run Finished** alerts (activated with the **Run Finished** setting in User Settings) operate only with Python scripts and remain turned off in Jupyter Notebook environments to avoid notifications for each cell execution. 

Use `run.alert()` in notebook environments instead.