---
tite: Do "Run Finished" alerts work in notebooks?
displayed_sidebar: support
tags:
- alerts
- notebooks
---
No. **Run Finished** alerts (turned on with the **Run Finished** setting in User Settings) only work with Python scripts and are turned off in Jupyter Notebook environments to prevent alert notifications on every cell execution. 

Use `wandb.alert()` in notebook environments instead.
