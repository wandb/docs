---
title: "What is the `Est. Runs` column?"
displayed_sidebar: support
tags:
   - sweeps
   - hyperparameter
---
W&B provides an estimated number of Runs generated when creating a W&B Sweep with a discrete search space. This total reflects the cartesian product of the search space.

For instance, consider the following search space:

![](/images/sweeps/sweeps_faq_whatisestruns_1.png)

In this case, the Cartesian product equals 9. W&B displays this value in the App UI as the estimated run count (**Est. Runs**):

![](/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp)

To retrieve the estimated Run count programmatically, use the `expected_run_count` attribute of the Sweep object within the W&B SDK:

```python
sweep_id = wandb.sweep(
    sweep_configs, project="your_project_name", entity="your_entity_name"
)
api = wandb.Api()
sweep = api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```